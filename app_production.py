"""
E-Consultation Sentiment Analysis Web Application
Production version for cloud deployment
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime
import json
import traceback
from pathlib import Path

# Import our modules
from src import (
    SentimentAnalyzer,
    TextSummarizer,
    WordCloudGenerator,
    DataProcessor,
    validate_data
)

# Production Configuration
class ProductionConfig:
    """Production configuration for cloud deployment"""
    
    # Basic Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'prod-secret-key-change-this-in-production-2024'
    DEBUG = False
    
    # Directories - use temp directories in production
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or str(BASE_DIR / 'uploads')
    RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER') or str(BASE_DIR / 'results')
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'txt'}
    
    # Analysis settings
    DEFAULT_SENTIMENT_METHOD = 'ensemble'
    MAX_SAMPLE_SIZE = 10000
    
    @staticmethod
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ProductionConfig.ALLOWED_EXTENSIONS

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(ProductionConfig)

# Create required directories
for directory in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(directory, exist_ok=True)

# Initialize analyzers
sentiment_analyzer = SentimentAnalyzer(method=app.config['DEFAULT_SENTIMENT_METHOD'])
text_summarizer = TextSummarizer()
wordcloud_generator = WordCloudGenerator()
data_processor = DataProcessor()

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and trigger analysis"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        # Check if file type is allowed
        if not ProductionConfig.allowed_file(file.filename):
            flash('Invalid file type. Please upload CSV, Excel, JSON, or TXT file', 'error')
            return redirect(url_for('index'))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file
        try:
            # Load and clean data
            df = data_processor.load_data(filepath)
            text_column = request.form.get('text_column', 'comment_text')
            
            # Validate data
            is_valid, issues = validate_data(df, text_column)
            if not is_valid and len(issues) > 0:
                flash(f"Data validation issues: {', '.join(issues)}", 'warning')
            
            # Prepare data for analysis
            df_prepared = data_processor.prepare_for_analysis(
                df, 
                text_column,
                sample_size=app.config['MAX_SAMPLE_SIZE']
            )
            
            # Store data in session for analysis
            session_data = {
                'filename': filename,
                'filepath': filepath,
                'text_column': text_column,
                'row_count': len(df_prepared),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save prepared data for analysis
            prepared_filepath = filepath.replace(os.path.splitext(filepath)[1], '_prepared.csv')
            df_prepared.to_csv(prepared_filepath, index=False)
            session_data['prepared_filepath'] = prepared_filepath
            
            # Save session data
            session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_session.json")
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
            
            return redirect(url_for('analyze', session_id=timestamp))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Upload error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/analyze/<session_id>')
def analyze(session_id):
    """Perform analysis and display results"""
    try:
        # Load session data
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_session.json")
        if not os.path.exists(session_file):
            flash('Session not found. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Load prepared data
        df = pd.read_csv(session_data['prepared_filepath'])
        text_column = session_data.get('text_column', 'comment_text')
        
        # Check if analysis already exists
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_results.json")
        
        if not os.path.exists(results_file):
            # Perform sentiment analysis
            df_with_sentiment = sentiment_analyzer.analyze_dataframe(df, text_column)
            
            # Get aggregate sentiment
            sentiment_results = []
            for _, row in df_with_sentiment.iterrows():
                sentiment_results.append(sentiment_analyzer.analyze_text(str(row[text_column])))
            
            aggregate_sentiment = sentiment_analyzer.get_aggregate_sentiment(sentiment_results)
            
            # Generate summaries
            texts = df[text_column].tolist()
            collective_summary = text_summarizer.generate_collective_summary(texts, num_points=5)
            thematic_categories = text_summarizer.categorize_by_theme(texts)
            
            # Generate word cloud
            keywords = wordcloud_generator.extract_keywords(texts, top_n=20)
            combined_text = ' '.join(texts)
            wordcloud = wordcloud_generator.generate_wordcloud(combined_text)
            wordcloud_base64 = wordcloud_generator.wordcloud_to_base64(wordcloud)
            
            # Generate sentiment-based word clouds
            sentiment_clouds = {}
            for sentiment in ['positive', 'negative', 'neutral']:
                sentiment_texts = df_with_sentiment[
                    df_with_sentiment['sentiment'] == sentiment
                ][text_column].tolist()
                if sentiment_texts:
                    sentiment_text = ' '.join([str(t) for t in sentiment_texts])
                    cloud = wordcloud_generator.generate_wordcloud(sentiment_text)
                    sentiment_clouds[sentiment] = wordcloud_generator.wordcloud_to_base64(cloud)
            
            # Prepare results
            results = {
                'session_id': session_id,
                'filename': session_data['filename'],
                'total_comments': len(df),
                'analysis_timestamp': datetime.now().isoformat(),
                'aggregate_sentiment': aggregate_sentiment,
                'collective_summary': collective_summary,
                'thematic_categories': thematic_categories,
                'keywords': keywords,
                'wordcloud': wordcloud_base64,
                'sentiment_clouds': sentiment_clouds,
                'sample_results': df_with_sentiment.head(10).to_dict('records')
            }
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump(results, f)
            
            # Save full results to Excel
            excel_file = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_full_results.xlsx")
            df_with_sentiment.to_excel(excel_file, index=False, engine='openpyxl')
            
        else:
            # Load existing results
            with open(results_file, 'r') as f:
                results = json.load(f)
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        flash(f'Analysis error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download/<session_id>')
def download_results(session_id):
    """Download analysis results"""
    try:
        # Check which format was requested
        format_type = request.args.get('format', 'excel')
        
        if format_type == 'excel':
            filepath = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_full_results.xlsx")
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            download_name = f"sentiment_analysis_results_{session_id}.xlsx"
        elif format_type == 'json':
            filepath = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_results.json")
            mimetype = 'application/json'
            download_name = f"sentiment_analysis_results_{session_id}.json"
        else:
            flash('Invalid download format', 'error')
            return redirect(url_for('index'))
        
        if not os.path.exists(filepath):
            flash('Results file not found', 'error')
            return redirect(url_for('index'))
        
        return send_file(
            filepath,
            mimetype=mimetype,
            as_attachment=True,
            download_name=download_name
        )
        
    except Exception as e:
        flash(f'Download error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic analysis"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'comments' not in data:
            return jsonify({'error': 'No comments provided'}), 400
        
        comments = data['comments']
        method = data.get('method', app.config['DEFAULT_SENTIMENT_METHOD'])
        
        # Initialize analyzer with specified method
        analyzer = SentimentAnalyzer(method=method)
        
        # Analyze comments
        results = []
        for comment in comments:
            if isinstance(comment, str):
                result = analyzer.analyze_text(comment)
                results.append({
                    'text': result.text,
                    'sentiment': result.sentiment_label,
                    'confidence': result.confidence,
                    'scores': {
                        'positive': result.positive,
                        'negative': result.negative,
                        'neutral': result.neutral,
                        'compound': result.compound
                    }
                })
        
        # Get aggregate results
        sentiment_results = [analyzer.analyze_text(c) for c in comments if isinstance(c, str)]
        aggregate = analyzer.get_aggregate_sentiment(sentiment_results)
        
        # Generate summary
        summarizer = TextSummarizer()
        collective_summary = summarizer.generate_collective_summary(comments)
        
        return jsonify({
            'results': results,
            'aggregate': aggregate,
            'summary': collective_summary
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample')
def sample_analysis():
    """Run analysis on sample data"""
    try:
        # Load sample data
        sample_file = os.path.join('data', 'sample', 'sample_comments.csv')
        
        if not os.path.exists(sample_file):
            flash('Sample data not found', 'error')
            return redirect(url_for('index'))
        
        # Create a session for sample analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_id = f"sample_{timestamp}"
        
        # Copy sample file to uploads
        import shutil
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}.csv")
        shutil.copy(sample_file, upload_path)
        
        # Create session data
        session_data = {
            'filename': 'sample_comments.csv',
            'filepath': upload_path,
            'text_column': 'comment_text',
            'prepared_filepath': upload_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save session
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_session.json")
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
        
        return redirect(url_for('analyze', session_id=session_id))
        
    except Exception as e:
        flash(f'Error loading sample data: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

# Health check endpoint for cloud platforms
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # For production, this is handled by the WSGI server
    # This is only for local testing
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)