"""
E-Consultation Sentiment Analysis Web Application - Minimal Version
Flask-based web interface for analyzing stakeholder comments
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime
import json
import traceback
from pathlib import Path

# Try to import our modules with graceful fallbacks
try:
    from src import SentimentAnalyzer, DataProcessor, validate_data
    HAS_SENTIMENT = True
except ImportError as e:
    print(f"Warning: Could not import sentiment analyzer: {e}")
    HAS_SENTIMENT = False

try:
    from src import TextSummarizer
    HAS_SUMMARIZER = True
except ImportError as e:
    print(f"Warning: Could not import text summarizer: {e}")
    HAS_SUMMARIZER = False

try:
    from src import WordCloudGenerator
    HAS_WORDCLOUD = True
except ImportError as e:
    print(f"Warning: Could not import wordcloud generator: {e}")
    HAS_WORDCLOUD = False

from config import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize analyzers only if available
if HAS_SENTIMENT:
    sentiment_analyzer = SentimentAnalyzer(method='vader')  # Use only VADER
    data_processor = DataProcessor()
else:
    sentiment_analyzer = None
    data_processor = None

if HAS_SUMMARIZER:
    text_summarizer = TextSummarizer()
else:
    text_summarizer = None

if HAS_WORDCLOUD:
    wordcloud_generator = WordCloudGenerator()
else:
    wordcloud_generator = None

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'sentiment_available': HAS_SENTIMENT,
        'summarizer_available': HAS_SUMMARIZER,
        'wordcloud_available': HAS_WORDCLOUD
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and trigger analysis"""
    try:
        if not HAS_SENTIMENT or not data_processor:
            flash('Sentiment analysis is not available due to missing dependencies', 'error')
            return redirect(url_for('index'))

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
        if not Config.allowed_file(file.filename):
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
                sample_size=min(1000, Config.MAX_SAMPLE_SIZE)  # Limit to 1000 for free tier
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
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Upload error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/analyze/<session_id>')
def analyze(session_id):
    """Perform analysis and display results"""
    try:
        if not HAS_SENTIMENT or not sentiment_analyzer:
            flash('Sentiment analysis is not available', 'error')
            return redirect(url_for('index'))

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
            
            # Generate summaries only if available
            texts = df[text_column].tolist()
            if HAS_SUMMARIZER and text_summarizer:
                collective_summary = text_summarizer.generate_collective_summary(texts, num_points=3)
                thematic_categories = text_summarizer.categorize_by_theme(texts)
            else:
                collective_summary = {'summary': 'Text summarization not available', 'key_points': []}
                thematic_categories = {}
            
            # Generate word cloud only if available
            if HAS_WORDCLOUD and wordcloud_generator:
                try:
                    keywords = wordcloud_generator.extract_keywords(texts, top_n=20)
                    combined_text = ' '.join(texts[:100])  # Limit text for wordcloud
                    wordcloud = wordcloud_generator.generate_wordcloud(combined_text)
                    wordcloud_base64 = wordcloud_generator.wordcloud_to_base64(wordcloud)
                    sentiment_clouds = {}
                except Exception as e:
                    print(f"Wordcloud generation failed: {e}")
                    keywords = []
                    wordcloud_base64 = None
                    sentiment_clouds = {}
            else:
                keywords = []
                wordcloud_base64 = None
                sentiment_clouds = {}
            
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
            
            # Save full results to Excel only if openpyxl is available
            try:
                excel_file = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_full_results.xlsx")
                df_with_sentiment.to_excel(excel_file, index=False, engine='openpyxl')
            except ImportError:
                print("Excel export not available - openpyxl not installed")
            
        else:
            # Load existing results
            with open(results_file, 'r') as f:
                results = json.load(f)
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        flash(f'Analysis error: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)