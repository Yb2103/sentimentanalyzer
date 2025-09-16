"""
E-Consultation Sentiment Analysis Web Application - Standalone Version
Completely self-contained Flask app for Render deployment
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime
import json
import traceback
from pathlib import Path
import re
from collections import Counter

# Basic sentiment analysis imports
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    import nltk
    SENTIMENT_AVAILABLE = True
    
    # Download required NLTK data on startup
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
except ImportError as e:
    print(f"Warning: Sentiment analysis not available: {e}")
    SENTIMENT_AVAILABLE = False

# Configuration
class AppConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = False
    
    # Directories
    BASE_DIR = Path(__file__).resolve().parent
    UPLOAD_FOLDER = str(BASE_DIR / 'uploads')
    RESULTS_FOLDER = str(BASE_DIR / 'results')
    
    # File settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'txt'}
    MAX_SAMPLE_SIZE = 1000  # Reduced for free tier
    
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in AppConfig.ALLOWED_EXTENSIONS

# Simple sentiment analyzer
class SimpleSentimentAnalyzer:
    def __init__(self):
        if SENTIMENT_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
    
    def analyze_text(self, text):
        if not SENTIMENT_AVAILABLE or not self.vader:
            return {
                'text': str(text),
                'sentiment': 'neutral',
                'confidence': 0.0,
                'compound': 0.0
            }
        
        if not text or pd.isna(text):
            return {
                'text': '',
                'sentiment': 'neutral',
                'confidence': 0.0,
                'compound': 0.0
            }
        
        text = str(text).strip()
        if not text:
            return {
                'text': '',
                'sentiment': 'neutral',
                'confidence': 0.0,
                'compound': 0.0
            }
        
        # VADER analysis
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        # Determine sentiment
        if compound >= 0.05:
            sentiment = 'positive'
            confidence = compound
        elif compound <= -0.05:
            sentiment = 'negative'
            confidence = abs(compound)
        else:
            sentiment = 'neutral'
            confidence = 1 - abs(compound)
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'compound': compound
        }
    
    def analyze_dataframe(self, df, text_column):
        results = []
        for _, row in df.iterrows():
            result = self.analyze_text(row[text_column])
            results.append(result)
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['sentiment'] = [r['sentiment'] for r in results]
        df_result['confidence'] = [r['confidence'] for r in results]
        df_result['compound'] = [r['compound'] for r in results]
        
        return df_result
    
    def get_aggregate_sentiment(self, results):
        if not results:
            return {'overall': 'neutral', 'distribution': {}, 'total_analyzed': 0}
        
        sentiments = [r['sentiment'] for r in results if 'sentiment' in r]
        if not sentiments:
            return {'overall': 'neutral', 'distribution': {}, 'total_analyzed': 0}
        
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        
        distribution = {k: round(v/total * 100, 1) for k, v in sentiment_counts.items()}
        overall = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'overall': overall,
            'distribution': distribution,
            'total_analyzed': total
        }

# Simple text processing functions
def extract_keywords(texts, top_n=20):
    if not SENTIMENT_AVAILABLE:
        return []
    
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    word_freq = Counter()
    
    for text in texts:
        if pd.isna(text):
            continue
        
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        word_freq.update(filtered_words)
    
    return [{'word': word, 'count': count} for word, count in word_freq.most_common(top_n)]

def generate_summary(texts, num_sentences=3):
    if not texts:
        return "No text available for summary."
    
    # Combine all texts
    combined = " ".join([str(t) for t in texts if pd.notna(t)])
    if not combined.strip():
        return "No valid text available for summary."
    
    # Simple sentence extraction
    sentences = re.split(r'[.!?]+', combined)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return "No sentences found for summary."
    
    if len(sentences) <= num_sentences:
        return ". ".join(sentences) + "."
    
    # Score sentences by length (basic approach)
    scored_sentences = [(s, len(s.split())) for s in sentences]
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    top_sentences = [s[0] for s in scored_sentences[:num_sentences]]
    return ". ".join(top_sentences) + "."

def load_data(filepath):
    """Load data from various file formats"""
    file_ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if file_ext == '.csv':
            return pd.read_csv(filepath, encoding='utf-8')
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        elif file_ext == '.json':
            return pd.read_json(filepath)
        elif file_ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return pd.DataFrame({'comment_text': [line.strip() for line in lines if line.strip()]})
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        # Try different encodings
        if file_ext == '.csv':
            try:
                return pd.read_csv(filepath, encoding='latin1')
            except:
                return pd.read_csv(filepath, encoding='cp1252')
        raise e

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(AppConfig)

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize analyzer
sentiment_analyzer = SimpleSentimentAnalyzer()

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'sentiment_available': SENTIMENT_AVAILABLE
    })

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
        if not AppConfig.allowed_file(file.filename):
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
            # Load data
            df = load_data(filepath)
            text_column = request.form.get('text_column', 'comment_text')
            
            # Check if text column exists
            if text_column not in df.columns:
                # Try to find a text-like column
                text_columns = [col for col in df.columns if any(word in col.lower() 
                               for word in ['text', 'comment', 'review', 'feedback', 'content'])]
                if text_columns:
                    text_column = text_columns[0]
                else:
                    text_column = df.columns[0]  # Use first column
            
            # Sample data if too large
            if len(df) > app.config['MAX_SAMPLE_SIZE']:
                df = df.sample(n=app.config['MAX_SAMPLE_SIZE']).reset_index(drop=True)
                flash(f'Large dataset sampled to {app.config["MAX_SAMPLE_SIZE"]} rows', 'info')
            
            # Store data for analysis
            session_data = {
                'filename': filename,
                'filepath': filepath,
                'text_column': text_column,
                'row_count': len(df),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save prepared data for analysis
            prepared_filepath = filepath.replace(os.path.splitext(filepath)[1], '_prepared.csv')
            df.to_csv(prepared_filepath, index=False)
            session_data['prepared_filepath'] = prepared_filepath
            
            # Save session data
            session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_session.json")
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
            
            return redirect(url_for('analyze', session_id=timestamp))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
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
            
            # Get analysis results
            analysis_results = []
            for _, row in df_with_sentiment.iterrows():
                result = sentiment_analyzer.analyze_text(str(row[text_column]))
                analysis_results.append(result)
            
            # Get aggregate sentiment
            aggregate_sentiment = sentiment_analyzer.get_aggregate_sentiment(analysis_results)
            
            # Generate summary and keywords
            texts = df[text_column].dropna().tolist()
            summary = generate_summary(texts, num_sentences=5)
            keywords = extract_keywords(texts, top_n=20)
            
            # Prepare results
            results = {
                'session_id': session_id,
                'filename': session_data['filename'],
                'total_comments': len(df),
                'analysis_timestamp': datetime.now().isoformat(),
                'aggregate_sentiment': aggregate_sentiment,
                'summary': summary,
                'keywords': keywords,
                'sample_results': df_with_sentiment.head(10).to_dict('records'),
                'sentiment_available': SENTIMENT_AVAILABLE
            }
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump(results, f)
            
            # Save full results to Excel if possible
            try:
                excel_file = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_full_results.xlsx")
                df_with_sentiment.to_excel(excel_file, index=False, engine='openpyxl')
            except Exception as e:
                print(f"Warning: Could not save Excel file: {e}")
            
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
        format_type = request.args.get('format', 'json')
        
        if format_type == 'excel':
            filepath = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_full_results.xlsx")
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            download_name = f"sentiment_analysis_results_{session_id}.xlsx"
        else:  # Default to JSON
            filepath = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_results.json")
            mimetype = 'application/json'
            download_name = f"sentiment_analysis_results_{session_id}.json"
        
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
        data = request.get_json()
        
        if not data or 'comments' not in data:
            return jsonify({'error': 'No comments provided'}), 400
        
        comments = data['comments']
        
        # Analyze comments
        results = []
        for comment in comments:
            if isinstance(comment, str):
                result = sentiment_analyzer.analyze_text(comment)
                results.append(result)
        
        # Get aggregate results
        aggregate = sentiment_analyzer.get_aggregate_sentiment(results)
        
        # Generate summary
        summary = generate_summary(comments)
        
        return jsonify({
            'results': results,
            'aggregate': aggregate,
            'summary': summary,
            'sentiment_available': SENTIMENT_AVAILABLE
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f"Internal error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # For production, this is handled by the WSGI server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)