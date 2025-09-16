"""
Sentiment Analysis Web Application with Real-Time Features
Flask-SocketIO based web interface with live sentiment tracking
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_socketio import SocketIO, emit
from flask_compress import Compress
from flask_caching import Cache
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime
import json
import traceback
import asyncio
from pathlib import Path
import time
from threading import Thread

# Import our modules
from src import (
    SentimentAnalyzer,
    TextSummarizer,
    WordCloudGenerator,
    DataProcessor,
    validate_data
)
from src.async_sentiment_analyzer import AsyncSentimentAnalyzer
from config import Config, ANALYSIS_CONFIG

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
compress = Compress(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 300})

# Initialize analyzers
sentiment_analyzer = SentimentAnalyzer(method=Config.DEFAULT_SENTIMENT_METHOD)
async_analyzer = AsyncSentimentAnalyzer(method=Config.DEFAULT_SENTIMENT_METHOD)
text_summarizer = TextSummarizer()
wordcloud_generator = WordCloudGenerator()
data_processor = DataProcessor()

# Track active sessions
active_sessions = {}

@app.route('/')
@cache.cached(timeout=3600)
def index():
    """Home page with upload form - cached for 1 hour"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and trigger analysis with real-time progress"""
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
            method = request.form.get('method', Config.DEFAULT_SENTIMENT_METHOD)
            
            # Validate data
            is_valid, issues = validate_data(df, text_column)
            if not is_valid and len(issues) > 0:
                flash(f"Data validation issues: {', '.join(issues)}", 'warning')
            
            # Prepare data for analysis
            df_prepared = data_processor.prepare_for_analysis(
                df, 
                text_column,
                sample_size=Config.MAX_SAMPLE_SIZE
            )
            
            # Store data in session for analysis
            session_data = {
                'filename': filename,
                'filepath': filepath,
                'text_column': text_column,
                'method': method,
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
            
            # Store in active sessions
            active_sessions[timestamp] = {
                'status': 'uploaded',
                'progress': 0,
                'session_data': session_data
            }
            
            return redirect(url_for('analyze', session_id=timestamp))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Upload error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/analyze/<session_id>')
def analyze(session_id):
    """Perform analysis with real-time progress updates"""
    try:
        # Load session data
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_session.json")
        if not os.path.exists(session_file):
            flash('Session not found. Please upload a file first.', 'error')
            return redirect(url_for('index'))
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Check if analysis already exists
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_results.json")
        
        if not os.path.exists(results_file):
            # Start async analysis in background
            thread = Thread(target=run_async_analysis, args=(session_id, session_data))
            thread.daemon = True
            thread.start()
        
        return render_template('results_realtime.html', session_id=session_id, session_data=session_data)
        
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        flash(f'Analysis error: {str(e)}', 'error')
        return redirect(url_for('index'))

def run_async_analysis(session_id, session_data):
    """Run analysis asynchronously with progress updates"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_analysis_worker(session_id, session_data))
    loop.close()

async def async_analysis_worker(session_id, session_data):
    """Async worker for sentiment analysis with progress updates"""
    try:
        # Update status
        active_sessions[session_id] = {'status': 'processing', 'progress': 0}
        socketio.emit('progress', {'session_id': session_id, 'progress': 0, 'status': 'Loading data...'})
        
        # Load prepared data
        df = pd.read_csv(session_data['prepared_filepath'])
        text_column = session_data.get('text_column', 'comment_text')
        method = session_data.get('method', 'ensemble')
        
        # Initialize async analyzer
        analyzer = AsyncSentimentAnalyzer(method=method)
        
        # Analyze with progress updates
        texts = df[text_column].fillna('').astype(str).tolist()
        total_texts = len(texts)
        results = []
        
        batch_size = 10
        for i in range(0, total_texts, batch_size):
            batch = texts[i:min(i + batch_size, total_texts)]
            batch_results = await analyzer.analyze_batch(batch, batch_size=batch_size)
            results.extend(batch_results)
            
            # Update progress
            progress = int((len(results) / total_texts) * 50)  # 50% for sentiment analysis
            active_sessions[session_id]['progress'] = progress
            socketio.emit('progress', {
                'session_id': session_id, 
                'progress': progress, 
                'status': f'Analyzing sentiments... ({len(results)}/{total_texts})'
            })
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.1)
        
        # Add results to DataFrame
        df['sentiment'] = [r.sentiment_label for r in results]
        df['sentiment_positive'] = [r.positive for r in results]
        df['sentiment_negative'] = [r.negative for r in results]
        df['sentiment_neutral'] = [r.neutral for r in results]
        df['sentiment_compound'] = [r.compound for r in results]
        df['sentiment_confidence'] = [r.confidence for r in results]
        
        # Get aggregate sentiment
        aggregate_sentiment = analyzer.get_aggregate_sentiment(results)
        
        socketio.emit('progress', {
            'session_id': session_id, 
            'progress': 60, 
            'status': 'Generating summaries...'
        })
        
        # Generate summaries
        collective_summary = text_summarizer.generate_collective_summary(texts, num_points=5)
        thematic_categories = text_summarizer.categorize_by_theme(texts[:100])  # Limit for performance
        
        socketio.emit('progress', {
            'session_id': session_id, 
            'progress': 75, 
            'status': 'Creating visualizations...'
        })
        
        # Generate word clouds
        keywords = wordcloud_generator.extract_keywords(texts, top_n=20)
        combined_text = ' '.join(texts[:500])  # Limit for performance
        wordcloud = wordcloud_generator.generate_wordcloud(combined_text)
        wordcloud_base64 = wordcloud_generator.wordcloud_to_base64(wordcloud)
        
        # Generate sentiment-based word clouds
        sentiment_clouds = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_texts = df[df['sentiment'] == sentiment][text_column].tolist()[:100]
            if sentiment_texts:
                sentiment_text = ' '.join([str(t) for t in sentiment_texts])
                cloud = wordcloud_generator.generate_wordcloud(
                    sentiment_text,
                    colormap=ANALYSIS_CONFIG['wordcloud']['colormaps'][sentiment]
                )
                sentiment_clouds[sentiment] = wordcloud_generator.wordcloud_to_base64(cloud)
        
        socketio.emit('progress', {
            'session_id': session_id, 
            'progress': 90, 
            'status': 'Finalizing results...'
        })
        
        # Calculate processing stats
        avg_processing_time = sum(r.processing_time for r in results) / len(results) if results else 0
        
        # Prepare results
        results_data = {
            'session_id': session_id,
            'filename': session_data['filename'],
            'total_comments': len(df),
            'analysis_timestamp': datetime.now().isoformat(),
            'method': method,
            'aggregate_sentiment': aggregate_sentiment,
            'collective_summary': collective_summary,
            'thematic_categories': thematic_categories,
            'keywords': keywords,
            'wordcloud': wordcloud_base64,
            'sentiment_clouds': sentiment_clouds,
            'sample_results': df.head(10).to_dict('records'),
            'processing_stats': {
                'avg_time_per_text': avg_processing_time,
                'total_time': sum(r.processing_time for r in results),
                'cache_hits': len([r for r in results if r.processing_time < 0.001])
            }
        }
        
        # Save results
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_data, f)
        
        # Save full results to Excel
        excel_file = os.path.join(app.config['RESULTS_FOLDER'], f"{session_id}_full_results.xlsx")
        df.to_excel(excel_file, index=False, engine='openpyxl')
        
        # Update status
        active_sessions[session_id] = {'status': 'completed', 'progress': 100}
        socketio.emit('progress', {
            'session_id': session_id, 
            'progress': 100, 
            'status': 'Analysis complete!'
        })
        
        # Emit results
        socketio.emit('results_ready', {
            'session_id': session_id,
            'results': results_data
        })
        
    except Exception as e:
        app.logger.error(f"Async analysis error: {str(e)}\n{traceback.format_exc()}")
        active_sessions[session_id] = {'status': 'error', 'error': str(e)}
        socketio.emit('analysis_error', {
            'session_id': session_id,
            'error': str(e)
        })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to real-time analysis server'})

@socketio.on('get_progress')
def handle_get_progress(data):
    """Get current progress for a session"""
    session_id = data.get('session_id')
    if session_id in active_sessions:
        session = active_sessions[session_id]
        emit('progress', {
            'session_id': session_id,
            'progress': session.get('progress', 0),
            'status': session.get('status', 'Unknown')
        })

@socketio.on('analyze_live')
def handle_live_analysis(data):
    """Handle live text analysis via WebSocket"""
    text = data.get('text', '')
    method = data.get('method', 'ensemble')
    
    if not text:
        emit('live_result', {'error': 'No text provided'})
        return
    
    try:
        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        analyzer = AsyncSentimentAnalyzer(method=method)
        result = loop.run_until_complete(analyzer.analyze_text(text))
        
        emit('live_result', {
            'text': result.text,
            'sentiment': result.sentiment_label,
            'confidence': result.confidence,
            'scores': {
                'positive': result.positive,
                'negative': result.negative,
                'neutral': result.neutral,
                'compound': result.compound
            },
            'processing_time': result.processing_time
        })
        
        loop.close()
        
    except Exception as e:
        emit('live_result', {'error': str(e)})

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
@cache.cached(timeout=60, key_prefix='api_analyze', 
              make_cache_key=lambda: json.dumps(request.get_json()))
def api_analyze():
    """API endpoint for programmatic analysis - cached for 1 minute"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'comments' not in data:
            return jsonify({'error': 'No comments provided'}), 400
        
        comments = data['comments']
        method = data.get('method', Config.DEFAULT_SENTIMENT_METHOD)
        
        # Run async analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        analyzer = AsyncSentimentAnalyzer(method=method)
        results_coro = analyzer.analyze_batch(comments)
        results = loop.run_until_complete(results_coro)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result.text,
                'sentiment': result.sentiment_label,
                'confidence': result.confidence,
                'scores': {
                    'positive': result.positive,
                    'negative': result.negative,
                    'neutral': result.neutral,
                    'compound': result.compound
                },
                'processing_time': result.processing_time
            })
        
        # Get aggregate results
        aggregate = analyzer.get_aggregate_sentiment(results)
        
        # Generate summary
        summarizer = TextSummarizer()
        collective_summary = summarizer.generate_collective_summary(comments)
        
        loop.close()
        
        return jsonify({
            'results': formatted_results,
            'aggregate': aggregate,
            'summary': collective_summary,
            'performance': {
                'total_processing_time': sum(r.processing_time for r in results),
                'avg_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sample')
def sample_analysis():
    """Run analysis on sample data"""
    try:
        # Create sample data if it doesn't exist
        sample_file = os.path.join('data', 'sample', 'sample_comments.csv')
        
        if not os.path.exists(sample_file):
            # Create sample data
            os.makedirs(os.path.dirname(sample_file), exist_ok=True)
            sample_data = pd.DataFrame({
                'comment_id': range(1, 21),
                'comment_text': [
                    "This product is absolutely amazing! Best purchase I've ever made.",
                    "Terrible experience. Would not recommend to anyone.",
                    "It's okay, nothing special but does the job.",
                    "Outstanding service and quality! Exceeded my expectations.",
                    "Completely disappointed. Waste of money.",
                    "Average product. Some good features, some bad.",
                    "Love it! Works perfectly and looks great too.",
                    "Not worth the price. Many better alternatives available.",
                    "Decent quality for the price point.",
                    "Fantastic! Highly recommended to everyone.",
                    "Poor quality and bad customer service.",
                    "It serves its purpose adequately.",
                    "Brilliant design and excellent functionality!",
                    "Regret buying this. Full of problems.",
                    "Acceptable but room for improvement.",
                    "Superb! Exactly what I was looking for.",
                    "Disappointing. Did not meet expectations.",
                    "Fair product with pros and cons.",
                    "Incredible value! So happy with this purchase.",
                    "Would return if I could. Not satisfied."
                ],
                'timestamp': pd.date_range(start='2024-01-01', periods=20, freq='D')
            })
            sample_data.to_csv(sample_file, index=False)
        
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
            'method': 'ensemble',
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

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f"Internal error: {str(error)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create required directories
    for directory in [Config.UPLOAD_FOLDER, Config.RESULTS_FOLDER, 'data/sample']:
        os.makedirs(directory, exist_ok=True)
    
    # Run the application with SocketIO
    socketio.run(app, debug=Config.DEBUG, host='0.0.0.0', port=5000)