"""
Simple Sentiment Analysis Web Application for debugging
"""

from flask import Flask, render_template, request, jsonify
import os
from src.sentiment_analyzer import SentimentAnalyzer

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-key-123'

# Initialize analyzer
sentiment_analyzer = SentimentAnalyzer(method='vader')

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for sentiment analysis"""
    try:
        # Get JSON data
        data = request.get_json()
        print(f"Received data: {data}")
        
        if not data or 'comments' not in data:
            return jsonify({'error': 'No comments provided'}), 400
        
        comments = data['comments']
        if not comments or len(comments) == 0:
            return jsonify({'error': 'Empty comments list'}), 400
            
        # Analyze comments
        results = []
        for comment in comments:
            if comment and isinstance(comment, str):
                result = sentiment_analyzer.analyze_text(comment)
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
        
        if not results:
            return jsonify({'error': 'No valid comments to analyze'}), 400
            
        # Get aggregate results
        sentiment_results = [sentiment_analyzer.analyze_text(c) for c in comments if isinstance(c, str)]
        aggregate = sentiment_analyzer.get_aggregate_sentiment(sentiment_results)
        
        response = {
            'results': results,
            'aggregate': aggregate,
            'summary': 'Analysis complete'
        }
        
        print(f"Sending response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in API: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    # Create required directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "="*50)
    print("Simple Sentiment Analysis App")
    print("="*50)
    print("\nEndpoints:")
    print("  Homepage: http://localhost:5000")
    print("  Test API: http://localhost:5000/test")
    print("  Analysis API: POST http://localhost:5000/api/analyze")
    print("\nPress Ctrl+C to stop")
    print("="*50 + "\n")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)