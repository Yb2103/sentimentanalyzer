#!/usr/bin/env python3
"""
Test deployment configuration
Verify that the application can start and basic functionality works
"""

import sys
import traceback
import requests
import time
import subprocess
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import flask
        import pandas
        import numpy
        import nltk
        import transformers
        import torch
        import textblob
        import vaderSentiment
        import wordcloud
        import matplotlib
        import plotly
        import seaborn
        import openpyxl
        import beautifulsoup4
        import lxml
        import gunicorn
        
        # Test our custom modules
        from src import SentimentAnalyzer, TextSummarizer, WordCloudGenerator, DataProcessor
        
        print("✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_nltk_data():
    """Test that NLTK data is available"""
    print("Testing NLTK data...")
    try:
        import nltk
        
        # Test punkt tokenizer
        nltk.data.find('tokenizers/punkt')
        print("✓ Punkt tokenizer available")
        
        # Test VADER lexicon
        nltk.data.find('vader_lexicon')
        print("✓ VADER lexicon available")
        
        # Test stopwords
        nltk.data.find('corpora/stopwords')
        print("✓ Stopwords available")
        
        print("✓ All NLTK data available!")
        return True
        
    except Exception as e:
        print(f"✗ NLTK data error: {e}")
        return False

def test_sentiment_analysis():
    """Test basic sentiment analysis functionality"""
    print("Testing sentiment analysis...")
    try:
        from src import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(method='vader')
        result = analyzer.analyze_text("This is a great project!")
        
        print(f"✓ Sentiment analysis working: {result.sentiment_label} (confidence: {result.confidence:.2f})")
        return True
        
    except Exception as e:
        print(f"✗ Sentiment analysis error: {e}")
        traceback.print_exc()
        return False

def test_flask_app():
    """Test that Flask app can be initialized"""
    print("Testing Flask app initialization...")
    try:
        # Import and create the app
        from app_production import app
        
        # Test basic route
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ Health check endpoint working")
                return True
            else:
                print(f"✗ Health check failed with status {response.status_code}")
                return False
                
    except Exception as e:
        print(f"✗ Flask app error: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all deployment tests"""
    print("=" * 50)
    print("DEPLOYMENT READINESS TESTS")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("NLTK Data Test", test_nltk_data),
        ("Sentiment Analysis Test", test_sentiment_analysis),
        ("Flask App Test", test_flask_app)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        try:
            success = test_func()
            results.append(success)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Check issues before deploying.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)