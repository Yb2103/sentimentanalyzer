"""
E-Consultation Sentiment Analysis Package
Core modules for sentiment analysis, text summarization, and visualization
"""

from .sentiment_analyzer import SentimentAnalyzer, quick_sentiment
from .text_summarizer import TextSummarizer
from .wordcloud_generator import WordCloudGenerator
from .data_processor import DataProcessor, validate_data

__version__ = "1.0.0"
__all__ = [
    'SentimentAnalyzer',
    'TextSummarizer', 
    'WordCloudGenerator',
    'DataProcessor',
    'quick_sentiment',
    'validate_data'
]