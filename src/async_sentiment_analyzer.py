"""
Async Sentiment Analysis Module for improved performance
Provides concurrent sentiment analysis with caching
"""

import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4)

@dataclass
class SentimentResult:
    """Data class for storing sentiment analysis results"""
    text: str
    positive: float
    negative: float
    neutral: float
    compound: float
    sentiment_label: str
    confidence: float
    processing_time: float = 0.0

class AsyncSentimentAnalyzer:
    """
    Async sentiment analyzer with caching for improved performance
    """
    
    def __init__(self, method: str = 'vader', cache_size: int = 1000):
        """
        Initialize the async sentiment analyzer
        
        Args:
            method: Analysis method ('vader', 'textblob', or 'ensemble')
            cache_size: Size of LRU cache for results
        """
        self.method = method
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.cache = {}
        self.cache_size = cache_size
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{text}_{self.method}".encode()).hexdigest()
    
    async def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text asynchronously
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentResult object containing analysis results
        """
        import time
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        if not text or not isinstance(text, str):
            return self._empty_result(text)
        
        # Run analysis in thread pool
        loop = asyncio.get_event_loop()
        
        if self.method == 'vader':
            result = await loop.run_in_executor(executor, self._vader_analysis, text)
        elif self.method == 'textblob':
            result = await loop.run_in_executor(executor, self._textblob_analysis, text)
        elif self.method == 'ensemble':
            result = await loop.run_in_executor(executor, self._ensemble_analysis, text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        result.processing_time = time.time() - start_time
        
        # Cache result
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = result
        
        return result
    
    def _vader_analysis(self, text: str) -> SentimentResult:
        """VADER sentiment analysis"""
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine sentiment label with refined thresholds
        if scores['compound'] >= 0.1:
            label = 'positive'
        elif scores['compound'] <= -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Calculate confidence
        confidence = min(abs(scores['compound']) * 1.5, 1.0)
        
        return SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            positive=scores['pos'],
            negative=scores['neg'],
            neutral=scores['neu'],
            compound=scores['compound'],
            sentiment_label=label,
            confidence=confidence
        )
    
    def _textblob_analysis(self, text: str) -> SentimentResult:
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert to VADER-like scores
        if polarity > 0.1:
            positive = polarity
            negative = 0
            label = 'positive'
        elif polarity < -0.1:
            positive = 0
            negative = abs(polarity)
            label = 'negative'
        else:
            positive = 0
            negative = 0
            label = 'neutral'
        
        neutral = 1 - (abs(polarity))
        
        return SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            positive=positive,
            negative=negative,
            neutral=neutral,
            compound=polarity,
            sentiment_label=label,
            confidence=1 - subjectivity
        )
    
    def _ensemble_analysis(self, text: str) -> SentimentResult:
        """Ensemble method combining VADER and TextBlob"""
        vader_result = self._vader_analysis(text)
        textblob_result = self._textblob_analysis(text)
        
        # Weighted average (VADER tends to be more accurate for social media)
        vader_weight = 0.6
        textblob_weight = 0.4
        
        positive = (vader_result.positive * vader_weight + textblob_result.positive * textblob_weight)
        negative = (vader_result.negative * vader_weight + textblob_result.negative * textblob_weight)
        neutral = (vader_result.neutral * vader_weight + textblob_result.neutral * textblob_weight)
        compound = (vader_result.compound * vader_weight + textblob_result.compound * textblob_weight)
        
        # Determine final label
        if compound >= 0.1:
            label = 'positive'
        elif compound <= -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        confidence = (vader_result.confidence * vader_weight + textblob_result.confidence * textblob_weight)
        
        return SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            positive=positive,
            negative=negative,
            neutral=neutral,
            compound=compound,
            sentiment_label=label,
            confidence=confidence
        )
    
    def _empty_result(self, text: str) -> SentimentResult:
        """Return empty/neutral result for invalid input"""
        return SentimentResult(
            text=str(text)[:100] if text else "",
            positive=0.0,
            negative=0.0,
            neutral=1.0,
            compound=0.0,
            sentiment_label='neutral',
            confidence=0.0
        )
    
    async def analyze_batch(self, texts: List[str], batch_size: int = 10) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts concurrently
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process concurrently
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_tasks = [self.analyze_text(text) for text in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    async def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Analyze sentiment for texts in a DataFrame asynchronously
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing texts
            
        Returns:
            DataFrame with added sentiment columns
        """
        df_copy = df.copy()
        
        # Extract texts
        texts = df_copy[text_column].fillna('').astype(str).tolist()
        
        # Analyze in batches
        results = await self.analyze_batch(texts)
        
        # Add results to DataFrame
        df_copy['sentiment'] = [r.sentiment_label for r in results]
        df_copy['sentiment_positive'] = [r.positive for r in results]
        df_copy['sentiment_negative'] = [r.negative for r in results]
        df_copy['sentiment_neutral'] = [r.neutral for r in results]
        df_copy['sentiment_compound'] = [r.compound for r in results]
        df_copy['sentiment_confidence'] = [r.confidence for r in results]
        
        return df_copy
    
    def get_aggregate_sentiment(self, results: List[SentimentResult]) -> Dict:
        """
        Calculate aggregate sentiment statistics
        
        Args:
            results: List of SentimentResult objects
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {
                'total_analyzed': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'average_compound': 0,
                'average_confidence': 0,
                'dominant_sentiment': 'neutral'
            }
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for result in results:
            sentiment_counts[result.sentiment_label] += 1
        
        avg_compound = np.mean([r.compound for r in results])
        avg_confidence = np.mean([r.confidence for r in results])
        
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'total_analyzed': len(results),
            'positive_count': sentiment_counts['positive'],
            'negative_count': sentiment_counts['negative'],
            'neutral_count': sentiment_counts['neutral'],
            'positive_percentage': (sentiment_counts['positive'] / len(results)) * 100,
            'negative_percentage': (sentiment_counts['negative'] / len(results)) * 100,
            'neutral_percentage': (sentiment_counts['neutral'] / len(results)) * 100,
            'average_compound': float(avg_compound),
            'average_confidence': float(avg_confidence),
            'dominant_sentiment': dominant_sentiment,
            'sentiment_distribution': sentiment_counts
        }
    
    def clear_cache(self):
        """Clear the results cache"""
        self.cache.clear()

# Quick analysis function
async def quick_sentiment_async(text: str, method: str = 'ensemble') -> Dict:
    """
    Quick async sentiment analysis for a single text
    
    Args:
        text: Input text
        method: Analysis method
        
    Returns:
        Dictionary with sentiment results
    """
    analyzer = AsyncSentimentAnalyzer(method=method)
    result = await analyzer.analyze_text(text)
    
    return {
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
    }