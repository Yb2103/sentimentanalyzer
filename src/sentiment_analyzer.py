"""
Sentiment Analysis Module for E-Consultation Comments
Provides sentiment analysis using VADER and optional transformer models
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

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

class SentimentAnalyzer:
    """
    Main sentiment analyzer class supporting multiple analysis methods
    """
    
    def __init__(self, method: str = 'vader'):
        """
        Initialize the sentiment analyzer
        
        Args:
            method: Analysis method ('vader', 'textblob', or 'ensemble')
        """
        self.method = method
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentResult object containing analysis results
        """
        if not text or not isinstance(text, str):
            return self._empty_result(text)
        
        if self.method == 'vader':
            return self._vader_analysis(text)
        elif self.method == 'textblob':
            return self._textblob_analysis(text)
        elif self.method == 'ensemble':
            return self._ensemble_analysis(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _vader_analysis(self, text: str) -> SentimentResult:
        """VADER sentiment analysis"""
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine sentiment label
        if scores['compound'] >= 0.05:
            label = 'positive'
        elif scores['compound'] <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Calculate confidence
        confidence = abs(scores['compound'])
        
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
        if polarity > 0:
            positive = polarity
            negative = 0
            label = 'positive'
        elif polarity < 0:
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
        
        # Average the scores
        positive = (vader_result.positive + textblob_result.positive) / 2
        negative = (vader_result.negative + textblob_result.negative) / 2
        neutral = (vader_result.neutral + textblob_result.neutral) / 2
        compound = (vader_result.compound + textblob_result.compound) / 2
        
        # Determine final label
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        confidence = (vader_result.confidence + textblob_result.confidence) / 2
        
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
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        for text in texts:
            results.append(self.analyze_text(text))
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Analyze sentiment for texts in a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing texts
            
        Returns:
            DataFrame with added sentiment columns
        """
        results = []
        for text in df[text_column]:
            result = self.analyze_text(str(text) if pd.notna(text) else "")
            results.append({
                'positive_score': result.positive,
                'negative_score': result.negative,
                'neutral_score': result.neutral,
                'compound_score': result.compound,
                'sentiment': result.sentiment_label,
                'confidence': result.confidence
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Concatenate with original DataFrame
        return pd.concat([df, results_df], axis=1)
    
    def get_aggregate_sentiment(self, results: List[SentimentResult]) -> Dict:
        """
        Calculate aggregate sentiment statistics
        
        Args:
            results: List of sentiment results
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {
                'total_comments': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'average_compound': 0,
                'sentiment_distribution': {}
            }
        
        sentiments = [r.sentiment_label for r in results]
        compounds = [r.compound for r in results]
        
        return {
            'total_comments': len(results),
            'positive_count': sentiments.count('positive'),
            'negative_count': sentiments.count('negative'),
            'neutral_count': sentiments.count('neutral'),
            'average_compound': np.mean(compounds),
            'sentiment_distribution': {
                'positive': round(sentiments.count('positive') / len(sentiments) * 100, 2),
                'negative': round(sentiments.count('negative') / len(sentiments) * 100, 2),
                'neutral': round(sentiments.count('neutral') / len(sentiments) * 100, 2)
            },
            'overall_sentiment': self._determine_overall_sentiment(compounds)
        }
    
    def _determine_overall_sentiment(self, compounds: List[float]) -> str:
        """Determine overall sentiment from compound scores"""
        avg_compound = np.mean(compounds)
        if avg_compound >= 0.05:
            return 'positive'
        elif avg_compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'

# Convenience functions
def quick_sentiment(text: str, method: str = 'vader') -> Dict:
    """
    Quick sentiment analysis for a single text
    
    Args:
        text: Input text
        method: Analysis method
        
    Returns:
        Dictionary with sentiment results
    """
    analyzer = SentimentAnalyzer(method=method)
    result = analyzer.analyze_text(text)
    return {
        'sentiment': result.sentiment_label,
        'confidence': result.confidence,
        'scores': {
            'positive': result.positive,
            'negative': result.negative,
            'neutral': result.neutral,
            'compound': result.compound
        }
    }

if __name__ == "__main__":
    # Test the sentiment analyzer
    test_texts = [
        "This legislation is excellent and will greatly benefit the industry.",
        "I strongly oppose this draft. It will harm small businesses.",
        "The proposed changes seem reasonable and balanced.",
        "Terrible idea! This will destroy our competitiveness.",
        "I have no strong opinion about this proposal."
    ]
    
    analyzer = SentimentAnalyzer(method='ensemble')
    
    print("Sentiment Analysis Test Results:")
    print("=" * 50)
    
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result.sentiment_label}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Scores - Pos: {result.positive:.2f}, Neg: {result.negative:.2f}, Neu: {result.neutral:.2f}")