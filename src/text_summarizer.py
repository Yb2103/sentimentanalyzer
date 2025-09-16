"""
Text Summarization Module for E-Consultation Comments
Provides extractive and abstractive summarization capabilities
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
from collections import Counter
import heapq

# Download required NLTK data
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

class TextSummarizer:
    """
    Text summarization class for generating concise summaries
    """
    
    def __init__(self, language='english'):
        """
        Initialize the text summarizer
        
        Args:
            language: Language for stopwords (default 'english')
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove special characters and digits
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra whitespaces
        text = text.strip()
        
        return text
    
    def extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """
        Generate extractive summary using frequency-based scoring
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences in summary
            
        Returns:
            Summary text
        """
        if not text or len(text.strip()) == 0:
            return ""
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize into sentences
        sentences = sent_tokenize(cleaned_text)
        
        if len(sentences) <= num_sentences:
            return cleaned_text
        
        # Tokenize into words and remove stopwords
        word_tokens = word_tokenize(cleaned_text.lower())
        filtered_words = [word for word in word_tokens 
                         if word.isalnum() and word not in self.stop_words]
        
        # Calculate word frequencies
        word_freq = FreqDist(filtered_words)
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            score = 0
            word_count = 0
            
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            # Normalize by sentence length
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores, 
                                      key=sentence_scores.get)
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
        
        return ' '.join(summary_sentences)
    
    def keyword_summary(self, text: str, num_keywords: int = 10) -> Dict:
        """
        Extract keywords and key phrases from text
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
            
        Returns:
            Dictionary with keywords and their frequencies
        """
        if not text:
            return {}
        
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize and filter
        words = word_tokenize(cleaned_text.lower())
        filtered_words = [word for word in words 
                         if word.isalnum() and 
                         word not in self.stop_words and 
                         len(word) > 2]
        
        # Get frequency distribution
        word_freq = Counter(filtered_words)
        
        # Get top keywords
        top_keywords = dict(word_freq.most_common(num_keywords))
        
        return top_keywords
    
    def summarize_batch(self, texts: List[str], max_length: int = 100) -> List[str]:
        """
        Summarize multiple texts
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum length for each summary (in words)
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for text in texts:
            if not text:
                summaries.append("")
                continue
            
            # Determine number of sentences based on text length
            word_count = len(text.split())
            if word_count < 50:
                num_sentences = 1
            elif word_count < 150:
                num_sentences = 2
            else:
                num_sentences = 3
            
            summary = self.extractive_summary(text, num_sentences)
            
            # Truncate if still too long
            words = summary.split()
            if len(words) > max_length:
                summary = ' '.join(words[:max_length]) + '...'
            
            summaries.append(summary)
        
        return summaries
    
    def generate_collective_summary(self, texts: List[str], 
                                  num_points: int = 5) -> Dict:
        """
        Generate a collective summary from multiple texts
        
        Args:
            texts: List of texts
            num_points: Number of key points to extract
            
        Returns:
            Dictionary with collective summary information
        """
        if not texts:
            return {
                'key_points': [],
                'common_themes': {},
                'summary': ""
            }
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Get overall summary
        overall_summary = self.extractive_summary(combined_text, num_points)
        
        # Extract common themes (keywords)
        themes = self.keyword_summary(combined_text, 15)
        
        # Extract key points (individual summaries)
        key_points = []
        for text in texts[:num_points]:
            if text:
                summary = self.extractive_summary(text, 1)
                if summary:
                    key_points.append(summary)
        
        return {
            'key_points': key_points,
            'common_themes': themes,
            'summary': overall_summary,
            'total_comments_analyzed': len(texts)
        }
    
    def categorize_by_theme(self, texts: List[str], 
                          categories: Optional[List[str]] = None) -> Dict:
        """
        Categorize texts by themes or topics
        
        Args:
            texts: List of texts
            categories: Optional predefined categories
            
        Returns:
            Dictionary with texts grouped by themes
        """
        if not categories:
            # Default categories for legislation comments
            categories = {
                'support': ['support', 'agree', 'good', 'excellent', 'benefit', 
                           'positive', 'favor', 'approve'],
                'oppose': ['oppose', 'disagree', 'bad', 'harm', 'negative', 
                          'against', 'reject', 'concern'],
                'suggestion': ['suggest', 'recommend', 'propose', 'should', 
                              'could', 'consider', 'alternative'],
                'clarification': ['unclear', 'clarify', 'explain', 'question', 
                                'what', 'how', 'why', 'confus']
            }
        
        categorized = {cat: [] for cat in categories.keys()}
        categorized['uncategorized'] = []
        
        for text in texts:
            if not text:
                continue
            
            text_lower = text.lower()
            categorized_flag = False
            
            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        categorized[category].append(text)
                        categorized_flag = True
                        break
                if categorized_flag:
                    break
            
            if not categorized_flag:
                categorized['uncategorized'].append(text)
        
        # Add summaries for each category
        result = {}
        for category, texts_list in categorized.items():
            if texts_list:
                summary = self.generate_collective_summary(texts_list, 3)
                result[category] = {
                    'count': len(texts_list),
                    'percentage': round(len(texts_list) / len(texts) * 100, 2),
                    'summary': summary['summary'],
                    'key_themes': list(summary['common_themes'].keys())[:5]
                }
        
        return result

def summarize_dataframe(df: pd.DataFrame, text_column: str, 
                       summarizer: Optional[TextSummarizer] = None) -> pd.DataFrame:
    """
    Add summaries to a dataframe
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        summarizer: Optional TextSummarizer instance
        
    Returns:
        DataFrame with added summary column
    """
    if summarizer is None:
        summarizer = TextSummarizer()
    
    df['summary'] = df[text_column].apply(
        lambda x: summarizer.extractive_summary(str(x) if pd.notna(x) else "", 2)
    )
    
    return df

if __name__ == "__main__":
    # Test the summarizer
    test_texts = [
        """The proposed legislation introduces significant changes to corporate governance 
        requirements. These changes will require companies to maintain higher transparency 
        standards and implement more stringent internal controls. While the intent is 
        commendable, small and medium enterprises may face challenges in compliance due 
        to resource constraints. The implementation timeline should be extended to allow 
        adequate preparation time.""",
        
        """I strongly support this initiative as it will enhance investor confidence and 
        market integrity. The requirements for independent directors and audit committees 
        are particularly important. This will bring our regulations in line with 
        international best practices.""",
        
        """The draft needs clarification on several points. What constitutes a material 
        transaction? How will the compliance be monitored? The penalty structure seems 
        disproportionate for minor violations."""
    ]
    
    summarizer = TextSummarizer()
    
    print("Text Summarization Test Results:")
    print("=" * 50)
    
    # Test individual summaries
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Text {i}]")
        print("Original length:", len(text.split()), "words")
        summary = summarizer.extractive_summary(text, 2)
        print("Summary:", summary)
        print("Summary length:", len(summary.split()), "words")
    
    # Test collective summary
    print("\n\nCollective Summary:")
    print("-" * 30)
    collective = summarizer.generate_collective_summary(test_texts)
    print("Overall Summary:", collective['summary'])
    print("\nCommon Themes:", list(collective['common_themes'].keys())[:5])
    
    # Test categorization
    print("\n\nThematic Categorization:")
    print("-" * 30)
    categorized = summarizer.categorize_by_theme(test_texts)
    for category, info in categorized.items():
        if info['count'] > 0:
            print(f"\n{category.upper()}: {info['count']} comments ({info['percentage']}%)")
            print(f"Key themes: {info['key_themes']}")