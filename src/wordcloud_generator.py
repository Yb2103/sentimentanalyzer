"""
Word Cloud Generator Module for E-Consultation Comments
Creates visual representations of frequently used keywords
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import Counter
import base64
from io import BytesIO

# Download required NLTK data
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

class WordCloudGenerator:
    """
    Generate word clouds from text data
    """
    
    def __init__(self, language='english'):
        """
        Initialize the word cloud generator
        
        Args:
            language: Language for stopwords filtering
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        # Add custom stopwords for legal/corporate text
        self.custom_stopwords = {
            'shall', 'may', 'must', 'section', 'clause', 'article',
            'paragraph', 'sub', 'act', 'regulation', 'rule', 'provision',
            'draft', 'proposed', 'amendment', 'legislation', 'bill'
        }
        self.stop_words.update(self.custom_stopwords)
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for word cloud
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def generate_wordcloud(self, text: str, max_words: int = 100,
                          width: int = 800, height: int = 400,
                          background_color: str = 'white',
                          colormap: str = 'viridis') -> WordCloud:
        """
        Generate a word cloud from text
        
        Args:
            text: Input text
            max_words: Maximum number of words in cloud
            width: Width of the word cloud image
            height: Height of the word cloud image
            background_color: Background color
            colormap: Color scheme for words
            
        Returns:
            WordCloud object
        """
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        if not cleaned_text:
            # Return empty word cloud
            return WordCloud(width=width, height=height, 
                           background_color=background_color).generate("empty")
        
        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color=background_color,
            colormap=colormap,
            stopwords=self.stop_words,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(cleaned_text)
        
        return wordcloud
    
    def generate_from_frequencies(self, word_freq: Dict[str, float],
                                 max_words: int = 100,
                                 width: int = 800, height: int = 400,
                                 background_color: str = 'white',
                                 colormap: str = 'viridis') -> WordCloud:
        """
        Generate word cloud from word frequencies
        
        Args:
            word_freq: Dictionary of word frequencies
            max_words: Maximum number of words
            width: Width of image
            height: Height of image
            background_color: Background color
            colormap: Color scheme
            
        Returns:
            WordCloud object
        """
        # Filter out stopwords
        filtered_freq = {word: freq for word, freq in word_freq.items()
                        if word.lower() not in self.stop_words}
        
        if not filtered_freq:
            return WordCloud(width=width, height=height,
                           background_color=background_color).generate("empty")
        
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color=background_color,
            colormap=colormap,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(filtered_freq)
        
        return wordcloud
    
    def save_wordcloud(self, wordcloud: WordCloud, filepath: str,
                      dpi: int = 100):
        """
        Save word cloud to file
        
        Args:
            wordcloud: WordCloud object
            filepath: Path to save the image
            dpi: Image resolution
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def wordcloud_to_base64(self, wordcloud: WordCloud) -> str:
        """
        Convert word cloud to base64 string for web display
        
        Args:
            wordcloud: WordCloud object
            
        Returns:
            Base64 encoded string
        """
        img_buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return img_str
    
    def extract_keywords(self, texts: List[str], 
                        top_n: int = 20) -> Dict[str, int]:
        """
        Extract top keywords from multiple texts
        
        Args:
            texts: List of texts
            top_n: Number of top keywords to return
            
        Returns:
            Dictionary of keywords and their frequencies
        """
        all_words = []
        
        for text in texts:
            if not text:
                continue
            
            cleaned_text = self.preprocess_text(text)
            words = word_tokenize(cleaned_text)
            
            # Filter words
            filtered = [word for word in words 
                       if len(word) > 2 and 
                       word not in self.stop_words]
            all_words.extend(filtered)
        
        # Count frequencies
        word_freq = Counter(all_words)
        
        # Get top keywords
        top_keywords = dict(word_freq.most_common(top_n))
        
        return top_keywords
    
    def generate_comparative_clouds(self, texts_dict: Dict[str, List[str]],
                                   max_words: int = 50) -> Dict[str, WordCloud]:
        """
        Generate separate word clouds for different categories
        
        Args:
            texts_dict: Dictionary with category names as keys and lists of texts as values
            max_words: Maximum words per cloud
            
        Returns:
            Dictionary of WordCloud objects
        """
        clouds = {}
        
        for category, texts in texts_dict.items():
            combined_text = ' '.join(texts)
            cloud = self.generate_wordcloud(combined_text, max_words=max_words)
            clouds[category] = cloud
        
        return clouds
    
    def generate_sentiment_cloud(self, df: pd.DataFrame,
                                text_column: str,
                                sentiment_column: str) -> Dict[str, WordCloud]:
        """
        Generate word clouds for different sentiment categories
        
        Args:
            df: DataFrame with text and sentiment columns
            text_column: Name of text column
            sentiment_column: Name of sentiment column
            
        Returns:
            Dictionary of word clouds by sentiment
        """
        sentiment_clouds = {}
        
        for sentiment in df[sentiment_column].unique():
            sentiment_texts = df[df[sentiment_column] == sentiment][text_column].tolist()
            combined_text = ' '.join([str(text) for text in sentiment_texts if pd.notna(text)])
            
            # Use different color schemes for different sentiments
            if sentiment == 'positive':
                colormap = 'Greens'
            elif sentiment == 'negative':
                colormap = 'Reds'
            else:
                colormap = 'Greys'
            
            cloud = self.generate_wordcloud(combined_text, colormap=colormap)
            sentiment_clouds[sentiment] = cloud
        
        return sentiment_clouds

def create_word_frequency_chart(word_freq: Dict[str, int], 
                               top_n: int = 15) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a bar chart of word frequencies
    
    Args:
        word_freq: Dictionary of word frequencies
        top_n: Number of top words to display
        
    Returns:
        Matplotlib figure and axes
    """
    # Get top words
    top_words = dict(Counter(word_freq).most_common(top_n))
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    words = list(top_words.keys())
    frequencies = list(top_words.values())
    
    bars = ax.bar(words, frequencies, color='steelblue')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    ax.set_xlabel('Keywords', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Top Keywords in Comments', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig, ax

if __name__ == "__main__":
    # Test the word cloud generator
    test_texts = [
        """The proposed corporate governance framework needs substantial revision. 
        Companies require clear guidelines on compliance requirements. The audit 
        committee structure proposed is excellent but implementation timeline is 
        concerning for small enterprises.""",
        
        """Strong support for transparency measures. Independent directors will 
        enhance accountability. Whistleblower protection mechanisms are essential. 
        The penalty framework seems appropriate for ensuring compliance.""",
        
        """Concerned about compliance costs for SMEs. The reporting requirements 
        are too stringent. Need clarification on materiality thresholds. Suggest 
        phased implementation approach for smaller companies."""
    ]
    
    generator = WordCloudGenerator()
    
    print("Word Cloud Generator Test")
    print("=" * 50)
    
    # Test keyword extraction
    keywords = generator.extract_keywords(test_texts, top_n=10)
    print("\nTop Keywords:")
    for word, freq in keywords.items():
        print(f"  {word}: {freq}")
    
    # Generate word cloud
    combined_text = ' '.join(test_texts)
    wordcloud = generator.generate_wordcloud(combined_text)
    
    # Save to file
    output_path = "test_wordcloud.png"
    generator.save_wordcloud(wordcloud, output_path)
    print(f"\nWord cloud saved to: {output_path}")
    
    # Test sentiment-based clouds
    print("\nGenerating sentiment-based word clouds...")
    sentiment_texts = {
        'positive': [text for text in test_texts[:2]],
        'negative': [test_texts[2]]
    }
    
    sentiment_clouds = generator.generate_comparative_clouds(sentiment_texts)
    print(f"Generated {len(sentiment_clouds)} sentiment word clouds")