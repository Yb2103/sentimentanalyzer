"""
Data Processing Module for E-Consultation Comments
Handles file I/O, data cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
import re
from datetime import datetime
import json

class DataProcessor:
    """
    Handle data loading, cleaning, and preprocessing
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.txt']
        
    def load_data(self, filepath: str, 
                  text_column: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            filepath: Path to the data file
            text_column: Name of the text column (auto-detect if None)
            
        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Load based on file type
        if file_ext == '.csv':
            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif file_ext == '.json':
            df = pd.read_json(filepath)
        elif file_ext == '.txt':
            # For text files, create a DataFrame with single column
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            df = pd.DataFrame({'comment_text': lines})
        else:
            raise ValueError(f"Cannot process file type: {file_ext}")
        
        # Auto-detect text column if not specified
        if text_column is None:
            text_column = self._detect_text_column(df)
        
        # Ensure text column exists
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        # Add comment_id if not present
        if 'comment_id' not in df.columns:
            df['comment_id'] = range(1, len(df) + 1)
        
        return df
    
    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """
        Auto-detect the text column in DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the likely text column
        """
        # Common text column names
        common_names = ['comment', 'text', 'comment_text', 'feedback', 
                       'suggestion', 'content', 'message', 'description',
                       'comments', 'review', 'response']
        
        # Check for exact matches (case-insensitive)
        for col in df.columns:
            if col.lower() in common_names:
                return col
        
        # Check for partial matches
        for col in df.columns:
            for name in common_names:
                if name in col.lower():
                    return col
        
        # Find column with longest average text length
        text_lengths = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                text_lengths[col] = avg_length
        
        if text_lengths:
            return max(text_lengths, key=text_lengths.get)
        
        # Default to first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        raise ValueError("Could not detect text column automatically")
    
    def clean_data(self, df: pd.DataFrame, 
                   text_column: str) -> pd.DataFrame:
        """
        Clean and preprocess the data
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy
        df_clean = df.copy()
        
        # Remove null values in text column
        df_clean = df_clean[df_clean[text_column].notna()]
        
        # Convert to string and strip whitespace
        df_clean[text_column] = df_clean[text_column].astype(str).str.strip()
        
        # Remove empty strings
        df_clean = df_clean[df_clean[text_column] != '']
        
        # Remove duplicates based on text
        df_clean = df_clean.drop_duplicates(subset=[text_column], keep='first')
        
        # Add cleaned text column
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.clean_text)
        
        # Add text statistics
        df_clean['word_count'] = df_clean['cleaned_text'].str.split().str.len()
        df_clean['char_count'] = df_clean['cleaned_text'].str.len()
        
        # Filter out very short comments (less than 3 words)
        df_clean = df_clean[df_clean['word_count'] >= 3]
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        return df_clean
    
    def clean_text(self, text: str) -> str:
        """
        Clean individual text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def prepare_for_analysis(self, df: pd.DataFrame,
                            text_column: str,
                            sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Prepare data for analysis
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            sample_size: Optional sample size for large datasets
            
        Returns:
            Prepared DataFrame
        """
        # Clean the data
        df_prepared = self.clean_data(df, text_column)
        
        # Sample if requested
        if sample_size and len(df_prepared) > sample_size:
            df_prepared = df_prepared.sample(n=sample_size, random_state=42)
        
        # Add timestamp if not present
        if 'timestamp' not in df_prepared.columns:
            df_prepared['timestamp'] = datetime.now()
        
        # Add processing metadata
        df_prepared['processed_at'] = datetime.now()
        
        return df_prepared
    
    def save_results(self, df: pd.DataFrame, 
                    output_path: str,
                    format: str = 'excel') -> str:
        """
        Save analysis results to file
        
        Args:
            df: DataFrame with results
            output_path: Output file path
            format: Output format ('excel', 'csv', 'json')
            
        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Add appropriate extension if not present
        base_path = os.path.splitext(output_path)[0]
        
        if format == 'excel':
            output_file = f"{base_path}.xlsx"
            df.to_excel(output_file, index=False, engine='openpyxl')
        elif format == 'csv':
            output_file = f"{base_path}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
        elif format == 'json':
            output_file = f"{base_path}.json"
            df.to_json(output_file, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        return output_file
    
    def generate_statistics(self, df: pd.DataFrame, 
                           text_column: str) -> Dict[str, Any]:
        """
        Generate statistics about the dataset
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_comments': len(df),
            'unique_comments': df[text_column].nunique(),
            'duplicate_count': len(df) - df[text_column].nunique(),
            'avg_word_count': df['word_count'].mean() if 'word_count' in df.columns else 0,
            'min_word_count': df['word_count'].min() if 'word_count' in df.columns else 0,
            'max_word_count': df['word_count'].max() if 'word_count' in df.columns else 0,
            'avg_char_count': df['char_count'].mean() if 'char_count' in df.columns else 0,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict()
        }
        
        # Add date range if timestamp column exists
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                stats['date_range'] = {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                }
            except:
                pass
        
        return stats
    
    def split_by_category(self, df: pd.DataFrame,
                         category_column: str) -> Dict[str, pd.DataFrame]:
        """
        Split DataFrame by category
        
        Args:
            df: Input DataFrame
            category_column: Name of category column
            
        Returns:
            Dictionary of DataFrames by category
        """
        if category_column not in df.columns:
            raise ValueError(f"Category column '{category_column}' not found")
        
        categories = {}
        for category in df[category_column].unique():
            categories[str(category)] = df[df[category_column] == category].copy()
        
        return categories

def validate_data(df: pd.DataFrame, text_column: str) -> Tuple[bool, List[str]]:
    """
    Validate data for analysis
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check if DataFrame is empty
    if df.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    # Check if text column exists
    if text_column not in df.columns:
        issues.append(f"Text column '{text_column}' not found")
        return False, issues
    
    # Check for null values
    null_count = df[text_column].isna().sum()
    if null_count > 0:
        issues.append(f"Found {null_count} null values in text column")
    
    # Check for empty strings
    empty_count = (df[text_column] == '').sum()
    if empty_count > 0:
        issues.append(f"Found {empty_count} empty strings in text column")
    
    # Check for duplicates
    duplicate_count = df[text_column].duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate texts")
    
    # Check minimum data size
    if len(df) < 5:
        issues.append("Dataset has less than 5 records")
    
    return len(issues) == 0, issues

if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'comment_id': range(1, 6),
        'comment_text': [
            "This legislation will greatly benefit small businesses.",
            "I'm concerned about the compliance costs.",
            "The timeline for implementation is too short.",
            "Excellent proposal with clear guidelines.",
            "Need more clarification on section 3.2"
        ],
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='D')
    })
    
    print("Data Processor Test")
    print("=" * 50)
    
    # Clean data
    cleaned_data = processor.clean_data(sample_data, 'comment_text')
    print("\nCleaned Data:")
    print(cleaned_data[['comment_text', 'cleaned_text', 'word_count']].head())
    
    # Generate statistics
    stats = processor.generate_statistics(cleaned_data, 'comment_text')
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if key not in ['columns', 'data_types']:
            print(f"  {key}: {value}")
    
    # Validate data
    is_valid, issues = validate_data(cleaned_data, 'cleaned_text')
    print(f"\nData Valid: {is_valid}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")