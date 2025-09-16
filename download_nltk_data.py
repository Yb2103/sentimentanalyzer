#!/usr/bin/env python3
"""
Download required NLTK data for deployment
Run this during the build process to ensure NLTK data is available
"""

import nltk
import ssl

def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Handle SSL issues that sometimes occur
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        nltk_downloads = [
            'punkt',
            'vader_lexicon',
            'stopwords',
            'wordnet',
            'omw-1.4',
            'averaged_perceptron_tagger'
        ]
        
        for item in nltk_downloads:
            try:
                print(f"Downloading NLTK data: {item}")
                nltk.download(item, quiet=True)
                print(f"Successfully downloaded: {item}")
            except Exception as e:
                print(f"Warning: Could not download {item}: {e}")
        
        print("NLTK data download completed!")
        
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

if __name__ == "__main__":
    download_nltk_data()