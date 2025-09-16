"""
Configuration settings for E-Consultation Sentiment Analysis Application
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = BASE_DIR / 'uploads'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'

# Create directories if they don't exist
for directory in [UPLOAD_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Flask configuration
class Config:
    """Flask application configuration"""
    
    # Basic Flask config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # File upload settings
    UPLOAD_FOLDER = str(UPLOAD_DIR)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'txt'}
    
    # Analysis settings
    DEFAULT_SENTIMENT_METHOD = 'ensemble'  # 'vader', 'textblob', or 'ensemble'
    MAX_WORDS_WORDCLOUD = 100
    WORDCLOUD_WIDTH = 800
    WORDCLOUD_HEIGHT = 400
    
    # Batch processing
    BATCH_SIZE = 100
    MAX_SAMPLE_SIZE = 10000  # Maximum samples for web interface
    
    # Results
    RESULTS_FOLDER = str(RESULTS_DIR)
    RESULT_EXPIRY_HOURS = 24  # Hours before results are deleted
    
    # UI Settings
    ITEMS_PER_PAGE = 20
    
    @staticmethod
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Analysis configuration
ANALYSIS_CONFIG = {
    'sentiment': {
        'methods': ['vader', 'textblob', 'ensemble'],
        'default_method': 'ensemble',
        'confidence_threshold': 0.5
    },
    'summarization': {
        'extractive_sentences': 3,
        'max_summary_length': 100,
        'collective_key_points': 5
    },
    'wordcloud': {
        'max_words': 100,
        'colormaps': {
            'positive': 'Greens',
            'negative': 'Reds',
            'neutral': 'Greys',
            'default': 'viridis'
        }
    },
    'preprocessing': {
        'min_word_count': 3,
        'remove_duplicates': True,
        'clean_html': True,
        'remove_urls': True
    }
}

# Export configuration
EXPORT_CONFIG = {
    'formats': ['excel', 'csv', 'json'],
    'default_format': 'excel',
    'include_metadata': True,
    'timestamp_format': '%Y-%m-%d %H:%M:%S'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'formatter': 'default'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file', 'console']
    }
}