#!/bin/bash

# Download required NLTK data
python -c "
import nltk
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

print('NLTK data downloaded successfully')
"

# Set environment variables
export FLASK_ENV=production
export FLASK_DEBUG=False

# Create necessary directories
mkdir -p uploads results data

# Start the application
exec gunicorn --bind 0.0.0.0:$PORT app:app --timeout 120 --workers 1