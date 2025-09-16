#!/bin/bash

echo "Starting Render build process..."

# Update pip and install build tools
python -m pip install --upgrade pip
python -m pip install wheel setuptools

# Install packages one by one to handle errors better
echo "Installing core packages..."
pip install Flask
pip install numpy
pip install pandas
pip install nltk
pip install vaderSentiment
pip install openpyxl || echo "openpyxl failed, continuing..."
pip install gunicorn
pip install Flask-Cors
pip install Werkzeug
pip install python-dotenv

# Download NLTK data
echo "Downloading NLTK data..."
python -c "
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'NLTK download failed: {e}')
"

echo "Build completed!"
