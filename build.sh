#!/bin/bash

echo "Starting Render build process..."

# Update pip and install build dependencies
python -m pip install --upgrade pip setuptools wheel

# Install requirements
pip install --no-cache-dir -r requirements.txt

# Download NLTK data
python -c "
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
print('NLTK data downloaded successfully')
"

echo "Build completed successfully!"