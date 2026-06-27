# src/preprocessing.py
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure NLTK data directory path dynamically for Vercel / serverless runtimes
if os.environ.get('VERCEL'):
    nltk_data_dir = '/tmp/nltk_data'
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
else:
    # Use standard NLTK path locally
    pass

# Ensure required NLTK resources are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    # If running on Vercel, download into the writable /tmp/nltk_data
    download_dir = '/tmp/nltk_data' if os.environ.get('VERCEL') else None
    nltk.download('stopwords', download_dir=download_dir)
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
    # Test lookup to trigger download if missing
    lemmatizer.lemmatize('emails')
except LookupError:
    download_dir = '/tmp/nltk_data' if os.environ.get('VERCEL') else None
    nltk.download('wordnet', download_dir=download_dir)
    nltk.download('omw-1.4', download_dir=download_dir)
    lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans raw email text for machine learning models:
    - Converts text to lowercase.
    - Replaces URL links with 'url' and digits with 'number' to reduce vocabulary entropy.
    - Removes punctuation and special characters.
    - Removes English stopwords.
    - Applies WordNet Lemmatization.
    """
    if not text:
        return ""
        
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Normalize URLs, emails, and numbers
    text = re.sub(r"https?://\S+|www\.\S+", "url", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email", text)
    text = re.sub(r"\d+", "number", text)
    
    # 3. Remove punctuation and special characters (keep only words and whitespace)
    text = re.sub(r"[^\w\s]", "", text)
    
    # 4. Tokenize and filter
    words = text.split()
    cleaned_words = []
    
    for word in words:
        # Remove stopwords and short tokens
        if word not in stop_words and len(word) > 1:
            # Apply WordNet Lemmatization
            lemma = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma)
            
    return " ".join(cleaned_words)