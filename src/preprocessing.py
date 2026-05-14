# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(url="https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"):
    """
    Load dataset directly from URL.
    Dataset: UCI SMS Spam Collection (tab-separated).
    Columns: 'label' (ham/spam), 'message'
    """
    df = pd.read_csv(url, sep='\t', names=['label', 'text'])
    df['label'] = df['label'].map({'ham':0, 'spam':1})
    return df

def preprocess(df):
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
