# src/train.py
import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from preprocessing import load_data, preprocess

def train_models():
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = preprocess(df)

    os.makedirs("../models", exist_ok=True)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    joblib.dump(nb, "../models/naive_bayes.pkl")

    svm = LinearSVC()
    svm.fit(X_train, y_train)
    joblib.dump(svm, "../models/svm.pkl")

    joblib.dump(vectorizer, "../models/vectorizer.pkl")

if __name__ == "__main__":
    train_models()
