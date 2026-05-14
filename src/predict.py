# src/predict.py
import joblib
import os
from preprocessing import clean_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
nb_model   = joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl"))
svm_model  = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))

def predict_message(message, model="Naive Bayes"):
    cleaned = clean_text(message)
    X = vectorizer.transform([cleaned])
    if model == "Naive Bayes":
        return nb_model.predict(X)[0]
    else:
        return svm_model.predict(X)[0]