# src/app.py
import os
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Dynamically resolve directory paths to prevent FileNotFoundError on serverless runtimes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "..", "models", "svm.pkl"))

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "active",
        "service": "ThreatMail Spam Intelligence API",
        "description": "An AI-powered service classifying input email content as Spam or Ham.",
        "usage": {
            "method": "POST",
            "endpoint": "/predict",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "text": "Your raw email text goes here"
            }
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Empty text provided"}), 400
        
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = model.decision_function(X)[0]
    return jsonify({"prediction": "Spam" if prediction == 1 else "Ham", "confidence": float(confidence)})

if __name__ == "__main__":
    app.run(debug=True)
