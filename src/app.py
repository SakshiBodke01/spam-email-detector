# src/app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

vectorizer = joblib.load("../models/vectorizer.pkl")
model = joblib.load("../models/svm.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = model.decision_function(X)[0]
    return jsonify({"prediction": "Spam" if prediction == 1 else "Ham", "confidence": float(confidence)})

if __name__ == "__main__":
    app.run(debug=True)
