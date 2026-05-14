# src/predict.py
import joblib

def predict_email(text, model_name="svm"):
    vectorizer = joblib.load("../models/vectorizer.pkl")
    model = joblib.load(f"../models/{model_name}.pkl")
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = model.decision_function(X)[0] if model_name == "svm" else None
    return {"prediction": "Spam" if prediction == 1 else "Ham", "confidence": confidence}

if __name__ == "__main__":
    sample = "Congratulations! You won a lottery. Claim now."
    print(predict_email(sample, "svm"))
