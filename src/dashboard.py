# src/dashboard.py
import streamlit as st
import joblib
import os

# Resolve path to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Load vectorizer and models
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
nb_model = joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl"))
svm_model = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))

st.title("Spam Email Detector")
st.subheader("Enter an email or SMS text below to check if it's Spam or Ham.")

user_input = st.text_area("Message:")
model_choice = st.selectbox("Choose model", ["Naive Bayes", "SVM"])

if st.button("Predict"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        if model_choice == "Naive Bayes":
            prediction = nb_model.predict(X)[0]
        else:
            prediction = svm_model.predict(X)[0]
        st.success(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")
    else:
        st.warning("Please enter a message to classify.")
