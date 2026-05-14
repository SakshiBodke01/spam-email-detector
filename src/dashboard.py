# src/dashboard.py
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from preprocessing import load_data

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

st.title("📧 Spam Email Detector")
st.write("Enter an email or SMS text below to check if it's Spam or Ham.")

vectorizer = joblib.load("../models/vectorizer.pkl")
model = joblib.load("../models/svm.pkl")

user_input = st.text_area("Email/SMS Text:")

if st.button("Predict"):
    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]
    confidence = model.decision_function(X)[0]
    st.success(f"Prediction: {'Spam' if prediction == 1 else 'Ham'} (Confidence: {confidence:.2f})")

# Extra Feature: WordCloud visualization
st.subheader("📊 Spam vs Ham WordCloud")
df = load_data()
spam_texts = " ".join(df[df['label']==1]['text'])
ham_texts = " ".join(df[df['label']==0]['text'])

col1, col2 = st.columns(2)
with col1:
    st.write("Spam WordCloud")
    wc_spam = WordCloud(width=400, height=300, background_color="black").generate(spam_texts)
    plt.imshow(wc_spam, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

with col2:
    st.write("Ham WordCloud")
    wc_ham = WordCloud(width=400, height=300, background_color="white").generate(ham_texts)
    plt.imshow(wc_ham, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
