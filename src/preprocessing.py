# src/preprocessing.py
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "url", text)
    text = re.sub(r"\d+", "number", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text