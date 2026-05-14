# src/train.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import joblib
import os

# ── Load dataset from URL (no local spam.csv needed) ──────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)

print("📥 Downloading dataset…")
df = pd.read_csv(DATASET_URL, sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(f"✅ Loaded {len(df)} rows  |  spam: {df['label'].sum()}  |  ham: {(df['label']==0).sum()}")

# ── Vectorize ─────────────────────────────────────────────────────────────────
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# ── Train ─────────────────────────────────────────────────────────────────────
nb_model = MultinomialNB()
nb_model.fit(X, y)

svm_model = LinearSVC()
svm_model.fit(X, y)

# ── Save ──────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
joblib.dump(nb_model,   os.path.join(MODEL_DIR, "naive_bayes.pkl"))
joblib.dump(svm_model,  os.path.join(MODEL_DIR, "svm.pkl"))

print("✅ Models trained and saved in models/")