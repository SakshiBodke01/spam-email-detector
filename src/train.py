# src/train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Import preprocessing cleaner
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text

# ── Load dataset from URL ─────────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)

print("[INFO] Downloading SMS Spam Collection dataset...")
df = pd.read_csv(DATASET_URL, sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(f"[INFO] Loaded {len(df)} rows  |  Spam: {df['label'].sum()}  |  Ham: {(df['label']==0).sum()}")

# ── Preprocess Text ───────────────────────────────────────────────────────────
print("[INFO] Preprocessing raw text using NLTK lemmatizer & stopwords removal...")
df["cleaned_message"] = df["message"].apply(clean_text)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_message"], df["label"], test_size=0.2, random_state=42
)

# ── Grid Settings ─────────────────────────────────────────────────────────────
vectorizers = {
    "Count Vectorizer": CountVectorizer(),
    "TF-IDF Vectorizer": TfidfVectorizer()
}

classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine (SVM)": LinearSVC(dual=False),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

# ── Train and Evaluate Combinations ───────────────────────────────────────────
print("\n[INFO] Evaluating Vectorizers and Classifiers Grid...")
for vec_name, vec in vectorizers.items():
    print(f"\n--- Feature Extractor: {vec_name} ---")
    # Transform
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    
    for clf_name, clf in classifiers.items():
        # Train
        clf.fit(X_train_vec, y_train)
        
        # Predict
        preds = clf.predict(X_test_vec)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        results.append({
            "Vectorizer": vec_name,
            "Classifier": clf_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Vectorizer_Obj": vec,
            "Classifier_Obj": clf
        })
        print(f"  > {clf_name} -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

# ── Print Comparison Table ────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("                       MODEL COMPARISON MATRIX")
print("="*80)
print(results_df[["Vectorizer", "Classifier", "Accuracy", "Precision", "Recall", "F1-Score"]].to_string(index=False))
print("="*80)

# Select best model based on F1-Score
best_row = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]
best_vec_name = best_row["Vectorizer"]
best_clf_name = best_row["Classifier"]
print(f"\n[BEST] Best Performing Combination: {best_vec_name} + {best_clf_name} (F1: {best_row['F1-Score']:.4f})")

# ── Retrain on Full Dataset for Production ─────────────────────────────────────
print("\n[INFO] Retraining the best configuration on full dataset for maximum coverage...")
final_vectorizer = vectorizers[best_vec_name]
X_full_vec = final_vectorizer.fit_transform(df["cleaned_message"])
y_full = df["label"]

# We fit a fresh classifier of the same type on the full dataset
if best_clf_name == "Naive Bayes":
    final_clf = MultinomialNB()
elif best_clf_name == "Logistic Regression":
    final_clf = LogisticRegression(max_iter=1000)
elif best_clf_name == "Support Vector Machine (SVM)":
    final_clf = LinearSVC(dual=False)
else:
    final_clf = RandomForestClassifier(n_estimators=100, random_state=42)

final_clf.fit(X_full_vec, y_full)

# ── Save Models ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Save
joblib.dump(final_vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
joblib.dump(final_clf, os.path.join(MODEL_DIR, "model.pkl"))

# Save a text report of comparisons for documentation
with open(os.path.join(MODEL_DIR, "comparison_report.txt"), "w") as f:
    f.write(results_df[["Vectorizer", "Classifier", "Accuracy", "Precision", "Recall", "F1-Score"]].to_string(index=False))

print(f"[INFO] Saved best vectorizer and model to {MODEL_DIR}/\n")