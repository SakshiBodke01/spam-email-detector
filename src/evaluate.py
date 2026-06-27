# src/evaluate.py
import os
import pandas as pd
import joblib
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Import preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

# ── Load chosen best model ────────────────────────────────────────────────────
if not os.path.exists(os.path.join(MODEL_DIR, "model.pkl")):
    print("[ERROR] No trained model found. Please run 'python src/train.py' first.")
    sys.exit(1)

print("[INFO] Loading production vectorizer and classifier...")
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
model      = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))

# ── Load dataset from URL ─────────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)

print("[INFO] Downloading dataset for evaluation...")
df = pd.read_csv(DATASET_URL, sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(f"[INFO] Loaded {len(df)} rows")

# Preprocess
print("[INFO] Cleaning and tokenizing evaluation messages...")
df["cleaned_message"] = df["message"].apply(clean_text)

X = vectorizer.transform(df["cleaned_message"])
y = df["label"]

# ── Predict ───────────────────────────────────────────────────────────────────
print("[INFO] Generating model predictions...")
preds = model.predict(X)

# ── Report Metrics ────────────────────────────────────────────────────────────
acc = accuracy_score(y, preds)
prec = precision_score(y, preds, zero_division=0)
rec = recall_score(y, preds, zero_division=0)
f1 = f1_score(y, preds, zero_division=0)

print("\n" + "="*50)
print("             PRODUCTION MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy:  {acc:.4%}")
print(f"Precision: {prec:.4%}")
print(f"Recall:    {rec:.4%}")
print(f"F1-Score:  {f1:.4%}")
print("="*50)

print("\nDetailed Classification Report:")
print(classification_report(y, preds, target_names=["Ham", "Spam"]))

# ── Confusion Matrix Visual ───────────────────────────────────────────────────
print("[INFO] Rendering confusion matrix plot...")
cm = confusion_matrix(y, preds)
plt.figure(figsize=(6, 5))
fig = plt.gcf()
fig.patch.set_facecolor("#0F172A")

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
    linewidths=0.5, linecolor="#1E293B", cbar=False,
    annot_kws={"size": 14, "weight": "bold"}
)

plt.title("Model Confusion Matrix", color="#F1F5F9", fontsize=14, pad=15)
plt.xlabel("Predicted Label", color="#CBD5E1", fontsize=12, labelpad=10)
plt.ylabel("Actual Label", color="#CBD5E1", fontsize=12, labelpad=10)
plt.tick_params(colors="#94A3B8")

# Style plot background
ax = plt.gca()
ax.set_facecolor("#1E293B")

plt.tight_layout()
plot_path = os.path.join(PROJECT_ROOT, "confusion_matrices.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0F172A")
print(f"[INFO] Plot successfully saved to {plot_path}\n")