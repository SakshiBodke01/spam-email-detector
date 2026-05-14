# src/evaluate.py
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

# ── Load models ───────────────────────────────────────────────────────────────
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
nb_model   = joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl"))
svm_model  = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))

# ── Load dataset from URL (no local spam.csv needed) ──────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)

print("📥 Downloading dataset…")
df = pd.read_csv(DATASET_URL, sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print(f"✅ Loaded {len(df)} rows")

X = vectorizer.transform(df["message"])
y = df["label"]

# ── Reports ───────────────────────────────────────────────────────────────────
print("\n── Naive Bayes ──────────────────────────────")
print(classification_report(y, nb_model.predict(X)))

print("── SVM ──────────────────────────────────────")
print(classification_report(y, svm_model.predict(X)))

# ── Confusion matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor("#0F172A")

for ax, model, title in zip(
    axes,
    [nb_model, svm_model],
    ["Naive Bayes", "SVM"],
):
    cm = confusion_matrix(y, model.predict(X))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        ax=ax, linewidths=0.5, linecolor="#1E293B",
        cbar=False,
    )
    ax.set_title(f"{title} Confusion Matrix", color="#CBD5E1", fontsize=11)
    ax.set_xlabel("Predicted", color="#4B6180")
    ax.set_ylabel("Actual",    color="#4B6180")
    ax.tick_params(colors="#4B6180")
    ax.set_facecolor("#1E293B")

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "confusion_matrices.png"), dpi=150,
            bbox_inches="tight", facecolor="#0F172A")
print("\n📊 Confusion matrices saved to confusion_matrices.png")
plt.show()