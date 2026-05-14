# src/evaluate.py
import joblib
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_data, preprocess

def evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test, vectorizer = preprocess(df)

    nb = joblib.load("../models/naive_bayes.pkl")
    svm = joblib.load("../models/svm.pkl")

    for model, name in [(nb, "Naive Bayes"), (svm, "SVM")]:
        y_pred = model.predict(X_test)
        print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
