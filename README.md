# ThreatMail Enterprise - Spam Email Detection System

An end-to-end Machine Learning spam classification and NLP threat audit system. It executes comparative training grid searches across 8 configurations (Count Vectorizer, TF-IDF Vectorizer vs Naive Bayes, Logistic Regression, Linear SVM, Random Forest), automatically serializes the best performing model, and serves it via an interactive Flask web console logging transactions to a local SQLite database.

---

## 🚀 Key Features
1. **Grid Comparison Training**: Evaluates 8 combinations of vectorization features and machine learning algorithms, tracking Accuracy, Precision, Recall, and F1-Scores.
2. **NLTK Preprocessing Pipeline**: Tokenization, regex normalizations (URLs, emails, digits), punctuation removal, English stopword stripping, and WordNet Lemmatization.
3. **Enterprise Web UI Console**: Two-column responsive dark console layout:
   - **Left**: Mail Inspector text area with pre-loaded mock email templates for testing.
   - **Right**: NLP Analysis metrics (word counts, capitalization density), dynamic risk checks, and model term weight contributions.
4. **SQLite Prediction Audit Log**: Exposes a real-time table at the bottom of the console pulling the latest prediction transactions logged to the local database file.
5. **Vercel Serverless Ready**: Dynamic model-loading paths, size-optimized dependency configurations, and writable database directories for zero-config serverless deployments.

---

## 📁 Repository Structure
```
spam-email-detector/
│
├── models/                   # Serialized ML model binaries (.pkl)
│   ├── model.pkl             # Best-performing trained classifier
│   ├── vectorizer.pkl        # Best-performing Count/TF-IDF Vectorizer
│   └── comparison_report.txt # Text file summarizing grid metric scores
│
├── src/                      # Source code
│   ├── app.py                # Serverless Flask application & SQLite logger
│   ├── preprocessing.py      # NLTK tokenization, stopwords, and lemmatization
│   ├── train.py              # Grid search training & auto-serialization script
│   └── evaluate.py           # Metrics calculation & confusion matrix generator
│
├── requirements.txt          # Production dependencies (optimized for Vercel size limit)
├── local_requirements.txt    # Local dependencies (for grid training and analytics)
├── vercel.json               # Vercel serverless functions configuration
├── PROJECT_REPORT.md         # Full project report containing architectural diagrams
└── confusion_matrices.png    # Exported Seaborn heatmap of the model outcomes
```

---

## 🛠️ Local Installation & Usage

### 1. Project Setup
Clone the repository and install local training/eval requirements:
```bash
git clone https://github.com/SakshiBodke01/spam-email-detector.git
cd spam-email-detector
pip install -r local_requirements.txt
```

### 2. Execute Grid Training
Run the training pipeline. It downloads the SMS Collection dataset, applies the preprocessing rules, evaluates all 8 model configurations, logs the comparison matrix, and saves the best model to the `models/` directory:
```bash
python src/train.py
```

### 3. Run Performance Evaluation
Print the classification report and save the confusion matrix plot:
```bash
python src/evaluate.py
```

### 4. Start the Web Console
Launch the Flask development server:
```bash
python src/app.py
```
Open `http://127.0.0.1:5000` in your browser. You can input text or click the test templates to observe real-time classifications, risk triggers, and SQLite logging.

---

## 📊 Visuals & System Architecture
Detailed flowcharts, system deployment layouts, database ER diagrams, and mathematical formulations are documented in the **[PROJECT_REPORT.md](PROJECT_REPORT.md)**.
- **System Architecture**: Browser Client $\leftrightarrow$ Flask Web Server $\leftrightarrow$ scikit-learn Predictor $\leftrightarrow$ SQLite DB logger.
- **Database ERD**: Models predictions inside a `prediction_logs` table (`id`, `email_text`, `prediction`, `confidence`, `timestamp`).

---

## 📡 Live Production API

- **GET `/`**: Serves the interactive Enterprise Threat Console UI.
- **POST `/predict`**: Analyzes the raw email text payload and logs to SQLite.
  - *Request Payload*: `{"text": "Your raw email body goes here"}`
  - *Response*:
    ```json
    {
      "prediction": "Spam",
      "confidence": 0.9852,
      "is_probability": true,
      "metadata": { "word_count": 12, "char_count": 84, "caps_ratio": 0.45 },
      "risk_flags": ["Financial / Promotional Keywords"],
      "keywords": [
        { "word": "won", "score": 1, "impact": 1.4820 }
      ]
    }
    ```
- **GET `/logs`**: Retrieves the latest 10 prediction transactions logged in the SQLite audit database.
