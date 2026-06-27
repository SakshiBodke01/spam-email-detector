# ThreatMail - Spam Email Detector & Intelligence Dashboard

ThreatMail is an AI-powered email classification and intelligence platform. It processes raw email content to filter out malicious communications (Spam) from legitimate messages (Ham) using trained machine learning classifiers (Naive Bayes and Support Vector Machines), achieving a classification accuracy of **95%+**.

---

## 🚀 Features

1. **Serverless REST API (Flask)**:
   - Lightweight API backend ready to serve real-time predictions.
   - Deployed on **Vercel** with fully resolved paths and size-optimized dependencies.
2. **Interactive Intelligence Dashboard (Streamlit)**:
   - **Real-Time Tester**: Paste any email body to test classification and receive model confidence scores.
   - **Exploratory Analytics**: Visually explore spam vs. ham token distributions and high-frequency terms.
   - **Dynamic Word Clouds**: Compare the most frequent vocabularies of legitimate vs. spam emails.
   - **Model Metrics**: Review precision, recall, F1-scores, and confusion matrices for Naive Bayes and SVM classifiers.

---

## 📁 Repository Structure

```
spam-email-detector/
│
├── models/                   # Serialized ML model binaries (.pkl)
│   ├── naive_bayes.pkl       # Naive Bayes classifier weights
│   ├── svm.pkl               # Trained SVM model binary
│   └── vectorizer.pkl        # TF-IDF CountVectorizer model
│
├── src/                      # Source code
│   ├── app.py                # Serverless Flask API entrypoint (Vercel ready)
│   ├── dashboard.py          # Interactive Streamlit Dashboard application
│   ├── preprocessing.py      # Token cleaning and regex helper methods
│   ├── train.py              # ML training script compiling models
│   └── evaluate.py           # Model testing and validation report generator
│
├── requirements.txt          # Production dependencies (pruned for Vercel)
├── local_requirements.txt    # Local dependencies (including Streamlit & Matplotlib)
└── vercel.json               # Vercel serverless configuration file
```

---

## 🛠️ Installation & Setup

### 1. Local Setup
Clone the repository and install the local requirements (which include the interactive dashboard dependencies):
```bash
git clone https://github.com/SakshiBodke01/spam-email-detector.git
cd spam-email-detector
pip install -r local_requirements.txt
```

### 2. Run the Interactive Dashboard
Launch the Streamlit dashboard on your local server:
```bash
streamlit run src/dashboard.py
```
This will open the intelligence UI in your web browser, typically at `http://localhost:8501`.

### 3. Run the Flask API Locally
Start the local development server for the REST API:
```bash
python src/app.py
```
The API will be active at `http://127.0.0.1:5000`.

---

## 📡 API Usage (Deployed on Vercel)

### 1. Home Endpoint (`GET /`)
Check the service status and usage instructions.
- **Request**:
  ```bash
  curl -X GET https://<your-vercel-app-url>/
  ```
- **Response**:
  ```json
  {
    "status": "active",
    "service": "ThreatMail Spam Intelligence API",
    "description": "An AI-powered service classifying input email content as Spam or Ham.",
    "usage": {
      "method": "POST",
      "endpoint": "/predict",
      "headers": {
        "Content-Type": "application/json"
      },
      "body": {
        "text": "Your raw email text goes here"
      }
    }
  }
  ```

### 2. Predict Endpoint (`POST /predict`)
Send email content to receive a classification and confidence score.
- **Request**:
  ```bash
  curl -X POST https://<your-vercel-app-url>/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Congratulations! You have won a free $1000 Walmart gift card. Click here to claim now."}'
  ```
- **Response**:
  ```json
  {
    "prediction": "Spam",
    "confidence": 1.482093
  }
  ```
