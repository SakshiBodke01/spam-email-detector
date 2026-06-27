# src/app.py
import os
from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)

# Dynamically resolve directory paths to prevent FileNotFoundError on serverless runtimes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "..", "models", "svm.pkl"))

@app.route("/", methods=["GET"])
def home():
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ThreatMail · Spam Email Intelligence</title>
        <!-- Bootstrap 5 CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <!-- Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0F172A;
                --surface: #1E293B;
                --border: #2D3F55;
                --text: #CBD5E1;
                --text-heading: #F1F5F9;
                --primary: #3B82F6;
                --success: #22C55E;
                --danger: #EF4444;
            }
            body {
                background-color: var(--bg);
                color: var(--text);
                font-family: 'Outfit', sans-serif;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 2rem 1rem;
            }
            .container {
                max-width: 650px;
            }
            .card-custom {
                background-color: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 2.5rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            }
            h1 {
                font-family: 'Outfit', sans-serif;
                font-weight: 700;
                color: var(--text-heading);
                letter-spacing: -0.02em;
            }
            .mono {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.85rem;
            }
            textarea.form-control {
                background-color: #0f172a;
                border: 1px solid var(--border);
                color: var(--text-heading);
                border-radius: 8px;
                padding: 1rem;
                resize: none;
            }
            textarea.form-control:focus {
                background-color: #0f172a;
                border-color: var(--primary);
                color: var(--text-heading);
                box-shadow: 0 0 15px rgba(59, 130, 246, 0.25);
            }
            .btn-primary-custom {
                background-color: var(--primary);
                color: white;
                font-weight: 600;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                transition: all 0.2s ease;
                text-decoration: none;
            }
            .btn-primary-custom:hover {
                background-color: #2563eb;
                transform: translateY(-1px);
                color: white;
            }
            .btn-primary-custom:active {
                transform: translateY(0);
            }
            .result-container {
                margin-top: 1.5rem;
                display: none;
                border-radius: 8px;
                padding: 1.25rem;
                border: 1px solid transparent;
                animation: fadeIn 0.3s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(5px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .result-spam {
                background-color: rgba(239, 68, 68, 0.1);
                border-color: rgba(239, 68, 68, 0.2);
                color: #fca5a5;
                box-shadow: 0 0 20px rgba(239, 68, 68, 0.1);
            }
            .result-ham {
                background-color: rgba(34, 197, 94, 0.1);
                border-color: rgba(34, 197, 94, 0.2);
                color: #86efac;
                box-shadow: 0 0 20px rgba(34, 197, 94, 0.1);
            }
            .badge-custom {
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                padding: 0.25rem 0.6rem;
                border-radius: 4px;
                font-size: 0.75rem;
            }
            .badge-spam {
                background-color: var(--danger);
                color: white;
            }
            .badge-ham {
                background-color: var(--success);
                color: white;
            }
            .footer {
                margin-top: 2rem;
                text-align: center;
                font-size: 0.8rem;
                color: #64748b;
            }
            .spinner-border-custom {
                width: 1.2rem;
                height: 1.2rem;
                border-width: 0.15em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card-custom">
                <div class="text-center mb-4">
                    <span class="badge bg-secondary mb-2 mono">THREATMAIL · INTEL API</span>
                    <h1>Spam Email Intelligence</h1>
                    <p class="text-muted small">Analyze raw email text instantly using an SVM classifier with TF-IDF vectorization.</p>
                </div>

                <form id="spamForm">
                    <div class="mb-3">
                        <label for="emailText" class="form-label text-white small">Email Content</label>
                        <textarea class="form-control" id="emailText" rows="6" placeholder="Paste the subject and body of the email here..." required></textarea>
                    </div>
                    
                    <button type="submit" class="btn-primary-custom w-100 d-flex align-items-center justify-content-center gap-2" id="submitBtn">
                        <span>Analyze Content</span>
                    </button>
                </form>

                <div class="result-container" id="resultContainer">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h5 class="m-0 fw-bold">Analysis Verdict</h5>
                        <span id="verdictBadge" class="badge-custom"></span>
                    </div>
                    <p id="verdictText" class="m-0 small"></p>
                    <div class="mt-2 pt-2 border-top border-secondary-subtle d-flex justify-content-between text-secondary mono" style="font-size: 0.75rem;">
                        <span>Classifier: SVM (Linear)</span>
                        <span id="confidenceVal"></span>
                    </div>
                </div>
            </div>

            <div class="footer text-secondary small">
                &copy; 2026 ThreatMail. Powered by scikit-learn, Flask, and Vercel.
            </div>
        </div>

        <script>
            document.getElementById('spamForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const text = document.getElementById('emailText').value.trim();
                const submitBtn = document.getElementById('submitBtn');
                const resultContainer = document.getElementById('resultContainer');
                const verdictBadge = document.getElementById('verdictBadge');
                const verdictText = document.getElementById('verdictText');
                const confidenceVal = document.getElementById('confidenceVal');

                // Show loading state
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-custom" role="status" aria-hidden="true"></span> Processing...';
                resultContainer.style.display = 'none';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        const isSpam = data.prediction === 'Spam';
                        
                        // Reset classes
                        resultContainer.className = 'result-container ' + (isSpam ? 'result-spam' : 'result-ham');
                        verdictBadge.className = 'badge-custom ' + (isSpam ? 'badge-spam' : 'badge-ham');
                        
                        // Populate data
                        verdictBadge.innerText = data.prediction;
                        verdictText.innerText = isSpam 
                            ? 'Warning: This message has been flagged as SPAM. It exhibits signature patterns commonly associated with phishing, scams, or unsolicited marketing.' 
                            : 'Safe: This message has been classified as HAM (legitimate). No security threat indicators were detected.';
                        
                        confidenceVal.innerText = 'Confidence: ' + data.confidence.toFixed(4);
                        resultContainer.style.display = 'block';
                    } else {
                        alert('Error: ' + (data.error || 'Failed to process request'));
                    }
                } catch (err) {
                    alert('Network error: ' + err.message);
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = 'Analyze Content';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Empty text provided"}), 400
        
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = model.decision_function(X)[0]
    return jsonify({"prediction": "Spam" if prediction == 1 else "Ham", "confidence": float(confidence)})

if __name__ == "__main__":
    app.run(debug=True)
