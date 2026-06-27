# src/app.py
import os
from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)

# Dynamically resolve directory paths to prevent FileNotFoundError on serverless runtimes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "..", "models", "svm.pkl"))

# src/app.py
import os
import re
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
        <title>ThreatMail Enterprise · Spam Intelligence Hub</title>
        <!-- Bootstrap 5 CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <!-- Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #090d16;
                --surface: #111827;
                --surface-card: #1f2937;
                --border: #1f2937;
                --border-light: #374151;
                --text: #9ca3af;
                --text-bright: #f3f4f6;
                --primary: #3b82f6;
                --primary-glow: rgba(59, 130, 246, 0.15);
                --success: #10b981;
                --success-glow: rgba(16, 185, 129, 0.1);
                --danger: #f43f5e;
                --danger-glow: rgba(244, 63, 94, 0.1);
                --mono: 'IBM Plex Mono', monospace;
            }
            body {
                background-color: var(--bg);
                color: var(--text);
                font-family: 'Outfit', sans-serif;
                min-height: 100vh;
                padding: 1.5rem 0;
            }
            .header-banner {
                background: linear-gradient(90deg, #1e293b, var(--surface));
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.5rem 2rem;
                margin-bottom: 2rem;
            }
            .card-glass {
                background-color: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            .section-title {
                font-size: 1.1rem;
                font-weight: 600;
                color: var(--text-bright);
                margin-bottom: 1.25rem;
                display: flex;
                align-items: center;
                gap: 8px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            textarea.form-control {
                background-color: #0b0f17;
                border: 1px solid var(--border-light);
                color: var(--text-bright);
                border-radius: 8px;
                padding: 1rem;
                resize: none;
                font-size: 0.9rem;
            }
            textarea.form-control:focus {
                background-color: #0b0f17;
                border-color: var(--primary);
                color: var(--text-bright);
                box-shadow: 0 0 15px var(--primary-glow);
            }
            .btn-primary-custom {
                background-color: var(--primary);
                color: white;
                font-weight: 600;
                border: none;
                padding: 0.85rem 1.5rem;
                border-radius: 8px;
                transition: all 0.2s ease;
                font-size: 0.95rem;
            }
            .btn-primary-custom:hover {
                background-color: #2563eb;
                transform: translateY(-1px);
            }
            .btn-primary-custom:disabled {
                background-color: #1d4ed8;
                opacity: 0.7;
                cursor: not-allowed;
            }
            .template-btn {
                background-color: #1e293b;
                border: 1px solid var(--border);
                color: var(--text-bright);
                font-size: 0.75rem;
                padding: 6px 12px;
                border-radius: 6px;
                transition: all 0.2s ease;
                cursor: pointer;
            }
            .template-btn:hover {
                background-color: #334155;
                border-color: var(--border-light);
            }
            .placeholder-box {
                border: 2px dashed var(--border-light);
                border-radius: 8px;
                padding: 4rem 2rem;
                text-align: center;
                color: #6b7280;
            }
            .verdict-card {
                padding: 1.25rem 1.5rem;
                border-radius: 8px;
                border: 1px solid transparent;
                margin-bottom: 1.5rem;
            }
            .bg-danger-subtle-custom {
                background-color: var(--danger-glow);
                border-color: rgba(244, 63, 94, 0.2);
            }
            .bg-success-subtle-custom {
                background-color: var(--success-glow);
                border-color: rgba(16, 185, 129, 0.2);
            }
            .badge-custom {
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                padding: 0.3rem 0.6rem;
                border-radius: 4px;
            }
            .badge-spam {
                background-color: var(--danger);
                color: white;
            }
            .badge-ham {
                background-color: var(--success);
                color: white;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
                margin-bottom: 1.5rem;
            }
            .metric-box {
                background-color: #0b0f17;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 0.85rem 1rem;
            }
            .metric-val {
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--text-bright);
                font-family: var(--mono);
            }
            .table-custom {
                width: 100%;
                margin-bottom: 0;
                border-collapse: collapse;
            }
            .table-custom th {
                color: #6b7280;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                padding: 8px 12px;
                border-bottom: 1px solid var(--border-light);
            }
            .table-custom td {
                padding: 10px 12px;
                border-bottom: 1px solid var(--border);
                font-size: 0.85rem;
            }
            .table-custom tr:last-child td {
                border-bottom: none;
            }
            .progress-bar-custom {
                background-color: #0f172a;
                border: 1px solid var(--border-light);
                height: 10px;
                border-radius: 6px;
                overflow: hidden;
            }
            .footer {
                text-align: center;
                font-size: 0.8rem;
                color: #4b5563;
                margin-top: 3rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Corporate Top Banner -->
            <div class="header-banner d-flex justify-content-between align-items-center flex-wrap gap-3">
                <div>
                    <span class="badge bg-primary mb-2" style="font-size: 0.7rem; font-family: var(--mono);">ENTERPRISE GRADE</span>
                    <h1 class="m-0 h2 text-white">ThreatMail Intelligence Console</h1>
                    <p class="text-muted m-0 small">Real-time NLP analysis and SVM boundary decision metrics for email classification.</p>
                </div>
                <div class="d-flex align-items-center gap-2">
                    <span class="badge bg-success" style="font-family: var(--mono);">SYSTEM ACTIVE</span>
                    <span class="badge bg-secondary" style="font-family: var(--mono);">SVM 95%+ ACCURACY</span>
                </div>
            </div>

            <!-- Main Layout Grid -->
            <div class="row">
                <!-- Left Input Panel -->
                <div class="col-lg-5 mb-4 mb-lg-0">
                    <div class="card-glass h-100 d-flex flex-column justify-content-between">
                        <div>
                            <div class="section-title">
                                <span>📥</span> Mail Inspector
                            </div>
                            <p class="text-muted small mb-3">Paste the raw email header, subject, or message body below to execute the classifier check.</p>
                            
                            <form id="spamForm">
                                <div class="mb-4">
                                    <textarea class="form-control" id="emailText" rows="10" placeholder="Type or paste suspicious text here..." required></textarea>
                                </div>
                                <button type="submit" class="btn-primary-custom w-100" id="submitBtn">
                                    Analyze Content
                                </button>
                            </form>
                        </div>
                        
                        <div class="mt-4 pt-3 border-top border-secondary-subtle">
                            <span class="text-white small d-block mb-2 font-monospace">Quick test templates:</span>
                            <div class="d-flex gap-2 flex-wrap">
                                <button class="template-btn" onclick="loadTemplate('clean')">Clean Mail</button>
                                <button class="template-btn" onclick="loadTemplate('phish')">Lottery Spam</button>
                                <button class="template-btn" onclick="loadTemplate('urgency')">Urgency Scam</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Right Output Metrics Panel -->
                <div class="col-lg-7">
                    <div class="card-glass h-100">
                        <div class="section-title">
                            <span>📊</span> NLP & Security Analysis
                        </div>

                        <!-- Awaiting Analysis Placeholder -->
                        <div id="verdictPlaceholder" class="placeholder-box">
                            <div class="fs-1 mb-2">⚡</div>
                            <h5 class="text-white mb-2">Awaiting Intelligence Stream</h5>
                            <p class="m-0 small text-muted" style="max-width: 400px; margin: 0 auto;">Input email content in the Inspector panel and submit to view SVM classifier weights, risk flags, and NLP attributes.</p>
                        </div>

                        <!-- Full Analytical Dashboard (Hidden initially) -->
                        <div id="analysisResultPanel" style="display: none;">
                            <!-- Verdict Section -->
                            <div id="verdictCard" class="verdict-card d-flex justify-content-between align-items-center">
                                <div>
                                    <div class="d-flex align-items-center gap-2 mb-1">
                                        <h4 id="verdictTitle" class="m-0"></h4>
                                        <span id="verdictBadge" class="badge-custom"></span>
                                    </div>
                                    <p id="verdictDesc" class="m-0 small text-white-50"></p>
                                </div>
                                <div class="text-end">
                                    <span class="text-muted d-block small font-monospace">SVM Boundary Dist</span>
                                    <span id="confidenceVal" class="h4 text-white font-monospace"></span>
                                </div>
                            </div>

                            <!-- Decision Meter -->
                            <div class="mb-4">
                                <div class="d-flex justify-content-between text-muted small mb-1 font-monospace" style="font-size: 0.75rem;">
                                    <span>← Ham (Safe Zone)</span>
                                    <span>SVM Decision Plane (0.0)</span>
                                    <span>Spam (Threat Zone) →</span>
                                </div>
                                <div class="progress-bar-custom">
                                    <div id="confidenceMeter" class="progress-bar" role="progressbar" style="width: 50%;"></div>
                                </div>
                            </div>

                            <!-- Metrics Grid -->
                            <div class="metrics-grid">
                                <div class="metric-box">
                                    <span class="text-muted small d-block font-monospace" style="font-size: 0.7rem;">Word Count</span>
                                    <span id="metaWords" class="metric-val"></span>
                                </div>
                                <div class="metric-box">
                                    <span class="text-muted small d-block font-monospace" style="font-size: 0.7rem;">Character Length</span>
                                    <span id="metaChars" class="metric-val"></span>
                                </div>
                                <div class="metric-box">
                                    <span class="text-muted small d-block font-monospace" style="font-size: 0.7rem;">Capitalization Ratio</span>
                                    <span id="metaCaps" class="metric-val"></span>
                                </div>
                                <div class="metric-box">
                                    <span class="text-muted small d-block font-monospace" style="font-size: 0.7rem;">Exclamations</span>
                                    <span id="metaExcl" class="metric-val"></span>
                                </div>
                            </div>

                            <!-- Threat Flags checklist -->
                            <div class="mb-4">
                                <span class="text-muted small d-block mb-2 font-monospace" style="font-size: 0.75rem;">Detected Security Risk Factors:</span>
                                <div class="card p-3" style="background-color: #0b0f17; border-color: var(--border);" id="riskFlagsContainer">
                                </div>
                            </div>

                            <!-- Feature Coefficients Table -->
                            <div>
                                <span class="text-muted small d-block mb-2 font-monospace" style="font-size: 0.75rem;">Top NLP Term Classifier Coefficients:</span>
                                <div class="table-responsive" style="border: 1px solid var(--border); border-radius: 8px; overflow: hidden;">
                                    <table class="table-custom">
                                        <thead>
                                            <tr>
                                                <th class="text-start">Vocabulary Term</th>
                                                <th class="text-center">Count</th>
                                                <th class="text-end">SVM Impact Weight</th>
                                            </tr>
                                        </thead>
                                        <tbody id="keywordsTableBody">
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <div class="footer">
                &copy; 2026 ThreatMail Enterprise. Built with Flask, scikit-learn, and Vercel.
            </div>
        </div>

        <script>
            const templates = {
                clean: "Hi Rohan, hope you are doing well. Just wanted to confirm our meeting scheduled for tomorrow at 10 AM in the conference room. Let me know if you need any adjustments.",
                phish: "CONGRATULATIONS! You have been selected as the grand prize winner of $1,000,000 CASH PRIZE! Click here to claim your free reward now: www.freeprizelink.com/reward. Deal expires soon!",
                urgency: "URGENT SECURITY ALERT: We detected unauthorized login attempts on your banking account. Please verify your identity immediately to prevent block. Expire in 10 minutes."
            };

            function loadTemplate(type) {
                document.getElementById('emailText').value = templates[type];
            }

            document.getElementById('spamForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const text = document.getElementById('emailText').value.trim();
                const submitBtn = document.getElementById('submitBtn');
                const verdictPlaceholder = document.getElementById('verdictPlaceholder');
                const analysisResultPanel = document.getElementById('analysisResultPanel');

                // Show loading state
                submitBtn.disabled = true;
                submitBtn.innerHTML = 'Executing NLP Classifier...';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Render panels
                        verdictPlaceholder.style.display = 'none';
                        analysisResultPanel.style.display = 'block';
                        
                        const isSpam = data.prediction === 'Spam';
                        
                        // Verdict card style
                        const verdictCard = document.getElementById('verdictCard');
                        verdictCard.className = 'verdict-card d-flex justify-content-between align-items-center ' + 
                            (isSpam ? 'bg-danger-subtle-custom' : 'bg-success-subtle-custom');
                        
                        // Verdict Title
                        const verdictTitle = document.getElementById('verdictTitle');
                        verdictTitle.innerText = isSpam ? 'Threat Flagged' : 'Legitimate';
                        verdictTitle.className = isSpam ? 'text-danger m-0 fw-bold h4' : 'text-success m-0 fw-bold h4';
                        
                        // Verdict Badge
                        const verdictBadge = document.getElementById('verdictBadge');
                        verdictBadge.innerText = data.prediction;
                        verdictBadge.className = 'badge-custom ' + (isSpam ? 'badge-spam' : 'badge-ham');
                        
                        // Verdict Desc
                        document.getElementById('verdictDesc').innerText = isSpam 
                            ? 'High probability of spam, phishing, or financial scan indicators.' 
                            : 'Normal linguistic structure with no threat patterns detected.';
                        
                        // SVM Value
                        document.getElementById('confidenceVal').innerText = (data.confidence > 0 ? '+' : '') + data.confidence.toFixed(4);
                        
                        // Meter Bar
                        // LinearSVC boundary decision value. Decision boundary is 0.0. 
                        // Map -2.0 to 2.0 to 10% to 90% range
                        let scorePercent = 50 + (data.confidence * 20);
                        scorePercent = Math.max(5, Math.min(95, scorePercent));
                        const meter = document.getElementById('confidenceMeter');
                        meter.style.width = scorePercent + '%';
                        meter.className = 'progress-bar ' + (isSpam ? 'bg-danger' : 'bg-success');
                        
                        // Metadata
                        document.getElementById('metaWords').innerText = data.metadata.word_count;
                        document.getElementById('metaChars').innerText = data.metadata.char_count;
                        document.getElementById('metaCaps').innerText = (data.metadata.caps_ratio * 100).toFixed(1) + '%';
                        document.getElementById('metaExcl').innerText = data.metadata.excl_count;
                        
                        // Risk Checklist
                        const flagsContainer = document.getElementById('riskFlagsContainer');
                        flagsContainer.innerHTML = '';
                        if (data.risk_flags.length === 0) {
                            flagsContainer.innerHTML = '<div class="text-success small">✓ Verified Clean: No suspicious threat vectors detected.</div>';
                        } else {
                            data.risk_flags.forEach(flag => {
                                flagsContainer.innerHTML += `<div class="d-flex align-items-center gap-2 mb-2">
                                    <span class="badge bg-danger-subtle text-danger px-2 py-1" style="font-size: 0.65rem;">FLAGGED</span>
                                    <span class="small text-white-50">${flag}</span>
                                </div>`;
                            });
                        }
                        
                        // Keywords coefficients table
                        const kwTable = document.getElementById('keywordsTableBody');
                        kwTable.innerHTML = '';
                        if (data.keywords.length === 0) {
                            kwTable.innerHTML = '<tr><td colspan="3" class="text-center text-muted small py-3">No matched key terms in model vocabulary.</td></tr>';
                        } else {
                            data.keywords.forEach(kw => {
                                const isPositive = kw.impact > 0;
                                const arrow = isPositive 
                                    ? '<span class="text-danger">Spam bias</span>' 
                                    : '<span class="text-success">Ham bias</span>';
                                kwTable.innerHTML += `<tr>
                                    <td class="font-monospace text-white">${kw.word}</td>
                                    <td class="text-center text-white-50 font-monospace">${kw.score}</td>
                                    <td class="text-end font-monospace">${(kw.impact > 0 ? '+' : '') + kw.impact.toFixed(4)} (${arrow})</td>
                                </tr>`;
                            });
                        }
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
        
    # Text metadata
    word_count = len(text.split())
    char_count = len(text)
    
    # Calculate caps ratio
    letters = [c for c in text if c.isalpha()]
    caps = [c for c in letters if c.isupper()]
    caps_ratio = len(caps) / len(letters) if len(letters) > 0 else 0.0
    
    # Exclamation count
    excl_count = text.count('!')
    
    # Currency symbols & cash terms
    money_symbols = ['$', '€', '£', '₹']
    money_count = sum(text.count(s) for s in money_symbols)
    
    # Link detection
    has_links = 1 if re.search(r"http\S+|www\S+|\.com\b|\.net\b|\.org\b", text.lower()) else 0
    
    # Urgency keywords
    urgency_words = ['urgent', 'immediate', 'act fast', 'limited time', 'now', 'hurry', 'last chance', 'expire']
    urgency_count = sum(1 for w in urgency_words if w in text.lower())
    
    # Build risk flags
    risk_flags = []
    if caps_ratio > 0.25:
        risk_flags.append(f"High Uppercase Density ({caps_ratio:.1%})")
    if excl_count > 2:
        risk_flags.append(f"Excessive Punctuation ({excl_count} exclamations)")
    if money_count > 0 or any(w in text.lower() for w in ['free', 'cash', 'won', 'prize', 'claim', 'credit']):
        risk_flags.append("Financial / Promotional Keywords")
    if has_links:
        risk_flags.append("External Link References")
    if urgency_count > 0:
        risk_flags.append("Urgency / Pressure Wording")
        
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = model.decision_function(X)[0]
    
    # Get feature weights for terms in the vocabulary
    feature_index = X.nonzero()[1]
    raw_counts = X.data
    
    keywords = []
    if len(feature_index) > 0:
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()
            
        for idx, count in zip(feature_index, raw_counts):
            word = feature_names[idx]
            # SVM model coefficients weight
            coef = float(model.coef_[0][idx])
            keywords.append({
                "word": word,
                "score": int(count), # count frequency
                "impact": coef
            })
        # Sort by coefficient magnitude descending (most impactful first)
        keywords = sorted(keywords, key=lambda x: abs(x["impact"]), reverse=True)[:5]
        
    return jsonify({
        "prediction": "Spam" if prediction == 1 else "Ham",
        "confidence": float(confidence),
        "metadata": {
            "word_count": word_count,
            "char_count": char_count,
            "caps_ratio": caps_ratio,
            "excl_count": excl_count,
            "money_count": money_count,
            "has_links": bool(has_links),
            "urgency_count": urgency_count
        },
        "risk_flags": risk_flags,
        "keywords": keywords
    })

if __name__ == "__main__":
    app.run(debug=True)
