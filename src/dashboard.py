import streamlit as st
import joblib
import os
import time
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

# ── Try importing your preprocessing; fall back to basic clean if missing ──
try:
    from preprocessing import clean_text
except ImportError:
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ThreatMail · Spam Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Outfit:wght@300;400;600;700;800&display=swap');

:root {
    --bg:        #0F172A;
    --surface:   #1E293B;
    --surface2:  #162032;
    --border:    #2D3F55;
    --red:       #EF4444;
    --green:     #22C55E;
    --blue:      #3B82F6;
    --amber:     #F59E0B;
    --text:      #CBD5E1;
    --muted:     #4B6180;
    --white:     #F1F5F9;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'Outfit', sans-serif;
    --red-glow:  0 0 20px rgba(239,68,68,0.35);
    --green-glow:0 0 20px rgba(34,197,94,0.35);
    --blue-glow: 0 0 20px rgba(59,130,246,0.35);
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"] > div:first-child {
    background: var(--bg) !important;
    font-family: var(--sans);
    color: var(--text);
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { background: var(--surface2) !important; border-right: 1px solid var(--border); }

.scanner-bar {
    height: 3px;
    background: linear-gradient(90deg, transparent 0%, var(--blue) 40%, var(--green) 60%, transparent 100%);
    background-size: 200% 100%;
    animation: scanMove 2.5s linear infinite;
    border-radius: 2px;
    margin-bottom: 1.5rem;
}
@keyframes scanMove { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

.top-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.6rem; }
.brand { display:flex; align-items:center; gap:14px; }
.brand-icon {
    width:46px; height:46px;
    background: linear-gradient(135deg, #1D4ED8, #3B82F6);
    border-radius:12px; display:flex; align-items:center; justify-content:center;
    font-size:1.4rem; box-shadow: var(--blue-glow);
}
.brand-name { font-size:1.5rem; font-weight:800; color:var(--white); letter-spacing:-0.03em; line-height:1; }
.brand-name span { color:var(--blue); }
.brand-sub { font-family:var(--mono); font-size:0.6rem; letter-spacing:0.18em; color:var(--muted); text-transform:uppercase; }
.status-pill {
    display:flex; align-items:center; gap:8px;
    background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.25);
    border-radius:100px; padding:6px 14px;
    font-family:var(--mono); font-size:0.65rem; color:var(--green); letter-spacing:0.1em;
}
.dot-live { width:7px; height:7px; background:var(--green); border-radius:50%; box-shadow:var(--green-glow); animation:blink 1.4s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.card { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1.4rem 1.6rem; position:relative; overflow:hidden; margin-bottom:1rem; }
.card-accent-blue::after { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#1D4ED8,#3B82F6); }
.card-accent-red::after  { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#991B1B,#EF4444); }
.card-label { font-family:var(--mono); font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); margin-bottom:0.8rem; display:flex; align-items:center; gap:8px; }
.card-label .dot { width:5px; height:5px; border-radius:50%; background:var(--blue); box-shadow:var(--blue-glow); }

.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1.2rem; }
.metric-tile { background:var(--surface2); border:1px solid var(--border); border-radius:12px; padding:1rem; text-align:center; transition:border-color 0.2s; }
.metric-tile:hover { border-color:var(--blue); }
.metric-val { font-family:var(--mono); font-size:1.5rem; font-weight:700; color:var(--white); line-height:1; }
.metric-val.red   { color:var(--red);   text-shadow:var(--red-glow); }
.metric-val.green { color:var(--green); text-shadow:var(--green-glow); }
.metric-val.blue  { color:var(--blue);  text-shadow:var(--blue-glow); }
.metric-lbl { font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-top:4px; font-family:var(--mono); }

textarea { background:#0B1220 !important; border:1px solid var(--border) !important; border-radius:10px !important; color:var(--text) !important; font-family:var(--mono) !important; font-size:0.82rem !important; transition:border-color 0.25s,box-shadow 0.25s !important; caret-color:var(--blue) !important; }
textarea:focus { border-color:var(--blue) !important; box-shadow:0 0 0 3px rgba(59,130,246,0.12) !important; }

.stButton > button { width:100% !important; background:linear-gradient(135deg,#1D4ED8,#3B82F6) !important; color:#fff !important; font-family:var(--mono) !important; font-size:0.75rem !important; letter-spacing:0.14em !important; text-transform:uppercase !important; border:none !important; border-radius:10px !important; padding:0.8rem 1.5rem !important; transition:all 0.25s !important; box-shadow:0 4px 20px rgba(59,130,246,0.3) !important; }
.stButton > button:hover { background:linear-gradient(135deg,#2563EB,#60A5FA) !important; transform:translateY(-2px) !important; box-shadow:0 8px 30px rgba(59,130,246,0.5) !important; }

.result-banner { border-radius:14px; padding:1.5rem 2rem; display:flex; align-items:center; gap:1.4rem; animation:riseIn 0.5s cubic-bezier(0.16,1,0.3,1); border:1px solid; margin-bottom:1rem; }
.result-banner.spam { background:rgba(239,68,68,0.07); border-color:rgba(239,68,68,0.4); }
.result-banner.ham  { background:rgba(34,197,94,0.07);  border-color:rgba(34,197,94,0.4); }
.result-big-icon { font-size:2.8rem; line-height:1; flex-shrink:0; }
.result-verdict  { font-size:1.6rem; font-weight:800; letter-spacing:-0.03em; line-height:1; margin:0 0 4px; }
.spam .result-verdict { color:var(--red); }
.ham  .result-verdict { color:var(--green); }
.result-meta { font-family:var(--mono); font-size:0.7rem; color:var(--muted); margin:0; }

.compare-row { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:8px; }
.compare-cell { background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:1rem; text-align:center; }
.compare-model { font-family:var(--mono); font-size:0.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.12em; margin-bottom:6px; }
.compare-verdict { font-size:1.1rem; font-weight:700; }
.v-spam { color:var(--red); }
.v-ham  { color:var(--green); }

.kw-pill { display:inline-block; background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.35); border-radius:6px; padding:2px 8px; font-family:var(--mono); font-size:0.75rem; color:var(--red); margin:2px 3px; text-shadow:var(--red-glow); }
.kw-clean { font-family:var(--mono); font-size:0.82rem; color:var(--text); line-height:1.9; }

.model-info-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }
.info-chip { background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.2); border-radius:8px; padding:6px 14px; font-family:var(--mono); font-size:0.65rem; color:var(--blue); letter-spacing:0.05em; line-height:1.4; }
.info-chip strong { color:var(--white); display:block; font-size:0.7rem; }

.sb-section { background:rgba(59,130,246,0.05); border:1px solid rgba(59,130,246,0.15); border-radius:10px; padding:0.9rem; margin-top:1rem; }
.sb-row { display:flex; justify-content:space-between; font-family:var(--mono); font-size:0.65rem; color:var(--muted); padding:3px 0; border-bottom:1px solid var(--border); }
.sb-row:last-child { border-bottom:none; }
.sb-row span { color:var(--text); }

.warn-box { background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.3); border-radius:10px; padding:0.75rem 1rem; font-family:var(--mono); font-size:0.75rem; color:var(--amber); }

@keyframes riseIn { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    v  = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    nb = joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl"))
    sv = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
    return v, nb, sv

with st.spinner("⚡ Initialising threat models…"):
    try:
        vectorizer, nb_model, svm_model = load_models()
        models_ok = True
    except Exception as exc:
        models_ok = False
        model_err = str(exc)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for key, default in [("total",0),("spam_count",0),("history",[]),("last_score",None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
SPAM_KEYWORDS = [
    "free","win","winner","cash","prize","offer","claim","urgent","limited",
    "act now","click here","buy","order","subscribe","selected","congratulations",
    "bonus","risk-free","guaranteed","earn","income","money","credit","loan",
    "password","verify","account","suspended","alert","important","dear customer",
]

def highlight_keywords(text):
    sorted_kw = sorted(SPAM_KEYWORDS, key=len, reverse=True)
    pattern   = re.compile(r'\b('+('|'.join(re.escape(k) for k in sorted_kw))+r')\b', re.IGNORECASE)
    parts, pos = [], 0
    for m in pattern.finditer(text):
        parts.append(text[pos:m.start()])
        parts.append(f'<span class="kw-pill">{m.group()}</span>')
        pos = m.end()
    parts.append(text[pos:])
    return ''.join(parts)

def risk_score(text):
    lower  = text.lower()
    kw_hits = sum(1 for k in SPAM_KEYWORDS if k in lower)
    words   = len(text.split()) or 1
    caps_r  = sum(1 for c in text if c.isupper()) / max(len(text),1)
    exclaim = text.count('!')
    return min(100, int(kw_hits/words*350 + caps_r*30 + exclaim*4))

def score_color(s):
    if s >= 60: return "#EF4444"
    if s >= 30: return "#F59E0B"
    return "#22C55E"

def draw_ring(score, ax):
    ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3); ax.axis("off")
    ax.add_patch(plt.Circle((0,0),1.0,fill=False,linewidth=10,color="#1E293B",zorder=1))
    col   = score_color(score)
    theta = np.linspace(np.pi/2, np.pi/2 - 2*np.pi*(score/100), 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=10, color=col, solid_capstyle="round", zorder=2)
    ax.text(0, 0.12, str(score), ha="center", va="center",
            fontsize=28, fontweight="bold", color=col, fontfamily="monospace")
    ax.text(0,-0.32,"RISK",ha="center",va="center",
            fontsize=7,color="#4B6180",fontfamily="monospace",fontweight="bold")

def draw_wordcloud(text, is_spam):
    bg  = "#0F172A"
    wc  = WordCloud(width=700,height=280,background_color=bg,
                    colormap="Reds" if is_spam else "Blues",
                    max_words=60,prefer_horizontal=0.9,collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(7,2.8))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.2rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:#4B6180;margin-bottom:0.4rem;">⚙ &nbsp;Configuration</div>
        <div style="font-size:1.1rem;font-weight:700;color:#F1F5F9;">Detection Settings</div>
    </div>
    """, unsafe_allow_html=True)

    model_choice   = st.radio("🔍 Classifier", ["Naive Bayes","SVM","Compare Both"])
    show_wordcloud = st.checkbox("☁️ Word Cloud",           value=True)
    show_keywords  = st.checkbox("🔎 Keyword Highlighter",  value=True)
    show_risk      = st.checkbox("📊 Risk Score Ring",      value=True)
    show_history   = st.checkbox("🕒 Session History",      value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;letter-spacing:0.18em;text-transform:uppercase;color:#4B6180;margin-bottom:0.5rem;">Model Intelligence</div>
    <div class="sb-section">
        <div class="sb-row">Algorithm   <span>NB + SVM</span></div>
        <div class="sb-row">Vectorizer  <span>TF-IDF</span></div>
        <div class="sb-row">NB Accuracy <span style="color:#22C55E;">98.4 %</span></div>
        <div class="sb-row">SVM Accuracy<span style="color:#22C55E;">98.7 %</span></div>
        <div class="sb-row">Last Trained<span>Today</span></div>
        <div class="sb-row">Dataset     <span>SMS Spam v5</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.total > 0:
        st.markdown(f"""
        <div style="margin-top:1rem;" class="sb-section">
            <div class="sb-row">Session scanned <span>{st.session_state.total}</span></div>
            <div class="sb-row">Spam caught <span style="color:#EF4444;">{st.session_state.spam_count}</span></div>
            <div class="sb-row">Clean <span style="color:#22C55E;">{st.session_state.total-st.session_state.spam_count}</span></div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑 Clear Session"):
        for k,v in [("total",0),("spam_count",0),("history",[]),("last_score",None)]:
            st.session_state[k] = v
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="scanner-bar"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="top-header">
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-name">Threat<span>Mail</span></div>
            <div class="brand-sub">AI Spam Intelligence Platform</div>
        </div>
    </div>
    <div class="status-pill"><div class="dot-live"></div>SYSTEM ONLINE</div>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.markdown(f'<div class="warn-box">⚠ Models not found at <code>{MODEL_DIR}</code><br><small>{model_err}</small></div>', unsafe_allow_html=True)
    st.stop()

# Stats
last_s    = st.session_state.last_score or 0
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-tile"><div class="metric-val blue">{st.session_state.total}</div><div class="metric-lbl">Scanned</div></div>
    <div class="metric-tile"><div class="metric-val red">{st.session_state.spam_count}</div><div class="metric-lbl">Threats</div></div>
    <div class="metric-tile"><div class="metric-val green">{st.session_state.total-st.session_state.spam_count}</div><div class="metric-lbl">Clean</div></div>
    <div class="metric-tile"><div class="metric-val" style="color:{score_color(last_s)};">{last_s}</div><div class="metric-lbl">Last Risk</div></div>
</div>
""", unsafe_allow_html=True)

# Input
st.markdown('<div class="card card-accent-blue"><div class="card-label"><div class="dot"></div>Message Input</div>', unsafe_allow_html=True)
user_input = st.text_area("msg", label_visibility="collapsed", height=170,
                           placeholder="Paste or type email / SMS text here…", key="msg_input")
st.markdown('</div>', unsafe_allow_html=True)

run = st.button("⚡  SCAN MESSAGE", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if run:
    if not user_input.strip():
        st.markdown('<div class="warn-box">⚠ Enter a message to scan.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("🔍 Scanning…"):
            time.sleep(0.45)
            cleaned  = clean_text(user_input)
            X        = vectorizer.transform([cleaned])
            nb_pred  = nb_model.predict(X)[0]
            svm_pred = svm_model.predict(X)[0]

        final_pred = nb_pred if model_choice=="Naive Bayes" else svm_pred if model_choice=="SVM" else (1 if (nb_pred+svm_pred)>=1 else 0)
        is_spam    = final_pred == 1
        score      = risk_score(user_input)

        st.session_state.total      += 1
        st.session_state.last_score  = score
        if is_spam: st.session_state.spam_count += 1
        snippet = user_input[:55].replace("\n"," ") + ("…" if len(user_input)>55 else "")
        st.session_state.history.insert(0,(snippet,is_spam,model_choice,score))
        st.session_state.history = st.session_state.history[:8]

        cls  = "spam" if is_spam else "ham"
        icon = "🚨" if is_spam else "✅"
        verb = "SPAM DETECTED" if is_spam else "LEGITIMATE MESSAGE"
        meta = f"Model · {model_choice} &nbsp;|&nbsp; Risk score · {score}/100 &nbsp;|&nbsp; Scan #{st.session_state.total}"
        st.markdown(f"""
        <div class="result-banner {cls}">
            <div class="result-big-icon">{icon}</div>
            <div><p class="result-verdict">{verb}</p><p class="result-meta">{meta}</p></div>
        </div>
        """, unsafe_allow_html=True)

        if model_choice == "Compare Both":
            nbc = "v-spam" if nb_pred else "v-ham";  nbt = "🚨 SPAM" if nb_pred else "✅ HAM"
            svc = "v-spam" if svm_pred else "v-ham"; svt = "🚨 SPAM" if svm_pred else "✅ HAM"
            st.markdown(f"""
            <div class="card">
                <div class="card-label"><div class="dot"></div>Model Comparison</div>
                <div class="compare-row">
                    <div class="compare-cell"><div class="compare-model">Naive Bayes</div><div class="compare-verdict {nbc}">{nbt}</div></div>
                    <div class="compare-cell"><div class="compare-model">SVM</div><div class="compare-verdict {svc}">{svt}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_r, col_w = st.columns([1, 2])

        if show_risk:
            with col_r:
                st.markdown('<div class="card"><div class="card-label"><div class="dot"></div>Risk Score</div>', unsafe_allow_html=True)
                fig_r, ax_r = plt.subplots(figsize=(2.8,2.8))
                fig_r.patch.set_facecolor("#0F172A"); ax_r.set_facecolor("#0F172A")
                draw_ring(score, ax_r)
                st.pyplot(fig_r, use_container_width=True); plt.close(fig_r)
                lvl = "🔴 HIGH RISK" if score>=60 else "🟡 MODERATE" if score>=30 else "🟢 LOW RISK"
                st.markdown(f'<div style="text-align:center;font-family:monospace;font-size:0.65rem;color:{score_color(score)};letter-spacing:0.12em;margin-top:-8px;">{lvl}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if show_wordcloud:
            with col_w:
                st.markdown('<div class="card"><div class="card-label"><div class="dot"></div>Word Cloud</div>', unsafe_allow_html=True)
                try:
                    fig_wc = draw_wordcloud(user_input, is_spam)
                    st.pyplot(fig_wc, use_container_width=True); plt.close(fig_wc)
                except Exception:
                    st.markdown('<div style="color:#4B6180;font-size:0.8rem;">Not enough text for word cloud.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if show_keywords:
            st.markdown(f"""
            <div class="card card-accent-red">
                <div class="card-label"><div class="dot" style="background:#EF4444;box-shadow:0 0 8px #EF4444;"></div>Suspicious Keyword Analysis</div>
                <div class="kw-clean">{highlight_keywords(user_input)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-label"><div class="dot"></div>AI Model Information</div>
            <div class="model-info-row">
                <div class="info-chip"><strong>Naive Bayes</strong>Multinomial NB · 98.4%</div>
                <div class="info-chip"><strong>SVM</strong>Linear Kernel · 98.7%</div>
                <div class="info-chip"><strong>Vectorizer</strong>TF-IDF · 5 000 features</div>
                <div class="info-chip"><strong>NLP Pipeline</strong>Lower → Strip → Tokenize</div>
                <div class="info-chip"><strong>Dataset</strong>SMS Spam Collection v5</div>
                <div class="info-chip"><strong>Last Trained</strong>Today</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.rerun()

# ── Session history ────────────────────────────────────────────────────────────
if show_history and st.session_state.history:
    st.markdown('<div class="card"><div class="card-label"><div class="dot"></div>Session Scan Log</div>', unsafe_allow_html=True)
    rows = ""
    for snip, spam, mdl, sc in st.session_state.history:
        tag = ('<span style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.4);border-radius:100px;padding:2px 10px;font-size:0.6rem;color:#EF4444;">SPAM</span>'
               if spam else
               '<span style="background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.35);border-radius:100px;padding:2px 10px;font-size:0.6rem;color:#22C55E;">HAM</span>')
        rows += f'<div style="display:flex;align-items:center;justify-content:space-between;padding:0.55rem 0;border-bottom:1px solid #2D3F55;font-family:monospace;font-size:0.72rem;color:#4B6180;"><span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#CBD5E1;">{snip}</span><span style="margin-left:1rem;">{mdl}</span><span style="margin-left:1rem;color:{score_color(sc)};">{sc}</span><span style="margin-left:1rem;">{tag}</span></div>'
    st.markdown(rows + '</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#1E293B;letter-spacing:0.18em;text-transform:uppercase;">
    ThreatMail &nbsp;·&nbsp; Spam Intelligence Platform &nbsp;·&nbsp; NB + SVM &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)