"""
Page 1: Phishing Message Detector
Real-time inference interface using fitted TF-IDF pipelines with heuristic rule diagnostics.
"""

import os
import streamlit as st
import pandas as pd
from src.predict import predict_message, load_model_artifact, DEFAULT_MODEL_PATH
from src.preprocessing import clean_text

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Phishing Detector — Live Inference",
    page_icon="🔍",
    layout="wide",
)

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #0b1120; color: #f1f5f9; }
    
    .status-banner-phish {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.3) 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        text-align: center;
    }
    .status-banner-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        text-align: center;
    }
    .status-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
    }
    .status-sub {
        font-size: 1rem;
        color: #cbd5e1;
    }
    .indicator-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .indicator-title {
        font-weight: 700;
        color: #38bdf8;
        font-size: 1rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fef3c7;
        font-size: 0.85rem;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


SAMPLE_PRESETS = {
    "🏦 High-Risk Bank Phishing": (
        "URGENT SECURITY ALERT: We have detected unauthorized attempts to access your Chase Bank account from an unrecognized IP address. "
        "Your access will be permanently suspended within 24 hours unless you verify your identity immediately. "
        "Click here to confirm your credentials, password, and SSN: http://192.168.1.105/auth/chase-verify"
    ),
    "💻 Urgent IT Helpdesk Spoof": (
        "Attention Employee: Your Microsoft Office 365 password expires today. "
        "Failure to update your password immediately will terminate your corporate email access. "
        "Please update your login credentials here: http://secure-corporate-auth-update.net/login"
    ),
    "🎁 Gift Card Scam": (
        "Congratulations! You have been selected as a Walmart $1,000 gift card winner. "
        "Claim your prize immediately before the offer expires in 12 hours. "
        "Enter your credit card number and billing address to verify delivery: http://free-gift-card-portal.com"
    ),
    "💼 Benign Work Project Email": (
        "Hi Alex,\n\nI reviewed the Q3 machine learning project roadmap. The proposed cross-validation and feature extraction steps look solid. "
        "Let's schedule a 30-minute sync on Thursday at 2:00 PM to finalize the evaluation metrics before our sprint planning meeting.\n\nBest regards,\nDavid"
    ),
    "📦 Benign Delivery Confirmation": (
        "Thank you for your order! Your package has shipped via FedEx and is scheduled for delivery on Monday. "
        "You can track your package progress anytime in your account dashboard. Thank you for shopping with us!"
    ),
}


def get_available_models():
    """List available joblib models in models/."""
    available = {}
    if os.path.exists(DEFAULT_MODEL_PATH):
        available["Best Generalizer (Selected Model)"] = DEFAULT_MODEL_PATH
    for fname in ["logistic_regression.joblib", "naive_bayes.joblib", "decision_tree.joblib", "random_forest.joblib"]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            label = fname.replace(".joblib", "").replace("_", " ").title()
            available[f"{label} Model"] = p
    return available


def main():
    st.title("🔍 Live Phishing Message Detector")
    st.markdown(
        "Enter any raw email body, text message, or notice. "
        "The message is evaluated through the fitted TF-IDF vectorizer and classification pipeline with zero refitting."
    )

    available_models = get_available_models()
    if not available_models:
        st.error(
            "⚠️ No trained model artifacts found in `models/`. "
            "Please run `python -m src.train` in your terminal to train and save baseline models."
        )
        return

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Inference Settings")
        selected_model_label = st.selectbox(
            "Select Model Pipeline:",
            options=list(available_models.keys()),
            index=0,
            help="Select which fitted scikit-learn model pipeline to use for inference.",
        )
        selected_model_path = available_models[selected_model_label]

        st.markdown("---")
        st.markdown("### 📋 Quick-Load Test Messages")
        for preset_name, preset_text in SAMPLE_PRESETS.items():
            if st.button(preset_name, use_container_width=True):
                st.session_state["input_message"] = preset_text

    # Default message in session state
    if "input_message" not in st.session_state:
        st.session_state["input_message"] = SAMPLE_PRESETS["🏦 High-Risk Bank Phishing"]

    # Input text area
    user_text = st.text_area(
        "Message Content:",
        value=st.session_state["input_message"],
        height=180,
        placeholder="Paste an email body, subject, or SMS text here...",
    )

    btn_col1, btn_col2, _ = st.columns([1.5, 1, 4])
    with btn_col1:
        detect_clicked = st.button("🚀 Detect Phishing", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🧹 Clear Input", use_container_width=True):
            st.session_state["input_message"] = ""
            st.rerun()

    if detect_clicked:
        if not user_text.strip():
            st.warning("Please enter some message text to analyze.")
            return

        with st.spinner("Analyzing text via TF-IDF vectorizer and classifier pipeline..."):
            try:
                artifact = load_model_artifact(selected_model_path)
                result = predict_message(user_text, model_artifact=artifact)
            except Exception as e:
                st.error(f"Inference error: {e}")
                return

        # Classification Banner
        is_phishing = result["prediction"] == 1
        prob = result["phishing_probability"]

        if is_phishing:
            st.markdown(f"""
            <div class="status-banner-phish">
                <div class="status-title" style="color: #ef4444;">🚨 CLASSIFIED AS PHISHING</div>
                <div class="status-sub">
                    Model evaluated high deceptive intent, social engineering keywords, or credential harvesting signals.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-banner-safe">
                <div class="status-title" style="color: #10b981;">🛡️ CLASSIFIED AS LEGITIMATE</div>
                <div class="status-sub">
                    Model found typical conversational, transactional, or benign workplace linguistic structures.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability & Confidence Meter
        score_col1, score_col2 = st.columns([1, 1])
        with score_col1:
            st.markdown(f"**Phishing Risk Probability:** `{prob*100:.2f}%`")
            st.progress(float(prob))
        with score_col2:
            st.markdown(f"**Model Used:** `{result['model_used']}` | **Overall Confidence:** `{result['confidence']*100:.2f}%`")

        st.markdown("<br>", unsafe_allow_html=True)

        # Heuristic Analysis Grid
        st.subheader("🔎 Rule-Based Security Heuristics")
        heur = result["heuristics"]

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown(f"""
            <div class="indicator-card">
                <div class="indicator-title">🔗 URLs Detected</div>
                <span style="font-size: 1.5rem; font-weight: 700; color: {'#ef4444' if heur['url_count']>0 else '#10b981'};">
                    {heur['url_count']}
                </span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    {'⚠️ IP Address in URL!' if heur['has_ip_url'] else 'No IP-based URLs'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with h2:
            urg_count = len(heur["urgency_triggers"])
            st.markdown(f"""
            <div class="indicator-card">
                <div class="indicator-title">⏰ Urgency Keywords</div>
                <span style="font-size: 1.5rem; font-weight: 700; color: {'#ef4444' if urg_count>0 else '#10b981'};">
                    {urg_count}
                </span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    {', '.join(heur['urgency_triggers'][:3]) if urg_count else 'None detected'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with h3:
            cred_count = len(heur["credential_triggers"])
            st.markdown(f"""
            <div class="indicator-card">
                <div class="indicator-title">💳 Credential Requests</div>
                <span style="font-size: 1.5rem; font-weight: 700; color: {'#ef4444' if cred_count>0 else '#10b981'};">
                    {cred_count}
                </span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    {', '.join(heur['credential_triggers'][:3]) if cred_count else 'None detected'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with h4:
            st.markdown(f"""
            <div class="indicator-card">
                <div class="indicator-title">📊 Heuristic Risk Index</div>
                <span style="font-size: 1.5rem; font-weight: 700; color: {'#ef4444' if heur['heuristic_risk_score']>50 else '#38bdf8'};">
                    {heur['heuristic_risk_score']} / 100
                </span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    Rule-based aggregate severity
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Sanitized URL inspection (never clickable!)
        if heur["urls_detected"]:
            st.markdown("**Sanitized URL Preview (Defanged for Safety):**")
            for u in heur["urls_detected"]:
                defanged = u.replace("http://", "hxxp://").replace("https://", "hxxps://").replace(".", "[.]")
                st.code(defanged, language="text")

        # Explainability Tokens Breakdown
        tokens = result["influential_tokens"]
        if tokens:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("💡 Token Impact Breakdown (TF-IDF Log-Odds)")
            st.markdown("Words from this email that exerted the greatest mathematical pull on the classification decision:")

            df_tok = pd.DataFrame(tokens)
            st.dataframe(
                df_tok[["token", "direction", "tfidf", "model_weight", "impact"]],
                column_config={
                    "token": "N-Gram Token",
                    "direction": "Pushes Toward",
                    "tfidf": st.column_config.NumberColumn("TF-IDF Score", format="%.4f"),
                    "model_weight": st.column_config.NumberColumn("Model Weight (Log-Odds)", format="%.4f"),
                    "impact": st.column_config.NumberColumn("Total Impact", format="%.4f"),
                },
                use_container_width=True,
                hide_index=True,
            )

        # Cybersecurity Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            🛡️ <b>Security Advisory:</b> Supporting indicators and machine learning classifications are statistical 
            and heuristic estimations. They do not constitute absolute proof of legitimacy or maliciousness. 
            Never execute suspicious links or submit sensitive credentials based solely on automated screening.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
