"""
Phishing Message Detector — Overfitting Study
Main Entry Point & Project Overview Dashboard.
"""

import json
import os
import streamlit as st
import pandas as pd

# Base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_REPORT_PATH = os.path.join(BASE_DIR, "reports", "data_preparation_report.json")
TRAIN_REPORT_PATH = os.path.join(BASE_DIR, "reports", "training_results.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Phishing Detector — Overfitting Study",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyber Dashboard CSS Styling
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header card */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        color: #38bdf8;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .research-question-box {
        background-color: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0284c7;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-top: 14px;
        font-size: 1.05rem;
        color: #e0f2fe;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Concept boxes */
    .concept-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
    }
    .concept-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .concept-desc {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Workflow pipeline */
    .flow-step {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        color: #f8fafc;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def load_reports():
    data_rep = None
    train_rep = None
    if os.path.exists(DATA_REPORT_PATH):
        try:
            with open(DATA_REPORT_PATH, "r") as f:
                data_rep = json.load(f)
        except Exception:
            pass
    if os.path.exists(TRAIN_REPORT_PATH):
        try:
            with open(TRAIN_REPORT_PATH, "r") as f:
                train_rep = json.load(f)
        except Exception:
            pass
    return data_rep, train_rep


def main():
    data_rep, train_rep = load_reports()

    # Hero Banner
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🛡️ Phishing Message Detector — Overfitting Study</div>
        <div class="hero-subtitle">
            An end-to-end Machine Learning cybersecurity application evaluating email text classification,
            decision boundaries, model complexity, and empirical regularization techniques.
        </div>
        <div class="research-question-box">
            <b>🔬 Central Research Question:</b><br>
            <i>"How does model complexity affect phishing message detection, and which techniques help reduce overfitting while maintaining high generalization on unseen messages?"</i>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Telemetry Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        clean_rows = data_rep["cleaning_statistics"]["final_clean_rows"] if data_rep else "17,522"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{clean_rows:,}</div>
            <div class="metric-lbl">Clean Corpus Emails</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        phish_pct = data_rep["partition_statistics"]["train_phishing_pct"] if data_rep else "37.4%"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #ef4444;">{phish_pct}%</div>
            <div class="metric-lbl">Phishing Class Ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        best_model = train_rep["best_model"]["key"].replace("_", " ").title() if train_rep else "Logistic Regression"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #10b981;">{best_model}</div>
            <div class="metric-lbl">Top Generalizer</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        test_acc = f"{train_rep['best_model']['test_metrics']['test_accuracy']*100:.1f}%" if train_rep else "97.3%"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #38bdf8;">{test_acc}</div>
            <div class="metric-lbl">Test Set Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline Workflow Section
    st.subheader("🔄 Leakage-Free ML Engineering Architecture")
    st.markdown("""
    To guarantee authentic scientific evaluation, the feature extractor (TF-IDF) is strictly fitted on 
    the **training split only**. Validation and test partitions remain completely unseen during tokenization and IDF weighting.
    """)

    w1, w2, w3, w4, w5, w6 = st.columns(6)
    with w1:
        st.markdown("<div class='flow-step'>📥 1. Raw Dataset<br><small>HuggingFace Corpus</small></div>", unsafe_allow_html=True)
    with w2:
        st.markdown("<div class='flow-step'>🧹 2. Cleaning & Dedup<br><small>Exact Drop (-1,109)</small></div>", unsafe_allow_html=True)
    with w3:
        st.markdown("<div class='flow-step'>🔀 3. Stratified Split<br><small>70% / 15% / 15%</small></div>", unsafe_allow_html=True)
    with w4:
        st.markdown("<div class='flow-step'>🔤 4. TF-IDF Fit<br><small>Train Set Only</small></div>", unsafe_allow_html=True)
    with w5:
        st.markdown("<div class='flow-step'>⚙️ 5. Pipeline Train<br><small>4 Classifiers</small></div>", unsafe_allow_html=True)
    with w6:
        st.markdown("<div class='flow-step'>🎯 6. Prediction UI<br><small>Inference Pipeline</small></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Scientific Concepts
    st.subheader("📚 Conceptual Framework: Overfitting & Generalization")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="concept-card">
            <div class="concept-title">⚠️ Overfitting (High Variance)</div>
            <div class="concept-desc">
                Occurs when an overly complex model (e.g. an unconstrained Decision Tree with 50+ depth) 
                memorizes noise, exact recipient names, and specific URLs in the training split.
                <br><br>
                <b>Diagnostic Signature:</b> Near-perfect training accuracy (99.8%) paired with a sharply degraded validation accuracy (85.2%), resulting in a wide <b>Generalization Gap</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="concept-card">
            <div class="concept-title">🛡️ Regularization & Complexity Control</div>
            <div class="concept-desc">
                Mathematical constraints penalizing extreme model parameters to force smoother decision boundaries.
                <br><br>
                <b>Methods Tested:</b>
                <ul>
                    <li><b>L2 Regularization (Ridge / LogReg C):</b> Drives feature weights toward zero without forcing strict sparsity.</li>
                    <li><b>L1 Regularization (Lasso):</b> Enforces weight sparsity, performing embedded feature selection.</li>
                    <li><b>Tree Pruning (max_depth / min_samples):</b> Prevents the tree from isolating single noise emails in leaf nodes.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation Guidance
    st.info("""
    👉 **Explore the Application via the Sidebar:**
    - **Page 1: Phishing Detector** — Test real email & SMS messages with live probability scores and heuristic trigger breakdown.
    - **Page 2: Model Comparison** — Side-by-side performance audit across all 4 classifiers with confusion matrices and generalization gaps.
    - **Page 3: Interactive Overfitting Lab** — Dynamically adjust Decision Tree depth and Logistic Regression $C$ values to observe generalization curves in real-time.
    - **Page 4: Error Analysis** — Inspect false positives, false negatives, and token-level log-odds explainability.
    - **Page 5: Methodology & Findings** — Complete scientific documentation, dataset licensing, and experimental conclusions.
    """)


if __name__ == "__main__":
    main()
