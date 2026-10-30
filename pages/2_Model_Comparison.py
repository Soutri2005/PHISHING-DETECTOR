"""
Page 2: Model Comparison Dashboard
Comparative evaluation of all four baseline classifiers across training, validation,
and test partitions.
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_REPORT_PATH = os.path.join(BASE_DIR, "reports", "training_results.json")

st.set_page_config(
    page_title="Model Comparison — Benchmark",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0b1120; color: #f1f5f9; }
    .card-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .card-title {
        color: #38bdf8;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }
    .cm-table {
        width: 100%;
        border-collapse: collapse;
        text-align: center;
        margin-top: 10px;
    }
    .cm-cell {
        padding: 12px;
        border: 1px solid #334155;
        font-weight: bold;
    }
    .cm-tp { background-color: rgba(16, 185, 129, 0.2); color: #34d399; }
    .cm-tn { background-color: rgba(56, 189, 248, 0.2); color: #38bdf8; }
    .cm-fp { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .cm-fn { background-color: rgba(239, 68, 68, 0.2); color: #f87171; }
</style>
""", unsafe_allow_html=True)


def load_report():
    if not os.path.exists(TRAIN_REPORT_PATH):
        return None
    try:
        with open(TRAIN_REPORT_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    st.title("⚖️ Baseline Model Comparison")
    st.markdown(
        "Direct comparison of four real machine learning classifiers trained on identical "
        "training partitions and evaluated on the held-out validation and test sets."
    )

    report = load_report()
    if not report:
        st.warning("⚠️ Training report not found. Please run `python -m src.train` to generate benchmark metrics.")
        return

    models_data = report["baseline_models"]
    best_info = report["best_model"]

    # Comparative Table Construction
    table_rows = []
    for key, data in models_data.items():
        name = key.replace("_", " ").title()
        table_rows.append({
            "Model": name,
            "Train Accuracy": f"{data['train_accuracy']*100:.2f}%",
            "Val Accuracy": f"{data['val_accuracy']*100:.2f}%",
            "Generalization Gap": f"{data['generalization_gap']*100:.2f}%",
            "Val Precision": f"{data['val_precision']*100:.2f}%",
            "Val Recall": f"{data['val_recall']*100:.2f}%",
            "Val F1-Score": f"{data['val_f1']*100:.2f}%",
            "Training Time (s)": f"{data['training_time_seconds']:.2f}s",
            "_raw_train_acc": data["train_accuracy"],
            "_raw_val_acc": data["val_accuracy"],
            "_raw_gap": data["generalization_gap"],
            "_raw_f1": data["val_f1"],
            "_raw_prec": data["val_precision"],
            "_raw_rec": data["val_recall"],
        })
    df_compare = pd.DataFrame(table_rows)

    # Highlight table
    st.subheader("📊 Performance Benchmark Table")
    st.dataframe(
        df_compare.drop(columns=[col for col in df_compare.columns if col.startswith("_")]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Comparison Charts
    st.subheader("📈 Diagnostic Charts: Accuracy vs Overfitting Gap")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Train vs Val Accuracy
        chart_data = []
        for row in table_rows:
            chart_data.append({"Model": row["Model"], "Split": "Training Accuracy", "Accuracy": row["_raw_train_acc"]})
            chart_data.append({"Model": row["Model"], "Split": "Validation Accuracy", "Accuracy": row["_raw_val_acc"]})
        df_chart_acc = pd.DataFrame(chart_data)

        c_acc = alt.Chart(df_chart_acc).mark_bar().encode(
            x=alt.X("Model:N", title=None),
            y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0.8, 1.0]), title="Accuracy Score"),
            color=alt.Color("Split:N", scale=alt.Scale(range=["#06b6d4", "#10b981"])),
            xOffset="Split:N",
            tooltip=["Model", "Split", alt.Tooltip("Accuracy:Q", format=".4f")],
        ).properties(title="Training vs Validation Accuracy", height=320)
        st.altair_chart(c_acc, use_container_width=True)

    with chart_col2:
        # Generalization Gap Bar Chart
        df_gap = pd.DataFrame([{
            "Model": r["Model"],
            "Generalization Gap": r["_raw_gap"]
        } for r in table_rows])

        c_gap = alt.Chart(df_gap).mark_bar().encode(
            x=alt.X("Model:N", title=None),
            y=alt.Y("Generalization Gap:Q", title="Gap (Train Acc - Val Acc)"),
            color=alt.Color(
                "Generalization Gap:Q",
                scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                legend=None,
            ),
            tooltip=["Model", alt.Tooltip("Generalization Gap:Q", format=".4f")],
        ).properties(title="Generalization Gap (Higher = More Overfitted)", height=320)
        st.altair_chart(c_gap, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix Section
    st.subheader("🎯 Confusion Matrices & Error Trade-offs")
    st.markdown("""
    In cybersecurity, not all errors are equal:
    - **False Positive (FP):** Legitimate email flagged as phishing. Causes annoyance, lost revenue, and communication failure.
    - **False Negative (FN):** Malicious phishing email allowed into the inbox. Causes credential theft, ransomware, and breach.
    """)

    cm_cols = st.columns(4)
    for idx, (m_key, m_data) in enumerate(models_data.items()):
        cm = m_data["val_confusion_matrix"]
        m_name = m_key.replace("_", " ").title()
        with cm_cols[idx]:
            st.markdown(f"**{m_name}**")
            st.markdown(f"""
            <table class="cm-table">
                <tr>
                    <td style="color:#94a3b8; font-size:0.75rem;">Actual \\ Pred</td>
                    <td style="color:#94a3b8; font-size:0.75rem;">Legitimate</td>
                    <td style="color:#94a3b8; font-size:0.75rem;">Phishing</td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; font-size:0.75rem;">Legit</td>
                    <td class="cm-cell cm-tn">TN: {cm['tn']}</td>
                    <td class="cm-cell cm-fp">FP: {cm['fp']}</td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; font-size:0.75rem;">Phish</td>
                    <td class="cm-cell cm-fn">FN: {cm['fn']}</td>
                    <td class="cm-cell cm-tp">TP: {cm['tp']}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Untouched Test Set Evaluation
    st.subheader("🔒 Untouched Test Set Verification (Best Model)")
    test_met = best_info["test_metrics"]
    best_name = best_info["key"].replace("_", " ").title()

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        st.metric("Final Model Selected", best_name)
    with t_col2:
        st.metric("Test Accuracy", f"{test_met['test_accuracy']*100:.2f}%")
    with t_col3:
        st.metric("Test Recall (Catch Rate)", f"{test_met['test_recall']*100:.2f}%")
    with t_col4:
        st.metric("Test F1-Score", f"{test_met['test_f1']*100:.2f}%")

    st.info(
        "💡 **Methodology Check:** The test set was touched exactly once after model selection to verify generalization. "
        "Hyperparameters were never tuned against these test scores."
    )


if __name__ == "__main__":
    main()
