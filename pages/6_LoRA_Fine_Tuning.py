"""
Page 6: Parameter-Efficient Fine-Tuning (PEFT / LoRA)
Demonstrates Low-Rank Adaptation of a Pretrained Language Model on Phishing Detection.
"""

import json
import os
import streamlit as st
import pandas as pd
import altair as alt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINETUNE_REPORT_PATH = os.path.join(BASE_DIR, "reports", "finetune_results.json")
DATASET_REPORT_PATH = os.path.join(BASE_DIR, "reports", "finetune_dataset_report.json")
LORA_MODEL_DIR = os.path.join(BASE_DIR, "models", "lora_bert_tiny")

st.set_page_config(
    page_title="PEFT / LoRA Fine-Tuning",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0b1120; color: #f1f5f9; }
    .hero-lora {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #4338ca;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .lora-title {
        color: #818cf8;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .lora-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    .concept-box {
        background: rgba(99, 102, 241, 0.08);
        border-left: 4px solid #6366f1;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-top: 14px;
        font-size: 1rem;
        color: #e0e7ff;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .cm-table {
        width: 100%;
        border-collapse: collapse;
        text-align: center;
        margin-top: 10px;
    }
    .cm-cell {
        padding: 10px;
        border: 1px solid #334155;
        font-weight: bold;
    }
    .cm-tp { background-color: rgba(16, 185, 129, 0.2); color: #34d399; }
    .cm-tn { background-color: rgba(56, 189, 248, 0.2); color: #38bdf8; }
    .cm-fp { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .cm-fn { background-color: rgba(239, 68, 68, 0.2); color: #f87171; }
</style>
""", unsafe_allow_html=True)


def load_json_safe(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    st.markdown("""
    <div class="hero-lora">
        <div class="lora-title">⚡ Parameter-Efficient Fine-Tuning (LoRA / PEFT)</div>
        <div class="lora-subtitle">
            Fine-tuning a pretrained Transformer Language Model (<code>prajjwal1/bert-tiny</code>) on a curated 
            800-email corpus using Low-Rank Adaptation (LoRA).
        </div>
        <div class="concept-box">
            <b>🔬 Core Research Comparison:</b><br>
            <i>"How does Low-Rank Adaptation (LoRA) compare to an untrained base model on a small curated dataset, and how effectively does parameter freezing prevent catastrophic forgetting?"</i>
        </div>
    </div>
    """, unsafe_allow_html=True)

    finetune_data = load_json_safe(FINETUNE_REPORT_PATH)
    dataset_data = load_json_safe(DATASET_REPORT_PATH)

    if not finetune_data:
        st.info("ℹ️ LoRA fine-tuning training results are currently processing or pending. Run `python -m src.finetune_lora` in the terminal to execute.")
        return

    meta = finetune_data["metadata"]
    base_res = finetune_data["baseline_model"]["test_metrics"]
    lora_res = finetune_data["lora_finetuned_model"]["test_metrics"]
    comp = finetune_data["controlled_comparison"]

    # Telemetry Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{meta['base_model']}</div>
            <div class="metric-lbl">Pretrained Base Model</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{meta['parameters_summary']['trainable_percentage']}%</div>
            <div class="metric-lbl">Trainable Parameters (LoRA)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #10b981;">{lora_res['accuracy']*100:.1f}%</div>
            <div class="metric-lbl">LoRA Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #38bdf8;">+{comp['accuracy_improvement']*100:.1f}%</div>
            <div class="metric-lbl">Improvement Over Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for in-depth inspection
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚖️ Baseline vs. LoRA Comparison",
        "🎛️ Architecture & LoRA Decomposition",
        "📈 Loss & Epoch Trajectory",
        "🧪 Live Inference with LoRA",
    ])

    # ==========================================
    # TAB 1: BASELINE VS LORA
    # ==========================================
    with tab1:
        st.subheader("⚖️ Controlled Comparison: Baseline vs. LoRA Fine-Tuned")
        st.markdown("""
        To evaluate parameter-efficient fine-tuning according to strict scientific standards:
        - **Baseline:** Pretrained `bert-tiny` model with an untrained classification head (zero-shot baseline).
        - **LoRA Model:** Identical base weights, but with Low-Rank adaptation matrices ($r=8, \alpha=16$) fine-tuned on the 600-sample training set.
        - **Evaluation:** Evaluated on the identical held-out test split of 100 emails.
        """)

        df_compare = pd.DataFrame([
            {
                "Model Variant": "1. Baseline (Raw Pretrained)",
                "Parameters Trained": "0 (Zero-Shot / Head Only)",
                "Test Accuracy": f"{base_res['accuracy']*100:.2f}%",
                "Test Precision": f"{base_res['precision']*100:.2f}%",
                "Test Recall": f"{base_res['recall']*100:.2f}%",
                "Test F1-Score": f"{base_res['f1']*100:.2f}%",
            },
            {
                "Model Variant": "2. LoRA Fine-Tuned (PEFT)",
                "Parameters Trained": f"{meta['parameters_summary']['trainable_parameters']:,} ({meta['parameters_summary']['trainable_percentage']}%)",
                "Test Accuracy": f"{lora_res['accuracy']*100:.2f}%",
                "Test Precision": f"{lora_res['precision']*100:.2f}%",
                "Test Recall": f"{lora_res['recall']*100:.2f}%",
                "Test F1-Score": f"{lora_res['f1']*100:.2f}%",
            },
        ])

        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎯 Side-by-Side Confusion Matrices")
        cm_col1, cm_col2 = st.columns(2)

        with cm_col1:
            st.markdown("**Baseline (Raw Pretrained Base):**")
            b_cm = base_res["confusion_matrix"]
            st.markdown(f"""
            <table class="cm-table">
                <tr>
                    <td style="color:#94a3b8; font-size:0.8rem;">Actual \\ Pred</td>
                    <td style="color:#94a3b8; font-size:0.8rem;">Legitimate</td>
                    <td style="color:#94a3b8; font-size:0.8rem;">Phishing</td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; font-size:0.8rem;">Legit</td>
                    <td class="cm-cell cm-tn">TN: {b_cm['tn']}</td>
                    <td class="cm-cell cm-fp">FP: {b_cm['fp']}</td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; font-size:0.8rem;">Phish</td>
                    <td class="cm-cell cm-fn">FN: {b_cm['fn']}</td>
                    <td class="cm-cell cm-tp">TP: {b_cm['tp']}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        with cm_col2:
            st.markdown("**LoRA Fine-Tuned (PEFT):**")
            l_cm = lora_res["confusion_matrix"]
            st.markdown(f"""
            <table class="cm-table">
                <tr>
                    <td style="color:#94a3b8; font-size:0.8rem;">Actual \\ Pred</td>
                    <td style="color:#94a3b8; font-size:0.8rem;">Legitimate</td>
                    <td style="color:#94a3b8; font-size:0.8rem;">Phishing</td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; font-size:0.8rem;">Legit</td>
                    <td class="cm-cell cm-tn">TN: {l_cm['tn']}</td>
                    <td class="cm-cell cm-fp">FP: {l_cm['fp']}</td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; font-size:0.8rem;">Phish</td>
                    <td class="cm-cell cm-fn">FN: {l_cm['fn']}</td>
                    <td class="cm-cell cm-tp">TP: {l_cm['tp']}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

    # ==========================================
    # TAB 2: ARCHITECTURE & LORA
    # ==========================================
    with tab2:
        st.subheader("🎛️ How LoRA (Low-Rank Adaptation) Works")
        st.markdown(r"""
        In full fine-tuning, every weight matrix $W \in \mathbb{R}^{d \times k}$ in all transformer layers is updated:
        $$W_{new} = W + \Delta W$$
        This is computationally expensive and risks **catastrophic forgetting**.

        **LoRA decomposes the weight update into two low-rank matrices:**
        $$W_{new} = W + \frac{\alpha}{r} (B \times A)$$
        where:
        - $W$ is **frozen** (zero gradient computation).
        - $A \in \mathbb{R}^{r \times k}$ is initialized with Gaussian noise.
        - $B \in \mathbb{R}^{d \times r}$ is initialized to zero so training starts exactly at the base model state.
        - $r$ is the **rank** (e.g. $r=8$). Because $r \ll \min(d, k)$, the number of trainable parameters is reduced by over **99%**!
        """)

        p_sum = meta["parameters_summary"]
        df_params = pd.DataFrame([
            {"Parameter Category": "Frozen Pretrained Weights", "Count": p_sum["total_parameters"] - p_sum["trainable_parameters"], "Type": "Frozen (Base Model)"},
            {"Parameter Category": "Trainable LoRA Adapters", "Count": p_sum["trainable_parameters"], "Type": "Trainable (Adapters)"},
        ])

        chart_params = alt.Chart(df_params).mark_arc(innerRadius=60).encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Parameter Category:N", scale=alt.Scale(range=["#334155", "#818cf8"])),
            tooltip=["Parameter Category", "Count:Q"],
        ).properties(title="Parameter Efficiency Breakdown", height=280)

        st.altair_chart(chart_params, use_container_width=True)

    # ==========================================
    # TAB 3: TRAINING TRAJECTORY
    # ==========================================
    with tab3:
        st.subheader("📈 Training Loss & Validation Performance Across Epochs")
        history = finetune_data.get("training_history", [])
        if history:
            df_hist = pd.DataFrame(history)

            h_col1, h_col2 = st.columns(2)
            with h_col1:
                c_loss = alt.Chart(df_hist).mark_line(point=True, color="#ef4444").encode(
                    x=alt.X("epoch:O", title="Epoch"),
                    y=alt.Y("train_loss:Q", title="Cross-Entropy Training Loss"),
                    tooltip=["epoch", "train_loss"],
                ).properties(title="Training Loss Trajectory", height=280)
                st.altair_chart(c_loss, use_container_width=True)

            with h_col2:
                c_acc = alt.Chart(df_hist).mark_line(point=True, color="#10b981").encode(
                    x=alt.X("epoch:O", title="Epoch"),
                    y=alt.Y("val_accuracy:Q", scale=alt.Scale(domain=[0.5, 1.0]), title="Validation Accuracy"),
                    tooltip=["epoch", "val_accuracy"],
                ).properties(title="Validation Accuracy Trajectory", height=280)
                st.altair_chart(c_acc, use_container_width=True)

    # ==========================================
    # TAB 4: LIVE INFERENCE
    # ==========================================
    with tab4:
        st.subheader("🧪 Live Inference with LoRA BERT-Tiny")
        st.markdown("Test any message directly through the tokenizer and fine-tuned LoRA neural network:")

        test_input = st.text_area(
            "Enter email or SMS text to test LoRA model:",
            value="URGENT: Your account has been compromised. Log in immediately to verify your identity.",
            height=120,
        )

        if st.button("🚀 Classify with LoRA BERT-Tiny", type="primary"):
            if not os.path.exists(LORA_MODEL_DIR):
                st.error("LoRA model artifact directory not found. Please train first.")
            else:
                with st.spinner("Tokenizing and inferencing with LoRA neural adapter..."):
                    try:
                        import torch
                        from transformers import BertTokenizer, BertForSequenceClassification
                        from peft import PeftModel

                        tokenizer = BertTokenizer.from_pretrained(LORA_MODEL_DIR)
                        base_m = BertForSequenceClassification.from_pretrained(meta["base_model"], num_labels=2)
                        model = PeftModel.from_pretrained(base_m, LORA_MODEL_DIR)
                        model.eval()

                        inputs = tokenizer(test_input, truncation=True, max_length=128, return_tensors="pt")
                        with torch.no_grad():
                            logits = model(**inputs).logits
                            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

                        phish_prob = probs[1]
                        pred_label = "Phishing" if phish_prob >= 0.5 else "Legitimate"

                        st.markdown(f"**Classification:** `{pred_label}` | **Phishing Probability:** `{phish_prob*100:.2f}%`")
                        st.progress(float(phish_prob))

                    except Exception as e:
                        st.error(f"Inference error: {e}")


if __name__ == "__main__":
    main()
