"""
Page 3: Interactive Overfitting Lab
Core research lab demonstrating empirical relationship between model complexity,
hyperparameters, regularization, and the generalization gap.
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from src.preprocessing import create_pipeline
from src.evaluate import evaluate_model
from src.train import load_partitions, MAX_FEATURES, NGRAM_RANGE, RANDOM_SEED

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

DT_CSV_PATH = os.path.join(REPORTS_DIR, "dt_complexity_experiments.csv")
LR_CSV_PATH = os.path.join(REPORTS_DIR, "lr_regularization_experiments.csv")
LC_CSV_PATH = os.path.join(REPORTS_DIR, "learning_curve_experiments.csv")

st.set_page_config(
    page_title="Overfitting Lab — Experiments",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0b1120; color: #f1f5f9; }
    .lab-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .lab-title {
        color: #38bdf8;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 8px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .badge-green { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-red { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-cyan { background-color: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #0284c7; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_partitions():
    return load_partitions()


def load_csv_experiment(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def main():
    st.title("🧪 Interactive Overfitting Lab")
    st.markdown(
        "Explore how **model capacity**, **tree depth**, and **regularization penalties** "
        "shape the boundary between healthy generalization and severe memorization of noise."
    )

    tab1, tab2, tab3 = st.tabs([
        "🌲 Decision Tree Complexity",
        "⚖️ Logistic Regression Regularization",
        "📈 Empirical Learning Curves",
    ])

    # ==========================================
    # TAB 1: DECISION TREE COMPLEXITY
    # ==========================================
    with tab1:
        st.subheader("🌲 Experiment A: Tree Depth & Capacity vs Generalization")
        st.markdown("""
        Decision Trees split features greedily. When given **unlimited depth**, the tree splits until nearly every 
        training email resides in its own pure leaf node. In text classification with 5,000 TF-IDF features, 
        this causes the tree to memorize noise words (e.g. random employee names, timestamps), driving 
        **Training Accuracy to ~100%** while **Validation Accuracy degrades**, yielding a severe **Generalization Gap**.
        """)

        df_dt = load_csv_experiment(DT_CSV_PATH)
        if not df_dt.empty:
            st.markdown("#### 📊 Pre-computed Depth Sweep Results")
            c_dt1, c_dt2 = st.columns(2)
            with c_dt1:
                # Accuracy curve
                chart_acc = []
                for _, row in df_dt.iterrows():
                    chart_acc.append({"Depth": row["max_depth_param"], "Split": "Train Accuracy", "Score": row["train_accuracy"]})
                    chart_acc.append({"Depth": row["max_depth_param"], "Split": "Val Accuracy", "Score": row["val_accuracy"]})
                df_c_acc = pd.DataFrame(chart_acc)

                c1 = alt.Chart(df_c_acc).mark_line(point=True).encode(
                    x=alt.X("Depth:N", sort=None, title="Max Depth"),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.75, 1.01]), title="Accuracy"),
                    color=alt.Color("Split:N", scale=alt.Scale(range=["#06b6d4", "#10b981"])),
                    tooltip=["Depth", "Split", alt.Tooltip("Score:Q", format=".4f")],
                ).properties(title="Accuracy Divergence: Overfitting Signature", height=320)
                st.altair_chart(c1, use_container_width=True)

            with c_dt2:
                # Gap curve
                c2 = alt.Chart(df_dt).mark_bar().encode(
                    x=alt.X("max_depth_param:N", sort=None, title="Max Depth"),
                    y=alt.Y("generalization_gap:Q", title="Generalization Gap (Train - Val)"),
                    color=alt.Color("generalization_gap:Q", scale=alt.Scale(scheme="reds"), legend=None),
                    tooltip=["max_depth_param", "leaf_count", alt.Tooltip("generalization_gap:Q", format=".4f")],
                ).properties(title="Generalization Gap vs Max Depth", height=320)
                st.altair_chart(c2, use_container_width=True)

            with st.expander("📄 View Decision Tree Experiment Data Table"):
                st.dataframe(
                    df_dt[["max_depth_param", "actual_depth", "leaf_count", "train_accuracy", "val_accuracy", "generalization_gap", "val_f1", "training_time"]],
                    use_container_width=True,
                    hide_index=True,
                )
                csv_bytes = df_dt.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Export Decision Tree Data (CSV)", csv_bytes, "dt_complexity_results.csv", "text/csv")

        st.markdown("---")
        st.markdown("#### 🎛️ Live Custom Decision Tree Trainer")
        st.markdown("Select custom pruning parameters to train an isolated tree on the fly:")

        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 1.5, 1.5])
        with ctrl_col1:
            custom_depth_choice = st.selectbox("Max Depth:", ["Unlimited (None)", 2, 4, 8, 12, 16, 25, 40], index=3)
            max_depth = None if custom_depth_choice == "Unlimited (None)" else int(custom_depth_choice)
        with ctrl_col2:
            min_samples_split = st.slider("Min Samples Split:", min_value=2, max_value=50, value=2, step=2)
        with ctrl_col3:
            min_samples_leaf = st.slider("Min Samples Leaf:", min_value=1, max_value=25, value=1, step=1)
        with ctrl_col4:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            run_dt_btn = st.button("🚀 Train Custom Tree", type="primary", use_container_width=True)

        if run_dt_btn:
            with st.spinner("Fitting Decision Tree with selected constraints..."):
                train_df, val_df, _ = get_cached_partitions()
                X_train, y_train = train_df["text"], train_df["label"]
                X_val, y_val = val_df["text"], val_df["label"]

                clf = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_SEED,
                )
                pipe = create_pipeline(clf, max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)

                t0 = time.time()
                pipe.fit(X_train, y_train)
                duration = time.time() - t0

                res = evaluate_model(pipe, X_train, y_train, X_val, y_val, training_time=duration)
                tree_ = pipe.named_steps["classifier"].tree_

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Train Accuracy", f"{res['train_accuracy']*100:.2f}%")
                m2.metric("Val Accuracy", f"{res['val_accuracy']*100:.2f}%")
                m3.metric("Generalization Gap", f"{res['generalization_gap']*100:.2f}%")
                m4.metric("Leaves Count", f"{tree_.n_leaves:,}")
                m5.metric("Training Time", f"{duration:.2f}s")

                gap = res["generalization_gap"]
                if gap > 0.05:
                    st.error(f"⚠️ **Heavy Overfitting Detected!** Generalization gap is {gap*100:.2f}%. Training accuracy is {res['train_accuracy']*100:.1f}%, but validation drops to {res['val_accuracy']*100:.1f}%. Increasing min_samples_split or capping max_depth will constrain overfitting.")
                elif gap > 0.02:
                    st.warning(f"⚡ **Moderate Gap:** Generalization gap is {gap*100:.2f}%. The tree demonstrates moderate capacity to generalize.")
                else:
                    st.success(f"✅ **Well Regularized:** Generalization gap is only {gap*100:.2f}%. Model complexity is well-matched to the data.")

    # ==========================================
    # TAB 2: LOGISTIC REGRESSION REGULARIZATION
    # ==========================================
    with tab2:
        st.subheader("⚖️ Experiment B: Regularization Strength C & Sparsity")
        st.markdown("""
        In Logistic Regression, the parameter $C$ controls the **inverse regularization strength**:
        - **Small $C$ (e.g. 0.01):** Strong penalty. Forces weights toward zero, producing a simpler model and preventing overfitting.
        - **Large $C$ (e.g. 100):** Weak penalty. Allows the model to fit extreme weights to training idiosyncrasies.
        - **L1 Penalty (Lasso):** Drives irrelevant coefficients strictly to **0**, executing automatic feature selection.
        - **L2 Penalty (Ridge):** Shrinks coefficients continuously toward zero without eliminating them completely.
        """)

        df_lr = load_csv_experiment(LR_CSV_PATH)
        if not df_lr.empty:
            st.markdown("#### 📊 Regularization Sweep Benchmark")
            c_lr1, c_lr2 = st.columns(2)
            with c_lr1:
                # C vs Accuracy
                df_l2 = df_lr[df_lr["penalty"] == "l2"]
                c_data = []
                for _, r in df_l2.iterrows():
                    c_data.append({"C": str(r["C"]), "Split": "Train Accuracy", "Score": r["train_accuracy"]})
                    c_data.append({"C": str(r["C"]), "Split": "Val Accuracy", "Score": r["val_accuracy"]})
                df_plot = pd.DataFrame(c_data)

                c_acc_chart = alt.Chart(df_plot).mark_line(point=True).encode(
                    x=alt.X("C:N", sort=None, title="Inverse Regularization C"),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.8, 1.0]), title="Accuracy"),
                    color=alt.Color("Split:N", scale=alt.Scale(range=["#06b6d4", "#10b981"])),
                    tooltip=["C", "Split", alt.Tooltip("Score:Q", format=".4f")],
                ).properties(title="C vs Accuracy (L2 Penalty)", height=320)
                st.altair_chart(c_acc_chart, use_container_width=True)

            with c_lr2:
                # C vs Non-zero weights
                c_sparse = alt.Chart(df_lr).mark_bar().encode(
                    x=alt.X("C:N", sort=None, title="C Parameter"),
                    y=alt.Y("non_zero_weights:Q", title="Active Non-Zero Features"),
                    color=alt.Color("penalty:N", scale=alt.Scale(range=["#38bdf8", "#f59e0b"])),
                    xOffset="penalty:N",
                    tooltip=["C", "penalty", "non_zero_weights"],
                ).properties(title="Feature Sparsity: L1 (Lasso) vs L2 (Ridge)", height=320)
                st.altair_chart(c_sparse, use_container_width=True)

            with st.expander("📄 View Logistic Regression Experiment Data Table"):
                st.dataframe(
                    df_lr[["C", "penalty", "non_zero_weights", "train_accuracy", "val_accuracy", "generalization_gap", "val_f1", "training_time"]],
                    use_container_width=True,
                    hide_index=True,
                )
                csv_bytes_lr = df_lr.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Export Regularization Data (CSV)", csv_bytes_lr, "lr_regularization_results.csv", "text/csv")

        st.markdown("---")
        st.markdown("#### 🎛️ Live Regularization Trainer")

        lr_c1, lr_c2, lr_c3 = st.columns([2, 2, 2])
        with lr_c1:
            custom_c = st.select_slider("Select Regularization C:", options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], value=1.0)
        with lr_c2:
            custom_pen = st.radio("Penalty Norm:", ["L2 (Ridge)", "L1 (Lasso / Sparse)"], horizontal=True)
            pen_val = "l1" if "L1" in custom_pen else "l2"
        with lr_c3:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            run_lr_btn = st.button("🚀 Train Logistic Model", type="primary", use_container_width=True)

        if run_lr_btn:
            with st.spinner("Fitting Logistic Regression..."):
                train_df, val_df, _ = get_cached_partitions()
                X_train, y_train = train_df["text"], train_df["label"]
                X_val, y_val = val_df["text"], val_df["label"]

                solver = "liblinear" if pen_val == "l1" else "lbfgs"
                clf = LogisticRegression(C=custom_c, penalty=pen_val, solver=solver, max_iter=1000, random_state=RANDOM_SEED)
                pipe = create_pipeline(clf, max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)

                t0 = time.time()
                pipe.fit(X_train, y_train)
                duration = time.time() - t0

                res = evaluate_model(pipe, X_train, y_train, X_val, y_val, training_time=duration)
                non_zeros = int(np.count_nonzero(pipe.named_steps["classifier"].coef_))

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Train Accuracy", f"{res['train_accuracy']*100:.2f}%")
                m2.metric("Val Accuracy", f"{res['val_accuracy']*100:.2f}%")
                m3.metric("Generalization Gap", f"{res['generalization_gap']*100:.2f}%")
                m4.metric("Non-Zero Features", f"{non_zeros:,} / {MAX_FEATURES:,}")
                m5.metric("Training Time", f"{duration:.2f}s")

    # ==========================================
    # TAB 3: LEARNING CURVES
    # ==========================================
    with tab3:
        st.subheader("📈 Experiment C: Learning Curves (Sample Size vs Generalization)")
        st.markdown("""
        Learning curves track model performance as the number of available training samples increases.
        - **At small sample sizes (<500 emails):** Training accuracy is artificially high (model easily fits few points), but validation accuracy is low.
        - **As sample size scales (>5,000 emails):** Training and validation scores converge toward the true asymptotic capacity.
        - **Narrow convergence gap:** Indicates that additional data collection yields diminishing returns and variance is well-controlled.
        """)

        df_lc = load_csv_experiment(LC_CSV_PATH)
        if not df_lc.empty:
            c_lc_data = []
            for _, r in df_lc.iterrows():
                c_lc_data.append({"Samples": int(r["train_samples"]), "Split": "Training Accuracy", "Score": r["train_accuracy"]})
                c_lc_data.append({"Samples": int(r["train_samples"]), "Split": "Validation Accuracy", "Score": r["val_accuracy"]})
            df_plot_lc = pd.DataFrame(c_lc_data)

            chart_lc = alt.Chart(df_plot_lc).mark_line(point=True).encode(
                x=alt.X("Samples:Q", title="Training Samples Count"),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.8, 1.01]), title="Accuracy Score"),
                color=alt.Color("Split:N", scale=alt.Scale(range=["#06b6d4", "#10b981"])),
                tooltip=["Samples", "Split", alt.Tooltip("Score:Q", format=".4f")],
            ).properties(title="Empirical Learning Curve: Logistic Regression (C=1.0)", height=380)

            st.altair_chart(chart_lc, use_container_width=True)

            with st.expander("📄 View Learning Curve Data"):
                st.dataframe(df_lc, use_container_width=True, hide_index=True)
                csv_bytes_lc = df_lc.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Export Learning Curve Data (CSV)", csv_bytes_lc, "learning_curve_results.csv", "text/csv")


if __name__ == "__main__":
    main()
