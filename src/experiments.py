"""
Overfitting Study and Controlled Experiment Tracking Module.

Executes real, reproducible experiments to investigate:
1. Decision Tree complexity vs Generalization Gap (max_depth, min_samples_split).
2. Logistic Regression regularization strength C (L1 vs L2 penalties).
3. Learning curves across varying training subset fractions.

Saves experiment metric records to CSV and renders publication-ready Matplotlib figures.
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Set MPLCONFIGDIR before importing matplotlib to avoid permission errors
_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_mpl_dir = os.path.join(_base_dir, ".mplconfig")
os.makedirs(_mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _mpl_dir

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_tfidf_vectorizer
from src.evaluate import compute_metrics
from src.train import load_partitions, RANDOM_SEED, MAX_FEATURES, NGRAM_RANGE

BASE_DIR = _base_dir
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Cybersecurity Dashboard Color Palette
STYLE_BG = "#0f172a"
STYLE_CARD = "#1e293b"
STYLE_CYAN = "#06b6d4"
STYLE_TEAL = "#14b8a6"
STYLE_RED = "#ef4444"
STYLE_GREEN = "#10b981"
STYLE_TEXT = "#f8fafc"
STYLE_MUTED = "#94a3b8"


def set_custom_plot_style():
    """Apply consistent cyber-dashboard theme to Matplotlib."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": STYLE_BG,
        "axes.facecolor": STYLE_CARD,
        "axes.edgecolor": "#334155",
        "axes.labelcolor": STYLE_TEXT,
        "text.color": STYLE_TEXT,
        "xtick.color": STYLE_MUTED,
        "ytick.color": STYLE_MUTED,
        "grid.color": "#334155",
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
    })


def evaluate_sparse_model(
    clf: Any,
    X_train_sp: Any,
    y_train: np.ndarray,
    X_val_sp: Any,
    y_val: np.ndarray,
    training_time: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate classifier fitted on pre-vectorized sparse matrices."""
    train_pred = clf.predict(X_train_sp)
    train_met = compute_metrics(y_train, train_pred, dataset_name="train")

    val_pred = clf.predict(X_val_sp)
    val_prob = None
    if hasattr(clf, "predict_proba"):
        try:
            val_prob = clf.predict_proba(X_val_sp)
        except Exception:
            pass
    elif hasattr(clf, "decision_function"):
        try:
            val_prob = clf.decision_function(X_val_sp)
        except Exception:
            pass

    val_met = compute_metrics(y_val, val_pred, val_prob, dataset_name="val")

    train_acc = train_met["train_accuracy"]
    val_acc = val_met["val_accuracy"]
    gap = round(train_acc - val_acc, 4)

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "generalization_gap": gap,
        "val_precision": val_met["val_precision"],
        "val_recall": val_met["val_recall"],
        "val_f1": val_met["val_f1"],
        "training_time_seconds": round(training_time, 4),
    }


def run_decision_tree_experiment(
    X_train_sp: Any,
    y_train: np.ndarray,
    X_val_sp: Any,
    y_val: np.ndarray,
    depths: Optional[List[Optional[int]]] = None,
    min_samples_split: int = 2,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run the Decision Tree Complexity Experiment across multiple depths."""
    if depths is None:
        depths = [2, 5, 10, 15, 20, 30, None]

    print("\n--- Running Decision Tree Complexity Study ---", flush=True)
    records = []
    for depth in depths:
        depth_label = "Unlimited" if depth is None else str(depth)
        clf = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=min_samples_split,
            random_state=random_seed,
        )

        t0 = time.time()
        clf.fit(X_train_sp, y_train)
        duration = time.time() - t0

        metrics = evaluate_sparse_model(clf, X_train_sp, y_train, X_val_sp, y_val, training_time=duration)
        tree_obj = clf.tree_

        record = {
            "experiment_id": f"DT_depth_{depth_label}_mss_{min_samples_split}",
            "model": "DecisionTreeClassifier",
            "max_depth_param": depth_label,
            "max_depth_numeric": 50 if depth is None else depth,
            "min_samples_split": min_samples_split,
            "actual_depth": tree_obj.max_depth,
            "node_count": tree_obj.node_count,
            "leaf_count": tree_obj.n_leaves,
            "train_accuracy": metrics["train_accuracy"],
            "val_accuracy": metrics["val_accuracy"],
            "generalization_gap": metrics["generalization_gap"],
            "val_precision": metrics["val_precision"],
            "val_recall": metrics["val_recall"],
            "val_f1": metrics["val_f1"],
            "training_time": metrics["training_time_seconds"],
            "random_seed": random_seed,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        records.append(record)
        print(
            f"--> Depth: {depth_label:9s} | Actual Depth: {tree_obj.max_depth:2d} | "
            f"Train Acc: {record['train_accuracy']:.4f} | Val Acc: {record['val_accuracy']:.4f} | "
            f"Gap: {record['generalization_gap']:.4f} | Leaves: {tree_obj.n_leaves}",
            flush=True,
        )

    df_results = pd.DataFrame(records)
    csv_path = os.path.join(REPORTS_DIR, "dt_complexity_experiments.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved Decision Tree experiment records to {csv_path}", flush=True)

    plot_decision_tree_results(df_results)
    return df_results


def plot_decision_tree_results(df: pd.DataFrame):
    """Plot and save Decision Tree Complexity curve."""
    set_custom_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Plot 1: Train vs Validation Accuracy
    ax1.plot(
        df["max_depth_param"],
        df["train_accuracy"],
        marker="o",
        linewidth=2.5,
        color=STYLE_CYAN,
        label="Training Accuracy",
    )
    ax1.plot(
        df["max_depth_param"],
        df["val_accuracy"],
        marker="s",
        linewidth=2.5,
        color=STYLE_GREEN,
        label="Validation Accuracy",
    )
    ax1.set_title("Decision Tree: Depth vs Accuracy (Overfitting Pattern)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Max Depth Parameter", fontsize=11)
    ax1.set_ylabel("Accuracy Score", fontsize=11)
    ax1.set_ylim(0.60, 1.02)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.8)

    # Plot 2: Generalization Gap vs Depth
    ax2.plot(
        df["max_depth_param"],
        df["generalization_gap"],
        marker="^",
        linewidth=2.5,
        color=STYLE_RED,
        label="Generalization Gap (Train - Val)",
    )
    ax2.set_title("Decision Tree: Generalization Gap (Overfitting Severity)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Max Depth Parameter", fontsize=11)
    ax2.set_ylabel("Generalization Gap", fontsize=11)
    ax2.grid(True)
    ax2.legend(loc="upper left", framealpha=0.8)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "dt_complexity_curve.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved Decision Tree complexity plot to {fig_path}", flush=True)


def run_logistic_regression_experiment(
    X_train_sp: Any,
    y_train: np.ndarray,
    X_val_sp: Any,
    y_val: np.ndarray,
    c_values: Optional[List[float]] = None,
    penalties: Optional[List[str]] = None,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run the Logistic Regression Regularization Study."""
    if c_values is None:
        c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    if penalties is None:
        penalties = ["l2", "l1"]

    print("\n--- Running Logistic Regression Regularization Study ---", flush=True)
    records = []
    for penalty in penalties:
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        for c_val in c_values:
            clf = LogisticRegression(
                C=c_val,
                penalty=penalty,
                solver=solver,
                max_iter=1000,
                random_state=random_seed,
            )

            t0 = time.time()
            clf.fit(X_train_sp, y_train)
            duration = time.time() - t0

            metrics = evaluate_sparse_model(clf, X_train_sp, y_train, X_val_sp, y_val, training_time=duration)
            non_zero_weights = int(np.count_nonzero(clf.coef_))

            record = {
                "experiment_id": f"LR_C_{c_val}_{penalty}",
                "model": "LogisticRegression",
                "C": c_val,
                "penalty": penalty,
                "solver": solver,
                "non_zero_weights": non_zero_weights,
                "train_accuracy": metrics["train_accuracy"],
                "val_accuracy": metrics["val_accuracy"],
                "generalization_gap": metrics["generalization_gap"],
                "val_precision": metrics["val_precision"],
                "val_recall": metrics["val_recall"],
                "val_f1": metrics["val_f1"],
                "training_time": metrics["training_time_seconds"],
                "random_seed": random_seed,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            records.append(record)
            print(
                f"--> C={c_val:7.3f} ({penalty}) | Non-zero Coefs: {non_zero_weights:5d} | "
                f"Train Acc: {record['train_accuracy']:.4f} | Val Acc: {record['val_accuracy']:.4f} | "
                f"Gap: {record['generalization_gap']:.4f}",
                flush=True,
            )

    df_results = pd.DataFrame(records)
    csv_path = os.path.join(REPORTS_DIR, "lr_regularization_experiments.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved Logistic Regression experiment records to {csv_path}", flush=True)

    plot_logistic_regression_results(df_results)
    return df_results


def plot_logistic_regression_results(df: pd.DataFrame):
    """Plot and save Logistic Regression Regularization curve."""
    set_custom_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    df_l2 = df[df["penalty"] == "l2"]
    df_l1 = df[df["penalty"] == "l1"]

    # Plot 1: C vs Accuracy
    ax1.plot(df_l2["C"], df_l2["train_accuracy"], marker="o", color=STYLE_CYAN, label="Train Acc (L2)")
    ax1.plot(df_l2["C"], df_l2["val_accuracy"], marker="s", color=STYLE_GREEN, label="Val Acc (L2)")
    if not df_l1.empty:
        ax1.plot(df_l1["C"], df_l1["train_accuracy"], marker="^", linestyle="--", color="#a855f7", label="Train Acc (L1)")
        ax1.plot(df_l1["C"], df_l1["val_accuracy"], marker="v", linestyle="--", color="#f59e0b", label="Val Acc (L1)")

    ax1.set_xscale("log")
    ax1.set_title("Logistic Regression: Regularization C vs Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Regularization Strength C (Log Scale)", fontsize=11)
    ax1.set_ylabel("Accuracy Score", fontsize=11)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.8)

    # Plot 2: C vs Generalization Gap
    ax2.plot(df_l2["C"], df_l2["generalization_gap"], marker="o", color=STYLE_RED, label="Generalization Gap (L2)")
    if not df_l1.empty:
        ax2.plot(df_l1["C"], df_l1["generalization_gap"], marker="^", linestyle="--", color="#f97316", label="Generalization Gap (L1)")

    ax2.set_xscale("log")
    ax2.set_title("Regularization Strength C vs Generalization Gap", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Regularization Strength C (Log Scale)", fontsize=11)
    ax2.set_ylabel("Gap (Train - Val)", fontsize=11)
    ax2.grid(True)
    ax2.legend(loc="upper left", framealpha=0.8)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "lr_regularization_curve.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved Logistic Regression regularization plot to {fig_path}", flush=True)


def run_learning_curve_experiment(
    X_train_sp: Any,
    y_train: np.ndarray,
    X_val_sp: Any,
    y_val: np.ndarray,
    fractions: Optional[List[float]] = None,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate Learning Curves by training on increasing fractions of the training split."""
    if fractions is None:
        fractions = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

    print("\n--- Running Learning Curve Study ---", flush=True)
    np.random.seed(random_seed)
    n_total = X_train_sp.shape[0]
    indices = np.random.permutation(n_total)

    records = []
    for frac in fractions:
        n_samples = max(int(n_total * frac), 50)
        sub_indices = indices[:n_samples]

        X_sub = X_train_sp[sub_indices]
        y_sub = y_train[sub_indices]

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_seed)

        t0 = time.time()
        clf.fit(X_sub, y_sub)
        duration = time.time() - t0

        metrics = evaluate_sparse_model(clf, X_sub, y_sub, X_val_sp, y_val, training_time=duration)

        record = {
            "fraction": frac,
            "train_samples": n_samples,
            "train_accuracy": metrics["train_accuracy"],
            "val_accuracy": metrics["val_accuracy"],
            "generalization_gap": metrics["generalization_gap"],
            "val_f1": metrics["val_f1"],
            "training_time": metrics["training_time_seconds"],
        }
        records.append(record)
        print(
            f"--> Samples: {n_samples:5d} ({frac*100:4.1f}%) | "
            f"Train Acc: {record['train_accuracy']:.4f} | Val Acc: {record['val_accuracy']:.4f} | "
            f"Gap: {record['generalization_gap']:.4f}",
            flush=True,
        )

    df_results = pd.DataFrame(records)
    csv_path = os.path.join(REPORTS_DIR, "learning_curve_experiments.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved Learning Curve experiment records to {csv_path}", flush=True)

    plot_learning_curve(df_results)
    return df_results


def plot_learning_curve(df: pd.DataFrame):
    """Plot and save Learning Curve figure."""
    set_custom_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)

    ax.plot(
        df["train_samples"],
        df["train_accuracy"],
        marker="o",
        linewidth=2.5,
        color=STYLE_CYAN,
        label="Training Accuracy",
    )
    ax.plot(
        df["train_samples"],
        df["val_accuracy"],
        marker="s",
        linewidth=2.5,
        color=STYLE_GREEN,
        label="Validation Accuracy",
    )
    ax.fill_between(
        df["train_samples"],
        df["train_accuracy"],
        df["val_accuracy"],
        color=STYLE_CYAN,
        alpha=0.15,
        label="Generalization Gap",
    )

    ax.set_title("Empirical Learning Curve: Samples vs Generalization", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Training Sample Size", fontsize=11)
    ax.set_ylabel("Accuracy Score", fontsize=11)
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.8)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "learning_curve.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved Learning Curve plot to {fig_path}", flush=True)


def run_all_experiments():
    """Run complete suite of overfitting experiments with high performance vectorization."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Loading data partitions...", flush=True)
    train_df, val_df, _ = load_partitions()
    X_train, y_train = train_df["text"], np.asarray(train_df["label"], dtype=int)
    X_val, y_val = val_df["text"], np.asarray(val_df["label"], dtype=int)

    print("Fitting TF-IDF strictly on training partition once (zero-leakage)...", flush=True)
    tfidf = build_tfidf_vectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X_train_sp = tfidf.fit_transform(X_train)
    X_val_sp = tfidf.transform(X_val)
    print(f"Vectorized training matrix: {X_train_sp.shape}, validation matrix: {X_val_sp.shape}", flush=True)

    print("\nStarting comprehensive Overfitting Experiments Suite...", flush=True)
    run_decision_tree_experiment(X_train_sp, y_train, X_val_sp, y_val)
    run_logistic_regression_experiment(X_train_sp, y_train, X_val_sp, y_val)
    run_learning_curve_experiment(X_train_sp, y_train, X_val_sp, y_val)
    print("\nAll overfitting experiments executed and recorded successfully!", flush=True)


if __name__ == "__main__":
    run_all_experiments()
