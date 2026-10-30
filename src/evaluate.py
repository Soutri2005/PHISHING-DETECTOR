"""
Evaluation module for computing standard classification and overfitting metrics.

Metrics calculated:
- Accuracy (Train and Validation / Test)
- Generalization Gap (Train Accuracy - Validation Accuracy)
- Precision (Phishing positive class)
- Recall (Phishing positive class)
- F1-Score
- Confusion Matrix: TN, FP, FN, TP
- Training duration
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    dataset_name: str = "validation",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for a given split.
    Phishing is the positive class (label 1).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics = {
        f"{dataset_name}_accuracy": round(acc, 4),
        f"{dataset_name}_precision": round(prec, 4),
        f"{dataset_name}_recall": round(rec, 4),
        f"{dataset_name}_f1": round(f1, 4),
        f"{dataset_name}_confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "raw": [[tn, fp], [fn, tp]],
        },
    }

    if y_prob is not None:
        try:
            # If 2D probability array passed
            if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
                auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_prob))
            metrics[f"{dataset_name}_roc_auc"] = round(auc, 4)
        except Exception:
            metrics[f"{dataset_name}_roc_auc"] = None

    return metrics


def evaluate_model(
    model: Any,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    training_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate fitted model pipeline on both training and validation sets.
    Computes train-val generalization gap and complete diagnostic metrics.
    """
    # Training set predictions
    train_pred = model.predict(X_train)
    train_metrics = compute_metrics(y_train, train_pred, dataset_name="train")

    # Validation set predictions
    val_pred = model.predict(X_val)
    val_prob = None
    if hasattr(model, "predict_proba"):
        try:
            val_prob = model.predict_proba(X_val)
        except Exception:
            val_prob = None
    elif hasattr(model, "decision_function"):
        try:
            val_prob = model.decision_function(X_val)
        except Exception:
            val_prob = None

    val_metrics = compute_metrics(y_val, val_pred, val_prob, dataset_name="val")

    # Generalization gap: Train Accuracy - Val Accuracy
    train_acc = train_metrics["train_accuracy"]
    val_acc = val_metrics["val_accuracy"]
    gap = round(train_acc - val_acc, 4)

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "generalization_gap": gap,
        "val_precision": val_metrics["val_precision"],
        "val_recall": val_metrics["val_recall"],
        "val_f1": val_metrics["val_f1"],
        "val_roc_auc": val_metrics.get("val_roc_auc"),
        "train_confusion_matrix": train_metrics["train_confusion_matrix"],
        "val_confusion_matrix": val_metrics["val_confusion_matrix"],
        "training_time_seconds": round(training_time, 3),
    }


def evaluate_test_set(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> Dict[str, Any]:
    """
    Evaluate the final selected model configuration on the untouched test partition.
    Strictly isolated: only called after model selection.
    """
    test_pred = model.predict(X_test)
    test_prob = None
    if hasattr(model, "predict_proba"):
        try:
            test_prob = model.predict_proba(X_test)
        except Exception:
            test_prob = None

    test_metrics = compute_metrics(y_test, test_pred, test_prob, dataset_name="test")
    return test_metrics
