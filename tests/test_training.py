"""Unit tests for model evaluation and training metrics."""

import numpy as np
import pytest
from src.evaluate import compute_metrics


def test_compute_metrics_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y_true, y_pred, dataset_name="val")

    assert metrics["val_accuracy"] == 1.0
    assert metrics["val_precision"] == 1.0
    assert metrics["val_recall"] == 1.0
    assert metrics["val_f1"] == 1.0
    assert metrics["val_confusion_matrix"]["tp"] == 2
    assert metrics["val_confusion_matrix"]["tn"] == 2
    assert metrics["val_confusion_matrix"]["fp"] == 0
    assert metrics["val_confusion_matrix"]["fn"] == 0


def test_compute_metrics_imperfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])  # 1 TP, 1 FN, 1 TN, 1 FP
    metrics = compute_metrics(y_true, y_pred, dataset_name="test")

    assert metrics["test_accuracy"] == 0.5
    assert metrics["test_precision"] == 0.5
    assert metrics["test_recall"] == 0.5
    assert metrics["test_f1"] == 0.5
