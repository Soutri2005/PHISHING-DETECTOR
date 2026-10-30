"""Unit tests for LoRA fine-tuning dataset preparation and evaluation."""

import os
import pytest
from src.dataset_subset import create_curated_subset, SUBSET_DIR


def test_dataset_subset_creation():
    report = create_curated_subset()
    assert report["total_examples"] == 800
    assert report["split_counts"]["train"] == 600
    assert report["split_counts"]["val"] == 100
    assert report["split_counts"]["test"] == 100

    assert os.path.exists(os.path.join(SUBSET_DIR, "train_800.csv"))
    assert os.path.exists(os.path.join(SUBSET_DIR, "val_800.csv"))
    assert os.path.exists(os.path.join(SUBSET_DIR, "test_800.csv"))


def test_eval_metric_computation():
    from src.finetune_lora import compute_eval_metrics

    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    metrics = compute_eval_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["tn"] == 2
