"""Unit tests for dataset loading, normalization, and splitting."""

import pandas as pd
import pytest
from src.data_loader import detect_columns, clean_and_normalize, split_data


def test_detect_columns():
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1],
        "Email Text": ["Hello test", "Urgent phishing"],
        "Email Type": ["Safe Email", "Phishing Email"],
    })
    text_col, label_col = detect_columns(df)
    assert text_col == "Email Text"
    assert label_col == "Email Type"


def test_clean_and_normalize():
    raw_df = pd.DataFrame({
        "Email Text": ["Hello test", "Urgent phishing", "Hello test", "", None],
        "Email Type": ["Safe Email", "Phishing Email", "Safe Email", "Safe Email", "Safe Email"],
    })
    cleaned_df, stats = clean_and_normalize(raw_df)

    # 1 duplicate dropped, 2 empty/null dropped -> 2 unique valid rows
    assert len(cleaned_df) == 2
    assert stats["exact_duplicates_removed"] == 1
    assert stats["empty_text_removed"] == 2
    assert set(cleaned_df["label"]) == {0, 1}


def test_split_data_leakage_audit():
    # Construct synthetic data of 100 rows
    texts = [f"Unique message content {i}" for i in range(100)]
    labels = [1 if i % 3 == 0 else 0 for i in range(100)]
    df = pd.DataFrame({"text": texts, "label": labels, "label_name": ["Phishing" if l == 1 else "Legitimate" for l in labels]})

    train_df, val_df, test_df, stats = split_data(df, test_size=0.15, val_size=0.15, random_state=42)

    assert stats["train_val_leak_count"] == 0
    assert stats["train_test_leak_count"] == 0
    assert stats["val_test_leak_count"] == 0
    assert len(train_df) + len(val_df) + len(test_df) == 100
