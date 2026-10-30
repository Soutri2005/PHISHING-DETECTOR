"""
Dataset loader, validation, and leakage-safe splitting module.

Implements:
1. Automated loading and inspection of the real Phishing Email dataset.
2. Label normalization ('Phishing Email' -> 1, 'Safe Email' -> 0).
3. Handling of nulls, empties, and duplicate detection.
4. Stratified 70/15/15 train/validation/test partitioning with fixed random seed.
5. Leakage audit verifying zero overlapping texts between splits.
6. Export of clean splits and a detailed data preparation report.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import pandas as pd
except ImportError:
    pd = None

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Phishing_Email.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_REPORT_PATH = os.path.join(REPORTS_DIR, "data_preparation_report.json")

LABEL_MAPPING = {
    "phishing email": 1,
    "phishing": 1,
    "1": 1,
    1: 1,
    "safe email": 0,
    "legitimate": 0,
    "safe": 0,
    "ham": 0,
    "0": 0,
    0: 0,
}

LABEL_NAMES = {1: "Phishing", 0: "Legitimate"}


def detect_columns(data: Any) -> Tuple[str, str]:
    """Detect text and label column names from raw DataFrame or dictionary/list."""
    if hasattr(data, "columns"):
        cols = list(data.columns)
    elif isinstance(data, dict):
        cols = list(data.keys())
    elif isinstance(data, (list, tuple)) and len(data) > 0:
        if isinstance(data[0], dict):
            cols = list(data[0].keys())
        else:
            cols = list(data)
    else:
        cols = list(data)

    cols_lower = {str(col).lower().strip(): col for col in cols}

    # Identify text column
    text_candidates = ["email text", "text_combined", "text", "body", "message", "content", "email_text"]
    text_col = None
    for cand in text_candidates:
        if cand in cols_lower:
            text_col = cols_lower[cand]
            break

    # Identify label column
    label_candidates = ["email type", "label", "type", "class", "category", "target", "email_type"]
    label_col = None
    for cand in label_candidates:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break

    if not text_col or not label_col:
        raise ValueError(
            f"Could not automatically detect text and label columns. Available columns: {cols}"
        )

    return text_col, label_col


def load_raw_dataset(path: Optional[str] = None) -> Any:
    """Load the raw dataset from disk."""
    file_path = path or RAW_DATA_PATH
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Raw dataset file not found at: {file_path}. "
            f"Run python -m src.download_dataset first or place Phishing_Email.csv in data/raw/."
        )
    if pd is not None:
        return pd.read_csv(file_path)
    
    csv.field_size_limit(min(sys.maxsize, 2147483647))
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def clean_and_normalize(data: Any) -> Tuple[Any, Dict[str, int]]:
    """
    Clean, validate, and normalize text and labels.
    Returns cleaned dataset (DataFrame or list of dicts) with fields ['text', 'label', 'label_name']
    and a statistics dictionary.
    """
    if pd is not None and isinstance(data, pd.DataFrame):
        text_col, label_col = detect_columns(data)
        initial_rows = len(data)

        df_clean = data[[text_col, label_col]].copy()
        df_clean.columns = ["text", "raw_label"]

        df_clean["norm_key"] = df_clean["raw_label"].astype(str).str.strip().str.lower()
        df_clean["label"] = df_clean["norm_key"].map(LABEL_MAPPING)

        valid_label_mask = df_clean["label"].notna()
        invalid_labels_count = int((~valid_label_mask).sum())
        df_clean = df_clean[valid_label_mask].copy()
        df_clean["label"] = df_clean["label"].astype(int)

        df_clean["text"] = df_clean["text"].fillna("").astype(str).str.strip()

        non_empty_mask = df_clean["text"].str.len() > 0
        empty_text_count = int((~non_empty_mask).sum())
        df_clean = df_clean[non_empty_mask].copy()

        exact_duplicates_count = int(df_clean.duplicated(subset=["text"]).sum())
        df_clean = df_clean.drop_duplicates(subset=["text"], keep="first").copy()

        df_clean["label_name"] = df_clean["label"].map(LABEL_NAMES)
        df_clean = df_clean[["text", "label", "label_name"]].reset_index(drop=True)

        stats = {
            "initial_rows": initial_rows,
            "invalid_labels_removed": invalid_labels_count,
            "empty_text_removed": empty_text_count,
            "exact_duplicates_removed": exact_duplicates_count,
            "total_removed": initial_rows - len(df_clean),
            "final_clean_rows": len(df_clean),
            "phishing_count": int((df_clean["label"] == 1).sum()),
            "legitimate_count": int((df_clean["label"] == 0).sum()),
        }
        return df_clean, stats

    # Pure Python dict / list handling
    text_col, label_col = detect_columns(data)
    if isinstance(data, dict):
        keys = list(data.keys())
        num_rows = len(data[text_col])
        rows = [{k: data[k][i] for k in keys} for i in range(num_rows)]
    elif isinstance(data, list):
        rows = data
    else:
        rows = list(data)

    initial_rows = len(rows)
    seen_texts = set()
    cleaned_rows = []
    invalid_labels_count = 0
    empty_text_count = 0
    exact_duplicates_count = 0

    for r in rows:
        raw_label = r.get(label_col)
        raw_text = r.get(text_col)

        norm_key = str(raw_label).strip().lower() if raw_label is not None else ""
        if norm_key not in LABEL_MAPPING:
            invalid_labels_count += 1
            continue
        label = LABEL_MAPPING[norm_key]

        text = str(raw_text).strip() if raw_text is not None else ""
        if not text or text == "None" or text == "nan":
            empty_text_count += 1
            continue

        if text in seen_texts:
            exact_duplicates_count += 1
            continue

        seen_texts.add(text)
        cleaned_rows.append({
            "text": text,
            "label": label,
            "label_name": LABEL_NAMES[label],
        })

    stats = {
        "initial_rows": initial_rows,
        "invalid_labels_removed": invalid_labels_count,
        "empty_text_removed": empty_text_count,
        "exact_duplicates_removed": exact_duplicates_count,
        "total_removed": initial_rows - len(cleaned_rows),
        "final_clean_rows": len(cleaned_rows),
        "phishing_count": sum(1 for r in cleaned_rows if r["label"] == 1),
        "legitimate_count": sum(1 for r in cleaned_rows if r["label"] == 0),
    }
    return cleaned_rows, stats


def split_data(
    data: Any,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Split data into stratified Train (70%), Validation (15%), and Test (15%) partitions.
    Fixes the random state for exact reproducibility.
    """
    if pd is not None and isinstance(data, pd.DataFrame):
        train_val_df, test_df = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data["label"],
        )
        val_fraction = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_fraction,
            random_state=random_state,
            stratify=train_val_df["label"],
        )

        train_set = set(train_df["text"])
        val_set = set(val_df["text"])
        test_set = set(test_df["text"])

        train_val_leak = len(train_set.intersection(val_set))
        train_test_leak = len(train_set.intersection(test_set))
        val_test_leak = len(val_set.intersection(test_set))

        leakage_stats = {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_phishing_pct": round(float((train_df["label"] == 1).mean()) * 100, 2),
            "val_phishing_pct": round(float((val_df["label"] == 1).mean()) * 100, 2),
            "test_phishing_pct": round(float((test_df["label"] == 1).mean()) * 100, 2),
            "train_val_leak_count": train_val_leak,
            "train_test_leak_count": train_test_leak,
            "val_test_leak_count": val_test_leak,
        }

        if train_val_leak > 0 or train_test_leak > 0 or val_test_leak > 0:
            raise ValueError(f"CRITICAL: Data leakage detected across partitions! {leakage_stats}")

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            leakage_stats,
        )

    # Pure Python records split
    rows = list(data)
    labels = [r["label"] for r in rows]
    indices = list(range(len(rows)))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    val_fraction = val_size / (1.0 - test_size)
    train_val_labels = [labels[i] for i in train_val_idx]
    train_sub_idx, val_sub_idx = train_test_split(
        list(range(len(train_val_idx))),
        test_size=val_fraction,
        random_state=random_state,
        stratify=train_val_labels,
    )

    train_rows = [rows[train_val_idx[i]] for i in train_sub_idx]
    val_rows = [rows[train_val_idx[i]] for i in val_sub_idx]
    test_rows = [rows[i] for i in test_idx]

    train_set = set(r["text"] for r in train_rows)
    val_set = set(r["text"] for r in val_rows)
    test_set = set(r["text"] for r in test_rows)

    train_val_leak = len(train_set.intersection(val_set))
    train_test_leak = len(train_set.intersection(test_set))
    val_test_leak = len(val_set.intersection(test_set))

    leakage_stats = {
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "train_phishing_pct": round(sum(1 for r in train_rows if r["label"] == 1) / len(train_rows) * 100, 2),
        "val_phishing_pct": round(sum(1 for r in val_rows if r["label"] == 1) / len(val_rows) * 100, 2),
        "test_phishing_pct": round(sum(1 for r in test_rows if r["label"] == 1) / len(test_rows) * 100, 2),
        "train_val_leak_count": train_val_leak,
        "train_test_leak_count": train_test_leak,
        "val_test_leak_count": val_test_leak,
    }

    if train_val_leak > 0 or train_test_leak > 0 or val_test_leak > 0:
        raise ValueError(f"CRITICAL: Data leakage detected across partitions! {leakage_stats}")

    return train_rows, val_rows, test_rows, leakage_stats


def prepare_and_save_dataset(
    raw_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    random_state: int = 42,
) -> Dict:
    """End-to-end dataset preparation and report generation."""
    out_dir = output_dir or PROCESSED_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Loading raw dataset from {raw_path or RAW_DATA_PATH}...")
    raw_df = load_raw_dataset(raw_path)

    print("Cleaning and validating data...")
    clean_df, clean_stats = clean_and_normalize(raw_df)

    print("Executing stratified 70/15/15 train/val/test split...")
    train_df, val_df, test_df, split_stats = split_data(
        clean_df, test_size=0.15, val_size=0.15, random_state=random_state
    )

    # Save processed CSV files
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved processed partitions to {out_dir}")

    # Compile comprehensive report
    report = {
        "dataset_metadata": {
            "source": "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset",
            "license": "Public Domain / CC0 / Research Mirror",
            "description": "Real phishing and legitimate email corpus containing text subjects and bodies.",
            "random_seed": random_state,
        },
        "cleaning_statistics": clean_stats,
        "partition_statistics": split_stats,
        "leakage_verification": {
            "status": "PASSED - ZERO OVERLAP",
            "train_val_overlap": split_stats["train_val_leak_count"],
            "train_test_overlap": split_stats["train_test_leak_count"],
            "val_test_overlap": split_stats["val_test_leak_count"],
        },
        "files_generated": {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        },
    }

    with open(DATA_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved data preparation report to {DATA_REPORT_PATH}")

    return report


if __name__ == "__main__":
    report = prepare_and_save_dataset()
    print("\n--- Data Preparation Summary ---")
    print(json.dumps(report, indent=2))
