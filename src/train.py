"""
Model training pipeline for Phishing Message Detector.

Trains four real scikit-learn models:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Decision Tree Classifier
4. Random Forest Classifier

Saves fitted joblib pipelines and a consolidated training report.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from typing import Any, Dict, Tuple
import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import create_pipeline
from src.evaluate import evaluate_model, evaluate_test_set

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
TRAINING_REPORT_PATH = os.path.join(REPORTS_DIR, "training_results.json")

RANDOM_SEED = 42
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)


def get_model_definitions(random_seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """Define the 4 baseline classifiers."""
    return {
        "naive_bayes": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_seed,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def load_partitions(data_dir: str = PROCESSED_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-split train, val, and test datasets."""
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Processed data partitions not found in {data_dir}. Run python -m src.data_loader first."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Clean any potential na in text
    train_df["text"] = train_df["text"].fillna("")
    val_df["text"] = val_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    return train_df, val_df, test_df


def train_and_evaluate_all(
    random_seed: int = RANDOM_SEED,
    max_features: int = MAX_FEATURES,
    ngram_range: Tuple[int, int] = NGRAM_RANGE,
) -> Dict[str, Any]:
    """
    Train and evaluate all four baseline classifiers.
    Saves models and produces comparative results.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading data partitions...")
    train_df, val_df, test_df = load_partitions()

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val = val_df["text"], val_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    models_dict = get_model_definitions(random_seed)
    results = {}
    fitted_pipelines = {}

    print(f"\nTraining {len(models_dict)} baseline classifiers...")
    for model_key, base_classifier in models_dict.items():
        print(f"--> Training {model_key}...")
        pipeline = create_pipeline(
            classifier=base_classifier,
            max_features=max_features,
            ngram_range=ngram_range,
        )

        # Measure training duration
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        duration = time.time() - start_time

        # Evaluate on train and validation
        metrics = evaluate_model(pipeline, X_train, y_train, X_val, y_val, training_time=duration)
        results[model_key] = metrics
        fitted_pipelines[model_key] = pipeline

        # Save individual model pipeline with metadata
        model_save_path = os.path.join(MODELS_DIR, f"{model_key}.joblib")
        model_artifact = {
            "pipeline": pipeline,
            "model_name": model_key,
            "classifier_class": base_classifier.__class__.__name__,
            "params": base_classifier.get_params(),
            "tfidf_params": {
                "max_features": max_features,
                "ngram_range": ngram_range,
            },
            "metrics": metrics,
            "random_seed": random_seed,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        joblib.dump(model_artifact, model_save_path)
        print(
            f"    Val Acc: {metrics['val_accuracy']:.4f} | "
            f"Train Acc: {metrics['train_accuracy']:.4f} | "
            f"Gap: {metrics['generalization_gap']:.4f} | "
            f"Val F1: {metrics['val_f1']:.4f} | "
            f"Time: {duration:.2f}s"
        )

    # Select best model based on validation F1 score (balance of precision & recall)
    best_key = max(results.keys(), key=lambda k: results[k]["val_f1"])
    best_pipeline = fitted_pipelines[best_key]
    print(f"\nBest model selected based on Validation F1: {best_key} (F1: {results[best_key]['val_f1']})")

    # Evaluate best model strictly on the untouched test partition
    print(f"Evaluating {best_key} on untouched test set...")
    test_metrics = evaluate_test_set(best_pipeline, X_test, y_test)
    print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f} | Test F1: {test_metrics['test_f1']:.4f}")

    # Save best model artifact
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    best_artifact = {
        "pipeline": best_pipeline,
        "model_name": best_key,
        "classifier_class": models_dict[best_key].__class__.__name__,
        "params": models_dict[best_key].get_params(),
        "tfidf_params": {
            "max_features": max_features,
            "ngram_range": ngram_range,
        },
        "validation_metrics": results[best_key],
        "test_metrics": test_metrics,
        "random_seed": random_seed,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    joblib.dump(best_artifact, best_model_path)
    print(f"Saved best model artifact to {best_model_path}")

    # Compile training report
    training_report = {
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "random_seed": random_seed,
            "max_features": max_features,
            "ngram_range": list(ngram_range),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "baseline_models": results,
        "best_model": {
            "key": best_key,
            "validation_metrics": results[best_key],
            "test_metrics": test_metrics,
        },
    }

    with open(TRAINING_REPORT_PATH, "w") as f:
        json.dump(training_report, f, indent=2)
    print(f"Saved training results report to {TRAINING_REPORT_PATH}")

    return training_report


if __name__ == "__main__":
    report = train_and_evaluate_all()
    print("\nBaseline training completed successfully.")
