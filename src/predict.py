"""
Inference and Prediction Pipeline for Phishing Message Detector.

Loads fitted scikit-learn pipelines with zero refitting, calculates class probabilities,
extracts rule-based heuristics, and generates keyword explainability.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import clean_text, extract_heuristics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")


def load_model_artifact(model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely load a trained model artifact dictionary containing the fitted pipeline
    and training metadata.
    """
    path = model_path or DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found at: {path}. "
            f"Please run 'python -m src.train' to train baseline models first."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Invalid model artifact structure in {path}")
    return artifact


def get_feature_contributions(
    pipeline: Any,
    text: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    For linear models (Logistic Regression, etc.), calculate the contribution of each word/n-gram
    in the input message to the phishing log-odds.
    """
    try:
        classifier = pipeline.named_steps.get("classifier")
        tfidf = pipeline.named_steps.get("tfidf")

        if not hasattr(classifier, "coef_"):
            return []

        # Transform message text using the fitted TF-IDF vectorizer (no fit, just transform)
        tfidf_vec = tfidf.transform([text])
        feature_names = tfidf.get_feature_names_out()
        coefs = classifier.coef_[0]

        # Non-zero entries in this document
        row_indices = tfidf_vec.nonzero()[1]
        contributions = []

        for idx in row_indices:
            word = feature_names[idx]
            tfidf_val = tfidf_vec[0, idx]
            weight = coefs[idx]
            impact = tfidf_val * weight

            contributions.append({
                "token": word,
                "tfidf": round(float(tfidf_val), 4),
                "model_weight": round(float(weight), 4),
                "impact": round(float(impact), 4),
                "direction": "Phishing" if impact > 0 else "Legitimate",
            })

        # Sort by absolute impact descending
        contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return contributions[:top_k]
    except Exception:
        return []


def predict_message(
    text: str,
    model_artifact: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify a message text as Phishing or Legitimate.
    Returns predicted label, probability, heuristics, and token explainability.
    """
    if not text or not text.strip():
        raise ValueError("Cannot classify an empty message.")

    cleaned = clean_text(text)

    # Load artifact if not provided
    if model_artifact is None:
        model_artifact = load_model_artifact(model_path)

    pipeline = model_artifact["pipeline"]
    model_name = model_artifact.get("model_name", "Unknown Classifier")

    # Predict using the fitted pipeline
    prediction_int = int(pipeline.predict([cleaned])[0])
    label_name = "Phishing" if prediction_int == 1 else "Legitimate"

    # Probability estimation
    probability = None
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([cleaned])[0]
        # Probability of Phishing (class 1)
        probability = float(probs[1])
        confidence = float(probs[prediction_int])
    elif hasattr(pipeline, "decision_function"):
        score = float(pipeline.decision_function([cleaned])[0])
        # Sigmoid conversion
        probability = float(1 / (1 + np.exp(-score)))
        confidence = probability if prediction_int == 1 else (1.0 - probability)

    # Heuristic analysis
    heuristics = extract_heuristics(cleaned)

    # Token contributions (if supported by model)
    contributions = get_feature_contributions(pipeline, cleaned)

    return {
        "text": cleaned,
        "prediction": prediction_int,
        "label": label_name,
        "phishing_probability": round(probability, 4) if probability is not None else None,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "model_used": model_name,
        "heuristics": heuristics,
        "influential_tokens": contributions,
    }
