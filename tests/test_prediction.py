"""Unit tests for inference and prediction pipeline."""

import pytest
from src.predict import predict_message
from src.preprocessing import create_pipeline
from sklearn.linear_model import LogisticRegression


def test_predict_message_empty_error():
    with pytest.raises(ValueError):
        predict_message("")


def test_predict_message_with_mock_model():
    # Build a tiny trained pipeline on synthetic data
    pipe = create_pipeline(LogisticRegression(), max_features=100, min_df=1)
    X = [
        "Urgent: verify your banking password now",
        "Security alert: reset your banking credentials immediately",
        "Let us meet for team lunch tomorrow",
        "Please check the meeting notes in the doc for our lunch",
    ]
    y = [1, 1, 0, 0]
    pipe.fit(X, y)

    artifact = {
        "pipeline": pipe,
        "model_name": "TestLogisticRegression",
        "metrics": {"val_accuracy": 1.0},
    }

    res = predict_message("Urgent: your banking password is reset", model_artifact=artifact)
    assert res["label"] in ["Phishing", "Legitimate"]
    assert "heuristics" in res
    assert res["model_used"] == "TestLogisticRegression"
    assert "phishing_probability" in res
    assert 0.0 <= res["phishing_probability"] <= 1.0
