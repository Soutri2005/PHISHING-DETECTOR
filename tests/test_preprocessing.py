"""Unit tests for text preprocessing, heuristics, and pipeline construction."""

import pytest
from src.preprocessing import clean_text, extract_heuristics, build_tfidf_vectorizer, create_pipeline
from sklearn.linear_model import LogisticRegression


def test_clean_text_basic():
    raw = "  Hello   world!  \n\t Check this: http://example.com  "
    cleaned = clean_text(raw)
    assert cleaned == "Hello world! Check this: http://example.com"


def test_clean_text_null_and_types():
    assert clean_text(None) == ""
    assert clean_text(12345) == "12345"
    assert clean_text("") == ""


def test_extract_heuristics_urgency_and_urls():
    phish_msg = (
        "URGENT: Your account has been suspended immediately! "
        "Verify your password and pin at http://192.168.1.1/login within 24 hours."
    )
    res = extract_heuristics(phish_msg)
    assert res["url_count"] >= 1
    assert res["has_ip_url"] is True
    assert "urgent" in res["urgency_triggers"]
    assert "suspended" in res["urgency_triggers"]
    assert "password" in res["credential_triggers"]
    assert res["heuristic_risk_score"] > 50


def test_extract_heuristics_benign():
    benign_msg = "Hey Sarah, let's catch up tomorrow at 2 PM for coffee and review the project slides."
    res = extract_heuristics(benign_msg)
    assert res["url_count"] == 0
    assert len(res["urgency_triggers"]) == 0
    assert len(res["credential_triggers"]) == 0
    assert res["heuristic_risk_score"] < 20


def test_pipeline_construction():
    pipe = create_pipeline(LogisticRegression(), max_features=100)
    assert "tfidf" in pipe.named_steps
    assert "classifier" in pipe.named_steps
    assert pipe.named_steps["tfidf"].max_features == 100
