"""
Preprocessing and feature extraction module for Phishing Message Detector.

This module provides:
1. Safe text cleaning and normalization functions.
2. Domain heuristic extractors (suspicious URLs, urgency keywords, credential requests).
3. Leakage-free TF-IDF pipeline builder.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


# Common regex patterns for phishing detection heuristics
URL_PATTERN = re.compile(
    r'(https?://[^\s<>"]+|www\.[^\s<>"]+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)',
    re.IGNORECASE,
)
IP_URL_PATTERN = re.compile(
    r'https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s]*)?',
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
    re.IGNORECASE,
)

URGENCY_KEYWORDS = [
    "urgent", "immediately", "immediate", "act now", "action required",
    "account suspended", "suspended", "unauthorized", "suspicious activity",
    "limited time", "expire", "expiration", "24 hours", "48 hours",
    "security alert", "compromised", "verify your account", "terminate",
    "critical", "failure to", "warning", "attention needed"
]

CREDENTIAL_KEYWORDS = [
    "password", "passcode", "pin", "credential", "ssn", "social security",
    "bank account", "credit card", "debit card", "cvv", "routing number",
    "billing", "update payment", "confirm identity", "verify identity",
    "login to", "sign in here", "wire transfer", "crypto", "bitcoin", "wallet"
]

AUTHORITY_KEYWORDS = [
    "it department", "helpdesk", "administrator", "system admin",
    "security team", "paypal", "microsoft", "google", "apple", "amazon",
    "netflix", "irs", "federal", "official notification", "dear customer",
    "dear client", "dear valued member"
]


def clean_text(text: Any) -> str:
    """
    Safely clean and normalize raw message text.
    Handles nulls, numbers, and converts to string.
    Removes non-printable characters and normalizes excess whitespace.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Replace null bytes or non-printable characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)

    # Normalize multiple whitespaces, tabs, newlines
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_heuristics(text: str) -> Dict[str, Any]:
    """
    Extract rule-based cybersecurity heuristics from an email or SMS message.
    IMPORTANT: These indicators are heuristic signals and NOT definitive proof
    of maliciousness or safety.
    """
    cleaned = clean_text(text)
    lower = cleaned.lower()

    # URL detection
    urls = URL_PATTERN.findall(cleaned)
    ip_urls = IP_URL_PATTERN.findall(cleaned)
    emails = EMAIL_PATTERN.findall(cleaned)

    # Keyword matching
    urgency_matches = [kw for kw in URGENCY_KEYWORDS if kw in lower]
    credential_matches = [kw for kw in CREDENTIAL_KEYWORDS if kw in lower]
    authority_matches = [kw for kw in AUTHORITY_KEYWORDS if kw in lower]

    # Structural signals
    char_count = len(cleaned)
    word_count = len(cleaned.split())
    uppercase_count = sum(1 for c in cleaned if c.isupper())
    uppercase_ratio = round(uppercase_count / max(char_count, 1), 3)
    exclamation_count = cleaned.count("!")
    question_count = cleaned.count("?")

    # Risk score calculation (0 - 100 heuristic index)
    risk_score = 0
    if len(urls) > 0:
        risk_score += 20
    if len(urls) > 3:
        risk_score += 15
    if len(ip_urls) > 0:
        risk_score += 35
    if len(urgency_matches) > 0:
        risk_score += min(len(urgency_matches) * 10, 25)
    if len(credential_matches) > 0:
        risk_score += min(len(credential_matches) * 12, 30)
    if uppercase_ratio > 0.25:
        risk_score += 10
    if exclamation_count > 3:
        risk_score += 10

    risk_score = min(risk_score, 100)

    return {
        "url_count": len(urls),
        "urls_detected": urls[:5],  # Top 5 for inspection (never executed)
        "has_ip_url": len(ip_urls) > 0,
        "email_address_count": len(emails),
        "urgency_triggers": urgency_matches,
        "credential_triggers": credential_matches,
        "authority_triggers": authority_matches,
        "char_count": char_count,
        "word_count": word_count,
        "uppercase_ratio": uppercase_ratio,
        "exclamation_count": exclamation_count,
        "heuristic_risk_score": risk_score,
    }


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Construct a TF-IDF vectorizer configured for text classification.
    Always fit this vectorizer ONLY on training data to prevent leakage.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        stop_words="english",
        strip_accents="unicode",
        lowercase=True,
    )


def create_pipeline(
    classifier: Any,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> Pipeline:
    """
    Wrap the vectorizer and classifier in a single scikit-learn Pipeline.
    This guarantees zero leakage during cross-validation and ensures that
    the identical vectorization vocabulary and IDF weights are preserved
    for future inference.
    """
    tfidf = build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    return Pipeline([
        ("tfidf", tfidf),
        ("classifier", classifier),
    ])
