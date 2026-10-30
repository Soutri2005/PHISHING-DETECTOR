"""
Predict router — POST /api/predict
Item #3:  No API keys or secrets on this endpoint.
Item #17: Server-side input validation (length limits, non-empty).
Item #18: Rate limited via slowapi limiter dependency.

Wraps the existing src/predict.py module unchanged.
"""

import os
import sys

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

# Ensure project root is importable (also done in main.py, but safe to repeat)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_message, load_model_artifact, DEFAULT_MODEL_PATH
from backend.middleware.limiter import limiter

router = APIRouter(tags=["Inference"])

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ─── Request / Response Schemas ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Item #17: Strict server-side validation of the predict payload."""
    message: str = Field(
        ...,
        min_length=10,
        max_length=50_000,
        description="The raw email/SMS text to analyze.",
    )
    model_name: str = Field(
        default="best",
        description="Model key: 'best', 'logistic_regression', 'naive_bayes', 'decision_tree', 'random_forest'.",
    )
    # Honeypot field — must always be empty (bot protection, Item #18)
    hp_field: str = Field(
        default="",
        description="Leave empty. Anti-bot honeypot.",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message must not be blank or whitespace only.")
        return v.strip()

    @field_validator("model_name")
    @classmethod
    def valid_model_name(cls, v: str) -> str:
        allowed = {"best", "logistic_regression", "naive_bayes", "decision_tree", "random_forest"}
        if v not in allowed:
            raise ValueError(f"model_name must be one of: {sorted(allowed)}")
        return v

    @field_validator("hp_field")
    @classmethod
    def honeypot_must_be_empty(cls, v: str) -> str:
        """Item #18: If honeypot is filled, reject the request silently."""
        if v:
            raise ValueError("Invalid request.")
        return v


class PredictResponse(BaseModel):
    prediction: int
    label: str
    phishing_probability: float
    confidence: float
    model_used: str
    heuristics: dict
    influential_tokens: list


# ─── Model resolution helper ──────────────────────────────────────────────────

def _resolve_model_path(model_name: str) -> str:
    if model_name == "best" or model_name == "":
        return DEFAULT_MODEL_PATH
    path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return path


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a message as phishing or legitimate",
)
@limiter.limit("30/minute")  # Item #18: rate limiting
async def predict_endpoint(request: Request, body: PredictRequest):
    """
    Runs the message through the fitted TF-IDF + classifier pipeline.
    Returns prediction, probabilities, heuristics, and token-level explanations.
    """
    try:
        model_path = _resolve_model_path(body.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model artifact found. "
                "Run `python -m src.train` to generate model files."
            ),
        )

    try:
        artifact = load_model_artifact(model_path)
        result = predict_message(body.message, model_artifact=artifact)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return PredictResponse(
        prediction=result["prediction"],
        label="phishing" if result["prediction"] == 1 else "legitimate",
        phishing_probability=result["phishing_probability"],
        confidence=result["confidence"],
        model_used=result["model_used"],
        heuristics=result["heuristics"],
        influential_tokens=result.get("influential_tokens", []),
    )
