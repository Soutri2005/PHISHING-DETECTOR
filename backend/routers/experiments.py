"""
Experiments router — GET /api/experiments/{experiment_type}
Serves pre-computed CSV experiment data from reports/ directory.
"""
import os

import pandas as pd
# pyrefly: ignore[missing-import, missing-module]
from fastapi import APIRouter, HTTPException, Request
# pyrefly: ignore[missing-import, missing-module]
from fastapi.responses import JSONResponse
from backend.middleware.limiter import limiter

router = APIRouter(tags=["Experiments"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

ALLOWED_EXPERIMENTS = {
    "decision_tree": "dt_complexity_experiments.csv",
    "logistic_regression": "lr_regularization_experiments.csv",
    "learning_curve": "learning_curve_experiments.csv",
}


@router.get("/experiments/{experiment_type}", summary="Fetch experiment CSV data as JSON")
@limiter.limit("60/minute")
async def get_experiment(request: Request, experiment_type: str):
    """
    Returns experiment sweep data as a JSON array of records.
    - `decision_tree`       → dt_complexity_experiments.csv
    - `logistic_regression` → lr_regularization_experiments.csv
    - `learning_curve`      → learning_curve_experiments.csv
    """
    if experiment_type not in ALLOWED_EXPERIMENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown experiment '{experiment_type}'. Available: {list(ALLOWED_EXPERIMENTS.keys())}",
        )

    path = os.path.join(REPORTS_DIR, ALLOWED_EXPERIMENTS[experiment_type])
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Experiment file not found. Run `python -m src.experiments` to generate it.",
        )

    try:
        df = pd.read_csv(path)
        return JSONResponse(content={"data": df.to_dict(orient="records")})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read experiment file: {exc}")
