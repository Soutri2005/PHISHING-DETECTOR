"""
Reports router — GET /api/reports/{report_type}
Serves pre-computed JSON reports from reports/ directory.
"""
import json
import os

# pyrefly: ignore[missing-import]
from fastapi import APIRouter, HTTPException, Request
from backend.middleware.limiter import limiter

router = APIRouter(tags=["Reports"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

ALLOWED_REPORTS = {
    "data": "data_preparation_report.json",
    "training": "training_results.json",
}


@router.get("/reports/{report_type}", summary="Fetch a pre-computed JSON report")
@limiter.limit("60/minute")
async def get_report(request: Request, report_type: str):
    """
    Returns one of the pre-computed analysis JSON reports.
    - `data`     → data_preparation_report.json
    - `training` → training_results.json
    """
    if report_type not in ALLOWED_REPORTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown report type '{report_type}'. Available: {list(ALLOWED_REPORTS.keys())}",
        )

    path = os.path.join(REPORTS_DIR, ALLOWED_REPORTS[report_type])
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"Report file not found. Run `python -m src.train` to generate it.",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read report: {exc}")
