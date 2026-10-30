"""
Health check router.
GET /api/health — liveness probe for the backend service.
"""
# pyrefly: ignore[missing-import, missing-module]
from fastapi import APIRouter
# pyrefly: ignore[missing-import, missing-module]
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Health"])


@router.get("/health", summary="API health check")
async def health_check():
    """Returns 200 OK when the backend is running."""
    return JSONResponse({"status": "ok", "service": "phishing-detector-api"})
