"""
Health check router.
GET /api/health — liveness probe for the backend service.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Health"])


@router.get("/health", summary="API health check")
async def health_check():
    """Returns 200 OK when the backend is running."""
    return JSONResponse({"status": "ok", "service": "phishing-detector-api"})
