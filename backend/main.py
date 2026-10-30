"""
FastAPI Backend — Phishing Detector Overfitting Study
Wraps existing src/ Python ML modules with a REST API.

Items addressed:
  #3  — Secrets via env vars (python-dotenv), never hardcoded
  #4  — HTTPS redirect (HTTPSRedirectMiddleware in production)
  #18 — Rate limiting (slowapi) + CORS whitelist + input length limits
"""

import os
import sys
from contextlib import asynccontextmanager

# pyrefly: ignore[missing-import]
from fastapi import FastAPI
# pyrefly: ignore[missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore[missing-import]
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
# pyrefly: ignore[missing-import]
from fastapi.responses import JSONResponse
# pyrefly: ignore[missing-import]
from slowapi import _rate_limit_exceeded_handler
# pyrefly: ignore[missing-import]
from slowapi.errors import RateLimitExceeded

# Ensure project root (parent of backend/) is on sys.path so src/ is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# pyrefly: ignore[missing-import]
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from backend.routers import predict, reports, experiments, health
from backend.middleware.security import add_security_headers
from backend.middleware.limiter import limiter

# ─── Configuration (all from environment — no secrets in code) ────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",")]


# ─── App Lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Phishing Detector API starting. Environment: {ENVIRONMENT}")
    print(f"[startup] Allowed CORS origins: {ALLOWED_ORIGINS}")
    yield
    print("[shutdown] Phishing Detector API shutting down.")


# ─── App Instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Phishing Detector API",
    description=(
        "REST API for the Phishing Message Detector Overfitting Study. "
        "Exposes ML inference, experiment data, and report endpoints."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ─── Rate Limiter ──────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── HTTPS Redirect (production only — avoids breaking local dev) ─────────────
if ENVIRONMENT == "production":
    app.add_middleware(HTTPSRedirectMiddleware)  # Item #4

# ─── CORS (whitelist only — never wildcard in production) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# ─── Security Headers Middleware ───────────────────────────────────────────────
app.middleware("http")(add_security_headers)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")


# ─── Root redirect ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse(
        {"message": "Phishing Detector API", "docs": "/api/docs", "status": "ok"}
    )
