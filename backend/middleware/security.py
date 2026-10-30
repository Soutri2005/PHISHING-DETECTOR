"""
Security headers middleware. Item #4 (HSTS) + general hardening.
Applied to every response from the FastAPI backend.
"""
# pyrefly: ignore[missing-import]
from fastapi import Request
# pyrefly: ignore[missing-import, parse-error]
from fastapi.responses import Response


async def add_security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)

    # HTTP Strict Transport Security (force HTTPS for 1 year) — Item #4
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains; preload"
    )
    # Prevent MIME sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Deny embedding in iframes
    response.headers["X-Frame-Options"] = "DENY"
    # Enable XSS protection (legacy browsers)
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Remove server fingerprint
    response.headers.pop("server", None)
    response.headers.pop("x-powered-by", None)

    return response
