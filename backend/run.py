#!/usr/bin/env python3
"""
Startup helper — launches the FastAPI backend with uvicorn.
Run from the project root:
    python backend/run.py
"""
import os
import subprocess
import sys
import uvicorn

ENV = os.getenv("ENVIRONMENT", "development")
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))
RELOAD = ENV == "development"

cmd = [
    sys.executable, "-m", "uvicorn",
    "backend.main:app",
    f"--host={HOST}",
    f"--port={PORT}",
]
if RELOAD:
    cmd.append("--reload")

print(f"[backend] Starting FastAPI {'(dev, hot-reload)' if RELOAD else '(production)'} on http://{HOST}:{PORT}")
subprocess.run(cmd)
