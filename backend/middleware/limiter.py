"""Rate limiter singleton shared across all routers. Item #18."""
import os
from slowapi import Limiter
from slowapi.util import get_remote_address

RATE_LIMIT = os.getenv("RATE_LIMIT_PER_MINUTE", "30")

limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT}/minute"])
