"""FastAPI Paper Trading application module."""

from trading.api.app import app, create_app, run_local

__all__ = ["app", "create_app", "run_local"]
