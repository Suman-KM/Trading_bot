"""FastAPI application factory and configuration."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from trading.api.routes import router
from trading.logging_config import setup_safety_logger

logger = setup_safety_logger("trading.api")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application for local DEMO/PAPER operations."""
    app = FastAPI(
        title="AI Trading System - Demo Safety Core",
        version="0.1.0",
        description=(
            "Deterministic Safety Core and Paper Trading API for Member 1. "
            "Operates strictly in DEMO/PAPER mode with zero external connectivity."
        ),
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # Global handler to sanitize internal errors and prevent traceback or secret leakage
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred. No trading orders were placed.",
            },
        )

    app.include_router(router)
    return app


# Default application instance
app = create_app()


def run_local(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run application strictly bound to localhost (127.0.0.1)."""
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(
            f"Security violation: server must only bind to 127.0.0.1, requested: {host}"
        )
    import uvicorn

    uvicorn.run("trading.api.app:app", host=host, port=port, log_level="info")
