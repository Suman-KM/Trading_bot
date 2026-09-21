"""Structured JSON logging configuration for trading safety audit trail."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredJSONFormatter(logging.Formatter):
    """Formats log records as strict JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured audit fields if present
        for field in ("symbol", "action", "risk_decision", "reason", "order_id", "confidence"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def setup_safety_logger(name: str = "trading.safety", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if reconfigured
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJSONFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def log_risk_audit(
    logger: logging.Logger,
    symbol: str,
    action: str,
    approved: bool,
    reason: Optional[str] = None,
    order_id: Optional[str] = None,
    confidence: Optional[float] = None,
    extra_msg: str = "",
) -> None:
    """Emit a structured audit log entry."""
    decision_str = "APPROVED" if approved else "REJECTED"
    msg = extra_msg or f"Risk evaluation: {decision_str}"
    logger.info(
        msg,
        extra={
            "symbol": symbol,
            "action": action,
            "risk_decision": decision_str,
            "reason": reason or "NONE",
            "order_id": order_id or "NONE",
            "confidence": confidence if confidence is not None else 0.0,
        },
    )
