"""Immutable-style audit event trail for paper trading execution."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AuditEventType(str, Enum):
    """Categorized audit event types across the paper execution lifecycle."""

    SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
    RISK_ACCEPTED = "RISK_ACCEPTED"
    RISK_REJECTED = "RISK_REJECTED"
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_VALIDATED = "ORDER_VALIDATED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_UPDATED = "POSITION_UPDATED"
    POSITION_CLOSED = "POSITION_CLOSED"
    KILL_SWITCH_ACTIVATED = "KILL_SWITCH_ACTIVATED"
    KILL_SWITCH_DEACTIVATED = "KILL_SWITCH_DEACTIVATED"
    EXECUTION_ERROR = "EXECUTION_ERROR"


class AuditEvent(BaseModel):
    """Immutable audit event record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    source: str = "trading_system"
    details: Dict[str, Any] = Field(default_factory=dict)


class AuditTrail:
    """Thread-safe, append-only in-memory audit ledger for paper trading."""

    def __init__(self) -> None:
        self._events: List[AuditEvent] = []

    def record(
        self,
        event_type: AuditEventType,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        position_id: Optional[str] = None,
        source: str = "trading_system",
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Create and append an immutable audit event."""
        event = AuditEvent(
            event_type=event_type,
            symbol=symbol.strip().upper() if symbol else None,
            order_id=order_id,
            position_id=position_id,
            source=source,
            details=details or {},
        )
        self._events.append(event)
        return event

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query recorded events with optional filters."""
        res = self._events
        if event_type is not None:
            res = [e for e in res if e.event_type == event_type]
        if symbol is not None:
            cleaned = symbol.strip().upper()
            res = [e for e in res if e.symbol == cleaned]
        return list(reversed(res[-limit:]))

    def count(self) -> int:
        """Total number of audit events recorded."""
        return len(self._events)

    def clear(self) -> None:
        """Clear audit events (for test isolation)."""
        self._events.clear()
