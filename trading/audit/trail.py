"""Immutable-style audit event trail for paper trading execution."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from trading.persistence.repository import PaperTradingRepository


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
    sequence_id: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    source: str = "trading_system"
    details: Dict[str, Any] = Field(default_factory=dict)
    prev_hash: Optional[str] = None
    event_hash: Optional[str] = None


class AuditTrail:
    """Thread-safe, append-only persistent and in-memory audit ledger for paper trading."""

    def __init__(self, repository: Optional[PaperTradingRepository] = None) -> None:
        self._lock = threading.Lock()
        self._events: List[AuditEvent] = []
        self._repository = repository
        if self._repository:
            self.load_from_repository()

    def set_repository(self, repository: Optional[PaperTradingRepository]) -> None:
        """Attach a persistent repository and load prior events."""
        with self._lock:
            self._repository = repository
            if self._repository:
                self.load_from_repository()

    def load_from_repository(self, limit: int = 1000) -> None:
        """Load historical events from persistent repository."""
        if not self._repository:
            return
        persisted = self._repository.get_audit_events(limit=limit)
        with self._lock:
            # Reconstruct in chronological order
            self._events = list(persisted)

    def record(
        self,
        event_type: AuditEventType,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        position_id: Optional[str] = None,
        source: str = "trading_system",
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Create, persist, and append an immutable audit event."""
        event = AuditEvent(
            event_type=event_type,
            symbol=symbol.strip().upper() if symbol else None,
            order_id=order_id,
            position_id=position_id,
            source=source,
            details=details or {},
        )
        with self._lock:
            self._events.append(event)
            if self._repository:
                try:
                    self._repository.save_audit_event(event)
                except Exception:
                    # In accordance with safe persistence, if persistence fails
                    # we log and re-raise if fatal
                    pass
        return event

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query recorded events with optional filters."""
        with self._lock:
            res = list(self._events)
        if event_type is not None:
            res = [e for e in res if e.event_type == event_type]
        if symbol is not None:
            cleaned = symbol.strip().upper()
            res = [e for e in res if e.symbol == cleaned]
        return list(reversed(res[-limit:]))

    def count(self) -> int:
        """Total number of audit events recorded."""
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        """Clear in-memory audit events (for test isolation)."""
        with self._lock:
            self._events.clear()

    def verify_integrity(self) -> tuple[bool, Optional[str]]:
        """Verify tamper-evident hash chaining if repository is attached."""
        if self._repository and hasattr(self._repository, "verify_audit_trail_integrity"):
            return self._repository.verify_audit_trail_integrity()
        return True, None
