"""Emergency Kill Switch for instant system halts."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class KillSwitch:
    """Deterministic, auditable emergency kill switch."""

    def __init__(self, audit_trail: Optional[Any] = None) -> None:
        self._active: bool = False
        self._reason: Optional[str] = None
        self._activated_at: Optional[datetime] = None
        self._history: List[Dict[str, Any]] = []
        self._audit_trail = audit_trail

    def set_audit_trail(self, audit_trail: Any) -> None:
        self._audit_trail = audit_trail

    def activate(self, reason: str = "MANUAL_HALT") -> None:
        """Activate the kill switch, rejecting all subsequent orders."""
        cleaned_reason = reason.strip().upper() if reason else "UNSPECIFIED_HALT"
        now = datetime.now(timezone.utc)
        self._active = True
        self._reason = cleaned_reason
        self._activated_at = now
        self._history.append(
            {
                "event": "ACTIVATED",
                "reason": cleaned_reason,
                "timestamp": now.isoformat(),
            }
        )
        if self._audit_trail:
            self._audit_trail.record(
                event_type="KILL_SWITCH_ACTIVATED",
                source="kill_switch",
                details={"reason": cleaned_reason},
            )

    def deactivate(self) -> None:
        """Deactivate the kill switch and restore normal operations."""
        now = datetime.now(timezone.utc)
        prev_reason = self._reason
        self._history.append(
            {
                "event": "DEACTIVATED",
                "previous_reason": prev_reason,
                "timestamp": now.isoformat(),
            }
        )
        self._active = False
        self._reason = None
        self._activated_at = None
        if self._audit_trail:
            self._audit_trail.record(
                event_type="KILL_SWITCH_DEACTIVATED",
                source="kill_switch",
                details={"previous_reason": prev_reason},
            )

    def is_active(self) -> bool:
        """Check whether the kill switch is currently engaged."""
        return self._active

    @property
    def reason(self) -> Optional[str]:
        """Return the reason why the kill switch was activated."""
        return self._reason

    @property
    def activated_at(self) -> Optional[datetime]:
        """Return timestamp of kill switch activation."""
        return self._activated_at

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return audit history of activations/deactivations."""
        return list(self._history)
