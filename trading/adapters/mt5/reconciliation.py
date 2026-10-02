"""Deterministic state reconciliation between internal trading platform and MT5 terminal."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from trading.audit.trail import AuditEventType, AuditTrail
from trading.models.portfolio import AccountInfo
from trading.models.position import Position


class ReconciliationStatus(str, Enum):
    """Reconciliation state classification."""

    HEALTHY = "HEALTHY"
    MISMATCH_DETECTED = "MISMATCH_DETECTED"
    ERROR = "ERROR"


class PositionReconciliationError(RuntimeError):
    """Raised when an irreconcilable discrepancy is detected between internal state and MT5."""

    pass


class ReconciliationItem(BaseModel):
    """Detailed reconciliation breakdown for a single symbol or position."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    internal_quantity: float
    broker_volume_lots: float
    broker_volume_units: float
    internal_side: Optional[str] = None
    broker_side: Optional[str] = None
    matched: bool
    ticket: Optional[int] = None
    notes: str


class ReconciliationReport(BaseModel):
    """Comprehensive snapshot reconciliation report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ReconciliationStatus
    healthy: bool
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    internal_open_positions_count: int
    broker_open_positions_count: int
    internal_balance: float
    broker_balance: float
    internal_equity: float
    broker_equity: float
    items: List[ReconciliationItem]
    discrepancies: List[str]


def reconcile_positions(
    internal_positions: Dict[str, Position],
    broker_positions: List[Dict[str, Any]],
    internal_account: Optional[AccountInfo] = None,
    broker_account: Optional[Dict[str, Any]] = None,
    contract_size: float = 100_000.0,
    audit_trail: Optional[AuditTrail] = None,
    strict_fail_closed: bool = False,
) -> ReconciliationReport:
    """Compare internal positions and account state against broker terminal state.

    Invariants:
    - Zero tolerance for phantom positions (positions in broker but not internal, or vice versa).
    - If mismatch detected, marks healthy=False and appends audit event.
    - If strict_fail_closed is True, raises PositionReconciliationError immediately.
    """
    discrepancies: List[str] = []
    items: List[ReconciliationItem] = []

    # Map broker positions by symbol
    broker_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for bp in broker_positions:
        sym = str(bp.get("symbol", "")).strip().upper()
        if sym:
            broker_by_symbol.setdefault(sym, []).append(bp)

    # All unique symbols across both sides
    all_symbols = set(internal_positions.keys()).union(set(broker_by_symbol.keys()))

    for sym in sorted(all_symbols):
        int_pos = internal_positions.get(sym)
        b_pos_list = broker_by_symbol.get(sym, [])

        int_qty = int_pos.quantity if int_pos else 0.0
        int_side = int_pos.side.value if int_pos else None

        b_volume_lots = sum(float(p.get("volume", 0.0)) for p in b_pos_list)
        b_volume_units = round(b_volume_lots * contract_size, 4)

        b_sides = set()
        tickets = []
        for p in b_pos_list:
            t = p.get("type", 0)
            b_sides.add("BUY" if t == 0 else "SELL")
            if "ticket" in p:
                tickets.append(int(p["ticket"]))

        b_side_str = ",".join(sorted(b_sides)) if b_sides else None

        # Check matching
        qty_matched = math.isclose(int_qty, b_volume_units, abs_tol=1e-3)
        side_matched = (int_side == b_side_str) if (int_qty > 0 or b_volume_units > 0) else True

        matched = qty_matched and side_matched
        notes_list = []
        if not qty_matched:
            notes_list.append(
                f"Quantity mismatch: internal={int_qty} units, "
                f"broker={b_volume_units} units ({b_volume_lots} lots)"
            )
        if not side_matched:
            notes_list.append(f"Side mismatch: internal={int_side}, broker={b_side_str}")

        if not matched:
            discrepancies.append(f"Symbol {sym}: {'; '.join(notes_list)}")

        items.append(
            ReconciliationItem(
                symbol=sym,
                internal_quantity=int_qty,
                broker_volume_lots=b_volume_lots,
                broker_volume_units=b_volume_units,
                internal_side=int_side,
                broker_side=b_side_str,
                matched=matched,
                ticket=tickets[0] if tickets else None,
                notes="; ".join(notes_list) if notes_list else "MATCHED_OK",
            )
        )

    # Balance and Equity comparison if provided
    int_bal = internal_account.cash_balance if internal_account else 0.0
    int_eq = internal_account.equity if internal_account else 0.0
    b_bal = float(broker_account.get("balance", 0.0)) if broker_account else int_bal
    b_eq = float(broker_account.get("equity", 0.0)) if broker_account else int_eq

    # Balance and Equity discrepancy check
    # (allow reasonable tolerance for broker swap/floating costs)
    if broker_account and internal_account:
        if abs(int_bal - b_bal) > 500.0:  # Material divergence check
            discrepancies.append(f"Balance divergence: internal={int_bal}, broker={b_bal}")
        if abs(int_eq - b_eq) > 500.0:  # Material divergence check
            discrepancies.append(f"Equity divergence: internal={int_eq}, broker={b_eq}")

    healthy = len(discrepancies) == 0
    status = ReconciliationStatus.HEALTHY if healthy else ReconciliationStatus.MISMATCH_DETECTED

    report = ReconciliationReport(
        status=status,
        healthy=healthy,
        internal_open_positions_count=len(internal_positions),
        broker_open_positions_count=len(broker_positions),
        internal_balance=int_bal,
        broker_balance=b_bal,
        internal_equity=int_eq,
        broker_equity=b_eq,
        items=items,
        discrepancies=discrepancies,
    )

    if not healthy and audit_trail:
        audit_trail.record(
            event_type=AuditEventType.EXECUTION_ERROR,
            details={
                "event": "POSITION_RECONCILIATION_MISMATCH",
                "discrepancies": discrepancies,
            },
        )

    if not healthy and strict_fail_closed:
        disc_str = "; ".join(discrepancies)
        raise PositionReconciliationError(
            f"RECONCILIATION_FAILED: {len(discrepancies)} discrepancy(ies) detected: {disc_str}"
        )

    return report
