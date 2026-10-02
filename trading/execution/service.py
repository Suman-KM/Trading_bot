from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.adapter import BrokerAdapter
from trading.execution.paper_broker import PaperBroker
from trading.logging_config import log_risk_audit, setup_safety_logger
from trading.models.order import Order, OrderStatus
from trading.models.signal import Signal
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskDecision, RiskEngine, RiskReason

if TYPE_CHECKING:
    pass


class ExecutionResult(BaseModel):
    """Result returned by the trading execution pipeline."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    reason: Optional[str] = None
    order: Optional[Order] = None
    signal_symbol: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_idempotent_replay: bool = False


class TradingExecutionService:
    """Coordinates signal validation, risk engine approval, and paper execution.

    Invariants:
    - Sovereign Risk Engine is the final non-bypassable authority.
    - Idempotent request handling: duplicate signal requests produce identical cached
      results without re-executing.
    - Comprehensive structured audit trail logging for all lifecycle events.
    - Safe failure recovery: exceptions are captured without portfolio corruption.
    - Concurrency-safe: serialized execution prevents race conditions against risk limits.
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        paper_broker: Union[PaperBroker, BrokerAdapter],
        portfolio_manager: PortfolioManager,
        logger: Optional[logging.Logger] = None,
        audit_trail: Optional[AuditTrail] = None,
        repository: Optional[Any] = None,
    ) -> None:
        self.risk_engine = risk_engine
        self.broker = paper_broker
        self.portfolio_manager = portfolio_manager
        self.logger = logger or setup_safety_logger()
        self.audit_trail = audit_trail or AuditTrail()
        self.repository = repository
        self._lock = threading.RLock()
        self._idempotency_cache: Dict[str, ExecutionResult] = {}

    def set_repository(self, repository: Any) -> None:
        """Attach persistent repository."""
        with self._lock:
            self.repository = repository

    def _compute_idempotency_key(self, signal: Signal) -> str:
        """Derive a deterministic idempotency key for an incoming signal."""
        if signal.client_request_id:
            return f"req_{signal.client_request_id}"
        fingerprint = (
            f"{signal.symbol}_{signal.action.value}_{signal.timestamp.isoformat()}_"
            f"{signal.model_version}_{signal.suggested_entry_price}_{signal.confidence}"
        )
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

    def _save_idempotent_result(
        self, idempotency_key: str, res: ExecutionResult, client_request_id: Optional[str] = None
    ) -> None:
        """Cache and persist execution result for idempotency deduplication."""
        self._idempotency_cache[idempotency_key] = res
        if self.repository:
            order_id = res.order.order_id if res.order else "NO_ORDER"
            exec_id = f"exec-{idempotency_key[:12]}"
            status = "APPROVED" if res.approved else "REJECTED"
            json_payload = res.model_dump_json()
            self.repository.save_idempotency_record(
                client_request_id=idempotency_key,
                order_id=order_id,
                execution_id=exec_id,
                status=status,
                execution_report_json=json_payload,
            )
            if client_request_id and client_request_id != idempotency_key:
                self.repository.save_idempotency_record(
                    client_request_id=client_request_id,
                    order_id=order_id,
                    execution_id=exec_id,
                    status=status,
                    execution_report_json=json_payload,
                )

    def process_signal(
        self,
        signal_input: Union[Signal, Dict[str, Any]],
        current_market_price: Optional[float] = None,
    ) -> ExecutionResult:
        """Process incoming signal through strict safety verification.

        INVARIANT: PaperBroker is NEVER invoked if the RiskEngine rejects or fails.
        """
        with self._lock:
            # Step 1: Strict Signal Schema Validation
            if isinstance(signal_input, Signal):
                signal = signal_input
            else:
                try:
                    signal = Signal.model_validate(signal_input)
                except ValidationError as err:
                    error_msg = f"SCHEMA_VALIDATION_ERROR: {err.errors()[0]['msg']}"
                    raw_symbol = (
                        str(signal_input.get("symbol", "UNKNOWN"))
                        if isinstance(signal_input, dict)
                        else "UNKNOWN"
                    )
                    self.audit_trail.record(
                        event_type=AuditEventType.EXECUTION_ERROR,
                        symbol=raw_symbol,
                        details={"error": error_msg},
                    )
                    log_risk_audit(
                        logger=self.logger,
                        symbol=raw_symbol,
                        action="UNKNOWN",
                        approved=False,
                        reason=RiskReason.INVALID_SIGNAL.value,
                        extra_msg=error_msg,
                    )
                    return ExecutionResult(
                        approved=False,
                        reason=RiskReason.INVALID_SIGNAL.value,
                        order=None,
                        signal_symbol=raw_symbol,
                        metadata={"error": error_msg},
                    )

            # Step 2: Idempotency Verification (In-Memory + Persistent)
            idempotency_key = self._compute_idempotency_key(signal)
            cached: Optional[ExecutionResult] = self._idempotency_cache.get(idempotency_key)
            if cached is None and self.repository:
                try:
                    rec = self.repository.get_idempotency_record(idempotency_key)
                    if not rec and signal.client_request_id:
                        rec = self.repository.get_idempotency_record(signal.client_request_id)
                    if rec:
                        try:
                            cached_dict = json.loads(rec["execution_report_json"])
                            cached = ExecutionResult.model_validate(cached_dict)
                            self._idempotency_cache[idempotency_key] = cached
                        except Exception:
                            cached = None
                except Exception as err:
                    return ExecutionResult(
                        approved=False,
                        reason=f"EXECUTION_EXCEPTION: {str(err)}",
                        order=None,
                        signal_symbol=signal.symbol,
                        metadata={"error": str(err)},
                    )

            if cached is not None:
                # Return idempotent replay copy
                return ExecutionResult(
                    approved=cached.approved,
                    reason=cached.reason,
                    order=cached.order.model_copy() if cached.order else None,
                    signal_symbol=cached.signal_symbol,
                    metadata={**cached.metadata, "idempotency_key": idempotency_key},
                    is_idempotent_replay=True,
                )

            # Step 3: Record Signal Received Audit Event
            self.audit_trail.record(
                event_type=AuditEventType.SIGNAL_RECEIVED,
                symbol=signal.symbol,
                details={
                    "action": signal.action.value,
                    "confidence": signal.confidence,
                    "idempotency_key": idempotency_key,
                    "client_request_id": signal.client_request_id,
                },
            )

            try:
                # Optional database transaction wrapping state mutations
                db_tx = (
                    self.repository.db.transaction()
                    if self.repository and hasattr(self.repository, "db")
                    else None
                )

                def _execute_pipeline() -> ExecutionResult:
                    # Step 4: Query current simulated account and daily baseline
                    account = self.broker.get_account()
                    daily_baseline = self.portfolio_manager.daily_start_equity

                    # Step 5: Sovereign Risk Engine Evaluation & Position Sizing
                    decision: RiskDecision = self.risk_engine.evaluate(
                        signal=signal,
                        account=account,
                        daily_start_equity=daily_baseline,
                        current_market_price=current_market_price,
                    )

                    # Step 6: Audit Logging
                    order_id = decision.order.order_id if decision.order else None
                    log_risk_audit(
                        logger=self.logger,
                        symbol=signal.symbol,
                        action=signal.action.value,
                        approved=decision.approved,
                        reason=decision.reason,
                        order_id=order_id,
                        confidence=signal.confidence,
                    )

                    # Step 7: Enforce Non-Bypassable Boundary
                    if not decision.approved or decision.order is None:
                        self.audit_trail.record(
                            event_type=AuditEventType.RISK_REJECTED,
                            symbol=signal.symbol,
                            details={"reason": decision.reason, "metadata": decision.metadata},
                        )
                        res = ExecutionResult(
                            approved=False,
                            reason=decision.reason,
                            order=None,
                            signal_symbol=signal.symbol,
                            metadata=decision.metadata,
                        )
                        self._save_idempotent_result(idempotency_key, res, signal.client_request_id)
                        return res

                    self.audit_trail.record(
                        event_type=AuditEventType.RISK_ACCEPTED,
                        symbol=signal.symbol,
                        order_id=decision.order.order_id,
                        details=decision.metadata,
                    )

                    # Assign client_request_id to order
                    decision.order.client_request_id = signal.client_request_id
                    decision.order.transition_to(OrderStatus.VALIDATED)

                    self.audit_trail.record(
                        event_type=AuditEventType.ORDER_CREATED,
                        symbol=decision.order.symbol,
                        order_id=decision.order.order_id,
                        details={
                            "quantity": decision.order.quantity,
                            "side": decision.order.side.value,
                            "price": decision.order.price,
                        },
                    )

                    # Step 8: Execute exclusively on Paper Broker Simulator
                    executed_order = self.broker.submit_order(
                        decision.order, current_market_price=current_market_price
                    )

                    res = ExecutionResult(
                        approved=True,
                        reason=None,
                        order=executed_order,
                        signal_symbol=signal.symbol,
                        metadata=decision.metadata,
                    )
                    self._save_idempotent_result(idempotency_key, res, signal.client_request_id)
                    return res

                if db_tx:
                    with db_tx:
                        return _execute_pipeline()
                else:
                    return _execute_pipeline()

            except Exception as exc:
                self.audit_trail.record(
                    event_type=AuditEventType.EXECUTION_ERROR,
                    symbol=signal.symbol,
                    details={"exception": str(exc)},
                )
                return ExecutionResult(
                    approved=False,
                    reason=f"EXECUTION_EXCEPTION: {str(exc)}",
                    order=None,
                    signal_symbol=signal.symbol,
                    metadata={"error": str(exc)},
                )
