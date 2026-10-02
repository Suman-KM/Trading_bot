"""Extended MT5 Demo Forward Validation runner and operational stability engine.

Validates the complete execution pipeline under realistic forward-market conditions:
Market Data -> Signal Generation -> Risk Engine -> Position Sizing -> Order Validation
-> MT5 Demo Execution -> Broker Response -> Position Reconciliation -> Persistence
-> Audit Trail -> Recovery -> Monitoring.

ABSOLUTE GOVERNANCE & SAFETY INVARIANTS:
1. DEMO ONLY. ZERO REAL MONEY.
2. LIVE accounts (trade_mode == 2) immediately raise LiveAccountForbiddenError.
3. Unknown or unverified account modes fail closed.
4. Strategy is frozen: zero forced trades, zero parameter modification.
5. Strict operational guards: max daily trades, max open positions, consecutive error limit,
   reconciliation failure limit, and data staleness limit.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from trading.execution.mt5_adapter import MT5BrokerAdapter
from pydantic import BaseModel, ConfigDict, Field

from trading.adapters.mt5.client import MT5ReadOnlyClient
from trading.adapters.mt5.safety import (
    DEMO_ONLY,
    DemoAccountVerifier,
    LiveAccountForbiddenError,
    assert_demo_execution_authorized,
)
from trading.adapters.mt5.schemas import MT5TickData
from trading.audit.trail import AuditTrail
from trading.execution.exceptions import (
    BrokerExecutionDisabledError,
    MT5ConnectionError,
)
from trading.execution.service import TradingExecutionService
from trading.models.order import OrderStatus
from trading.models.signal import Signal, SignalAction
from trading.persistence.repository import PaperTradingRepository
from trading.risk.engine import RiskEngine
from trading.risk.kill_switch import KillSwitch

logger = logging.getLogger("trading.adapters.mt5.forward")


class ForwardValidationConfig(BaseModel):
    """Configuration for forward demo operational validation sessions."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = "EURUSD"
    timeframe: str = "M15"
    max_demo_trades_per_day: int = 3
    max_demo_open_positions: int = 1
    max_consecutive_execution_errors: int = 3
    max_reconciliation_failures: int = 1
    max_data_staleness_seconds: float = 120.0
    confidence_threshold: float = 0.60
    stop_loss_pips: float = 50.0
    take_profit_pips: float = 50.0
    poll_interval_seconds: float = 1.0
    close_on_stop: bool = False


class ForwardPreflightReport(BaseModel):
    """Preflight diagnostic report for forward session start gate."""

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    broker: str
    server: str
    account_login_masked: str
    account_mode: str
    demo_verified: bool
    live_account: bool
    symbol: str
    bid: float
    ask: float
    spread: float
    data_timestamp_utc: str
    data_fresh: bool
    risk_engine_ready: bool
    kill_switch_active: bool
    reconciliation_healthy: bool
    execution_enabled: bool
    forward_validation_enabled: bool
    can_start: bool
    notes: List[str] = Field(default_factory=list)


class ForwardSignalRecord(BaseModel):
    """Structured record for a generated trading signal."""

    model_config = ConfigDict(extra="forbid")

    timestamp_utc: str
    symbol: str
    timeframe: str
    action: str
    confidence: float
    entry_estimate: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_decision: str
    risk_rejection_reason: Optional[str] = None
    spread: float
    market_state: str
    execution_decision: str


class ForwardExecutionRecord(BaseModel):
    """Structured record for an executed trade on the demo broker."""

    model_config = ConfigDict(extra="forbid")

    internal_order_id: str
    broker_order_ticket: str
    deal_ticket: str
    symbol: str
    side: str
    requested_volume: float
    executed_volume: float
    requested_price: Optional[float] = None
    executed_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    spread: float
    broker_retcode: int
    submission_timestamp: str
    fill_timestamp: Optional[str] = None
    execution_latency_ms: float


class ForwardOperationalMetrics(BaseModel):
    """Observability metrics tracking forward demo operation."""

    model_config = ConfigDict(extra="forbid")

    signals_generated: int = 0
    signals_approved: int = 0
    signals_rejected: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_timeout: int = 0
    execution_errors: int = 0
    consecutive_execution_errors: int = 0
    reconciliation_failures: int = 0
    duplicate_prevented: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    daily_pnl: float = 0.0
    cumulative_demo_pnl: float = 0.0
    max_demo_drawdown: float = 0.0
    average_execution_latency_ms: float = 0.0
    maximum_execution_latency_ms: float = 0.0
    data_staleness_events: int = 0
    kill_switch_events: int = 0
    session_start_utc: Optional[str] = None
    session_end_utc: Optional[str] = None
    session_duration_seconds: float = 0.0


class ForwardDemoRunner:
    """Orchestrator for extended MT5 demo forward validation and stability monitoring.

    Enforces sovereign risk, demo account safety gates, market data freshness,
    reconciliation health, and deterministic audit persistence.
    """

    def __init__(
        self,
        adapter: MT5BrokerAdapter,
        execution_service: TradingExecutionService,
        client: Optional[MT5ReadOnlyClient] = None,
        repository: Optional[PaperTradingRepository] = None,
        audit_trail: Optional[AuditTrail] = None,
        risk_engine: Optional[RiskEngine] = None,
        kill_switch: Optional[KillSwitch] = None,
        config: Optional[ForwardValidationConfig] = None,
        strategy_fn: Optional[Callable[[MT5TickData], Optional[Signal]]] = None,
    ) -> None:
        self.adapter = adapter
        self.execution_service = execution_service
        self.client = client
        self.repository = repository
        self.audit_trail = audit_trail
        self.risk_engine = risk_engine
        self.kill_switch = kill_switch
        self.config = config or ForwardValidationConfig()
        self.strategy_fn = strategy_fn

        self.metrics = ForwardOperationalMetrics()
        self.signal_logs: List[ForwardSignalRecord] = []
        self.execution_logs: List[ForwardExecutionRecord] = []
        self._latencies: List[float] = []

        self._stop_requested = False
        self._is_running = False
        self._seen_client_request_ids: set[str] = set()
        self._peak_equity: float = 0.0

    @property
    def is_running(self) -> bool:
        """Return True if session is currently active."""
        return self._is_running

    def preflight_inspection(self) -> ForwardPreflightReport:
        """Perform comprehensive preflight safety inspection before session start."""
        notes: List[str] = []
        now_utc = datetime.now(timezone.utc)

        # 1. Assert Demo Only Global Flag
        if not DEMO_ONLY:
            notes.append("DEMO_ONLY global constant is False.")
            raise BrokerExecutionDisabledError("FATAL: DEMO_ONLY invariant violated.")

        # 2. Inspect Client Metadata
        if self.client is None or not self.client.is_connected:
            notes.append("MT5 client is not connected.")
            raise MT5ConnectionError("MT5 terminal client session is not connected.")

        acc_meta = self.client.get_account_metadata()
        term_meta = self.client.get_terminal_metadata()
        sym_spec = self.client.get_symbol_specification(self.config.symbol)

        # Fail-closed if live account detected
        if acc_meta.trade_mode == 2:
            notes.append("FATAL: Real capital / live account detected.")
            raise LiveAccountForbiddenError("LIVE_ACCOUNT_FORBIDDEN: trade_mode == 2")

        if acc_meta.trade_mode != 0 or not acc_meta.is_demo:
            notes.append(f"Account mode {acc_meta.trade_mode} is not DEMO.")
            raise LiveAccountForbiddenError("Account is not confirmed as DEMO.")

        # Positive Demo Verification
        demo_ver_rep = DemoAccountVerifier.verify(acc_meta, term_meta, sym_spec)
        assert_demo_execution_authorized(
            is_live=self.adapter.is_live,
            execution_enabled=self.adapter.execution_enabled,
            demo_execution_enabled=self.adapter.demo_execution_enabled,
            account_meta=acc_meta,
        )

        # 3. Market Data Freshness
        tick = self.client.get_latest_tick(self.config.symbol)
        staleness = (now_utc - tick.timestamp_utc).total_seconds()
        data_fresh = staleness <= self.config.max_data_staleness_seconds
        if not data_fresh:
            notes.append(
                f"Market data is stale: {staleness:.1f}s > "
                f"{self.config.max_data_staleness_seconds}s."
            )

        # 4. Risk Engine & Kill Switch
        ks_active = self.kill_switch.is_active() if self.kill_switch else False
        risk_ready = (self.risk_engine is not None) and not ks_active
        if ks_active:
            notes.append("Kill switch is active.")

        # 5. Reconciliation Health
        rec_report = self.adapter.reconcile()
        rec_healthy = rec_report.healthy
        if not rec_healthy:
            notes.append(f"Initial reconciliation failed: {rec_report.discrepancies}")

        can_start = (
            demo_ver_rep.is_demo
            and acc_meta.trade_mode == 0
            and not self.adapter.is_live
            and data_fresh
            and risk_ready
            and not ks_active
            and rec_healthy
            and self.adapter.execution_enabled
        )

        report = ForwardPreflightReport(
            timestamp_utc=now_utc.isoformat(),
            broker=acc_meta.company,
            server=acc_meta.server,
            account_login_masked=acc_meta.login_masked,
            account_mode="DEMO (0)",
            demo_verified=demo_ver_rep.is_demo,
            live_account=acc_meta.trade_mode == 2,
            symbol=self.config.symbol,
            bid=tick.bid,
            ask=tick.ask,
            spread=tick.spread,
            data_timestamp_utc=tick.timestamp_utc.isoformat(),
            data_fresh=data_fresh,
            risk_engine_ready=risk_ready,
            kill_switch_active=ks_active,
            reconciliation_healthy=rec_healthy,
            execution_enabled=self.adapter.execution_enabled,
            forward_validation_enabled=can_start,
            can_start=can_start,
            notes=notes,
        )

        return report

    def evaluate_frozen_strategy(self, tick: MT5TickData) -> Optional[Signal]:
        """Evaluate the frozen strategy against live market data.

        Strict invariant: If no valid signal condition is met, returns None (0 signals).
        Zero forced trades, zero synthetic signals.
        """
        if self.strategy_fn is not None:
            return self.strategy_fn(tick)

        # In live forward demo mode with no external strategy callback,
        # the frozen strategy passively monitors without improvising.
        return None

    def run_iteration(self) -> Dict[str, Any]:
        """Execute a single deterministic iteration of the forward validation loop."""
        if self._stop_requested:
            return {"status": "STOPPED", "reason": "STOP_REQUESTED"}

        now_utc = datetime.now(timezone.utc)

        # 1. Periodic Real-Money Verification
        if self.client and self.client.is_connected:
            acc_meta = self.client.get_account_metadata()
            if acc_meta.trade_mode == 2:
                if self.kill_switch:
                    self.kill_switch.activate("FATAL_SAFETY_VIOLATION: Live account mode detected.")
                raise LiveAccountForbiddenError("LIVE_ACCOUNT_DETECTED: trade_mode == 2.")
            if acc_meta.trade_mode != 0:
                raise LiveAccountForbiddenError(
                    f"NON_DEMO_ACCOUNT: trade_mode == {acc_meta.trade_mode}"
                )

        # 2. Market Data Freshness
        if self.client and self.client.is_connected:
            tick = self.client.get_latest_tick(self.config.symbol)
            staleness = (now_utc - tick.timestamp_utc).total_seconds()
            if staleness > self.config.max_data_staleness_seconds:
                self.metrics.data_staleness_events += 1
                logger.warning(
                    f"Market data stale ({staleness:.1f}s). Skipping new signal generation."
                )
                return {"status": "SKIPPED", "reason": "STALE_MARKET_DATA", "staleness": staleness}
        else:
            tick = MT5TickData(
                symbol=self.config.symbol,
                timestamp_utc=now_utc,
                bid=1.12500,
                ask=1.12510,
                spread=1e-04,
            )

        # 3. Position Reconciliation Check
        rec_report = self.adapter.reconcile()
        if not rec_report.healthy:
            self.metrics.reconciliation_failures += 1
            logger.error(f"Position reconciliation mismatch: {rec_report.discrepancies}")
            if self.metrics.reconciliation_failures >= self.config.max_reconciliation_failures:
                if self.kill_switch:
                    self.kill_switch.activate("RECONCILIATION_FAILURE_THRESHOLD_EXCEEDED")
                return {
                    "status": "ERROR",
                    "reason": "RECONCILIATION_FAILURE",
                    "discrepancies": rec_report.discrepancies,
                }

        # 4. Strategy Signal Evaluation
        signal = self.evaluate_frozen_strategy(tick)
        spread_val = round(tick.ask - tick.bid, 5)

        if signal is None:
            self._update_pnl_and_drawdown()
            return {"status": "IDLE", "reason": "NO_SIGNAL"}

        # Signal Generated - Log Signal Record
        self.metrics.signals_generated += 1
        sig_id = signal.client_request_id or f"p40-{int(time.time() * 1000)}"

        # 5. Idempotency Check
        if sig_id in self._seen_client_request_ids:
            self.metrics.duplicate_prevented += 1
            self.signal_logs.append(
                ForwardSignalRecord(
                    timestamp_utc=now_utc.isoformat(),
                    symbol=signal.symbol,
                    timeframe=signal.timeframe or self.config.timeframe,
                    action=signal.action.value,
                    confidence=signal.confidence,
                    entry_estimate=signal.suggested_entry_price,
                    stop_loss=signal.suggested_stop_loss,
                    take_profit=signal.suggested_take_profit,
                    risk_decision="REJECTED",
                    risk_rejection_reason="DUPLICATE_CLIENT_REQUEST_ID",
                    spread=spread_val,
                    market_state="ACTIVE",
                    execution_decision="DUPLICATE_PREVENTED",
                )
            )
            return {"status": "SKIPPED", "reason": "DUPLICATE_ORDER_PREVENTED"}

        self._seen_client_request_ids.add(sig_id)

        # 6. Operational Safety Limits Check
        if self.metrics.orders_submitted >= self.config.max_demo_trades_per_day:
            self.metrics.signals_rejected += 1
            self.signal_logs.append(
                ForwardSignalRecord(
                    timestamp_utc=now_utc.isoformat(),
                    symbol=signal.symbol,
                    timeframe=signal.timeframe or self.config.timeframe,
                    action=signal.action.value,
                    confidence=signal.confidence,
                    entry_estimate=signal.suggested_entry_price,
                    stop_loss=signal.suggested_stop_loss,
                    take_profit=signal.suggested_take_profit,
                    risk_decision="REJECTED",
                    risk_rejection_reason="MAX_DEMO_TRADES_PER_DAY_EXCEEDED",
                    spread=spread_val,
                    market_state="ACTIVE",
                    execution_decision="OPERATIONAL_LIMIT_REJECTED",
                )
            )
            return {"status": "REJECTED", "reason": "MAX_DEMO_TRADES_PER_DAY_EXCEEDED"}

        if len(self.adapter.get_positions()) >= self.config.max_demo_open_positions:
            self.metrics.signals_rejected += 1
            self.signal_logs.append(
                ForwardSignalRecord(
                    timestamp_utc=now_utc.isoformat(),
                    symbol=signal.symbol,
                    timeframe=signal.timeframe or self.config.timeframe,
                    action=signal.action.value,
                    confidence=signal.confidence,
                    entry_estimate=signal.suggested_entry_price,
                    stop_loss=signal.suggested_stop_loss,
                    take_profit=signal.suggested_take_profit,
                    risk_decision="REJECTED",
                    risk_rejection_reason="MAX_DEMO_OPEN_POSITIONS_REACHED",
                    spread=spread_val,
                    market_state="ACTIVE",
                    execution_decision="OPERATIONAL_LIMIT_REJECTED",
                )
            )
            return {"status": "REJECTED", "reason": "MAX_DEMO_OPEN_POSITIONS_REACHED"}

        if (
            self.metrics.consecutive_execution_errors
            >= self.config.max_consecutive_execution_errors
        ):
            if self.kill_switch:
                self.kill_switch.activate("MAX_CONSECUTIVE_EXECUTION_ERRORS_EXCEEDED")
            self.metrics.signals_rejected += 1
            return {"status": "REJECTED", "reason": "CONSECUTIVE_ERRORS_KILL_SWITCH"}

        # 7. Execute Through Sovereign Pipeline
        target_price = tick.ask if signal.action == SignalAction.BUY else tick.bid
        t_start = time.perf_counter()

        try:
            exec_res = self.execution_service.process_signal(
                signal, current_market_price=target_price
            )
        except Exception as exc:
            self.metrics.execution_errors += 1
            self.metrics.consecutive_execution_errors += 1
            logger.error(f"Execution service exception: {exc}")
            return {"status": "ERROR", "error": str(exc)}

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self._latencies.append(latency_ms)
        self.metrics.average_execution_latency_ms = round(
            sum(self._latencies) / len(self._latencies), 2
        )
        self.metrics.maximum_execution_latency_ms = round(max(self._latencies), 2)

        # 8. Record Signal & Execution Log
        if not exec_res.approved or exec_res.order is None:
            if exec_res.reason and "EXECUTION_EXCEPTION" in exec_res.reason:
                self.metrics.execution_errors += 1
                self.metrics.consecutive_execution_errors += 1
                logger.error(f"Execution error encountered: {exec_res.reason}")
                return {"status": "ERROR", "reason": exec_res.reason}

            self.metrics.signals_rejected += 1
            self.signal_logs.append(
                ForwardSignalRecord(
                    timestamp_utc=now_utc.isoformat(),
                    symbol=signal.symbol,
                    timeframe=signal.timeframe or self.config.timeframe,
                    action=signal.action.value,
                    confidence=signal.confidence,
                    entry_estimate=signal.suggested_entry_price,
                    stop_loss=signal.suggested_stop_loss,
                    take_profit=signal.suggested_take_profit,
                    risk_decision="REJECTED",
                    risk_rejection_reason=exec_res.reason,
                    spread=spread_val,
                    market_state="ACTIVE",
                    execution_decision="RISK_REJECTED",
                )
            )
            return {"status": "REJECTED", "reason": exec_res.reason}

        self.metrics.signals_approved += 1
        self.metrics.orders_submitted += 1
        order = exec_res.order

        self.signal_logs.append(
            ForwardSignalRecord(
                timestamp_utc=now_utc.isoformat(),
                symbol=signal.symbol,
                timeframe=signal.timeframe or self.config.timeframe,
                action=signal.action.value,
                confidence=signal.confidence,
                entry_estimate=signal.suggested_entry_price,
                stop_loss=signal.suggested_stop_loss,
                take_profit=signal.suggested_take_profit,
                risk_decision="APPROVED",
                risk_rejection_reason=None,
                spread=spread_val,
                market_state="ACTIVE",
                execution_decision="ORDER_SUBMITTED",
            )
        )

        if order.status == OrderStatus.FILLED:
            self.metrics.orders_filled += 1
            self.metrics.positions_opened += 1
            self.metrics.consecutive_execution_errors = 0
            self.execution_logs.append(
                ForwardExecutionRecord(
                    internal_order_id=order.order_id,
                    broker_order_ticket=str(getattr(order, "broker_order_id", order.order_id)),
                    deal_ticket=str(getattr(order, "broker_execution_id", order.order_id)),
                    symbol=order.symbol,
                    side=order.side.value,
                    requested_volume=round(order.quantity / 100_000.0, 4),
                    executed_volume=round(order.quantity / 100_000.0, 4),
                    requested_price=order.price,
                    executed_price=order.fill_price,
                    sl=order.stop_loss,
                    tp=order.take_profit,
                    spread=spread_val,
                    broker_retcode=10009,
                    submission_timestamp=now_utc.isoformat(),
                    fill_timestamp=(
                        order.fill_timestamp.isoformat()
                        if order.fill_timestamp
                        else now_utc.isoformat()
                    ),
                    execution_latency_ms=round(latency_ms, 2),
                )
            )
        elif order.status == OrderStatus.REJECTED:
            self.metrics.orders_rejected += 1
            self.metrics.execution_errors += 1
            self.metrics.consecutive_execution_errors += 1

        self._update_pnl_and_drawdown()
        return {
            "status": "EXECUTED",
            "order_id": order.order_id,
            "order_status": order.status.value,
        }

    def _update_pnl_and_drawdown(self) -> None:
        """Update P&L and drawdown metrics from adapter account state."""
        acc = self.adapter.get_account()
        self.metrics.daily_pnl = round(acc.realized_pnl, 4)
        self.metrics.cumulative_demo_pnl = round(acc.realized_pnl, 4)

        if self._peak_equity <= 0.0:
            self._peak_equity = acc.equity

        if acc.equity > self._peak_equity:
            self._peak_equity = acc.equity

        drawdown = (
            (self._peak_equity - acc.equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        )
        if drawdown > self.metrics.max_demo_drawdown:
            self.metrics.max_demo_drawdown = round(drawdown, 4)

    def run_session(
        self,
        duration_seconds: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run an active forward validation session until duration or iteration limit."""
        self._is_running = True
        self._stop_requested = False
        t_session_start = datetime.now(timezone.utc)
        self.metrics.session_start_utc = t_session_start.isoformat()

        # Execute preflight start gate
        preflight = self.preflight_inspection()
        logger.info(f"Preflight inspection passed. Server: {preflight.server}")

        iteration_count = 0
        t0 = time.time()

        try:
            while not self._stop_requested:
                if max_iterations is not None and iteration_count >= max_iterations:
                    break

                if duration_seconds is not None and (time.time() - t0) >= duration_seconds:
                    break

                self.run_iteration()
                iteration_count += 1

                if self.config.poll_interval_seconds > 0:
                    time.sleep(self.config.poll_interval_seconds)
        finally:
            return self.stop()

    def stop(self) -> Dict[str, Any]:
        """Emergency stop mechanism: blocks new entries, reconciles, persists state."""
        self._stop_requested = True
        self._is_running = False
        t_session_end = datetime.now(timezone.utc)
        self.metrics.session_end_utc = t_session_end.isoformat()

        if self.metrics.session_start_utc:
            t_start = datetime.fromisoformat(self.metrics.session_start_utc)
            self.metrics.session_duration_seconds = round(
                (t_session_end - t_start).total_seconds(), 2
            )

        # 1. Close open positions if configured
        if self.config.close_on_stop:
            for sym, pos in list(self.adapter.get_positions().items()):
                try:
                    fresh_tick = (
                        self.client.get_latest_tick(sym)
                        if self.client and self.client.is_connected
                        else None
                    )
                    if fresh_tick:
                        close_price = (
                            fresh_tick.bid if pos.side == SignalAction.BUY else fresh_tick.ask
                        )
                    else:
                        close_price = pos.entry_price
                    self.adapter.close_position(sym, exit_price=close_price)
                    self.metrics.positions_closed += 1
                except Exception as exc:
                    logger.error(f"Error closing position {sym} on stop: {exc}")

        # 2. Final Reconciliation
        final_rec = self.adapter.reconcile()
        self._update_pnl_and_drawdown()

        # 3. Cryptographic Audit Verification
        audit_valid, audit_err = (True, None)
        if self.repository:
            audit_valid, audit_err = self.repository.verify_audit_trail_integrity()

        summary = {
            "session_start_utc": self.metrics.session_start_utc,
            "session_end_utc": self.metrics.session_end_utc,
            "session_duration_seconds": self.metrics.session_duration_seconds,
            "account_mode": "DEMO (0)",
            "real_money_orders": 0,
            "real_capital_exposure": "$0.00",
            "metrics": self.metrics.model_dump(),
            "final_reconciliation": {
                "status": final_rec.status.value,
                "healthy": final_rec.healthy,
                "internal_positions_count": final_rec.internal_open_positions_count,
                "broker_positions_count": final_rec.broker_open_positions_count,
                "discrepancies": final_rec.discrepancies,
            },
            "audit_trail": {
                "chain_valid": audit_valid,
                "integrity_error": audit_err,
            },
            "signals_logged_count": len(self.signal_logs),
            "executions_logged_count": len(self.execution_logs),
        }
        return summary


__all__ = [
    "ForwardDemoRunner",
    "ForwardExecutionRecord",
    "ForwardOperationalMetrics",
    "ForwardPreflightReport",
    "ForwardSignalRecord",
    "ForwardValidationConfig",
]
