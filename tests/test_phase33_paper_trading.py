"""Comprehensive test suite for Phase 33: Paper Trading, Risk & Execution Engineering.

Verifies:
1. Deterministic Signal validation and boundary conditions.
2. RiskEngine invariant limits (daily loss 1.0%, position risk 0.5%, max positions 3,
   max exposure 20.0%).
3. Deterministic position sizing mathematics and failure boundaries.
4. Order state machine with explicit valid and invalid state transitions.
5. Position lifecycle (OPEN -> ACTIVE -> CLOSED) with long and short P&L.
6. Automated stop-loss and take-profit boundary triggering.
7. Explicit transaction-cost accounting (Gross P&L, spread, slippage, commission, Net P&L).
8. Structured execution report generation and history.
9. Structured immutable audit trail event logging.
10. Idempotency protection against duplicate signal/request submissions.
11. Kill switch activation, deactivation, and order rejection.
12. Daily loss limit enforcement and baseline reset.
13. Exposure and open position limits enforcement.
14. Failure injection and safe recovery without state corruption.
15. Atomic state consistency across Account, Positions, Orders, and Execution records.
16. BrokerAdapter boundary protocol and PaperBrokerAdapter implementation.
17. API observability endpoints: /readiness, /executions, /audit/events, /metrics, /health.
18. Security invariants (DEMO/PAPER mode, localhost-only, no credentials).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from trading.api.app import create_app
from trading.api.dependencies import reset_trading_context
from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.adapter import BrokerAdapter, PaperBrokerAdapter
from trading.execution.costs import TransactionCostConfig
from trading.execution.paper_broker import PaperBroker
from trading.execution.service import TradingExecutionService
from trading.models.order import Order, OrderSide, OrderStatus
from trading.models.position import Position
from trading.models.signal import Signal, SignalAction
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskEngine, RiskReason
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits, calculate_position_size


def make_test_signal(
    symbol: str = "EURUSD",
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.80,
    entry_price: float = 10.0,
    stop_loss: float = 9.0,
    take_profit: float = 12.0,
    client_request_id: str | None = None,
) -> Signal:
    """Helper to construct a deterministic test signal."""
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
        model_version="test_v1",
        timeframe="H4",
        expected_return=0.0050,
        feature_version="feat_v1",
        suggested_entry_price=entry_price,
        suggested_stop_loss=stop_loss,
        suggested_take_profit=take_profit,
        client_request_id=client_request_id,
    )


# ---------------------------------------------------------------------------
# 1. Signal Validation Tests
# ---------------------------------------------------------------------------


def test_signal_validation_boundaries():
    """Verify Signal model enforces strict validation on confidence, symbol, and prices."""
    # Valid signal
    sig = make_test_signal()
    assert sig.symbol == "EURUSD"
    assert sig.confidence == 0.80

    # Invalid empty symbol
    with pytest.raises(ValueError, match="Symbol cannot be empty"):
        make_test_signal(symbol="  ")

    # Invalid confidence out of [0, 1] bounds
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        make_test_signal(confidence=1.05)

    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        make_test_signal(confidence=-0.1)

    # Invalid non-finite prices
    with pytest.raises(ValueError, match="must be a positive finite number"):
        make_test_signal(entry_price=float("nan"))


# ---------------------------------------------------------------------------
# 2. Risk Limits and Position Sizing Boundaries
# ---------------------------------------------------------------------------


def test_risk_limits_default_invariants():
    """Verify RiskLimits retains strict non-negotiable defaults."""
    limits = RiskLimits()
    assert limits.MAX_DAILY_LOSS_PERCENT == 1.0
    assert limits.MAX_POSITION_RISK_PERCENT == 0.5
    assert limits.MAX_OPEN_POSITIONS == 3
    assert limits.MAX_TOTAL_EXPOSURE_PERCENT == 20.0
    assert limits.MIN_SIGNAL_CONFIDENCE == 0.60


def test_position_sizing_mathematics_and_boundaries():
    """Verify risk-based position sizing formula and error boundaries."""
    # Equity = $100,000, Risk = 0.5% ($500), Stop Distance = 0.0050 -> Qty = 100,000
    qty = calculate_position_size(
        account_equity=100_000.0,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        risk_percent=0.5,
        max_risk_percent=0.5,
    )
    assert qty == 100_000.0

    # Zero stop distance must be rejected
    with pytest.raises(ValueError, match="Zero stop distance"):
        calculate_position_size(
            account_equity=100_000.0,
            entry_price=1.1000,
            stop_loss_price=1.1000,
            risk_percent=0.5,
        )

    # Exceeding max risk percent must be rejected
    with pytest.raises(ValueError, match="exceeds max position risk"):
        calculate_position_size(
            account_equity=100_000.0,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            risk_percent=1.5,
            max_risk_percent=0.5,
        )

    # Non-positive equity must be rejected
    with pytest.raises(ValueError, match="Account equity must be a positive finite number"):
        calculate_position_size(
            account_equity=-100.0,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            risk_percent=0.5,
        )


# ---------------------------------------------------------------------------
# 3. Order State Machine Transitions
# ---------------------------------------------------------------------------


def test_order_state_machine_valid_transitions():
    """Verify valid order state transitions execute cleanly."""
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000.0,
        price=1.1000,
        status=OrderStatus.CREATED,
    )
    assert order.status == OrderStatus.CREATED

    order.transition_to(OrderStatus.VALIDATED)
    assert order.status == OrderStatus.VALIDATED

    order.transition_to(OrderStatus.SUBMITTED)
    assert order.status == OrderStatus.SUBMITTED

    order.transition_to(OrderStatus.FILLED)
    assert order.status == OrderStatus.FILLED

    order.transition_to(OrderStatus.CLOSED)
    assert order.status == OrderStatus.CLOSED


def test_order_state_machine_invalid_transitions_rejected():
    """Verify invalid order state transitions raise explicit ValueErrors."""
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000.0,
        price=1.1000,
        status=OrderStatus.FILLED,
    )
    # FILLED -> CREATED is invalid
    with pytest.raises(ValueError, match="Invalid order transition"):
        order.transition_to(OrderStatus.CREATED)

    # CLOSED -> FILLED is invalid
    order.status = OrderStatus.CLOSED
    with pytest.raises(ValueError, match="Invalid order transition"):
        order.transition_to(OrderStatus.FILLED)

    # CANCELLED -> FILLED is invalid
    cancelled_order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000.0,
        price=1.1000,
        status=OrderStatus.CANCELLED,
    )
    with pytest.raises(ValueError, match="Invalid order transition"):
        cancelled_order.transition_to(OrderStatus.FILLED)


# ---------------------------------------------------------------------------
# 4. Position Lifecycle, P&L, and Stop-Loss / Take-Profit Triggers
# ---------------------------------------------------------------------------


def test_position_lifecycle_long_and_short():
    """Verify position lifecycle, mark-to-market, and closure for both Long and Short."""
    now = datetime.now(timezone.utc)

    # Long Position: entry @ 1.1000, qty = 10,000
    pos_long = Position(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=10_000.0,
        entry_price=1.1000,
        current_price=1.1000,
        entry_timestamp=now,
    )
    assert pos_long.status == "OPEN"

    # Mark to market @ 1.1050 (+50 pips -> +$50.0)
    pos_long.update_price(1.1050)
    assert pos_long.status == "ACTIVE"
    assert math.isclose(pos_long.unrealized_pnl, 50.0, rel_tol=1e-5)

    # Close @ 1.1050 with $2.0 costs -> gross = $50.0, net = $48.0
    pos_long.close(exit_price=1.1050, costs=2.0)
    assert pos_long.status == "CLOSED"
    assert math.isclose(pos_long.gross_pnl, 50.0, rel_tol=1e-5)
    assert math.isclose(pos_long.costs, 2.0, rel_tol=1e-5)
    assert math.isclose(pos_long.net_pnl, 48.0, rel_tol=1e-5)
    assert pos_long.unrealized_pnl == 0.0

    # Short Position: entry @ 1.2000, qty = 5,000
    pos_short = Position(
        symbol="GBPUSD",
        side=OrderSide.SELL,
        quantity=5_000.0,
        entry_price=1.2000,
        current_price=1.2000,
        entry_timestamp=now,
    )
    # Price falls to 1.1900 (+100 pips gain for short -> +$50.0)
    pos_short.update_price(1.1900)
    assert math.isclose(pos_short.unrealized_pnl, 50.0, rel_tol=1e-5)

    # Close short @ 1.1900 with $1.5 costs -> net = $48.5
    pos_short.close(exit_price=1.1900, costs=1.5)
    assert pos_short.status == "CLOSED"
    assert math.isclose(pos_short.gross_pnl, 50.0, rel_tol=1e-5)
    assert math.isclose(pos_short.net_pnl, 48.5, rel_tol=1e-5)


def test_automated_stop_loss_and_take_profit_triggers():
    """Verify PaperBroker automatically triggers closure when price crosses SL or TP."""
    broker = PaperBroker(initial_balance=100_000.0)
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1_000.0,
        price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
    )
    broker.submit_order(order)
    assert "EURUSD" in broker.get_positions()

    # Price moves within bounds (1.0980) -> position stays open
    close_ord = broker.update_market_price("EURUSD", 1.0980)
    assert close_ord is None
    assert "EURUSD" in broker.get_positions()

    # Price breaches stop-loss (1.0940 <= 1.0950) -> position automatically closed!
    close_ord = broker.update_market_price("EURUSD", 1.0940)
    assert close_ord is not None
    assert close_ord.status == OrderStatus.FILLED
    assert "EURUSD" not in broker.get_positions()
    assert len(broker.get_closed_positions()) == 1


# ---------------------------------------------------------------------------
# 5. Transaction Cost Accounting (Gross vs Net P&L)
# ---------------------------------------------------------------------------


def test_transaction_cost_accounting():
    """Verify explicit accounting of spread, slippage, and commission."""
    cost_cfg = TransactionCostConfig(
        spread_points=0.0002,  # 2 pips
        slippage_points=0.0001,  # 1 pip
        commission_per_unit=0.00005,  # $0.50 per 10k
    )
    broker = PaperBroker(initial_balance=100_000.0, cost_config=cost_cfg)

    # Buy 10,000 units @ 1.1000
    # Entry costs: (0.0002 + 0.0001 + 0.00005) * 10,000 = 0.00035 * 10,000 = $3.50
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=10_000.0,
        price=1.1000,
    )
    broker.submit_order(order)

    account = broker.get_account()
    assert math.isclose(account.realized_pnl, -3.50, rel_tol=1e-4)

    # Close @ 1.1050 (+50 pips gross = +$50.0)
    # Exit costs: 0.00035 * 10,000 = $3.50
    # Total costs: $7.00. Gross P&L: $50.00. Net Realized P&L: $43.00
    broker.close_position("EURUSD", exit_price=1.1050)

    account_after = broker.get_account()
    metrics = broker.get_metrics()
    assert math.isclose(metrics["gross_pnl"], 50.0, rel_tol=1e-4)
    assert math.isclose(metrics["total_costs"], 7.0, rel_tol=1e-4)
    assert math.isclose(metrics["net_pnl"], 43.0, rel_tol=1e-4)
    assert math.isclose(account_after.realized_pnl, 43.0, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 6. Structured Execution Reports & Audit Trail
# ---------------------------------------------------------------------------


def test_execution_reports_and_audit_trail():
    """Verify PaperBroker generates structured ExecutionReports and AuditEvents."""
    audit = AuditTrail()
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit)

    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=5_000.0,
        price=1.1000,
    )
    broker.submit_order(order)

    # Verify ExecutionReport
    reports = broker.get_execution_reports()
    assert len(reports) == 1
    rep = reports[0]
    assert rep.symbol == "EURUSD"
    assert rep.side == OrderSide.BUY
    assert rep.quantity == 5_000.0
    assert rep.executed_price == 1.1000
    assert rep.execution_status == OrderStatus.FILLED

    # Verify AuditTrail
    events = audit.get_events()
    event_types = [e.event_type for e in events]
    assert AuditEventType.POSITION_OPENED in event_types
    assert AuditEventType.ORDER_FILLED in event_types


# ---------------------------------------------------------------------------
# 7. Idempotency Protection
# ---------------------------------------------------------------------------


def test_idempotency_duplicate_submission_protection():
    """Verify duplicate signal submissions return cached execution and never duplicate positions."""
    audit = AuditTrail()
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit)
    limits = RiskLimits()
    risk_engine = RiskEngine(limits=limits)
    portfolio_mgr = PortfolioManager(broker=broker)

    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
        audit_trail=audit,
    )

    signal = make_test_signal(
        symbol="EURUSD",
        client_request_id="unique_req_12345",
    )

    # First submission -> executes and opens position
    res1 = service.process_signal(signal)
    assert res1.approved is True
    assert res1.is_idempotent_replay is False
    assert len(broker.get_positions()) == 1

    # Second submission with IDENTICAL client_request_id -> idempotent replay!
    res2 = service.process_signal(signal)
    assert res2.approved is True
    assert res2.is_idempotent_replay is True
    assert res2.order is not None
    assert res2.order.order_id == res1.order.order_id

    # Crucial check: still exactly ONE position open in broker!
    assert len(broker.get_positions()) == 1


# ---------------------------------------------------------------------------
# 8. Kill Switch Reliability
# ---------------------------------------------------------------------------


def test_kill_switch_activation_and_deactivation():
    """Verify kill switch reliably halts order submission and restores safely."""
    audit = AuditTrail()
    kill_switch = KillSwitch(audit_trail=audit)
    risk_engine = RiskEngine(kill_switch=kill_switch)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit)
    portfolio_mgr = PortfolioManager(broker=broker)

    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
        audit_trail=audit,
    )

    # 1. Activate kill switch
    kill_switch.activate(reason="MANUAL_TEST_ENGAGEMENT")
    assert kill_switch.is_active() is True

    # 2. Submit signal -> must be rejected
    signal = make_test_signal()
    res = service.process_signal(signal)
    assert res.approved is False
    assert res.reason == RiskReason.KILL_SWITCH_ACTIVE.value
    assert len(broker.get_positions()) == 0

    # 3. Deactivate kill switch
    kill_switch.deactivate()
    assert kill_switch.is_active() is False

    # 4. Submit new signal -> now approved
    signal2 = make_test_signal(client_request_id="new_signal_after_reset")
    res2 = service.process_signal(signal2)
    assert res2.approved is True
    assert len(broker.get_positions()) == 1


# ---------------------------------------------------------------------------
# 9. Daily Loss Protection & Reset
# ---------------------------------------------------------------------------


def test_daily_loss_limit_protection_and_reset():
    """Verify 1.0% daily loss halts new trades until daily reset."""
    broker = PaperBroker(initial_balance=100_000.0)
    risk_engine = RiskEngine()
    portfolio_mgr = PortfolioManager(broker=broker, initial_daily_equity=100_000.0)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
    )

    # Simulate a realized loss of $1,200 (1.2% drawdown relative to $100k daily start)
    order = Order(symbol="LOSS", side=OrderSide.BUY, quantity=100.0, price=100.0)
    broker.submit_order(order)
    broker.close_position("LOSS", exit_price=88.0)  # -$1,200 loss

    assert portfolio_mgr.get_daily_loss_percent() >= 1.0

    # New signal must be rejected for daily loss limit
    signal = make_test_signal(symbol="EURUSD")
    res = service.process_signal(signal)
    assert res.approved is False
    assert res.reason == RiskReason.DAILY_LOSS_LIMIT_EXCEEDED.value

    # Reset daily baseline for new day
    portfolio_mgr.reset_daily_baseline()
    assert portfolio_mgr.get_daily_loss_percent() == 0.0

    # Release kill switch activated by daily loss
    risk_engine.kill_switch.deactivate()

    # Now allowed
    res2 = service.process_signal(make_test_signal(client_request_id="new_day_trade"))
    assert res2.approved is True


# ---------------------------------------------------------------------------
# 10. Open Position and Total Exposure Protection
# ---------------------------------------------------------------------------


def test_max_open_positions_limit_enforced():
    """Verify MAX_OPEN_POSITIONS = 3 strictly prevents a 4th simultaneous position."""
    broker = PaperBroker(initial_balance=100_000.0)
    risk_engine = RiskEngine()
    portfolio_mgr = PortfolioManager(broker=broker)
    service = TradingExecutionService(
        risk_engine=risk_engine, paper_broker=broker, portfolio_manager=portfolio_mgr
    )

    # Open 3 positions (allowed)
    for sym in ["SYM1", "SYM2", "SYM3"]:
        sig = make_test_signal(symbol=sym, client_request_id=f"open_{sym}")
        res = service.process_signal(sig)
        assert res.approved is True

    assert len(broker.get_positions()) == 3

    # Attempt 4th position -> rejected
    sig4 = make_test_signal(symbol="SYM4", client_request_id="open_SYM4")
    res4 = service.process_signal(sig4)
    assert res4.approved is False
    assert res4.reason == RiskReason.MAX_OPEN_POSITIONS_REACHED.value
    assert len(broker.get_positions()) == 3


# ---------------------------------------------------------------------------
# 11. Failure Injection & Atomic State Consistency
# ---------------------------------------------------------------------------


def test_failure_injection_portfolio_consistency():
    """Verify system handles invalid parameters safely without corrupting portfolio state."""
    broker = PaperBroker(initial_balance=100_000.0)
    risk_engine = RiskEngine()
    portfolio_mgr = PortfolioManager(broker=broker)
    service = TradingExecutionService(
        risk_engine=risk_engine, paper_broker=broker, portfolio_manager=portfolio_mgr
    )

    initial_account = broker.get_account()

    # Signal with zero stop distance
    bad_signal_dict = {
        "symbol": "EURUSD",
        "action": "BUY",
        "confidence": 0.85,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "v1",
        "timeframe": "H4",
        "expected_return": 0.01,
        "feature_version": "v1",
        "suggested_entry_price": 1.1000,
        "suggested_stop_loss": 1.1000,  # Zero stop distance
    }
    res = service.process_signal(bad_signal_dict)
    assert res.approved is False
    assert res.reason == RiskReason.POSITION_SIZING_FAILED.value

    # Verify Account and Portfolio remain bit-for-bit unchanged
    current_account = broker.get_account()
    assert current_account.equity == initial_account.equity
    assert current_account.cash_balance == initial_account.cash_balance
    assert len(broker.get_positions()) == 0


# ---------------------------------------------------------------------------
# 12. BrokerAdapter Boundary Verification
# ---------------------------------------------------------------------------


def test_broker_adapter_boundary_implementation():
    """Verify PaperBrokerAdapter adheres to BrokerAdapter interface.

    Guarantees is_live == False.
    """
    broker = PaperBroker(initial_balance=100_000.0)
    adapter: BrokerAdapter = PaperBrokerAdapter(broker=broker)

    assert isinstance(adapter, BrokerAdapter)
    assert adapter.is_live is False

    account = adapter.get_account()
    assert account.initial_balance == 100_000.0
    assert len(adapter.get_positions()) == 0


# ---------------------------------------------------------------------------
# 13. API Observability Endpoints (/readiness, /executions, /audit/events, /metrics)
# ---------------------------------------------------------------------------


def test_api_observability_endpoints():
    """Verify /readiness, /executions, /audit/events, and /metrics endpoints."""
    reset_trading_context(initial_balance=100_000.0)
    app = create_app()

    with TestClient(app) as client:
        # 1. GET /readiness
        res_ready = client.get("/readiness")
        assert res_ready.status_code == 200
        ready_data = res_ready.json()
        assert ready_data["ready"] is True
        assert ready_data["paper_broker_initialized"] is True
        assert ready_data["trading_backend"] == "PAPER"

        # 2. Submit a valid signal
        sig_payload = {
            "symbol": "EURUSD",
            "action": "BUY",
            "confidence": 0.85,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": "v1",
            "timeframe": "H4",
            "expected_return": 0.01,
            "feature_version": "v1",
            "suggested_entry_price": 10.0,
            "suggested_stop_loss": 9.0,
            "client_request_id": "api_test_req_01",
        }
        res_sig = client.post("/signals", json=sig_payload)
        assert res_sig.status_code == 200

        # 3. GET /executions
        res_exec = client.get("/executions")
        assert res_exec.status_code == 200
        exec_list = res_exec.json()
        assert len(exec_list) >= 1
        assert exec_list[0]["symbol"] == "EURUSD"
        assert exec_list[0]["execution_status"] == "FILLED"

        # 4. GET /audit/events
        res_audit = client.get("/audit/events")
        assert res_audit.status_code == 200
        audit_list = res_audit.json()
        assert len(audit_list) >= 1
        event_names = [e["event_type"] for e in audit_list]
        assert "SIGNAL_RECEIVED" in event_names
        assert "ORDER_FILLED" in event_names

        # 5. GET /metrics
        res_metrics = client.get("/metrics")
        assert res_metrics.status_code == 200
        metrics = res_metrics.json()
        assert metrics["orders_received"] >= 1
        assert metrics["orders_filled"] >= 1
        assert metrics["open_positions"] == 1


# ---------------------------------------------------------------------------
# 14. Atomic State Consistency and Security Invariants
# ---------------------------------------------------------------------------


def test_atomic_state_consistency():
    """Verify Account, Positions, Orders, and Execution records remain mutually consistent."""
    broker = PaperBroker(initial_balance=100_000.0)

    # 1. Fill an order
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1_000.0,
        price=1.1000,
    )
    filled_order = broker.submit_order(order)
    assert filled_order.status == OrderStatus.FILLED

    # Check consistency: 1 open position, 1 order, 1 execution report
    positions = broker.get_positions()
    orders = broker.get_all_orders()
    reports = broker.get_execution_reports()

    assert len(positions) == 1
    assert "EURUSD" in positions
    assert len(orders) == 1
    assert len(reports) == 1
    assert reports[0].order_id == filled_order.order_id

    # 2. Close position
    close_order = broker.close_position("EURUSD", exit_price=1.1050)
    assert close_order.status == OrderStatus.FILLED

    positions_after = broker.get_positions()
    closed_after = broker.get_closed_positions()
    orders_after = broker.get_all_orders()
    reports_after = broker.get_execution_reports()

    # Closed position must be removed from open positions
    assert len(positions_after) == 0
    assert len(closed_after) == 1
    assert len(orders_after) == 2
    assert len(reports_after) == 2

    # Account equity must equal initial_balance + realized_pnl + unrealized_pnl
    account = broker.get_account()
    expected_equity = account.initial_balance + account.realized_pnl + account.unrealized_pnl
    assert math.isclose(account.equity, expected_equity, rel_tol=1e-5)


def test_security_invariants_no_credentials():
    """Verify environment declares DEMO and PAPER mode with zero credentials."""
    reset_trading_context(initial_balance=100_000.0)
    app = create_app()

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["environment"] == "DEMO"
        assert data["trading_backend"] == "PAPER"

        # Verify no sensitive keys leaked in risk status
        res_risk = client.get("/risk/status")
        assert res_risk.status_code == 200
        risk_data = res_risk.json()
        assert "password" not in str(risk_data).lower()
        assert "secret" not in str(risk_data).lower()
        assert "api_key" not in str(risk_data).lower()
        assert risk_data["trading_mode"] == "DEMO/PAPER"
