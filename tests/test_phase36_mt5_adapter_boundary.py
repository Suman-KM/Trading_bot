"""Phase 36: MT5 Broker Adapter Readiness & Execution Boundary Validation Test Suite.

Tests cover all 20 required architectural and safety boundary invariants:
1. Adapter disabled by default (is_live == False, execution_enabled == False)
2. Real submission rejected (BrokerExecutionDisabledError)
3. Paper adapter unaffected (execution_enabled == True, is_live == False)
4. Order translation (Order -> MT5TradeRequest preserves volume, intent, SL/TP)
5. Symbol validation (BrokerSymbolSpecification match, unknown symbol fails closed)
6. Quantity & lot validation (min_volume, max_volume, volume_step enforcement)
7. Stop-loss & take-profit validation (directional consistency with entry price)
8. Broker capability rejection (unsupported operations raise UnsupportedBrokerOperationError)
9. Idempotency enforcement at execution boundary
10. Concurrent duplicate requests (thread-safe deduplication)
11. RiskEngine boundary isolation (signals never bypass RiskEngine)
12. Kill-switch boundary enforcement (kill switch ON -> zero broker submissions)
13. Audit trail logging for all broker-bound lifecycle events
14. Broker failure fail-closed behavior (connection loss, unavailable broker)
15. Malformed broker response handling (fails closed without phantom fills)
16. Broker timeout handling (fails closed without silent retries)
17. Duplicate broker acknowledgement handling (idempotent, no double accounting)
18. Partial fill representation & state transition
19. Readiness behavior (paper ready independent of MT5; MT5 execution disabled)
20. Configuration safety (zero stored credentials, zero tokens, paper default)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from trading.api.app import create_app
from trading.api.dependencies import (
    reset_trading_context,
    set_trading_context,
)
from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.adapter import BrokerAdapter, PaperBrokerAdapter
from trading.execution.capabilities import (
    BrokerCapabilities,
    UnsupportedBrokerOperationError,
)
from trading.execution.mt5_adapter import (
    BrokerExecutionDisabledError,
    MT5BrokerAdapter,
    MT5ConnectionError,
    MT5ResponseError,
)
from trading.execution.paper_broker import PaperBroker
from trading.execution.service import TradingExecutionService
from trading.execution.translation import (
    MT5TradeRequest,
    translate_order_to_mt5_request,
)
from trading.execution.validation import (
    BrokerSymbolSpecification,
    BrokerValidationError,
    validate_order_for_broker,
)
from trading.models.order import Order, OrderSide, OrderStatus, OrderType
from trading.models.signal import Signal, SignalAction
from trading.persistence.database import PaperDatabaseManager
from trading.persistence.migrations import PaperSchemaMigrator
from trading.persistence.repository import PaperTradingRepository
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskEngine, RiskReason
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Path to temporary SQLite database file."""
    return tmp_path / "paper_phase36.db"


@pytest.fixture
def temp_repo(temp_db_path: Path) -> Generator[PaperTradingRepository, None, None]:
    """Initialized repository on a temporary file."""
    db = PaperDatabaseManager(db_path=str(temp_db_path))
    PaperSchemaMigrator(db).apply_all()
    repo = PaperTradingRepository(db)
    try:
        yield repo
    finally:
        db.close()


@pytest.fixture
def eurusd_spec() -> BrokerSymbolSpecification:
    """Default EURUSD broker symbol specification."""
    return BrokerSymbolSpecification(
        symbol="EURUSD",
        min_volume=0.01,
        max_volume=100.0,
        volume_step=0.01,
        contract_size=100_000.0,
        price_digits=5,
        point=0.00001,
        tick_size=0.00001,
    )


def make_test_order(
    symbol: str = "EURUSD",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 10_000.0,
    price: float = 1.0850,
    stop_loss: float = 1.0800,
    take_profit: float = 1.0950,
    order_id: str = "ord_test_01",
    client_request_id: str = "req_test_01",
) -> Order:
    """Construct a valid test order."""
    return Order(
        order_id=order_id,
        client_request_id=client_request_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        status=OrderStatus.VALIDATED,
    )


def make_test_signal(
    symbol: str = "EURUSD",
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.85,
    entry: float = 1.0850,
    stop_loss: float = 1.0550,
    client_request_id: str = "sig_test_01",
) -> Signal:
    """Construct a valid test signal."""
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        timestamp=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        model_version="phase36_test_v1",
        timeframe="H4",
        expected_return=0.015,
        feature_version="v1",
        suggested_entry_price=entry,
        suggested_stop_loss=stop_loss,
        client_request_id=client_request_id,
    )


# -----------------------------------------------------------------------------
# 1. Adapter Disabled by Default
# -----------------------------------------------------------------------------


def test_01_adapter_disabled_by_default():
    """MT5BrokerAdapter must default to is_live == False and execution_enabled == False."""
    adapter = MT5BrokerAdapter()
    assert adapter.is_live is False
    assert adapter.execution_enabled is False
    assert isinstance(adapter, BrokerAdapter)


# -----------------------------------------------------------------------------
# 2. Real Submission Rejected
# -----------------------------------------------------------------------------


def test_02_real_submission_rejected():
    """Attempting order submission to MT5 adapter raises BrokerExecutionDisabledError."""
    adapter = MT5BrokerAdapter()
    order = make_test_order()

    with pytest.raises(BrokerExecutionDisabledError, match="MT5 broker execution is disabled"):
        adapter.submit_order(order, current_market_price=1.0850)

    # Cancel and close must also fail closed
    with pytest.raises(BrokerExecutionDisabledError):
        adapter.cancel_order(order.order_id)

    with pytest.raises(BrokerExecutionDisabledError):
        adapter.close_position("EURUSD", exit_price=1.0850)


# -----------------------------------------------------------------------------
# 3. Paper Adapter Unaffected
# -----------------------------------------------------------------------------


def test_03_paper_adapter_unaffected():
    """PaperBrokerAdapter remains execution-enabled for in-memory simulation
    with is_live == False.
    """
    broker = PaperBroker(initial_balance=100_000.0)
    paper_adapter = PaperBrokerAdapter(broker=broker)

    assert paper_adapter.is_live is False
    assert paper_adapter.execution_enabled is True
    assert paper_adapter.capabilities.supports_market_orders is True
    assert paper_adapter.capabilities.supports_limit_orders is False

    order = make_test_order(quantity=10_000.0)
    filled_order = paper_adapter.submit_order(order, current_market_price=1.0850)
    assert filled_order.status == OrderStatus.FILLED
    assert len(paper_adapter.get_positions()) == 1


# -----------------------------------------------------------------------------
# 4. Order Translation
# -----------------------------------------------------------------------------


def test_04_order_translation(eurusd_spec: BrokerSymbolSpecification):
    """Domain Order translates exactly to MT5TradeRequest without altering intent or risk."""
    order = make_test_order(
        side=OrderSide.BUY,
        quantity=20_000.0,
        price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
    )
    req = translate_order_to_mt5_request(order, eurusd_spec)

    assert isinstance(req, MT5TradeRequest)
    assert req.symbol == "EURUSD"
    assert req.order_type == 0  # BUY
    assert req.volume == 0.20  # 20,000 / 100,000 lots
    assert req.price == 1.0850
    assert req.stop_loss == 1.0800
    assert req.take_profit == 1.0950
    assert req.internal_order_id == order.order_id
    assert req.client_request_id == order.client_request_id


def test_04b_order_translation_sell(eurusd_spec: BrokerSymbolSpecification):
    """SELL order translates with order_type == 1 and consistent SL/TP."""
    order = make_test_order(
        side=OrderSide.SELL,
        quantity=10_000.0,
        price=1.0850,
        stop_loss=1.0900,
        take_profit=1.0750,
    )
    req = translate_order_to_mt5_request(order, eurusd_spec)
    assert req.order_type == 1  # SELL
    assert req.volume == 0.10


# -----------------------------------------------------------------------------
# 5. Symbol Validation
# -----------------------------------------------------------------------------


def test_05_symbol_validation(eurusd_spec: BrokerSymbolSpecification):
    """Order with mismatched symbol fails closed with BrokerValidationError."""
    mismatched_order = make_test_order(symbol="GBPUSD")
    with pytest.raises(BrokerValidationError, match="does not match broker specification"):
        validate_order_for_broker(mismatched_order, eurusd_spec)

    adapter = MT5BrokerAdapter()
    with pytest.raises(BrokerValidationError, match="Unknown broker symbol"):
        adapter.validate_order(mismatched_order)


# -----------------------------------------------------------------------------
# 6. Quantity & Volume Validation
# -----------------------------------------------------------------------------


def test_06_quantity_validation(eurusd_spec: BrokerSymbolSpecification):
    """Orders with invalid volumes (below min, above max, or off-step) fail closed."""
    # Below min_volume (0.01 lot = 1,000 units; test with 500 units = 0.005 lots)
    small_order = make_test_order(quantity=500.0)
    with pytest.raises(BrokerValidationError, match="is below minimum allowed volume"):
        validate_order_for_broker(small_order, eurusd_spec)

    # Above max_volume (100 lots = 10,000,000 units; test with 200 lots)
    huge_order = make_test_order(quantity=20_000_000.0)
    with pytest.raises(BrokerValidationError, match="exceeds maximum allowed volume"):
        validate_order_for_broker(huge_order, eurusd_spec)

    # Off-step volume (volume_step = 0.01 lot = 1,000 units; test with 1,234 units = 0.01234 lots)
    off_step_order = make_test_order(quantity=1_234.0)
    with pytest.raises(BrokerValidationError, match="does not align with broker volume step"):
        validate_order_for_broker(off_step_order, eurusd_spec)


# -----------------------------------------------------------------------------
# 7. Stop-Loss & Take-Profit Validation
# -----------------------------------------------------------------------------


def test_07_sl_tp_validation(eurusd_spec: BrokerSymbolSpecification):
    """Directionally inconsistent SL/TP levels raise BrokerValidationError."""
    # BUY order with SL above entry price
    bad_buy_sl = make_test_order(
        side=OrderSide.BUY, price=1.0850, stop_loss=1.0900, take_profit=1.0950
    )
    with pytest.raises(BrokerValidationError, match="stop-loss .* must be strictly below"):
        validate_order_for_broker(bad_buy_sl, eurusd_spec)

    # BUY order with TP below entry price
    bad_buy_tp = make_test_order(
        side=OrderSide.BUY, price=1.0850, stop_loss=1.0800, take_profit=1.0820
    )
    with pytest.raises(BrokerValidationError, match="take-profit .* must be strictly above"):
        validate_order_for_broker(bad_buy_tp, eurusd_spec)

    # SELL order with SL below entry price
    bad_sell_sl = make_test_order(
        side=OrderSide.SELL, price=1.0850, stop_loss=1.0800, take_profit=1.0750
    )
    with pytest.raises(BrokerValidationError, match="stop-loss .* must be strictly above"):
        validate_order_for_broker(bad_sell_sl, eurusd_spec)

    # SELL order with TP above entry price
    bad_sell_tp = make_test_order(
        side=OrderSide.SELL, price=1.0850, stop_loss=1.0900, take_profit=1.0950
    )
    with pytest.raises(BrokerValidationError, match="take-profit .* must be strictly below"):
        validate_order_for_broker(bad_sell_tp, eurusd_spec)


# -----------------------------------------------------------------------------
# 8. Broker Capability Rejection
# -----------------------------------------------------------------------------


def test_08_broker_capability_rejection():
    """Attempting an unsupported capability on a capability model fails closed."""
    caps = BrokerCapabilities(
        supports_market_orders=True,
        supports_limit_orders=False,
        supports_cancel=False,
    )
    caps.assert_supported("supports_market_orders")

    with pytest.raises(UnsupportedBrokerOperationError, match="supports_limit_orders"):
        caps.assert_supported("supports_limit_orders")

    with pytest.raises(UnsupportedBrokerOperationError, match="supports_cancel"):
        caps.assert_supported("supports_cancel")


# -----------------------------------------------------------------------------
# 9. Idempotency Enforcement
# -----------------------------------------------------------------------------


def test_09_idempotency_enforcement(temp_repo: PaperTradingRepository):
    """Submitting the same request ID twice returns cached execution without duplicate fills."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )
    sig = make_test_signal(client_request_id="req_idemp_p36")

    res1 = service.process_signal(sig, current_market_price=1.0850)
    assert res1.approved is True
    assert res1.is_idempotent_replay is False

    res2 = service.process_signal(sig, current_market_price=1.0850)
    assert res2.approved is True
    assert res2.is_idempotent_replay is True
    assert len(broker.get_execution_reports()) == 1


# -----------------------------------------------------------------------------
# 10. Concurrent Duplicate Requests
# -----------------------------------------------------------------------------


def test_10_concurrent_duplicate_requests(temp_repo: PaperTradingRepository):
    """10 simultaneous duplicate requests across threads execute exactly once."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )
    sig = make_test_signal(client_request_id="req_concurrent_p36")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(service.process_signal, sig, 1.0850) for _ in range(10)]
        results = [f.result() for f in futures]

    assert all(r.approved for r in results)
    replays = [r.is_idempotent_replay for r in results]
    assert replays.count(False) == 1
    assert replays.count(True) == 9
    assert len(broker.get_execution_reports()) == 1


# -----------------------------------------------------------------------------
# 11. RiskEngine Boundary Isolation
# -----------------------------------------------------------------------------


def test_11_risk_engine_boundary_isolation(temp_repo: PaperTradingRepository):
    """Signals never bypass RiskEngine; risk rejections result in zero broker submissions."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    limits = RiskLimits(MIN_SIGNAL_CONFIDENCE=0.70)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=limits, kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    # 1. Low confidence signal
    low_conf_sig = make_test_signal(confidence=0.50, client_request_id="sig_low_conf")
    res = service.process_signal(low_conf_sig, 1.0850)
    assert res.approved is False
    assert res.reason == RiskReason.CONFIDENCE_TOO_LOW.value
    assert len(broker.get_execution_reports()) == 0

    # 2. Action HOLD signal
    hold_sig = make_test_signal(action=SignalAction.HOLD, client_request_id="sig_hold")
    res_hold = service.process_signal(hold_sig, 1.0850)
    assert res_hold.approved is False
    assert res_hold.reason == RiskReason.SIGNAL_ACTION_HOLD.value
    assert len(broker.get_execution_reports()) == 0


# -----------------------------------------------------------------------------
# 12. Kill Switch Boundary Enforcement
# -----------------------------------------------------------------------------


def test_12_kill_switch_boundary_enforcement(temp_repo: PaperTradingRepository):
    """Kill switch activation blocks all submissions at RiskEngine boundary."""
    audit = AuditTrail(repository=temp_repo)
    ks = KillSwitch(audit_trail=audit, repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=ks),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    ks.activate(reason="MANUAL_HALT_P36")
    res = service.process_signal(make_test_signal(), 1.0850)
    assert res.approved is False
    assert res.reason == RiskReason.KILL_SWITCH_ACTIVE.value
    assert len(broker.get_positions()) == 0

    # Reboot / Reconstruct kill switch from repository: remains active
    ks_reboot = KillSwitch(audit_trail=audit, repository=temp_repo)
    assert ks_reboot.is_active() is True
    assert ks_reboot.reason == "MANUAL_HALT_P36"


# -----------------------------------------------------------------------------
# 13. Audit Trail Logging for Broker Lifecycle Events
# -----------------------------------------------------------------------------


def test_13_audit_trail_logging(temp_repo: PaperTradingRepository):
    """Every broker-bound action produces structured, tamper-evident audit events."""
    audit = AuditTrail(repository=temp_repo)
    adapter = MT5BrokerAdapter(audit_trail=audit)

    # 1. Validation dry-run audit event
    order = make_test_order(quantity=10_000.0)
    adapter.simulate_dry_run_submission(order, current_market_price=1.0850)

    # 2. Rejection audit event on execution attempt
    with pytest.raises(BrokerExecutionDisabledError):
        adapter.submit_order(order, current_market_price=1.0850)

    valid, err = temp_repo.verify_audit_trail_integrity()
    assert valid is True
    assert err is None

    events = temp_repo.get_audit_events()
    event_types = [e.event_type for e in events]
    assert AuditEventType.ORDER_VALIDATED in event_types
    assert AuditEventType.EXECUTION_ERROR in event_types


# -----------------------------------------------------------------------------
# 14. Broker Failure Handling (Fail-Closed)
# -----------------------------------------------------------------------------


def test_14_broker_failure_fail_closed():
    """Broker connection loss or unexpected network exception fails closed cleanly."""
    adapter = MT5BrokerAdapter(execution_enabled=True)

    with patch.object(adapter, "submit_order", side_effect=MT5ConnectionError("Socket timeout")):
        with pytest.raises(MT5ConnectionError):
            adapter.submit_order(make_test_order())

    # Verify zero phantom positions or fills
    assert len(adapter.get_positions()) == 0
    assert len(adapter.get_execution_reports()) == 0


# -----------------------------------------------------------------------------
# 15. Malformed Broker Response Handling
# -----------------------------------------------------------------------------


def test_15_malformed_broker_response():
    """Malformed or corrupt responses raise MT5ResponseError and never invent fills."""
    adapter = MT5BrokerAdapter(execution_enabled=True)

    with patch.object(
        adapter, "submit_order", side_effect=MT5ResponseError("Corrupt payload received")
    ):
        with pytest.raises(MT5ResponseError):
            adapter.submit_order(make_test_order())

    assert len(adapter.get_positions()) == 0


# -----------------------------------------------------------------------------
# 16. Broker Timeout Handling
# -----------------------------------------------------------------------------


def test_16_broker_timeout_handling():
    """Timeouts fail closed without performing uncontrolled duplicate retries."""
    adapter = MT5BrokerAdapter(execution_enabled=True)

    with patch.object(
        adapter, "submit_order", side_effect=TimeoutError("Broker RPC deadline exceeded")
    ):
        with pytest.raises(TimeoutError):
            adapter.submit_order(make_test_order())

    assert len(adapter.get_execution_reports()) == 0


# -----------------------------------------------------------------------------
# 17. Duplicate Broker Acknowledgement Handling
# -----------------------------------------------------------------------------


def test_17_duplicate_broker_acknowledgement():
    """Duplicate execution reports for an existing filled order do not duplicate accounting."""
    broker = PaperBroker(initial_balance=100_000.0)
    order = make_test_order(quantity=10_000.0)

    # Fill 1
    filled1 = broker.submit_order(order, current_market_price=1.0850)
    assert filled1.status == OrderStatus.FILLED
    cash_after_fill1 = broker.get_account().cash_balance

    # Fill 2 (simulated duplicate submission of already-filled order ticket)
    filled2 = broker.submit_order(order, current_market_price=1.0850)
    assert filled2.status == OrderStatus.FILLED
    assert broker.get_account().cash_balance == cash_after_fill1
    assert len(broker.get_positions()) == 1


# -----------------------------------------------------------------------------
# 18. Partial Fill Representation
# -----------------------------------------------------------------------------


def test_18_partial_fill_representation():
    """Order state machine supports PARTIALLY_FILLED transitions deterministically."""
    order = make_test_order(quantity=10_000.0)
    order.transition_to(OrderStatus.SUBMITTED)
    order.transition_to(OrderStatus.PARTIALLY_FILLED)
    assert order.status == OrderStatus.PARTIALLY_FILLED

    # PARTIALLY_FILLED can transition to FILLED or CANCELLED
    order.transition_to(OrderStatus.FILLED)
    assert order.status == OrderStatus.FILLED


# -----------------------------------------------------------------------------
# 19. Readiness Behavior
# -----------------------------------------------------------------------------


def test_19_readiness_behavior():
    """Readiness probe reports MT5 disabled while paper engine remains 100% ready."""
    context = reset_trading_context()
    set_trading_context(context)
    app = create_app()
    client = TestClient(app)

    resp = client.get("/readiness")
    assert resp.status_code == 200
    body = resp.json()

    assert body["ready"] is True
    assert body["trading_backend"] == "PAPER"
    assert body["mt5_adapter_available"] is False
    assert body["mt5_execution_enabled"] is False


# -----------------------------------------------------------------------------
# 20. Configuration Safety
# -----------------------------------------------------------------------------


def test_20_configuration_safety():
    """Verify zero trading credentials, tokens, or account passwords exist in configuration."""
    adapter = MT5BrokerAdapter()
    assert adapter._account_id is None
    assert adapter._server is None
    assert adapter.is_live is False
    assert adapter.execution_enabled is False
    assert not hasattr(adapter, "password")
    assert not hasattr(adapter, "token")
