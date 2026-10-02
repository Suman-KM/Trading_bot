"""Phase 39 test suite: Controlled MT5 Demo Execution and Live Broker Pipeline Validation.

Verifies:
1. Multi-condition demo authorization and fatal rejection of live accounts.
2. Positive verification of MetaQuotes-Demo account metadata and EURUSD symbol spec.
3. End-to-end signal -> risk -> sizing -> validation -> MT5 demo execution -> persistence.
4. Fail-closed error handling (timeout, check rejection, malformed response).
5. State reconciliation (healthy match, phantom detection, strict fail-closed).
6. Atomicity, snapshot rollback on persistence error, and restart recovery.
7. Enhanced API readiness telemetry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from trading.adapters.mt5.reconciliation import (
    PositionReconciliationError,
    ReconciliationStatus,
    reconcile_positions,
)
from trading.adapters.mt5.safety import (
    ACCOUNT_TRADE_MODE_CONTEST,
    ACCOUNT_TRADE_MODE_DEMO,
    ACCOUNT_TRADE_MODE_REAL,
    DemoAccountVerificationError,
    DemoAccountVerifier,
    DemoExecutionNotAuthorizedError,
    LiveAccountForbiddenError,
    assert_demo_execution_authorized,
)
from trading.adapters.mt5.schemas import MT5AccountMetadata, MT5TerminalMetadata
from trading.adapters.mt5.transport import (
    MT5DemoExecutionTransport,
)
from trading.api.app import create_app
from trading.api.dependencies import (
    TradingContext,
    reset_trading_context,
    set_trading_context,
)
from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.mt5_adapter import (
    BrokerExecutionDisabledError,
    MT5BrokerAdapter,
    MT5ConnectionError,
    MT5ResponseError,
)
from trading.execution.service import TradingExecutionService
from trading.execution.validation import BrokerSymbolSpecification
from trading.models.order import Order, OrderSide, OrderStatus, OrderType
from trading.models.position import Position
from trading.models.signal import Signal, SignalAction
from trading.persistence.database import PaperDatabaseManager
from trading.persistence.migrations import PaperSchemaMigrator
from trading.persistence.repository import PaperTradingRepository
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskEngine
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_phase39.db"


@pytest.fixture
def temp_repo(temp_db_path: Path) -> PaperTradingRepository:
    db = PaperDatabaseManager(db_path=temp_db_path)
    PaperSchemaMigrator.apply_migrations(db)
    return PaperTradingRepository(db)


def make_demo_account_meta(
    trade_mode: int = ACCOUNT_TRADE_MODE_DEMO,
    is_demo: bool = True,
    balance: float = 100_000.0,
    equity: float = 100_000.0,
) -> MT5AccountMetadata:
    return MT5AccountMetadata(
        login_masked="123***78",
        trade_mode=trade_mode,
        is_demo=is_demo,
        balance=balance,
        equity=equity,
        margin=0.0,
        margin_free=equity,
        leverage=100,
        currency="USD",
        company="MetaQuotes Software Corp.",
        server="MetaQuotes-Demo",
    )


def make_symbol_spec(symbol: str = "EURUSD", min_vol: float = 0.01) -> BrokerSymbolSpecification:
    return BrokerSymbolSpecification(
        symbol=symbol,
        min_volume=min_vol,
        max_volume=500.0,
        volume_step=0.01,
        contract_size=100_000.0,
        price_digits=5,
        point=0.00001,
        tick_size=0.00001,
    )


def make_signal(symbol: str = "EURUSD", action: SignalAction = SignalAction.BUY) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=0.85,
        suggested_entry_price=1.1250,
        suggested_stop_loss=1.0750 if action == SignalAction.BUY else 1.1750,
        suggested_take_profit=1.2000 if action == SignalAction.BUY else 1.0500,
        model_version="rule-based-test-v1",
        timeframe="H4",
        expected_return=0.0050,
        feature_version="1.0.0",
        timestamp=datetime(2026, 2, 19, 11, 0, 0, tzinfo=timezone.utc),
    )


def make_terminal_meta(connected: bool = True) -> MT5TerminalMetadata:
    return MT5TerminalMetadata(
        name="MetaTrader 5",
        company="MetaQuotes Ltd.",
        build=5000,
        version="5.00",
        connected=connected,
        trade_allowed=True,
        ping_last=15000,
        path="C:\\Program Files\\MetaTrader 5",
    )


# ---------------------------------------------------------------------------
# 1. Demo Verification & Safety Gate Tests
# ---------------------------------------------------------------------------


def test_01_demo_account_positive_verification():
    """Positive verification passes for MetaQuotes-Demo account."""
    meta = make_demo_account_meta()
    term = make_terminal_meta(connected=True)
    spec = make_symbol_spec()

    report = DemoAccountVerifier.verify(meta, terminal_meta=term, symbol_spec=spec)
    assert report.verified is True
    assert report.is_demo is True
    assert report.is_live_detected is False
    assert report.trade_mode_str == "DEMO"
    assert report.min_volume == 0.01


def test_02_live_account_forbidden():
    """Live account (trade_mode == 2) immediately raises LiveAccountForbiddenError."""
    meta = make_demo_account_meta(trade_mode=ACCOUNT_TRADE_MODE_REAL, is_demo=False)
    with pytest.raises(LiveAccountForbiddenError, match="LIVE/REAL"):
        DemoAccountVerifier.verify(meta)

    with pytest.raises(LiveAccountForbiddenError):
        assert_demo_execution_authorized(
            is_live=False,
            execution_enabled=True,
            demo_execution_enabled=True,
            account_meta=meta,
        )


def test_03_unknown_trade_mode_fails_closed():
    """Non-demo trade modes fail closed with DemoAccountVerificationError."""
    meta_contest = make_demo_account_meta(trade_mode=ACCOUNT_TRADE_MODE_CONTEST, is_demo=False)
    with pytest.raises(DemoAccountVerificationError, match="DEMO_VERIFICATION_FAILED"):
        DemoAccountVerifier.verify(meta_contest)

    meta_none = None
    with pytest.raises(DemoAccountVerificationError, match="Account metadata is None"):
        DemoAccountVerifier.verify(meta_none)


def test_04_demo_authorization_gate():
    """assert_demo_execution_authorized enforces all invariants fail-closed."""
    meta = make_demo_account_meta()

    # Rule 1: is_live must be False
    with pytest.raises(LiveAccountForbiddenError):
        assert_demo_execution_authorized(
            is_live=True,
            execution_enabled=True,
            demo_execution_enabled=True,
            account_meta=meta,
        )

    # Rule 2: execution_enabled must be True
    with pytest.raises(BrokerExecutionDisabledError):
        assert_demo_execution_authorized(
            is_live=False,
            execution_enabled=False,
            demo_execution_enabled=True,
            account_meta=meta,
        )

    # Rule 3: demo_execution_enabled must be True
    with pytest.raises(DemoExecutionNotAuthorizedError):
        assert_demo_execution_authorized(
            is_live=False,
            execution_enabled=True,
            demo_execution_enabled=False,
            account_meta=meta,
        )

    # Valid execution authorization
    report = assert_demo_execution_authorized(
        is_live=False,
        execution_enabled=True,
        demo_execution_enabled=True,
        account_meta=meta,
    )
    assert report.verified is True


def test_05_minimum_volume_discovery():
    """Minimum broker volume is discovered and validated dynamically."""
    meta = make_demo_account_meta()
    spec = make_symbol_spec(min_vol=0.01)
    rep = DemoAccountVerifier.verify(meta, symbol_spec=spec)
    assert rep.min_volume == 0.01

    bad_spec = make_symbol_spec(min_vol=-0.05)
    with pytest.raises(DemoAccountVerificationError, match="Invalid min_volume"):
        DemoAccountVerifier.verify(meta, symbol_spec=bad_spec)


def test_06_non_eurusd_symbol_rejected():
    """Only EURUSD is authorized for demo execution; other symbols fail verification."""
    meta = make_demo_account_meta()
    spec_gbp = make_symbol_spec(symbol="GBPUSD")
    with pytest.raises(DemoAccountVerificationError, match="Only EURUSD authorized"):
        DemoAccountVerifier.verify(meta, symbol_spec=spec_gbp)


# ---------------------------------------------------------------------------
# 2. Pipeline Execution & Safety Invariant Tests
# ---------------------------------------------------------------------------


class MockMT5Backend:
    """Mock backend providing controllable order_check and order_send responses."""

    def __init__(
        self,
        retcode: int = 10009,
        check_retcode: int = 0,
        deal: int = 7771,
        order: int = 8881,
        price: float = 1.1250,
        volume: float = 0.01,
        trade_mode: int = 0,
    ) -> None:
        self.retcode = retcode
        self.check_retcode = check_retcode
        self.deal = deal
        self.order = order
        self.price = price
        self.volume = volume
        self.trade_mode = trade_mode
        self.sent_requests: List[Dict[str, Any]] = []

    def account_info(self) -> Any:
        obj = MagicMock()
        obj.trade_mode = self.trade_mode
        obj.balance = 100_000.0
        obj.equity = 100_000.0
        obj.margin = 0.0
        obj.is_demo = self.trade_mode == 0
        obj.trade_allowed = True
        obj.trade_expert = True
        return obj

    def order_check(self, req: Dict[str, Any]) -> Any:
        chk = MagicMock()
        chk.retcode = self.check_retcode
        chk.comment = "Done" if self.check_retcode == 0 else "Check rejected"
        return chk

    def order_send(self, req: Dict[str, Any]) -> Any:
        self.sent_requests.append(req)
        res = MagicMock()
        res.retcode = self.retcode
        res.deal = self.deal
        res.order = self.order
        res.price = self.price
        res.volume = self.volume
        res.comment = "Done" if self.retcode == 10009 else "Rejected"
        return res


def test_07_full_buy_order_pipeline(temp_repo: PaperTradingRepository):
    """End-to-end signal -> risk -> MT5 demo adapter -> fill -> persistence."""
    mock_backend = MockMT5Backend(retcode=10009, price=1.1250, volume=0.10)
    transport = MT5DemoExecutionTransport(mock_backend=mock_backend)

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()

    audit = AuditTrail(repository=temp_repo)
    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        audit_trail=audit,
        repository=temp_repo,
    )
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=adapter,
        portfolio_manager=PortfolioManager(broker=adapter, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_signal(action=SignalAction.BUY)
    result = service.process_signal(sig, current_market_price=1.1250)

    assert result.approved is True
    assert result.order is not None
    assert result.order.status == OrderStatus.FILLED
    assert result.order.fill_price == 1.1250
    assert len(mock_backend.sent_requests) == 1

    # Position updated
    pos = adapter.get_position("EURUSD")
    assert pos is not None
    assert pos.quantity == 10_000.0  # 0.10 lots * 100,000

    # Persistence validated
    saved_orders = temp_repo.get_all_orders()
    assert len(saved_orders) == 1
    assert saved_orders[0].order_id == result.order.order_id


def test_08_full_sell_order_pipeline(temp_repo: PaperTradingRepository):
    """End-to-end SELL signal executes and persists correctly."""
    mock_backend = MockMT5Backend(retcode=10009, price=1.1250, volume=0.10)
    transport = MT5DemoExecutionTransport(mock_backend=mock_backend)

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()

    audit = AuditTrail(repository=temp_repo)
    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        audit_trail=audit,
        repository=temp_repo,
    )
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=adapter,
        portfolio_manager=PortfolioManager(broker=adapter, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_signal(action=SignalAction.SELL)
    result = service.process_signal(sig, current_market_price=1.1250)

    assert result.approved is True
    assert result.order.side == OrderSide.SELL
    assert result.order.status == OrderStatus.FILLED


def test_09_risk_engine_rejection_before_broker(temp_repo: PaperTradingRepository):
    """RiskEngine rejection blocks order from ever reaching MT5 transport."""
    mock_backend = MockMT5Backend()
    transport = MT5DemoExecutionTransport(mock_backend=mock_backend)

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()

    audit = AuditTrail(repository=temp_repo)
    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        audit_trail=audit,
        repository=temp_repo,
    )
    # High confidence requirement forces immediate risk rejection (signal has 0.85)
    limits = RiskLimits(MIN_SIGNAL_CONFIDENCE=0.95)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=limits, kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=adapter,
        portfolio_manager=PortfolioManager(broker=adapter, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_signal(action=SignalAction.BUY)
    result = service.process_signal(sig, current_market_price=1.1250)

    assert result.approved is False
    assert result.reason == "CONFIDENCE_TOO_LOW"
    assert len(mock_backend.sent_requests) == 0  # 0 calls to broker


def test_10_kill_switch_active_rejection(temp_repo: PaperTradingRepository):
    """KillSwitch active state prevents order submission to broker."""
    mock_backend = MockMT5Backend()
    transport = MT5DemoExecutionTransport(mock_backend=mock_backend)

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()

    audit = AuditTrail(repository=temp_repo)
    ks = KillSwitch(audit, temp_repo)
    ks.activate(reason="EMERGENCY_HALT")

    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        audit_trail=audit,
        repository=temp_repo,
    )
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=ks),
        paper_broker=adapter,
        portfolio_manager=PortfolioManager(broker=adapter, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_signal()
    result = service.process_signal(sig, current_market_price=1.1250)

    assert result.approved is False
    assert "KILL_SWITCH_ACTIVE" in (result.reason or "")
    assert len(mock_backend.sent_requests) == 0


def test_11_broker_check_rejection(temp_repo: PaperTradingRepository):
    """Broker order_check failure results in REJECTED order without position creation."""
    mock_backend = MockMT5Backend(check_retcode=10014)  # Invalid volume
    transport = MT5DemoExecutionTransport(mock_backend=mock_backend)

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()

    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        repository=temp_repo,
    )
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    res_order = adapter.submit_order(order, current_market_price=1.1250)

    assert res_order.status == OrderStatus.REJECTED
    assert "MOCK_CHECK_REJECTED" in (res_order.rejection_reason or "")
    assert adapter.get_positions() == {}


def test_12_broker_timeout_handling():
    """Simulated broker timeout raises MT5ConnectionError and restores snapshot."""
    mock_transport = MagicMock()
    mock_transport.send_trade_request.side_effect = MT5ConnectionError("MT5_TIMEOUT")

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()

    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=mock_transport,
        client=mock_client,
    )
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    with pytest.raises(MT5ConnectionError, match="MT5_TIMEOUT"):
        adapter.submit_order(order, current_market_price=1.1250)

    assert adapter.get_positions() == {}


def test_13_malformed_response_handling():
    """Malformed response raises MT5ResponseError and leaves state intact."""
    mock_transport = MagicMock()
    mock_transport.send_trade_request.side_effect = MT5ResponseError("MALFORMED_JSON")

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()

    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=mock_transport,
        client=mock_client,
    )
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    with pytest.raises(MT5ResponseError):
        adapter.submit_order(order, current_market_price=1.1250)

    assert adapter.get_positions() == {}


def test_14_duplicate_acknowledgement_idempotency(temp_repo: PaperTradingRepository):
    """Duplicate execution acknowledgment does not double-count positions."""
    mock_backend = MockMT5Backend(retcode=10009, deal=9999, order=8888)
    transport = MT5DemoExecutionTransport(mock_backend=mock_backend)

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()

    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        repository=temp_repo,
    )
    order1 = Order(
        order_id="ord-dup-1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    res1 = adapter.submit_order(order1, current_market_price=1.1250)
    assert res1.status == OrderStatus.FILLED
    assert adapter.get_position("EURUSD").quantity == 1000.0

    # Second submission returns identical deal ID
    order2 = Order(
        order_id="ord-dup-2",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    res2 = adapter.submit_order(order2, current_market_price=1.1250)
    assert res2 is not None
    # Quantity must remain 1000.0, NOT 2000.0
    assert adapter.get_position("EURUSD").quantity == 1000.0


def test_15_persistence_failure_rollback(temp_repo: PaperTradingRepository):
    """Atomic rollback on repository failure preserves memory consistency."""
    mock_backend = MockMT5Backend(retcode=10009)
    transport = MT5DemoExecutionTransport(mock_backend=mock_backend)

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()

    broken_repo = MagicMock(spec=temp_repo)
    broken_repo.save_order.side_effect = RuntimeError("DISK_IO_FAILURE")

    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        repository=broken_repo,
    )
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    with pytest.raises(RuntimeError, match="DISK_IO_FAILURE"):
        adapter.submit_order(order, current_market_price=1.1250)

    # In-memory positions restored to empty
    assert adapter.get_positions() == {}


def test_16_controlled_exit_position_close(temp_repo: PaperTradingRepository):
    """Controlled position close creates opposite deal, updates PnL, and closes position."""
    mock_backend = MockMT5Backend(retcode=10009, deal=101, price=1.1250)
    transport = MT5DemoExecutionTransport(mock_backend=mock_backend)

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()

    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        repository=temp_repo,
    )

    # 1. Open BUY position
    buy_order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    adapter.submit_order(buy_order, current_market_price=1.1250)
    assert "EURUSD" in adapter.get_positions()

    # 2. Close position at 1.1280 (profitable exit)
    mock_backend.deal = 102
    mock_backend.price = 1.1280
    close_order = adapter.close_position("EURUSD", exit_price=1.1280)

    assert close_order.status == OrderStatus.FILLED
    assert adapter.get_position("EURUSD") is None
    # Realized PnL: (1.1280 - 1.1250) * 1000 = +3.0 USD
    assert pytest.approx(adapter.get_account().realized_pnl, abs=0.01) == 3.0


# ---------------------------------------------------------------------------
# 3. State Reconciliation Tests
# ---------------------------------------------------------------------------


def test_17_reconciliation_healthy_matched():
    """Matching positions and balances report HEALTHY."""
    pos = Position(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000.0,
        entry_price=1.1250,
        current_price=1.1250,
    )
    int_positions = {"EURUSD": pos}
    broker_positions = [{"symbol": "EURUSD", "type": 0, "volume": 0.01, "ticket": 501}]

    rep = reconcile_positions(
        internal_positions=int_positions,
        broker_positions=broker_positions,
        contract_size=100_000.0,
    )
    assert rep.healthy is True
    assert rep.status == ReconciliationStatus.HEALTHY
    assert len(rep.discrepancies) == 0


def test_18_reconciliation_mismatch_detected(temp_repo: PaperTradingRepository):
    """Discrepancies report MISMATCH_DETECTED and record an audit event."""
    audit = AuditTrail(repository=temp_repo)
    int_positions = {}
    broker_positions = [{"symbol": "EURUSD", "type": 0, "volume": 0.01, "ticket": 501}]

    rep = reconcile_positions(
        internal_positions=int_positions,
        broker_positions=broker_positions,
        contract_size=100_000.0,
        audit_trail=audit,
    )
    assert rep.healthy is False
    assert rep.status == ReconciliationStatus.MISMATCH_DETECTED
    assert len(rep.discrepancies) > 0

    events = temp_repo.get_audit_events()
    assert any(e.event_type == AuditEventType.EXECUTION_ERROR for e in events)


def test_19_reconciliation_strict_fail_closed():
    """strict_fail_closed=True raises PositionReconciliationError immediately."""
    int_positions = {}
    broker_positions = [{"symbol": "EURUSD", "type": 0, "volume": 0.01, "ticket": 501}]

    with pytest.raises(PositionReconciliationError, match="RECONCILIATION_FAILED"):
        reconcile_positions(
            internal_positions=int_positions,
            broker_positions=broker_positions,
            strict_fail_closed=True,
        )


# ---------------------------------------------------------------------------
# 4. API Observability & Governance Invariants
# ---------------------------------------------------------------------------


def test_20_api_readiness_exposure():
    """Readiness endpoint exposes complete Phase 39 MT5 telemetry."""
    reset_trading_context()
    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()
    mock_client.get_open_positions.return_value = []

    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        client=mock_client,
    )
    ctx = TradingContext(mt5_adapter=adapter)
    set_trading_context(ctx)

    app = create_app()
    client = TestClient(app)
    resp = client.get("/readiness")

    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert data["mt5_adapter_available"] is True
    assert data["mt5_execution_enabled"] is True
    assert data["mt5_readonly_connected"] is True
    assert data["mt5_demo_account_verified"] is True
    assert data["mt5_live_account_detected"] is False
    assert data["mt5_trade_permissions_verified"] is True
    assert data["mt5_symbol_verified"] is True
    assert data["reconciliation_healthy"] is True
    reset_trading_context()


def test_21_execution_disabled_when_not_authorized():
    """Attempting order submission when demo_execution_enabled is False raises error."""
    adapter = MT5BrokerAdapter(execution_enabled=True, demo_execution_enabled=False)
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    with pytest.raises(DemoExecutionNotAuthorizedError):
        adapter.submit_order(order)


def test_22_restart_recovery_from_persistence(temp_repo: PaperTradingRepository):
    """Crash recovery restores active positions and execution history from persistent store."""
    mock_backend = MockMT5Backend(retcode=10009, deal=901, price=1.1250)
    transport = MT5DemoExecutionTransport(mock_backend=mock_backend)

    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.get_account_metadata.return_value = make_demo_account_meta()
    mock_client.get_symbol_specification.return_value = make_symbol_spec()

    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        repository=temp_repo,
    )
    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    adapter.submit_order(order, current_market_price=1.1250)

    # Simulate restart by creating new adapter pointing to same repository
    adapter_restarted = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=mock_client,
        repository=temp_repo,
    )
    assert adapter_restarted.execution_enabled is True
    # Recover positions from repo
    active_positions = temp_repo.get_active_positions()
    assert "EURUSD" in active_positions
    assert active_positions["EURUSD"].quantity == 1000.0
    assert len(temp_repo.get_all_executions()) == 1
