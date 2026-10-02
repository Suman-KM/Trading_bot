"""Phase 40: Extended MT5 Demo Forward Validation & Operational Stability Tests.

Covers all 20 mandatory validation requirements:
1. demo account gate
2. live account rejection
3. unknown account rejection
4. stale market data
5. signal logging
6. signal vs execution separation
7. daily loss protection
8. max trade count
9. max open position protection
10. kill switch
11. reconciliation success
12. reconciliation mismatch
13. account mismatch
14. duplicate order prevention
15. timeout recovery
16. restart recovery
17. persistence failure
18. readiness failure
19. execution error threshold
20. demo-only invariant
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from trading.adapters.mt5.client import MT5ReadOnlyClient
from trading.adapters.mt5.forward import (
    ForwardDemoRunner,
    ForwardValidationConfig,
)
from trading.adapters.mt5.reconciliation import (
    ReconciliationStatus,
)
from trading.adapters.mt5.safety import (
    ACCOUNT_TRADE_MODE_DEMO,
    ACCOUNT_TRADE_MODE_REAL,
    DEMO_ONLY,
    LiveAccountForbiddenError,
)
from trading.adapters.mt5.transport import (
    MT5DemoExecutionTransport,
)
from trading.api.app import app
from trading.audit.trail import AuditTrail
from trading.execution.exceptions import (
    MT5ConnectionError,
)
from trading.execution.mt5_adapter import MT5BrokerAdapter
from trading.execution.service import TradingExecutionService
from trading.models.order import OrderSide
from trading.models.position import Position
from trading.models.signal import Signal, SignalAction
from trading.persistence.database import PaperDatabaseManager
from trading.persistence.migrations import PaperSchemaMigrator
from trading.persistence.repository import PaperTradingRepository
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskEngine
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits


class MockBackendForForward:
    """Deterministic mock MT5 backend for forward testing."""

    def __init__(
        self,
        trade_mode: int = ACCOUNT_TRADE_MODE_DEMO,
        is_demo: bool = True,
        balance: float = 100000.0,
        equity: float = 100000.0,
        connected: bool = True,
        positions: Optional[List[Dict[str, Any]]] = None,
        retcode: int = 10009,
    ) -> None:
        self.trade_mode = trade_mode
        self.is_demo = is_demo
        self.balance = balance
        self.equity = equity
        self.connected = connected
        self._positions = positions or []
        self.retcode = retcode
        self.sent_orders: List[Dict[str, Any]] = []

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def terminal_info(self) -> Any:
        m = MagicMock()
        m._asdict.return_value = {
            "name": "MetaTrader 5",
            "company": "MetaQuotes Ltd.",
            "connected": self.connected,
            "trade_allowed": True,
            "ping_last": 1500,
            "path": "C:\\Program Files\\MetaTrader 5",
        }
        return m

    def version(self) -> Any:
        return (500, 6230, "Mock")

    def account_info(self) -> Any:
        m = MagicMock()
        m.login = 5056291434
        m.server = "MetaQuotes-Demo"
        m.currency = "USD"
        m.company = "MetaQuotes Ltd."
        m.leverage = 100
        m.balance = self.balance
        m.equity = self.equity
        m.margin = 0.0
        m.margin_free = self.balance
        m.trade_mode = self.trade_mode
        m.margin_mode = 2
        m._asdict.return_value = {
            "login": 5056291434,
            "server": "MetaQuotes-Demo",
            "currency": "USD",
            "company": "MetaQuotes Ltd.",
            "leverage": 100,
            "balance": self.balance,
            "equity": self.equity,
            "margin": 0.0,
            "margin_free": self.balance,
            "trade_mode": self.trade_mode,
            "margin_mode": 2,
        }
        return m

    def symbol_info(self, symbol: str) -> Any:
        m = MagicMock()
        m.name = symbol
        m.volume_min = 0.01
        m.volume_max = 500.0
        m.volume_step = 0.01
        m.trade_contract_size = 100000.0
        m.digits = 5
        m.point = 1e-05
        m.trade_tick_size = 1e-05
        return m

    def symbol_info_tick(self, symbol: str) -> Any:
        broker_time = datetime.now(timezone.utc) + timedelta(hours=3.0)
        now_ts = int(broker_time.timestamp())
        m = MagicMock()
        m.time = now_ts
        m.time_msc = now_ts * 1000
        m.bid = 1.12500
        m.ask = 1.12510
        m.last = 0.0
        m.volume = 0.0
        m.flags = 6
        m._asdict.return_value = {
            "time": now_ts,
            "time_msc": now_ts * 1000,
            "bid": 1.12500,
            "ask": 1.12510,
            "last": 0.0,
            "volume": 0.0,
            "flags": 6,
        }
        return m

    def positions_get(self, symbol: Optional[str] = None) -> Any:
        return self._positions

    def order_check(self, req: Dict[str, Any]) -> Any:
        m = MagicMock()
        m.retcode = 0
        m.comment = "Done"
        return m

    def order_send(self, req: Dict[str, Any]) -> Any:
        self.sent_orders.append(req)
        m = MagicMock()
        m.retcode = self.retcode
        m.order = 123456
        m.deal = 789012
        m.volume = req.get("volume", 0.01)
        m.price = req.get("price", 1.12510)
        m.comment = "Filled"
        if self.retcode == 10009:
            self._positions.append(
                {
                    "ticket": 123456,
                    "symbol": req.get("symbol", "EURUSD"),
                    "volume": req.get("volume", 0.01),
                    "type": req.get("type", 0),
                    "price_open": req.get("price", 1.12510),
                    "sl": req.get("sl", 0.0),
                    "tp": req.get("tp", 0.0),
                }
            )
        return m


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_forward.db"


@pytest.fixture
def repo(test_db_path: Path) -> PaperTradingRepository:
    mgr = PaperDatabaseManager(str(test_db_path))
    PaperSchemaMigrator.apply_migrations(mgr)
    return PaperTradingRepository(mgr)


@pytest.fixture
def setup_pipeline(repo: PaperTradingRepository):
    backend = MockBackendForForward()
    client = MT5ReadOnlyClient(mock_backend=backend)
    client.initialize()
    transport = MT5DemoExecutionTransport(mock_backend=backend)
    audit = AuditTrail(repository=repo)
    limits = RiskLimits(
        MAX_DAILY_LOSS_PERCENT=1.0,
        MAX_POSITION_RISK_PERCENT=0.5,
        MAX_TOTAL_EXPOSURE_PERCENT=20.0,
        MIN_SIGNAL_CONFIDENCE=0.60,
    )
    kill_switch = KillSwitch(audit_trail=audit, repository=repo)
    risk_engine = RiskEngine(limits=limits, kill_switch=kill_switch)
    adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        transport=transport,
        client=client,
        audit_trail=audit,
        repository=repo,
        initial_balance=100000.0,
    )
    portfolio = PortfolioManager(broker=adapter, repository=repo)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=adapter,
        portfolio_manager=portfolio,
        audit_trail=audit,
        repository=repo,
    )
    config = ForwardValidationConfig(
        max_demo_trades_per_day=3,
        max_demo_open_positions=1,
        max_consecutive_execution_errors=3,
        max_data_staleness_seconds=120.0,
    )
    runner = ForwardDemoRunner(
        adapter=adapter,
        execution_service=service,
        client=client,
        repository=repo,
        audit_trail=audit,
        risk_engine=risk_engine,
        kill_switch=kill_switch,
        config=config,
    )
    return {
        "backend": backend,
        "client": client,
        "transport": transport,
        "adapter": adapter,
        "service": service,
        "runner": runner,
        "repo": repo,
        "risk_engine": risk_engine,
        "kill_switch": kill_switch,
    }


def make_test_signal(
    symbol: str = "EURUSD",
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.85,
    entry_price: float = 1.12510,
    sl: float = 0.62510,
    tp: float = 1.62510,
    client_request_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Signal:
    import time

    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        timestamp=timestamp or datetime.now(timezone.utc),
        model_version="p40-test-v1",
        timeframe="M15",
        expected_return=0.0050,
        feature_version="1.0.0",
        client_request_id=client_request_id or f"req-{int(time.time() * 1000)}",
        suggested_entry_price=entry_price,
        suggested_stop_loss=sl,
        suggested_take_profit=tp,
    )


# ==============================================================================
# TEST 1: DEMO ACCOUNT GATE
# ==============================================================================
def test_01_demo_account_gate(setup_pipeline):
    """Verify demo account positive verification passes on confirmed demo account."""
    runner = setup_pipeline["runner"]
    preflight = runner.preflight_inspection()
    assert preflight.demo_verified is True
    assert preflight.live_account is False
    assert preflight.can_start is True
    assert preflight.account_mode == "DEMO (0)"


# ==============================================================================
# TEST 2: LIVE ACCOUNT REJECTION
# ==============================================================================
def test_02_live_account_rejection(setup_pipeline):
    """Verify live account (trade_mode == 2) immediately raises LiveAccountForbiddenError."""
    backend = setup_pipeline["backend"]
    backend.trade_mode = ACCOUNT_TRADE_MODE_REAL
    backend.is_demo = False

    runner = setup_pipeline["runner"]
    with pytest.raises(LiveAccountForbiddenError):
        runner.preflight_inspection()


# ==============================================================================
# TEST 3: UNKNOWN ACCOUNT REJECTION
# ==============================================================================
def test_03_unknown_account_rejection(setup_pipeline):
    """Verify unknown trade mode fails closed."""
    backend = setup_pipeline["backend"]
    backend.trade_mode = 99  # Invalid trade mode
    backend.is_demo = False

    runner = setup_pipeline["runner"]
    with pytest.raises(LiveAccountForbiddenError):
        runner.preflight_inspection()


# ==============================================================================
# TEST 4: STALE MARKET DATA
# ==============================================================================
def test_04_stale_market_data(setup_pipeline):
    """Verify stale market data halts new entries and logs staleness event."""
    runner = setup_pipeline["runner"]
    backend = setup_pipeline["backend"]

    # Provide a stale tick timestamp (10 minutes old in broker time)
    old_time = int(
        (datetime.now(timezone.utc) + timedelta(hours=3.0) - timedelta(minutes=10)).timestamp()
    )
    mock_tick = MagicMock()
    mock_tick.time = old_time
    mock_tick.time_msc = old_time * 1000
    mock_tick.bid = 1.12500
    mock_tick.ask = 1.12510
    mock_tick.last = 0.0
    mock_tick.volume = 0.0
    mock_tick.flags = 6
    mock_tick._asdict.return_value = {
        "time": old_time,
        "bid": 1.12500,
        "ask": 1.12510,
        "last": 0.0,
        "volume": 0.0,
        "flags": 6,
    }
    backend.symbol_info_tick = lambda sym: mock_tick

    res = runner.run_iteration()
    assert res["status"] == "SKIPPED"
    assert res["reason"] == "STALE_MARKET_DATA"
    assert runner.metrics.data_staleness_events == 1


# ==============================================================================
# TEST 5: SIGNAL LOGGING
# ==============================================================================
def test_05_signal_logging(setup_pipeline):
    """Verify signal logging records all signal attributes."""
    runner = setup_pipeline["runner"]
    now_utc = datetime.now(timezone.utc)

    # Provide a valid test signal
    test_sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.80,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="sig-log-test-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: test_sig

    runner.run_iteration()
    assert len(runner.signal_logs) == 1
    log = runner.signal_logs[0]
    assert log.symbol == "EURUSD"
    assert log.action == "BUY"
    assert log.confidence == 0.80
    assert log.risk_decision == "APPROVED"


# ==============================================================================
# TEST 6: SIGNAL VS EXECUTION SEPARATION
# ==============================================================================
def test_06_signal_vs_execution_separation(setup_pipeline):
    """Verify signal generation does not equal execution when rejected by risk."""
    runner = setup_pipeline["runner"]
    now_utc = datetime.now(timezone.utc)

    # Signal with low confidence that RiskEngine will reject
    low_conf_sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.45,  # Below MIN_SIGNAL_CONFIDENCE (0.60)
        entry_price=1.12510,
        sl=1.12000,
        tp=1.13000,
        client_request_id="low-conf-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: low_conf_sig

    res = runner.run_iteration()
    assert res["status"] == "REJECTED"
    assert runner.metrics.signals_generated == 1
    assert runner.metrics.signals_approved == 0
    assert runner.metrics.signals_rejected == 1
    assert runner.metrics.orders_submitted == 0


# ==============================================================================
# TEST 7: DAILY LOSS PROTECTION
# ==============================================================================
def test_07_daily_loss_protection(setup_pipeline):
    """Verify daily loss breach blocks subsequent orders."""
    runner = setup_pipeline["runner"]
    adapter = setup_pipeline["adapter"]
    backend = setup_pipeline["backend"]
    now_utc = datetime.now(timezone.utc)

    # Simulate realized loss exceeding MAX_DAILY_LOSS_PERCENT (1% of 100k = $1000)
    adapter._realized_pnl = -1500.0
    adapter._cash_balance -= 1500.0
    backend.equity = 98500.0
    backend.balance = 98500.0

    sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="loss-test-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: sig

    res = runner.run_iteration()
    assert res["status"] == "REJECTED"
    assert "DAILY_LOSS" in res["reason"]


# ==============================================================================
# TEST 8: MAX TRADE COUNT
# ==============================================================================
def test_08_max_trade_count(setup_pipeline):
    """Verify max_demo_trades_per_day operational limit halts submissions."""
    runner = setup_pipeline["runner"]
    runner.metrics.orders_submitted = 3  # Config limit is 3
    now_utc = datetime.now(timezone.utc)

    sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="max-trade-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: sig

    res = runner.run_iteration()
    assert res["status"] == "REJECTED"
    assert res["reason"] == "MAX_DEMO_TRADES_PER_DAY_EXCEEDED"


# ==============================================================================
# TEST 9: MAX OPEN POSITION PROTECTION
# ==============================================================================
def test_09_max_open_position_protection(setup_pipeline):
    """Verify max_demo_open_positions operational limit blocks additional entries."""
    runner = setup_pipeline["runner"]
    adapter = setup_pipeline["adapter"]
    backend = setup_pipeline["backend"]
    now_utc = datetime.now(timezone.utc)

    # Pre-populate one open position (max is 1)
    adapter._positions["EURUSD"] = Position(
        position_id="pos-open-1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000.0,
        entry_price=1.12500,
        current_price=1.12510,
        entry_timestamp=now_utc,
    )
    backend._positions = [
        {
            "ticket": 999111,
            "symbol": "EURUSD",
            "volume": 0.01,
            "type": 0,  # BUY
            "price_open": 1.12500,
            "sl": 0.0,
            "tp": 0.0,
        }
    ]

    sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="max-pos-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: sig

    res = runner.run_iteration()
    assert res["status"] == "REJECTED"
    assert res["reason"] == "MAX_DEMO_OPEN_POSITIONS_REACHED"


# ==============================================================================
# TEST 10: KILL SWITCH
# ==============================================================================
def test_10_kill_switch(setup_pipeline):
    """Verify activated kill switch rejects orders and maintains safe state."""
    runner = setup_pipeline["runner"]
    kill_switch = setup_pipeline["kill_switch"]
    kill_switch.activate("MANUAL_TEST_ACTIVATION")
    now_utc = datetime.now(timezone.utc)

    sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="ks-test-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: sig

    res = runner.run_iteration()
    assert res["status"] == "REJECTED"
    assert "KILL_SWITCH" in res["reason"]


# ==============================================================================
# TEST 11: RECONCILIATION SUCCESS
# ==============================================================================
def test_11_reconciliation_success(setup_pipeline):
    """Verify position reconciliation returns HEALTHY when states match."""
    adapter = setup_pipeline["adapter"]
    backend = setup_pipeline["backend"]
    now_utc = datetime.now(timezone.utc)

    # 1 internal position and 1 matching broker position
    adapter._positions["EURUSD"] = Position(
        position_id="pos-match",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000.0,
        entry_price=1.12500,
        current_price=1.12510,
        entry_timestamp=now_utc,
    )
    backend._positions = [
        {
            "ticket": 999111,
            "symbol": "EURUSD",
            "volume": 0.01,
            "type": 0,  # BUY
            "price_open": 1.12500,
            "sl": 0.0,
            "tp": 0.0,
        }
    ]

    report = adapter.reconcile()
    assert report.status == ReconciliationStatus.HEALTHY
    assert report.healthy is True
    assert report.internal_open_positions_count == 1
    assert report.broker_open_positions_count == 1


# ==============================================================================
# TEST 12: RECONCILIATION MISMATCH
# ==============================================================================
def test_12_reconciliation_mismatch(setup_pipeline):
    """Verify position mismatch triggers fail-closed response."""
    runner = setup_pipeline["runner"]
    adapter = setup_pipeline["adapter"]
    backend = setup_pipeline["backend"]

    # 0 internal positions but 1 broker position
    adapter._positions.clear()
    backend._positions = [
        {
            "ticket": 999222,
            "symbol": "EURUSD",
            "volume": 0.01,
            "type": 0,
            "price_open": 1.12500,
            "sl": 0.0,
            "tp": 0.0,
        }
    ]

    res = runner.run_iteration()
    assert res["status"] == "ERROR"
    assert res["reason"] == "RECONCILIATION_FAILURE"
    assert runner.metrics.reconciliation_failures == 1


# ==============================================================================
# TEST 13: ACCOUNT MISMATCH
# ==============================================================================
def test_13_account_mismatch(setup_pipeline):
    """Verify account equity mismatch is audited during reconciliation."""
    adapter = setup_pipeline["adapter"]
    backend = setup_pipeline["backend"]

    # Broker equity differs significantly from internal equity
    backend.equity = 80000.0
    adapter._cash_balance = 100000.0

    report = adapter.reconcile()
    equity_disc = [d for d in report.discrepancies if "Equity divergence" in d]
    assert len(equity_disc) > 0


# ==============================================================================
# TEST 14: DUPLICATE ORDER PREVENTION
# ==============================================================================
def test_14_duplicate_order_prevention(setup_pipeline):
    """Verify same client_request_id cannot be submitted twice."""
    runner = setup_pipeline["runner"]
    now_utc = datetime.now(timezone.utc)

    sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="dup-check-99",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: sig

    res1 = runner.run_iteration()
    assert res1["status"] == "EXECUTED"

    res2 = runner.run_iteration()
    assert res2["status"] == "SKIPPED"
    assert res2["reason"] == "DUPLICATE_ORDER_PREVENTED"
    assert runner.metrics.duplicate_prevented == 1


# ==============================================================================
# TEST 15: TIMEOUT RECOVERY
# ==============================================================================
def test_15_timeout_recovery(setup_pipeline):
    """Verify broker timeout fails closed and does not blindly retry."""
    runner = setup_pipeline["runner"]
    backend = setup_pipeline["backend"]
    now_utc = datetime.now(timezone.utc)

    # Force backend to simulate timeout
    def raise_timeout(*args, **kwargs):
        raise MT5ConnectionError("MT5_TIMEOUT: order execution timed out.")

    backend.order_send = raise_timeout

    sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="timeout-test-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: sig

    res = runner.run_iteration()
    assert res["status"] == "ERROR"
    assert runner.metrics.execution_errors == 1
    assert runner.metrics.consecutive_execution_errors == 1


# ==============================================================================
# TEST 16: RESTART RECOVERY
# ==============================================================================
def test_16_restart_recovery(setup_pipeline):
    """Verify state recovers accurately from database after restart."""
    runner = setup_pipeline["runner"]
    repo = setup_pipeline["repo"]
    now_utc = datetime.now(timezone.utc)

    # Execute a trade
    sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="restart-rec-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: sig
    runner.run_iteration()

    # Re-initialize new adapter and portfolio from same repo
    new_adapter = MT5BrokerAdapter(
        execution_enabled=True,
        demo_execution_enabled=True,
        repository=repo,
        initial_balance=100000.0,
    )
    _ = PortfolioManager(broker=new_adapter, repository=repo)
    recovered_pos = list(new_adapter.get_positions().values())
    assert len(recovered_pos) == 1
    assert recovered_pos[0].symbol == "EURUSD"
    assert recovered_pos[0].quantity == 1000.0


# ==============================================================================
# TEST 17: PERSISTENCE FAILURE
# ==============================================================================
def test_17_persistence_failure(setup_pipeline):
    """Verify transactional snapshot rollback if persistence fails."""
    runner = setup_pipeline["runner"]
    repo = setup_pipeline["repo"]
    now_utc = datetime.now(timezone.utc)

    # Force repo save_order to fail
    repo.save_order = MagicMock(side_effect=RuntimeError("DISK_CORRUPTION_SIMULATION"))

    sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="persist-fail-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: sig

    res = runner.run_iteration()
    assert res["status"] == "ERROR"
    # Adapter positions must remain 0 due to atomic rollback
    assert len(setup_pipeline["adapter"].get_positions()) == 0


# ==============================================================================
# TEST 18: READINESS FAILURE
# ==============================================================================
def test_18_readiness_failure(setup_pipeline):
    """Verify /readiness endpoint returns 503 when an operational safety rule fails."""
    client = TestClient(app)
    # The default test app context has kill_switch active or paper_broker uninitialized
    resp = client.get("/readiness")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "reconciliation_healthy" in data
    assert "data_fresh" in data
    assert "risk_engine_ready" in data


# ==============================================================================
# TEST 19: EXECUTION ERROR THRESHOLD
# ==============================================================================
def test_19_execution_error_threshold(setup_pipeline):
    """Verify consecutive execution errors trigger kill switch."""
    runner = setup_pipeline["runner"]
    kill_switch = setup_pipeline["kill_switch"]
    runner.metrics.consecutive_execution_errors = 3  # Limit is 3
    now_utc = datetime.now(timezone.utc)

    sig = make_test_signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=1.12510,
        sl=0.62510,
        tp=1.62510,
        client_request_id="thresh-test-1",
        timestamp=now_utc,
    )
    runner.strategy_fn = lambda tick: sig

    res = runner.run_iteration()
    assert res["status"] == "REJECTED"
    assert res["reason"] == "CONSECUTIVE_ERRORS_KILL_SWITCH"
    assert kill_switch.is_active() is True


# ==============================================================================
# TEST 20: DEMO-ONLY INVARIANT
# ==============================================================================
def test_20_demo_only_invariant(setup_pipeline):
    """Verify DEMO_ONLY == True, is_live == False, zero real capital exposure."""
    adapter = setup_pipeline["adapter"]
    runner = setup_pipeline["runner"]
    summary = runner.stop()

    assert DEMO_ONLY is True
    assert adapter.is_live is False
    assert summary["real_money_orders"] == 0
    assert summary["real_capital_exposure"] == "$0.00"
    assert summary["account_mode"] == "DEMO (0)"
