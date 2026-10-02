"""Phase 38: Controlled MT5 Execution Simulation & End-to-End Order Pipeline Validation.

Verifies the complete execution pipeline from Signal through RiskEngine, Position Sizing,
Broker Validation, Order Translation, Simulated MT5 Transport, Fill Simulation, Persistence,
Portfolio Update, and Audit Trail without calling order_send() or submitting real orders.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.mt5_adapter import (
    BrokerExecutionDisabledError,
    MT5BrokerAdapter,
    MT5ResponseError,
)
from trading.execution.mt5_simulator import (
    SimulatedMT5BrokerAdapter,
    SimulatedMT5Transport,
)
from trading.execution.paper_broker import PaperBroker
from trading.execution.service import TradingExecutionService
from trading.execution.validation import (
    BrokerValidationError,
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
def temp_repo(tmp_path: Path) -> Generator[PaperTradingRepository, None, None]:
    """Provide a clean, migrated, persistent SQLite database for testing."""
    db_path = tmp_path / "paper_trading_phase38.db"
    db_mgr = PaperDatabaseManager(db_path=str(db_path))
    PaperSchemaMigrator.apply_migrations(db_mgr)
    repo = PaperTradingRepository(db_mgr)
    yield repo
    db_mgr.close()


def make_valid_buy_signal(
    client_request_id: str = "req-buy-001",
    confidence: float = 0.85,
    entry_price: float = 1.1250,
    stop_loss: float = 1.0750,
    take_profit: float = 1.2000,
    symbol: str = "EURUSD",
) -> Signal:
    """Helper to construct a valid BUY signal for EURUSD."""
    return Signal(
        symbol=symbol,
        action=SignalAction.BUY,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
        model_version="1.0.0",
        timeframe="H4",
        expected_return=0.0050,
        feature_version="1.0.0",
        client_request_id=client_request_id,
        suggested_entry_price=entry_price,
        suggested_stop_loss=stop_loss,
        suggested_take_profit=take_profit,
    )


def make_valid_sell_signal(
    client_request_id: str = "req-sell-001",
    confidence: float = 0.85,
    entry_price: float = 1.1250,
    stop_loss: float = 1.1750,
    take_profit: float = 1.0500,
) -> Signal:
    """Helper to construct a valid SELL signal for EURUSD."""
    return Signal(
        symbol="EURUSD",
        action=SignalAction.SELL,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
        model_version="1.0.0",
        timeframe="H4",
        expected_return=0.0050,
        feature_version="1.0.0",
        client_request_id=client_request_id,
        suggested_entry_price=entry_price,
        suggested_stop_loss=stop_loss,
        suggested_take_profit=take_profit,
    )


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------


def test_01_buy_end_to_end(temp_repo: PaperTradingRepository):
    """Step 4: End-to-end BUY signal execution through simulated MT5 transport."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        initial_balance=100_000.0,
        audit_trail=audit,
        repository=temp_repo,
    )
    risk_engine = RiskEngine(
        limits=RiskLimits(),
        kill_switch=KillSwitch(audit_trail=audit, repository=temp_repo),
    )
    portfolio_mgr = PortfolioManager(broker=broker, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_valid_buy_signal("buy-e2e-01", entry_price=1.1250, stop_loss=1.0750)
    result = service.process_signal(sig, current_market_price=1.1250)

    # 1. Pipeline result checks
    assert result.approved is True
    assert result.order is not None
    assert result.order.symbol == "EURUSD"
    assert result.order.side == OrderSide.BUY
    assert result.order.status == OrderStatus.FILLED
    assert result.order.fill_price == 1.1250
    assert result.order.quantity > 0

    # 2. Simulated transport checks
    assert transport.submission_count == 1
    req = transport.submissions[0]
    assert req.symbol == "EURUSD"
    assert req.order_type == 0  # BUY
    assert req.price == 1.1250
    assert req.volume == round(result.order.quantity / 100_000.0, 4)

    # 3. Position & Portfolio update
    positions = broker.get_positions()
    assert "EURUSD" in positions
    pos = positions["EURUSD"]
    assert pos.side == OrderSide.BUY
    assert pos.quantity == result.order.quantity
    assert pos.entry_price == 1.1250

    # 4. Persistence verification
    persisted_order = temp_repo.get_order(result.order.order_id)
    assert persisted_order is not None
    assert persisted_order.status == OrderStatus.FILLED

    # 5. Audit trail verification
    events = temp_repo.get_audit_events(limit=50)
    event_types = [e.event_type.value for e in events]
    assert AuditEventType.SIGNAL_RECEIVED.value in event_types
    assert AuditEventType.RISK_ACCEPTED.value in event_types
    assert AuditEventType.ORDER_CREATED.value in event_types
    assert AuditEventType.ORDER_FILLED.value in event_types


def test_02_sell_end_to_end(temp_repo: PaperTradingRepository):
    """Step 5: End-to-end SELL signal execution through simulated MT5 transport."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        initial_balance=100_000.0,
        audit_trail=audit,
        repository=temp_repo,
    )
    risk_engine = RiskEngine(
        limits=RiskLimits(),
        kill_switch=KillSwitch(audit_trail=audit, repository=temp_repo),
    )
    portfolio_mgr = PortfolioManager(broker=broker, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_valid_sell_signal("sell-e2e-01", entry_price=1.1250, stop_loss=1.1750)
    result = service.process_signal(sig, current_market_price=1.1250)

    assert result.approved is True
    assert result.order is not None
    assert result.order.side == OrderSide.SELL
    assert result.order.status == OrderStatus.FILLED

    assert transport.submission_count == 1
    req = transport.submissions[0]
    assert req.order_type == 1  # SELL
    assert req.stop_loss == 1.1750
    assert req.take_profit == 1.0500

    # Directional sanity: for SELL, TP < entry < SL
    assert req.take_profit < req.price < req.stop_loss

    positions = broker.get_positions()
    assert "EURUSD" in positions
    assert positions["EURUSD"].side == OrderSide.SELL


def test_03_risk_rejections_prevent_broker_submission(temp_repo: PaperTradingRepository):
    """Step 6: Risk rejections prevent broker submission across all risk rules."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        initial_balance=100_000.0,
        audit_trail=audit,
        repository=temp_repo,
    )
    limits = RiskLimits(
        MIN_SIGNAL_CONFIDENCE=0.65,
        MAX_POSITION_RISK_PERCENT=0.5,
        MAX_DAILY_LOSS_PERCENT=1.0,
        MAX_OPEN_POSITIONS=1,
    )
    risk_engine = RiskEngine(
        limits=limits,
        kill_switch=KillSwitch(audit_trail=audit, repository=temp_repo),
    )
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    # 1. Low Confidence Rejection
    low_conf_sig = make_valid_buy_signal("low-conf", confidence=0.50)
    res1 = service.process_signal(low_conf_sig, current_market_price=1.1250)
    assert res1.approved is False
    assert res1.reason == RiskReason.CONFIDENCE_TOO_LOW.value
    assert transport.submission_count == 0

    # 2. Kill Switch Active Rejection
    risk_engine.kill_switch.activate(reason="MANUAL_TEST")
    valid_sig = make_valid_buy_signal("kill-active")
    res2 = service.process_signal(valid_sig, current_market_price=1.1250)
    assert res2.approved is False
    assert res2.reason == RiskReason.KILL_SWITCH_ACTIVE.value
    assert transport.submission_count == 0
    risk_engine.kill_switch.deactivate()

    # 3. Maximum Positions Rejection
    # Fill one position
    res3 = service.process_signal(make_valid_buy_signal("fill-first"), current_market_price=1.1250)
    assert res3.approved is True
    assert transport.submission_count == 1

    # Attempt second position when limit is 1
    res4 = service.process_signal(
        make_valid_buy_signal("fill-second", symbol="GBPUSD"), current_market_price=1.1250
    )
    assert res4.approved is False
    assert res4.reason == RiskReason.MAX_OPEN_POSITIONS_REACHED.value
    assert transport.submission_count == 1  # No new submission!


def test_04_kill_switch_persistence_and_restart(temp_repo: PaperTradingRepository):
    """Step 14: Kill switch halt persists across restarts and blocks simulated submission."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(transport=transport, repository=temp_repo)
    ks = KillSwitch(audit_trail=audit, repository=temp_repo)
    ks.activate("TEST_HALT")

    risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=ks)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    res = service.process_signal(make_valid_buy_signal("ks-block-01"), current_market_price=1.1250)
    assert res.approved is False
    assert res.reason == RiskReason.KILL_SWITCH_ACTIVE.value
    assert transport.submission_count == 0

    # Simulate restart by creating new KillSwitch from persistent repo
    ks_restarted = KillSwitch(audit_trail=audit, repository=temp_repo)
    assert ks_restarted.is_active() is True
    assert ks_restarted.reason == "TEST_HALT"

    # Deactivate and submit
    ks_restarted.deactivate()
    risk_engine_new = RiskEngine(limits=RiskLimits(), kill_switch=ks_restarted)
    service_new = TradingExecutionService(
        risk_engine=risk_engine_new,
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )
    res_after = service_new.process_signal(
        make_valid_buy_signal("ks-after-01"), current_market_price=1.1250
    )
    assert res_after.approved is True
    assert transport.submission_count == 1


def test_05_daily_loss_enforcement(temp_repo: PaperTradingRepository):
    """Step 15: Reaching daily loss limit blocks subsequent simulated broker orders."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        initial_balance=100_000.0,
        audit_trail=audit,
        repository=temp_repo,
    )
    limits = RiskLimits(MAX_DAILY_LOSS_PERCENT=1.0)
    risk_engine = RiskEngine(limits=limits, kill_switch=KillSwitch(audit, temp_repo))
    portfolio_mgr = PortfolioManager(broker=broker, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
        audit_trail=audit,
        repository=temp_repo,
    )

    # Inject controlled simulated loss of $1,200 (1.2% > 1.0% limit)
    broker._realized_pnl = -1200.0
    broker._cash_balance -= 1200.0

    sig = make_valid_buy_signal("daily-loss-check")
    res = service.process_signal(sig, current_market_price=1.1250)
    assert res.approved is False
    assert res.reason == RiskReason.DAILY_LOSS_LIMIT_EXCEEDED.value
    assert transport.submission_count == 0


def test_06_broker_validation_rejection(temp_repo: PaperTradingRepository):
    """Step 7: Orders failing broker specifications are rejected fail-closed."""
    transport = SimulatedMT5Transport()
    broker = SimulatedMT5BrokerAdapter(transport=transport, repository=temp_repo)

    # 1. Unknown symbol
    bad_sym_order = Order(
        order_id="bad-sym",
        symbol="BTCUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10_000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    try:
        broker.submit_order(bad_sym_order)
        assert False
    except BrokerValidationError as exc:
        assert "Unknown symbol" in str(exc)
    assert transport.submission_count == 0

    # 2. Volume below min_volume (0.01 lot = 1,000 units)
    bad_vol_order = Order(
        order_id="bad-vol",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,  # 0.0005 lots < 0.01
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    try:
        broker.submit_order(bad_vol_order)
        assert False
    except BrokerValidationError:
        pass
    assert transport.submission_count == 0

    # 3. Inverted SL/TP (BUY with SL > price)
    bad_sl_order = Order(
        order_id="bad-sl",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10_000.0,
        price=1.1250,
        stop_loss=1.1300,  # SL above entry for BUY!
        status=OrderStatus.VALIDATED,
    )
    try:
        broker.submit_order(bad_sl_order)
        assert False
    except BrokerValidationError:
        pass
    assert transport.submission_count == 0


def test_07_simulated_broker_rejection(temp_repo: PaperTradingRepository):
    """Step 8: Simulated broker rejection results in REJECTED order without phantom fill."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="REJECTED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        audit_trail=audit,
        repository=temp_repo,
    )
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_valid_buy_signal("reject-01")
    result = service.process_signal(sig, current_market_price=1.1250)

    assert result.approved is True
    assert result.order is not None
    assert result.order.status == OrderStatus.REJECTED
    assert "SIMULATED_BROKER_REJECTION" in (result.order.rejection_reason or "")

    # Zero positions created
    assert len(broker.get_positions()) == 0


def test_08_simulated_partial_fill(temp_repo: PaperTradingRepository):
    """Step 9: Simulated partial fill tracks filled vs remaining volume accurately."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="PARTIALLY_FILLED")
    transport.partial_fill_fraction = 0.4
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        audit_trail=audit,
        repository=temp_repo,
    )

    order = Order(
        order_id="partial-test-01",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10_000.0,  # 0.10 lots
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )

    # First fill: 40% (4,000 units = 0.04 lots)
    filled_order_1 = broker.submit_order(order, current_market_price=1.1250)
    assert filled_order_1.status == OrderStatus.PARTIALLY_FILLED
    positions = broker.get_positions()
    assert positions["EURUSD"].quantity == 4000.0

    # Second fill: remaining 60% (6,000 units = 0.06 lots)
    filled_order_2 = broker.submit_order(order, current_market_price=1.1250)
    assert filled_order_2.status == OrderStatus.FILLED
    positions_after = broker.get_positions()
    assert positions_after["EURUSD"].quantity == 10_000.0


def test_09_timeout_handling(temp_repo: PaperTradingRepository):
    """Step 10: Broker timeout fails closed without assuming execution or phantom positions."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="TIMEOUT")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        audit_trail=audit,
        repository=temp_repo,
    )
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_valid_buy_signal("timeout-01")
    result = service.process_signal(sig, current_market_price=1.1250)

    assert result.approved is False
    assert "SIMULATED_BROKER_TIMEOUT" in result.reason
    assert len(broker.get_positions()) == 0


def test_10_malformed_response_fail_closed(temp_repo: PaperTradingRepository):
    """Step 11: Malformed broker response fails closed with MT5ResponseError."""
    transport = SimulatedMT5Transport(mode="MALFORMED_RESPONSE")
    broker = SimulatedMT5BrokerAdapter(transport=transport, repository=temp_repo)

    order = Order(
        order_id="malformed-01",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10_000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    try:
        broker.submit_order(order)
        assert False
    except MT5ResponseError:
        pass
    assert len(broker.get_positions()) == 0


def test_11_duplicate_broker_acknowledgement(temp_repo: PaperTradingRepository):
    """Step 12: Duplicate broker fill acknowledgement does not double position quantity."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        audit_trail=audit,
        repository=temp_repo,
    )

    order = Order(
        order_id="dup-ack-01",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10_000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    broker.submit_order(order, current_market_price=1.1250)
    assert broker.get_positions()["EURUSD"].quantity == 10_000.0

    # Repeat submit with identical broker_execution_id
    # Force transport to reuse the same execution ID
    last_resp = transport.responses[-1]
    with patch.object(transport, "send_trade_request", return_value=last_resp):
        broker.submit_order(order, current_market_price=1.1250)

    # Quantity must remain 10,000 (not 20,000)
    assert broker.get_positions()["EURUSD"].quantity == 10_000.0


def test_12_idempotency_sequential(temp_repo: PaperTradingRepository):
    """Step 13: Repeating signal returns cached result without duplicate broker submission."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        audit_trail=audit,
        repository=temp_repo,
    )
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_valid_buy_signal("idempotent-01")
    res1 = service.process_signal(sig, current_market_price=1.1250)
    res2 = service.process_signal(sig, current_market_price=1.1250)

    assert res1.approved is True
    assert res2.approved is True
    assert res2.is_idempotent_replay is True
    assert transport.submission_count == 1
    assert broker.get_positions()["EURUSD"].quantity == res1.order.quantity


def test_13_multithreaded_concurrency_stress(temp_repo: PaperTradingRepository):
    """Step 13: 10 concurrent threads submitting identical signal produce exactly 1 submission."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        audit_trail=audit,
        repository=temp_repo,
    )
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_valid_buy_signal("concurrent-storm-01")

    def _worker():
        return service.process_signal(sig, current_market_price=1.1250)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_worker) for _ in range(10)]
        results = [f.result() for f in futures]

    assert all(r.approved is True for r in results)
    assert transport.submission_count == 1
    replays = sum(1 for r in results if r.is_idempotent_replay)
    assert replays == 9
    assert len(broker.get_positions()) == 1


def test_14_persistence_failure_rollback(temp_repo: PaperTradingRepository):
    """Step 18: Injected persistence failure triggers atomic rollback."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        initial_balance=100_000.0,
        audit_trail=audit,
        repository=temp_repo,
    )
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    sig = make_valid_buy_signal("rollback-01")

    # Inject database failure during order execution
    with patch.object(temp_repo, "save_order", side_effect=RuntimeError("DISK_FULL_SIMULATED")):
        res = service.process_signal(sig, current_market_price=1.1250)
        assert res.approved is False
        assert "EXECUTION_EXCEPTION" in res.reason

    # Snapshot restored: zero positions, balance intact
    assert len(broker.get_positions()) == 0
    assert broker._cash_balance == 100_000.0


def test_15_restart_recovery(temp_repo: PaperTradingRepository):
    """Step 19: Process restart restores deterministic state from repository."""
    audit = AuditTrail(repository=temp_repo)
    transport = SimulatedMT5Transport(mode="FILLED")
    broker = SimulatedMT5BrokerAdapter(
        transport=transport,
        initial_balance=100_000.0,
        audit_trail=audit,
        repository=temp_repo,
    )
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )

    res = service.process_signal(make_valid_buy_signal("restart-01"), current_market_price=1.1250)
    assert res.approved is True

    # Reconstruct from repository
    persisted_pos = temp_repo.get_active_positions()
    assert len(persisted_pos) == 1
    assert "EURUSD" in persisted_pos
    assert persisted_pos["EURUSD"].quantity == res.order.quantity


def test_16_replay_determinism():
    """Step 20: Deterministic replay yields identical state and matches checkpoint resume."""
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    from trading.models.order import Order, OrderSide
    from trading.replay.data_loader import MarketReplayDataLoader
    from trading.replay.engine import ReplayEngine

    base_time = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    candles = []
    p = 1.1200
    for i in range(30):
        t = base_time + timedelta(hours=4 * i)
        p += 0.0004 if i % 3 == 0 else -0.0002
        candles.append(
            {
                "timestamp": t.isoformat(),
                "open": p,
                "high": p + 0.0008,
                "low": p - 0.0008,
                "close": p + 0.0001,
                "volume": 500.0,
            }
        )
    events = MarketReplayDataLoader.load_from_dataframe(pd.DataFrame(candles))

    # Run 1: Uninterrupted run
    b1 = PaperBroker(initial_balance=100_000.0)
    e1 = ReplayEngine(events=events, paper_broker=b1)
    b1.submit_order(
        Order(
            order_id="chk_ord1",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=10_000.0,
            price=1.1200,
        )
    )
    s1 = e1.run()

    # Run 2: Checkpoint at bar 15, then resume
    b2 = PaperBroker(initial_balance=100_000.0)
    e2 = ReplayEngine(events=events, paper_broker=b2)
    b2.submit_order(
        Order(
            order_id="chk_ord2",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=10_000.0,
            price=1.1200,
        )
    )

    e2.run(max_steps=15)
    chkpt = e2.create_checkpoint()
    assert chkpt.cursor_index == 15

    e2.resume_from_checkpoint(chkpt)
    s2 = e2.run()

    assert s1["final_equity"] == s2["final_equity"]
    assert s1["cash_balance"] == s2["cash_balance"]
    assert s1["unrealized_pnl"] == s2["unrealized_pnl"]


def test_17_mt5_order_send_safety_assertion():
    """Step 21: MT5 execution boundary safety assertion: order_send() is NEVER called."""

    # Assert no order_send() attribute exists on any internal adapter or client
    from trading.adapters.mt5.client import MT5ReadOnlyClient

    client = MT5ReadOnlyClient()

    try:
        client.order_send()
        assert False
    except BrokerExecutionDisabledError as exc:
        assert "FORBIDDEN" in str(exc)

    try:
        client.order_check()
        assert False
    except BrokerExecutionDisabledError as exc:
        assert "FORBIDDEN" in str(exc)

    # Ensure SimulatedMT5Transport also lacks order_send
    transport = SimulatedMT5Transport()
    assert not hasattr(transport, "order_send")
    assert not hasattr(transport, "order_check")


def test_18_execution_disabled_invariant():
    """Step 21: MT5BrokerAdapter has execution_enabled == False and is_live == False."""
    adapter = MT5BrokerAdapter()
    assert adapter.is_live is False
    assert adapter.execution_enabled is False

    order = Order(
        order_id="ord-safety-check",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10_000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    try:
        adapter.submit_order(order)
        assert False
    except BrokerExecutionDisabledError as exc:
        assert "MT5_EXECUTION_DISABLED" in str(exc)


def test_19_readiness_exposure():
    """Step 22: System readiness exposes MT5 and simulated broker states clearly."""
    from fastapi.testclient import TestClient

    from trading.api.app import create_app
    from trading.api.dependencies import (
        TradingContext,
        reset_trading_context,
        set_trading_context,
    )

    reset_trading_context()
    real_adapter = MT5BrokerAdapter()
    ctx = TradingContext(mt5_adapter=real_adapter)
    set_trading_context(ctx)

    app = create_app()
    client_http = TestClient(app)
    resp = client_http.get("/readiness")
    assert resp.status_code == 200
    data = resp.json()

    assert data["ready"] is True
    assert data["trading_backend"] == "PAPER"
    assert data["mt5_adapter_available"] is True
    assert data["mt5_execution_enabled"] is False  # ALWAYS FALSE
    reset_trading_context()
