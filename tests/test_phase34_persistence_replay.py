"""Phase 34 Comprehensive Test Suite: Persistent Paper Trading, Market-Data Replay & Monitoring.

Tests cover all 20 formal verification requirements (A through T):
- A. Database schema & migrations
- B. Order persistence & status updates
- C. Execution persistence & TCA (gross vs net P&L, swap)
- D. Position persistence (active vs closed positions)
- E. Audit persistence & cryptographic hash chaining / tamper-evidence
- F. Idempotency persistence across restarts
- G. Transaction rollback on simulated crash
- H. Restart recovery (reconstructed equity matches persisted equity)
- I. Kill-switch recovery across restart
- J. Daily-loss recovery across restart (same day vs new UTC day)
- K. Exposure recovery across restart
- L. Replay ordering & non-monotonic validation (DataIntegrityError)
- M. Replay determinism (identical final balances)
- N. Replay checkpoint/resume determinism
- O. Stop-loss replay trigger
- P. Take-profit replay trigger & collision rule (SL executes first)
- Q. Concurrent duplicate request deduplication
- R. Persistence failure fail-closed behavior
- S. Readiness failure detection on corrupt/unreachable state
- T. Locked-test quarantine enforcement (>= 2026-02-19 12:00:00 UTC)
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from trading.api.app import create_app
from trading.api.dependencies import (
    get_trading_context,
    reset_trading_context,
)
from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.paper_broker import PaperBroker
from trading.execution.service import TradingExecutionService
from trading.models.execution import ExecutionReport
from trading.models.order import Order, OrderSide, OrderStatus
from trading.models.position import Position, PositionStatus
from trading.models.signal import Signal, SignalAction
from trading.persistence.database import PaperDatabaseManager
from trading.persistence.migrations import PaperSchemaMigrator
from trading.persistence.repository import PaperTradingRepository
from trading.portfolio.manager import PortfolioManager
from trading.replay.data_loader import (
    LOCKED_TEST_CUTOFF,
    DataIntegrityError,
    MarketReplayDataLoader,
    QuarantinedPartitionError,
)
from trading.replay.engine import ReplayEngine
from trading.replay.models import MarketEvent
from trading.risk.engine import RiskEngine
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide a path to a temporary SQLite database file."""
    return tmp_path / "paper_test.db"


@pytest.fixture
def temp_repo(temp_db_path: Path) -> Generator[PaperTradingRepository, None, None]:
    """Provide an initialized repository on a temporary file."""
    db = PaperDatabaseManager(db_path=str(temp_db_path))
    migrator = PaperSchemaMigrator(db)
    migrator.apply_all()
    repo = PaperTradingRepository(db)
    try:
        yield repo
    finally:
        db.close()


@pytest.fixture
def memory_repo() -> Generator[PaperTradingRepository, None, None]:
    """Provide an in-memory repository for fast isolated tests."""
    db = PaperDatabaseManager(db_path=":memory:")
    migrator = PaperSchemaMigrator(db)
    migrator.apply_all()
    repo = PaperTradingRepository(db)
    try:
        yield repo
    finally:
        db.close()


def make_test_signal(
    symbol: str = "EURUSD",
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.85,
    entry: float = 100.0,
    stop_loss: float = 95.0,
    client_request_id: str = "req_test_01",
) -> Signal:
    """Helper to construct valid validated signals."""
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        timestamp=datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        model_version="test_v1",
        timeframe="H4",
        expected_return=0.015,
        feature_version="v1",
        suggested_entry_price=entry,
        suggested_stop_loss=stop_loss,
        client_request_id=client_request_id,
    )


# -----------------------------------------------------------------------------
# Test A: Database Schema & Migrations
# -----------------------------------------------------------------------------


def test_a_database_schema_and_migrations(temp_db_path: Path):
    """Test A: Verify table creation, version tracking, and migration idempotency."""
    db = PaperDatabaseManager(db_path=str(temp_db_path))
    migrator = PaperSchemaMigrator(db)

    # First migration run
    applied = migrator.apply_all()
    assert applied == 1
    assert migrator.get_current_version() == 1

    # Second migration run must be idempotent (0 applied)
    applied_again = migrator.apply_all()
    assert applied_again == 0
    assert migrator.get_current_version() == 1

    # Verify tables exist
    with db.connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
        table_names = {t["name"] for t in tables}

    required_tables = {
        "paper_schema_migrations",
        "paper_orders",
        "paper_executions",
        "paper_positions",
        "paper_audit_events",
        "paper_account_snapshots",
        "paper_risk_state",
        "paper_idempotency_records",
    }
    for tbl in required_tables:
        assert tbl in table_names, f"Missing table: {tbl}"

    db.close()


# -----------------------------------------------------------------------------
# Test B: Order Persistence & Status Updates
# -----------------------------------------------------------------------------


def test_b_order_persistence_and_status_updates(memory_repo: PaperTradingRepository):
    """Test B: Save orders, update status transitions, and query history."""
    now = datetime.now(timezone.utc)
    order = Order(
        order_id="ord-b1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=10_000.0,
        price=1.1000,
        status=OrderStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    memory_repo.save_order(order)

    # Query back
    fetched = memory_repo.get_order("ord-b1")
    assert fetched is not None
    assert fetched.symbol == "EURUSD"
    assert fetched.status == OrderStatus.PENDING

    # Transition to FILLED
    order.transition_to(OrderStatus.FILLED)
    order.fill_price = 1.1002
    order.fill_timestamp = now + timedelta(seconds=1)
    order.updated_at = now + timedelta(seconds=1)
    memory_repo.save_order(order)

    # Re-query
    updated = memory_repo.get_order("ord-b1")
    assert updated is not None
    assert updated.status == OrderStatus.FILLED
    assert updated.fill_price == 1.1002
    assert updated.fill_timestamp is not None

    all_orders = memory_repo.get_all_orders()
    assert len(all_orders) == 1
    assert all_orders[0].order_id == "ord-b1"


# -----------------------------------------------------------------------------
# Test C: Execution Persistence, TCA Breakdown & Swap
# -----------------------------------------------------------------------------


def test_c_execution_persistence_tca_and_swap(memory_repo: PaperTradingRepository):
    """Test C: Persist execution reports with full TCA breakdown and swap."""
    now = datetime.now(timezone.utc)
    report = ExecutionReport(
        execution_id="exec-c1",
        order_id="ord-c1",
        client_request_id="req-c1",
        timestamp=now,
        symbol="EURUSD",
        side=OrderSide.BUY,
        requested_price=1.1000,
        executed_price=1.1001,
        quantity=50_000.0,
        spread=0.0001,
        slippage=0.0001,
        commission=1.50,
        swap=-0.75,
        gross_pnl=120.00,
        net_pnl=117.75,
        execution_status=OrderStatus.FILLED,
        rejection_reason=None,
    )
    memory_repo.save_execution(report)

    all_execs = memory_repo.get_all_executions()
    assert len(all_execs) == 1
    persisted = all_execs[0]
    assert persisted.execution_id == "exec-c1"
    assert persisted.swap == -0.75
    assert persisted.commission == 1.50
    assert persisted.gross_pnl == 120.00
    assert persisted.net_pnl == 117.75
    assert persisted.execution_status == OrderStatus.FILLED


# -----------------------------------------------------------------------------
# Test D: Position Persistence (Active vs Closed)
# -----------------------------------------------------------------------------


def test_d_position_persistence_active_and_closed(memory_repo: PaperTradingRepository):
    """Test D: Position lifecycle persistence, active query, and closed archive query."""
    now = datetime.now(timezone.utc)
    pos = Position(
        position_id="pos-d1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=10_000.0,
        entry_price=1.1000,
        current_price=1.1020,
        stop_loss=1.0950,
        take_profit=1.1100,
        entry_timestamp=now,
        unrealized_pnl=20.0,
        realized_pnl=0.0,
        gross_pnl=0.0,
        costs=1.5,
        net_pnl=-1.5,
        status=PositionStatus.OPEN,
    )
    memory_repo.save_position(pos)

    # Active positions must contain pos
    active = memory_repo.get_active_positions()
    assert "EURUSD" in active
    assert active["EURUSD"].unrealized_pnl == 20.0
    assert len(memory_repo.get_closed_positions()) == 0

    # Close the position
    pos.status = PositionStatus.CLOSED
    pos.exit_price = 1.1030
    pos.exit_timestamp = now + timedelta(hours=2)
    pos.realized_pnl = 30.0
    pos.gross_pnl = 30.0
    pos.net_pnl = 28.5
    pos.unrealized_pnl = 0.0
    memory_repo.save_position(pos)

    # Active positions must now be empty
    active_after = memory_repo.get_active_positions()
    assert "EURUSD" not in active_after

    # Closed positions must contain pos
    closed = memory_repo.get_closed_positions()
    assert len(closed) == 1
    assert closed[0].position_id == "pos-d1"
    assert closed[0].exit_price == 1.1030
    assert closed[0].net_pnl == 28.5


# -----------------------------------------------------------------------------
# Test E: Audit Persistence & Cryptographic Hash Chaining
# -----------------------------------------------------------------------------


def test_e_audit_persistence_and_hash_chaining(memory_repo: PaperTradingRepository):
    """Test E: Append-only audit events, chronological sequence, and tamper detection."""
    audit = AuditTrail(repository=memory_repo)

    # Record 3 events
    audit.record(
        event_type=AuditEventType.SIGNAL_RECEIVED,
        symbol="EURUSD",
        details={"model": "v1"},
    )
    ev2 = audit.record(
        event_type=AuditEventType.RISK_ACCEPTED,
        symbol="EURUSD",
        details={"approved": True},
    )
    audit.record(
        event_type=AuditEventType.ORDER_FILLED,
        symbol="EURUSD",
        details={"fill_price": 1.1000},
    )

    # Verify query by event type
    risk_events = memory_repo.get_audit_events(event_type=AuditEventType.RISK_ACCEPTED.value)
    assert len(risk_events) == 1
    assert risk_events[0].event_type == AuditEventType.RISK_ACCEPTED

    # Verify integrity passes
    is_valid, err = memory_repo.verify_audit_trail_integrity()
    assert is_valid is True
    assert err is None

    # Simulate database tampering (modify details_json of ev2 directly in SQL)
    with memory_repo.db.connection() as conn:
        conn.execute(
            "UPDATE paper_audit_events SET details_json = ? WHERE event_id = ?;",
            ('{"tampered": true}', ev2.event_id),
        )

    # Verify integrity check catches the tampering
    tampered_valid, tamper_err = memory_repo.verify_audit_trail_integrity()
    assert tampered_valid is False
    assert "Tamper detected" in (tamper_err or "")


# -----------------------------------------------------------------------------
# Test F: Idempotency Persistence Across Restarts
# -----------------------------------------------------------------------------


def test_f_idempotency_persistence_across_restarts(temp_db_path: Path):
    """Test F: Submit signal, crash/restart service, resubmit same signal, verify deduplication."""
    # Run 1: Initial Service Instance
    db1 = PaperDatabaseManager(db_path=str(temp_db_path))
    PaperSchemaMigrator(db1).apply_all()
    repo1 = PaperTradingRepository(db1)

    broker1 = PaperBroker(initial_balance=100_000.0, repository=repo1)
    risk1 = RiskEngine(limits=RiskLimits())
    audit1 = AuditTrail(repository=repo1)
    pm1 = PortfolioManager(broker=broker1, repository=repo1)
    service1 = TradingExecutionService(
        paper_broker=broker1,
        risk_engine=risk1,
        audit_trail=audit1,
        portfolio_manager=pm1,
        repository=repo1,
    )

    signal = make_test_signal(client_request_id="req_restart_dedup_01")
    res1 = service1.process_signal(signal)
    assert res1.approved is True
    assert res1.is_idempotent_replay is False
    assert res1.order is not None
    orig_order_id = res1.order.order_id

    # Simulate shutdown
    db1.close()

    # Run 2: Re-open database with completely new service instances
    db2 = PaperDatabaseManager(db_path=str(temp_db_path))
    repo2 = PaperTradingRepository(db2)

    broker2 = PaperBroker(initial_balance=100_000.0, repository=repo2)
    broker2.recover_from_repository()
    risk2 = RiskEngine(limits=RiskLimits())
    audit2 = AuditTrail(repository=repo2)
    pm2 = PortfolioManager(broker=broker2, repository=repo2)
    service2 = TradingExecutionService(
        paper_broker=broker2,
        risk_engine=risk2,
        audit_trail=audit2,
        portfolio_manager=pm2,
        repository=repo2,
    )

    # Resubmit identical signal
    res2 = service2.process_signal(signal)
    assert res2.approved is True
    assert res2.is_idempotent_replay is True
    assert res2.order is not None
    assert res2.order.order_id == orig_order_id
    assert len(broker2.get_all_orders()) == 1  # No duplicate order created

    db2.close()


# -----------------------------------------------------------------------------
# Test G: Transaction Rollback on Simulated Crash
# -----------------------------------------------------------------------------


def test_g_transaction_rollback_on_simulated_crash(memory_repo: PaperTradingRepository):
    """Test G: Transaction failure midway through mutations rolls back all tables."""
    now = datetime.now(timezone.utc)
    order = Order(
        order_id="ord-g1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=10_000.0,
        price=1.1000,
        status=OrderStatus.PENDING,
        created_at=now,
    )

    # Begin transaction, insert order, then simulate crash exception
    try:
        with memory_repo.db.transaction() as conn:
            memory_repo.save_order(order, conn=conn)
            # Verify order is visible inside transaction
            assert memory_repo.get_order("ord-g1", conn=conn) is not None
            # Simulate crash midway
            raise RuntimeError("SIMULATED_SYSTEM_CRASH_MIDWAY")
    except RuntimeError:
        pass

    # Verify order was NOT committed
    committed_order = memory_repo.get_order("ord-g1")
    assert committed_order is None


# -----------------------------------------------------------------------------
# Test H: Restart Recovery (Portfolio Equity Matches)
# -----------------------------------------------------------------------------


def test_h_restart_recovery_portfolio_equity(temp_db_path: Path):
    """Test H: System reconstructs exact balance, positions, and equity from persistence."""
    db1 = PaperDatabaseManager(db_path=str(temp_db_path))
    PaperSchemaMigrator(db1).apply_all()
    repo1 = PaperTradingRepository(db1)

    broker1 = PaperBroker(initial_balance=100_000.0, repository=repo1)
    order = Order(
        order_id="ord-h1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=20_000.0,
        price=1.1000,
        status=OrderStatus.PENDING,
    )
    broker1.submit_order(order, current_market_price=1.1000)
    broker1.update_market_price("EURUSD", price=1.1050)

    acc1 = broker1.get_account()
    expected_equity = acc1.equity
    expected_cash = acc1.cash_balance
    expected_unrealized = acc1.unrealized_pnl
    expected_realized = acc1.realized_pnl
    db1.close()

    # Reconstruct from new instance
    db2 = PaperDatabaseManager(db_path=str(temp_db_path))
    repo2 = PaperTradingRepository(db2)
    broker2 = PaperBroker(initial_balance=100_000.0, repository=repo2)
    broker2.recover_from_repository()

    acc2 = broker2.get_account()
    assert acc2.equity == expected_equity
    assert acc2.cash_balance == expected_cash
    assert acc2.unrealized_pnl == expected_unrealized
    assert acc2.realized_pnl == expected_realized
    assert "EURUSD" in acc2.positions
    assert acc2.positions["EURUSD"].current_price == 1.1050

    db2.close()


# -----------------------------------------------------------------------------
# Test I: Kill-Switch Recovery Across Restart
# -----------------------------------------------------------------------------


def test_i_kill_switch_recovery_across_restart(temp_db_path: Path):
    """Test I: Activated kill switch remains active with reason intact after restart."""
    db1 = PaperDatabaseManager(db_path=str(temp_db_path))
    PaperSchemaMigrator(db1).apply_all()
    repo1 = PaperTradingRepository(db1)

    ks1 = KillSwitch(repository=repo1)
    ks1.activate(reason="MANUAL_HALT_SAFETY_TEST")
    assert ks1.is_active() is True
    db1.close()

    # Reconnect
    db2 = PaperDatabaseManager(db_path=str(temp_db_path))
    repo2 = PaperTradingRepository(db2)
    ks2 = KillSwitch(repository=repo2)
    ks2.recover_from_repository()

    assert ks2.is_active() is True
    assert ks2.reason == "MANUAL_HALT_SAFETY_TEST"

    db2.close()


# -----------------------------------------------------------------------------
# Test J: Daily-Loss Recovery Across Restart (Same Day vs New Day)
# -----------------------------------------------------------------------------


def test_j_daily_loss_recovery_across_restart(temp_db_path: Path):
    """Test J: Daily loss accumulator restored on same UTC day; resets on new UTC day."""
    db1 = PaperDatabaseManager(db_path=str(temp_db_path))
    PaperSchemaMigrator(db1).apply_all()
    repo1 = PaperTradingRepository(db1)

    today = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc).date()
    broker1 = PaperBroker(initial_balance=100_000.0, repository=repo1)
    pm1 = PortfolioManager(broker=broker1, repository=repo1, today_utc=today)

    # Simulate daily realized loss of $1,500
    broker1._realized_pnl = -1500.0
    repo1.save_account_snapshot(account=broker1.get_account(), daily_pnl=-1500.0)
    pm1._sync_to_repository()
    db1.close()

    # Scenario 1: Restart on SAME UTC day -> loss accumulator restored
    db2 = PaperDatabaseManager(db_path=str(temp_db_path))
    repo2 = PaperTradingRepository(db2)
    broker2 = PaperBroker(initial_balance=100_000.0, repository=repo2)
    broker2.recover_from_repository()
    pm2 = PortfolioManager(broker=broker2, repository=repo2, today_utc=today)
    pm2.recover_from_repository()

    assert pm2.daily_start_equity == 100_000.0
    assert pm2.get_daily_loss_percent() == 1.5

    # Scenario 2: Restart on NEXT UTC day -> daily loss resets
    tomorrow = today + timedelta(days=1)
    pm3 = PortfolioManager(broker=broker2, repository=repo2, today_utc=tomorrow)
    pm3.recover_from_repository()

    # Equity carried over becomes new start equity, daily loss resets to 0.0
    assert pm3.daily_start_equity == broker2.get_account().equity
    assert pm3.get_daily_loss_percent() == 0.0

    db2.close()


# -----------------------------------------------------------------------------
# Test K: Exposure Recovery Across Restart
# -----------------------------------------------------------------------------


def test_k_exposure_recovery_across_restart(temp_db_path: Path):
    """Test K: Portfolio gross exposure correctly reconstructed from persistent positions."""
    db1 = PaperDatabaseManager(db_path=str(temp_db_path))
    PaperSchemaMigrator(db1).apply_all()
    repo1 = PaperTradingRepository(db1)

    broker1 = PaperBroker(initial_balance=100_000.0, repository=repo1)
    pm1 = PortfolioManager(broker=broker1, repository=repo1)

    # Submit 2 positions
    order1 = Order(
        order_id="ord-k1",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=20_000.0,
        price=1.1000,
        status=OrderStatus.PENDING,
    )
    order2 = Order(
        order_id="ord-k2",
        symbol="GBPUSD",
        side=OrderSide.BUY,
        quantity=10_000.0,
        price=1.3000,
        status=OrderStatus.PENDING,
    )
    broker1.submit_order(order1, current_market_price=1.1000)
    broker1.submit_order(order2, current_market_price=1.3000)

    # Initial exposure: (20000*1.10 + 10000*1.30) / 100000 = 35000 / 100000 = 35.0%
    expected_exp = pm1.get_total_exposure_percent()
    assert abs(expected_exp - 35.0) < 0.05
    db1.close()

    # Reconstruct
    db2 = PaperDatabaseManager(db_path=str(temp_db_path))
    repo2 = PaperTradingRepository(db2)
    broker2 = PaperBroker(initial_balance=100_000.0, repository=repo2)
    broker2.recover_from_repository()
    pm2 = PortfolioManager(broker=broker2, repository=repo2)
    pm2.recover_from_repository()

    recovered_exp = pm2.get_total_exposure_percent()
    assert abs(recovered_exp - expected_exp) < 0.001

    db2.close()


# -----------------------------------------------------------------------------
# Test L: Replay Ordering & Non-Monotonic Validation
# -----------------------------------------------------------------------------


def test_l_replay_ordering_and_non_monotonic_validation():
    """Test L: Replay rejects out-of-order or duplicate timestamps with DataIntegrityError."""
    # Case 1: Non-monotonic (t2 < t1)
    df_non_monotonic = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-01 10:00:00",
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
            },
            {
                "timestamp": "2025-01-01 09:00:00",
                "open": 1.105,
                "high": 1.11,
                "low": 1.10,
                "close": 1.108,
            },
        ]
    )
    with pytest.raises(DataIntegrityError) as exc_nm:
        MarketReplayDataLoader.load_from_dataframe(df_non_monotonic)
    assert "Non-monotonic timestamp" in str(exc_nm.value)

    # Case 2: Duplicate timestamps (t2 == t1)
    df_duplicate = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-01 10:00:00",
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
            },
            {
                "timestamp": "2025-01-01 10:00:00",
                "open": 1.105,
                "high": 1.11,
                "low": 1.10,
                "close": 1.108,
            },
        ]
    )
    with pytest.raises(DataIntegrityError) as exc_dup:
        MarketReplayDataLoader.load_from_dataframe(df_duplicate)
    assert "Duplicate timestamp" in str(exc_dup.value)


# -----------------------------------------------------------------------------
# Test M: Replay Determinism
# -----------------------------------------------------------------------------


def test_m_replay_determinism():
    """Test M: Replaying identical bars with scheduled orders produces byte-identical results."""
    # Generate 20 candles
    base_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(20):
        t = base_time + timedelta(hours=4 * i)
        price += 0.5 if i % 2 == 0 else -0.3
        candles.append(
            {
                "timestamp": t.isoformat(),
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.2,
                "volume": 1000.0,
            }
        )
    events = MarketReplayDataLoader.load_from_dataframe(pd.DataFrame(candles))

    def run_simulation():
        broker = PaperBroker(initial_balance=100_000.0)
        risk = RiskEngine(limits=RiskLimits())
        audit = AuditTrail()
        pm = PortfolioManager(broker=broker)
        service = TradingExecutionService(
            paper_broker=broker,
            risk_engine=risk,
            audit_trail=audit,
            portfolio_manager=pm,
        )
        engine = ReplayEngine(events=events, paper_broker=broker, execution_service=service)
        # Schedule an order at bar 5
        sig = make_test_signal(client_request_id=f"det_{uuid.uuid4().hex[:6]}")
        engine.schedule_order(trigger_time=events[5].timestamp, signal=sig)
        summary = engine.run()
        return summary, broker.get_all_orders()

    summary1, orders1 = run_simulation()
    summary2, orders2 = run_simulation()

    assert summary1["final_equity"] == summary2["final_equity"]
    assert summary1["cash_balance"] == summary2["cash_balance"]
    assert summary1["realized_pnl"] == summary2["realized_pnl"]
    assert len(orders1) == len(orders2) == 1
    assert orders1[0].quantity == orders2[0].quantity
    assert orders1[0].fill_price == orders2[0].fill_price


# -----------------------------------------------------------------------------
# Test N: Replay Checkpoint & Resume Determinism
# -----------------------------------------------------------------------------


def test_n_replay_checkpoint_resume_determinism():
    """Test N: Checkpoint at bar N and resume produces identical state to uninterrupted run."""
    base_time = datetime(2025, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    candles = []
    p = 1.2000
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
    # Open position at start
    b1.submit_order(
        Order(
            order_id="chk_ord1",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=10_000.0,
            price=1.2000,
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
            price=1.2000,
        )
    )

    # Run first 15 steps
    e2.run(max_steps=15)
    chkpt = e2.create_checkpoint()
    assert chkpt.cursor_index == 15

    # Resume remainder
    e2.resume_from_checkpoint(chkpt)
    s2 = e2.run()

    assert s1["final_equity"] == s2["final_equity"]
    assert s1["cash_balance"] == s2["cash_balance"]
    assert s1["unrealized_pnl"] == s2["unrealized_pnl"]


# -----------------------------------------------------------------------------
# Test O: Stop-Loss Replay Trigger
# -----------------------------------------------------------------------------


def test_o_stop_loss_replay_trigger():
    """Test O: Replaying a candle that crosses Stop Loss executes position exit at SL price."""
    b = PaperBroker(initial_balance=100_000.0)
    # Open LONG EURUSD at 1.1000 with SL at 1.0950, TP at 1.1100
    b.submit_order(
        Order(
            order_id="ord-sl",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=10_000.0,
            price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
        )
    )
    assert "EURUSD" in b.get_positions()

    # Create market bar where Low drops through Stop Loss (Low=1.0940, High=1.0980)
    sl_event = MarketEvent(
        symbol="EURUSD",
        timestamp=datetime(2025, 3, 1, 10, 0, 0, tzinfo=timezone.utc),
        open=1.0970,
        high=1.0980,
        low=1.0940,
        close=1.0960,
    )
    engine = ReplayEngine(events=[sl_event], paper_broker=b)
    engine.step()

    # Position must be closed at SL price (1.0950)
    assert "EURUSD" not in b.get_positions()
    closed = b.get_closed_positions()
    assert len(closed) == 1
    assert closed[0].exit_price == 1.0950
    assert engine.get_summary()["brackets_triggered"] == 1


# -----------------------------------------------------------------------------
# Test P: Take-Profit Replay Trigger & Collision Rule (SL First)
# -----------------------------------------------------------------------------


def test_p_take_profit_replay_trigger_and_collision_rule():
    """Test P: TP executes when hit; when BOTH SL and TP hit in same bar, SL executes first."""
    # Case 1: TP triggered cleanly
    b1 = PaperBroker(initial_balance=100_000.0)
    b1.submit_order(
        Order(
            order_id="ord-tp",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=10_000.0,
            price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1050,
        )
    )
    tp_event = MarketEvent(
        symbol="EURUSD",
        timestamp=datetime(2025, 3, 1, 10, 0, 0, tzinfo=timezone.utc),
        open=1.1010,
        high=1.1060,  # High crosses TP 1.1050
        low=1.0990,  # Low remains safely above SL 1.0950
        close=1.1040,
    )
    e1 = ReplayEngine(events=[tp_event], paper_broker=b1)
    e1.step()
    assert "EURUSD" not in b1.get_positions()
    assert b1.get_closed_positions()[0].exit_price == 1.1050

    # Case 2: Bracket Collision (Both SL and TP touched in same candle)
    # Conservative capital preservation rule: STOP-LOSS executes FIRST.
    b2 = PaperBroker(initial_balance=100_000.0)
    b2.submit_order(
        Order(
            order_id="ord-collision",
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=10_000.0,
            price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1050,
        )
    )
    collision_event = MarketEvent(
        symbol="EURUSD",
        timestamp=datetime(2025, 3, 1, 11, 0, 0, tzinfo=timezone.utc),
        open=1.1000,
        high=1.1060,  # Crosses TP 1.1050
        low=1.0940,  # Crosses SL 1.0950
        close=1.1010,
    )
    e2 = ReplayEngine(events=[collision_event], paper_broker=b2)
    e2.step()

    # Position must be closed at Stop-Loss price, preserving capital
    assert "EURUSD" not in b2.get_positions()
    closed_pos = b2.get_closed_positions()
    assert len(closed_pos) == 1
    assert closed_pos[0].exit_price == 1.0950  # Must be SL price, NOT TP


# -----------------------------------------------------------------------------
# Test Q: Concurrent Duplicate Request Deduplication
# -----------------------------------------------------------------------------


def test_q_concurrent_duplicate_request_deduplication(memory_repo: PaperTradingRepository):
    """Test Q: 10 concurrent requests with identical client_request_id result in 1 execution."""
    broker = PaperBroker(initial_balance=100_000.0, repository=memory_repo)
    risk = RiskEngine(limits=RiskLimits())
    audit = AuditTrail(repository=memory_repo)
    pm = PortfolioManager(broker=broker, repository=memory_repo)
    service = TradingExecutionService(
        paper_broker=broker,
        risk_engine=risk,
        audit_trail=audit,
        portfolio_manager=pm,
        repository=memory_repo,
    )

    req_id = f"concurrent_req_{uuid.uuid4().hex[:8]}"
    signal = make_test_signal(client_request_id=req_id)

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(service.process_signal, signal) for _ in range(10)]
        for f in futures:
            results.append(f.result())

    # All 10 returned approved
    assert all(r.approved for r in results)
    # Exactly one non-replay execution, exactly 9 idempotent replays
    non_replay_count = sum(1 for r in results if not r.is_idempotent_replay)
    replay_count = sum(1 for r in results if r.is_idempotent_replay)
    assert non_replay_count == 1
    assert replay_count == 9

    # All returned identical order ID
    order_ids = {r.order.order_id for r in results if r.order}
    assert len(order_ids) == 1

    # Exactly 1 order in broker
    assert len(broker.get_all_orders()) == 1


# -----------------------------------------------------------------------------
# Test R: Persistence Failure Fail-Closed Behavior
# -----------------------------------------------------------------------------


def test_r_persistence_failure_fail_closed_behavior():
    """Test R: Simulated DB failure rejects order and maintains clean in-memory broker state."""
    db = PaperDatabaseManager(db_path=":memory:")
    PaperSchemaMigrator(db).apply_all()
    repo = PaperTradingRepository(db)

    broker = PaperBroker(initial_balance=100_000.0, repository=repo)
    risk = RiskEngine(limits=RiskLimits())
    audit = AuditTrail(repository=repo)
    pm = PortfolioManager(broker=broker, repository=repo)
    service = TradingExecutionService(
        paper_broker=broker,
        risk_engine=risk,
        audit_trail=audit,
        portfolio_manager=pm,
        repository=repo,
    )

    # Close DB connection to simulate I/O or network partition
    db.close()

    signal = make_test_signal(client_request_id="req_fail_closed_01")
    res = service.process_signal(signal)

    # Must fail-closed: rejected, not approved
    assert res.approved is False
    assert "EXECUTION_EXCEPTION" in (res.reason or "")
    assert len(broker.get_all_orders()) == 0


# -----------------------------------------------------------------------------
# Test S: Readiness Failure Detection on Corrupt State
# -----------------------------------------------------------------------------


def test_s_readiness_failure_detection_on_corrupt_state():
    """Test S: Corrupt/unreachable database causes /readiness to return HTTP 503."""
    reset_trading_context(initial_balance=100_000.0)
    ctx = get_trading_context()
    app = create_app()

    with TestClient(app) as client:
        # Healthy first
        res_ok = client.get("/readiness")
        assert res_ok.status_code == 200
        assert res_ok.json()["ready"] is True

        # Now simulate persistence failure / unreachable DB
        ctx.persistence_healthy = False
        ctx.recovery_error = "SIMULATED_DB_CORRUPTION"
        if ctx.database_manager:
            ctx.database_manager.close()

        res_unready = client.get("/readiness")
        assert res_unready.status_code == 503
        data = res_unready.json()
        assert data["ready"] is False
        assert data["persistence_healthy"] is False


# -----------------------------------------------------------------------------
# Test T: Locked-Test Quarantine Enforcement
# -----------------------------------------------------------------------------


def test_t_locked_test_quarantine_enforcement():
    """Test T: Replay engine refuses to load data overlapping with locked test partition."""
    # Bar at or after cutoff 2026-02-19 12:00:00 UTC
    quarantined_bars = pd.DataFrame(
        [
            {
                "timestamp": "2026-02-19 12:00:00",
                "open": 1.0850,
                "high": 1.0870,
                "low": 1.0840,
                "close": 1.0860,
            }
        ]
    )
    with pytest.raises(QuarantinedPartitionError) as exc_info:
        MarketReplayDataLoader.load_from_dataframe(quarantined_bars)

    assert "Quarantined partition violation" in str(exc_info.value)

    # Future date (2026-03-01)
    future_bars = pd.DataFrame(
        [
            {
                "timestamp": "2026-03-01 00:00:00",
                "open": 1.0850,
                "high": 1.0870,
                "low": 1.0840,
                "close": 1.0860,
            }
        ]
    )
    with pytest.raises(QuarantinedPartitionError):
        MarketReplayDataLoader.load_from_dataframe(future_bars)

    # Valid pre-cutoff bar succeeds
    valid_bars = pd.DataFrame(
        [
            {
                "timestamp": "2026-02-19 11:59:59",
                "open": 1.0850,
                "high": 1.0870,
                "low": 1.0840,
                "close": 1.0860,
            }
        ]
    )
    events = MarketReplayDataLoader.load_from_dataframe(valid_bars)
    assert len(events) == 1
    assert events[0].timestamp < LOCKED_TEST_CUTOFF
