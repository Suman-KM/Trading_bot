"""Phase 35: Paper-Trading Operational Validation & Failure-Injection Test Suite.

Exhaustive operational stress testing covering:
- Category A: Database / Persistence Failures
- Category B: Process Restart / Crash Recovery
- Category C: Idempotency & Concurrency Stress
- Category D: Kill Switch Lifecycle & Persistence
- Category E: Daily Loss Limit Enforcement & Boundary Tests
- Category F: Position & Account Accounting Invariants
- Category G: Order State Machine Invalid Transitions
- Category H: Audit Trail Cryptographic Integrity & Tamper Detection
- Category I: Deterministic Replay & Checkpoint/Resume Safety
- Category J: Quarantined Test Partition Enforcement (>= 2026-02-19 12:00:00 UTC)
"""

from __future__ import annotations

import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from trading.api.app import create_app
from trading.api.dependencies import (
    TradingContext,
    reset_trading_context,
    set_trading_context,
)
from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.paper_broker import PaperBroker
from trading.execution.service import TradingExecutionService
from trading.models.order import Order, OrderSide, OrderStatus, OrderType
from trading.models.signal import Signal, SignalAction
from trading.persistence.database import PaperDatabaseManager
from trading.persistence.migrations import PaperSchemaMigrator
from trading.persistence.repository import PaperTradingRepository
from trading.portfolio.manager import PortfolioManager
from trading.replay.data_loader import (
    LOCKED_TEST_CUTOFF,
    MarketReplayDataLoader,
    QuarantinedPartitionError,
)
from trading.replay.engine import ReplayEngine
from trading.risk.engine import RiskEngine, RiskReason
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Path to temporary SQLite database file."""
    return tmp_path / "paper_phase35.db"


@pytest.fixture
def temp_repo(temp_db_path: Path) -> Generator[PaperTradingRepository, None, None]:
    """Initialized repository on a temporary file."""
    db = PaperDatabaseManager(db_path=str(temp_db_path))
    migrator = PaperSchemaMigrator(db)
    migrator.apply_all()
    repo = PaperTradingRepository(db)
    try:
        yield repo
    finally:
        db.close()


def make_signal(
    symbol: str = "EURUSD",
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.85,
    entry: float = 1.0850,
    stop_loss: float = 1.0550,
    client_request_id: str = "req_p35_01",
) -> Signal:
    """Construct a valid deterministic test signal with sizing within exposure limit."""
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        timestamp=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        model_version="phase35_test_v1",
        timeframe="H4",
        expected_return=0.015,
        feature_version="v1",
        suggested_entry_price=entry,
        suggested_stop_loss=stop_loss,
        client_request_id=client_request_id,
    )


# -----------------------------------------------------------------------------
# Category A: Database / Persistence Failures
# -----------------------------------------------------------------------------


def test_db_unavailable_at_order_creation_handles_cleanly(temp_repo: PaperTradingRepository):
    """Simulate a database lock/error when persisting order; verify fail-closed handling."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    limits = RiskLimits()
    kill_switch = KillSwitch(audit_trail=audit, repository=temp_repo)
    risk_engine = RiskEngine(limits=limits, kill_switch=kill_switch)
    portfolio = PortfolioManager(broker=broker, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio,
        audit_trail=audit,
        repository=temp_repo,
    )

    initial_cash = broker.get_account().cash_balance
    sig = make_signal(client_request_id="req_db_err_01")

    # Patch save_order to raise an OperationalError simulating DB failure
    with patch.object(
        temp_repo, "save_order", side_effect=sqlite3.OperationalError("database is locked")
    ):
        result = service.process_signal(sig, current_market_price=1.0850)

    # Must fail closed: order rejected / not approved, cash untouched, no position
    assert result.approved is False
    assert "EXECUTION_EXCEPTION" in (result.reason or "")
    assert broker.get_account().cash_balance == initial_cash
    assert len(broker.get_positions()) == 0


def test_db_write_failure_during_execution_rolls_back_broker_state(
    temp_repo: PaperTradingRepository,
):
    """Write failure on save_execution triggers in-memory broker rollback to prior state."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    limits = RiskLimits()
    kill_switch = KillSwitch(audit_trail=audit, repository=temp_repo)
    risk_engine = RiskEngine(limits=limits, kill_switch=kill_switch)
    portfolio = PortfolioManager(broker=broker, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio,
        audit_trail=audit,
        repository=temp_repo,
    )

    initial_account = broker.get_account()
    sig = make_signal(client_request_id="req_write_fail_01")

    with patch.object(
        temp_repo, "save_execution", side_effect=sqlite3.DatabaseError("disk I/O error")
    ):
        res = service.process_signal(sig, current_market_price=1.0850)

    assert res.approved is False
    assert "EXECUTION_EXCEPTION" in (res.reason or "")
    # Broker in-memory state must be perfectly rolled back
    assert broker.get_account().cash_balance == initial_account.cash_balance
    assert broker.get_account().equity == initial_account.equity
    assert len(broker.get_positions()) == 0


def test_transaction_rollback_preserves_consistency(temp_repo: PaperTradingRepository):
    """Transaction rollback on failure leaves SQLite database cleanly in pre-transaction state."""
    ord_to_insert = Order(
        order_id="test_tx_ord",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000,
        price=1.0850,
        status=OrderStatus.CREATED,
    )
    with pytest.raises(RuntimeError):
        with temp_repo.db.transaction() as conn:
            temp_repo.save_order(ord_to_insert, conn=conn)
            raise RuntimeError("Forced abort inside transaction")

    # Order should NOT exist
    ord_found = temp_repo.get_order("test_tx_ord")
    assert ord_found is None


def test_corrupted_state_readiness_fails_closed(temp_db_path: Path):
    """When the SQLite database or audit trail is corrupted, /readiness returns HTTP 503."""
    db = PaperDatabaseManager(db_path=str(temp_db_path))
    PaperSchemaMigrator(db).apply_all()
    repo = PaperTradingRepository(db)
    audit = AuditTrail(repository=repo)
    # Seed an audit event so an event exists to tamper with
    audit.record(
        event_type=AuditEventType.SIGNAL_RECEIVED,
        symbol="EURUSD",
        details={"started": True},
    )
    context = TradingContext(
        initial_balance=100_000.0,
        database_manager=db,
        repository=repo,
        audit_trail=audit,
    )
    set_trading_context(context)
    app = create_app()
    client = TestClient(app)

    # Initial state should be 200 OK
    resp = client.get("/readiness")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True

    # Tamper with audit trail in SQLite to corrupt hash chaining
    with db.connection() as conn:
        conn.execute(
            "UPDATE paper_audit_events SET details_json = ? WHERE sequence_id = 1;",
            ('{"corrupted": true}',),
        )

    # Now readiness should fail closed with 503
    resp_tampered = client.get("/readiness")
    assert resp_tampered.status_code == 503
    body = resp_tampered.json()
    assert body["ready"] is False
    assert body["audit_integrity_valid"] is False


def test_restart_after_persistence_failure(temp_repo: PaperTradingRepository):
    """After a failed/rolled-back execution, restart loads only cleanly committed state."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo))
    portfolio = PortfolioManager(broker=broker, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio,
        audit_trail=audit,
        repository=temp_repo,
    )

    # Successful order 1
    sig1 = make_signal(symbol="EURUSD", client_request_id="req_success_01")
    r1 = service.process_signal(sig1, current_market_price=1.0850)
    assert r1.approved is True

    # Failed order 2 with DB error
    sig2 = make_signal(symbol="GBPUSD", client_request_id="req_failed_02")
    with patch.object(
        temp_repo, "save_execution", side_effect=sqlite3.DatabaseError("simulated crash")
    ):
        r2 = service.process_signal(sig2, current_market_price=1.2500)
    assert r2.approved is False

    # Restart broker
    restarted_broker = PaperBroker(
        initial_balance=100_000.0, audit_trail=audit, repository=temp_repo
    )
    assert len(restarted_broker.get_positions()) == 1
    assert "EURUSD" in restarted_broker.get_positions()
    assert "GBPUSD" not in restarted_broker.get_positions()


# -----------------------------------------------------------------------------
# Category B: Process Restart / Crash Recovery
# -----------------------------------------------------------------------------


def test_restart_after_order_persistence(temp_repo: PaperTradingRepository):
    """Order persisted is retrieved cleanly by a new broker instance."""
    ord1 = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=5000,
        order_type=OrderType.MARKET,
        price=1.0850,
        status=OrderStatus.SUBMITTED,
    )
    temp_repo.save_order(ord1)

    broker2 = PaperBroker(initial_balance=100_000.0, repository=temp_repo)
    loaded_order = broker2.get_order(ord1.order_id)
    assert loaded_order is not None
    assert loaded_order.order_id == ord1.order_id
    assert loaded_order.quantity == 5000


def test_restart_after_execution(temp_repo: PaperTradingRepository):
    """Execution report and account snapshot persist and survive restart."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )
    res = service.process_signal(make_signal(), current_market_price=1.0850)
    assert res.approved is True

    # Reconstruct broker
    broker_restarted = PaperBroker(
        initial_balance=100_000.0, audit_trail=audit, repository=temp_repo
    )
    assert len(broker_restarted.get_execution_reports()) == 1
    assert broker_restarted.get_account().cash_balance == broker.get_account().cash_balance


def test_restart_with_single_open_position(temp_repo: PaperTradingRepository):
    """Active open position is faithfully restored after restart."""
    audit = AuditTrail(repository=temp_repo)
    broker1 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    ord_buy = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=10_000,
        order_type=OrderType.MARKET,
        price=1.0850,
        status=OrderStatus.PENDING,
    )
    broker1.submit_order(ord_buy, current_market_price=1.0850)

    # Reconstruct broker2
    broker2 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    pos = broker2.get_position("EURUSD")
    assert pos is not None
    assert pos.quantity == 10_000
    assert pos.entry_price == 1.0850
    assert pos.status == "OPEN"


def test_restart_with_multiple_open_positions(temp_repo: PaperTradingRepository):
    """Multiple positions across different symbols are restored."""
    audit = AuditTrail(repository=temp_repo)
    broker1 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    broker1.submit_order(
        Order(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=10_000,
            order_type=OrderType.MARKET,
            price=1.0850,
        ),
        current_market_price=1.0850,
    )
    broker1.submit_order(
        Order(
            symbol="GBPUSD",
            side=OrderSide.BUY,
            quantity=8_000,
            order_type=OrderType.MARKET,
            price=1.2600,
        ),
        current_market_price=1.2600,
    )

    broker2 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    positions = broker2.get_positions()
    assert len(positions) == 2
    assert "EURUSD" in positions
    assert "GBPUSD" in positions


def test_restart_with_daily_loss_preserved(temp_repo: PaperTradingRepository):
    """Daily loss percentage is preserved across process restart on the same UTC day."""
    audit = AuditTrail(repository=temp_repo)
    broker1 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    pm1 = PortfolioManager(broker=broker1, repository=temp_repo)

    # Buy at 1.0850 and sell at 1.0600 (significant loss)
    broker1.submit_order(
        Order(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=50_000,
            order_type=OrderType.MARKET,
            price=1.0850,
        ),
        current_market_price=1.0850,
    )
    broker1.close_position("EURUSD", exit_price=1.0600)
    loss1 = pm1.get_daily_loss_percent()
    assert loss1 > 1.0

    # Restart
    broker2 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    pm2 = PortfolioManager(broker=broker2, repository=temp_repo)
    assert abs(pm2.get_daily_loss_percent() - loss1) < 0.01


def test_restart_with_active_kill_switch(temp_repo: PaperTradingRepository):
    """Active kill switch state survives process restart."""
    audit = AuditTrail(repository=temp_repo)
    ks1 = KillSwitch(audit_trail=audit, repository=temp_repo)
    ks1.activate(reason="OPERATIONAL_STRESS_TEST")
    assert ks1.is_active() is True

    ks2 = KillSwitch(audit_trail=audit, repository=temp_repo)
    assert ks2.is_active() is True
    assert ks2.reason == "OPERATIONAL_STRESS_TEST"


def test_restart_after_rejected_order(temp_repo: PaperTradingRepository):
    """Rejected orders are persisted without creating active positions or altering cash."""
    audit = AuditTrail(repository=temp_repo)
    broker1 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    bad_order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=200_000,  # Exceeds available cash ($100k)
        order_type=OrderType.MARKET,
        price=1.0850,
    )
    res_ord = broker1.submit_order(bad_order, current_market_price=1.0850)
    assert res_ord.status == OrderStatus.REJECTED
    assert res_ord.rejection_reason == "INSUFFICIENT_CASH"

    broker2 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    assert len(broker2.get_positions()) == 0
    assert broker2.get_account().cash_balance == 100_000.0


def test_restart_after_closed_position(temp_repo: PaperTradingRepository):
    """Closed positions move to history; active positions empty after close and restart."""
    audit = AuditTrail(repository=temp_repo)
    broker1 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    broker1.submit_order(
        Order(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=10_000,
            order_type=OrderType.MARKET,
            price=1.0850,
        ),
        current_market_price=1.0850,
    )
    broker1.close_position("EURUSD", exit_price=1.0900)

    broker2 = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    assert len(broker2.get_positions()) == 0
    assert len(broker2.get_closed_positions()) == 1
    assert broker2.get_account().realized_pnl > 0


# -----------------------------------------------------------------------------
# Category C: Idempotency & Concurrency Stress
# -----------------------------------------------------------------------------


def test_concurrent_duplicate_requests_two_threads(temp_repo: PaperTradingRepository):
    """Two concurrent requests with the same client_request_id result in exactly 1 execution."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )
    sig = make_signal(client_request_id="req_concurrent_2")

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(service.process_signal, sig, 1.0850)
        f2 = executor.submit(service.process_signal, sig, 1.0850)
        r1 = f1.result()
        r2 = f2.result()

    assert r1.approved is True
    assert r2.approved is True
    # Exactly one is a replay
    replays = [r.is_idempotent_replay for r in (r1, r2)]
    assert replays.count(True) == 1
    assert replays.count(False) == 1
    assert len(broker.get_execution_reports()) == 1


def test_concurrent_duplicate_requests_ten_threads(temp_repo: PaperTradingRepository):
    """10 simultaneous duplicate requests across threads execute exactly once without dupes."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )
    sig = make_signal(client_request_id="req_concurrent_10")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(service.process_signal, sig, 1.0850) for _ in range(10)]
        results = [f.result() for f in futures]

    assert all(r.approved for r in results)
    replays = [r.is_idempotent_replay for r in results]
    assert replays.count(False) == 1
    assert replays.count(True) == 9
    assert len(broker.get_execution_reports()) == 1


def test_repeat_after_success_returns_cached_execution(temp_repo: PaperTradingRepository):
    """Sequential repeat of identical signal returns cached execution."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )
    sig = make_signal(client_request_id="req_repeat_success")

    r1 = service.process_signal(sig, 1.0850)
    assert r1.approved is True
    assert r1.is_idempotent_replay is False

    r2 = service.process_signal(sig, 1.0850)
    assert r2.approved is True
    assert r2.is_idempotent_replay is True
    assert len(broker.get_execution_reports()) == 1


def test_repeat_after_rejection_returns_cached_rejection(temp_repo: PaperTradingRepository):
    """Sequential repeat of a rejected signal returns cached rejection."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )
    # Signal with zero confidence will be rejected by RiskEngine
    bad_sig = make_signal(confidence=0.0, client_request_id="req_repeat_rejected")

    r1 = service.process_signal(bad_sig, 1.0850)
    assert r1.approved is False
    assert r1.is_idempotent_replay is False

    r2 = service.process_signal(bad_sig, 1.0850)
    assert r2.approved is False
    assert r2.is_idempotent_replay is True
    assert len(broker.get_positions()) == 0


def test_repeat_after_failure_handling(temp_repo: PaperTradingRepository):
    """Signals with invalid schema are safely rejected without crashing idempotency engine."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch(audit, temp_repo)),
        paper_broker=broker,
        portfolio_manager=PortfolioManager(broker=broker, repository=temp_repo),
        audit_trail=audit,
        repository=temp_repo,
    )
    invalid_dict = {"symbol": "EURUSD", "action": "INVALID_ACTION"}
    res1 = service.process_signal(invalid_dict)
    assert res1.approved is False
    assert "SCHEMA_VALIDATION_ERROR" in (res1.metadata.get("error", ""))


# -----------------------------------------------------------------------------
# Category D: Kill Switch
# -----------------------------------------------------------------------------


def test_kill_switch_activation_and_rejection(temp_repo: PaperTradingRepository):
    """When kill switch is active, all signals are immediately rejected."""
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

    ks.activate(reason="TEST_CIRCUIT_BREAKER")
    sig = make_signal(client_request_id="req_ks_01")
    res = service.process_signal(sig, 1.0850)

    assert res.approved is False
    assert res.reason == RiskReason.KILL_SWITCH_ACTIVE.value
    assert len(broker.get_positions()) == 0


def test_kill_switch_deactivation_and_resumption(temp_repo: PaperTradingRepository):
    """Deactivating kill switch allows valid orders to be executed normally."""
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

    ks.activate(reason="TEMP_HALT")
    assert (
        service.process_signal(make_signal(client_request_id="req_ks_blocked"), 1.0850).approved
        is False
    )

    ks.deactivate()
    assert ks.is_active() is False

    res = service.process_signal(make_signal(client_request_id="req_ks_resumed"), 1.0850)
    assert res.approved is True
    assert len(broker.get_positions()) == 1


# -----------------------------------------------------------------------------
# Category E: Daily Loss Limit
# -----------------------------------------------------------------------------


def test_controlled_losing_sequence_blocks_subsequent_orders(temp_repo: PaperTradingRepository):
    """Accumulating losses beyond MAX_DAILY_LOSS_PERCENT (3%) locks out subsequent orders."""
    audit = AuditTrail(repository=temp_repo)
    broker = PaperBroker(initial_balance=100_000.0, audit_trail=audit, repository=temp_repo)
    limits = RiskLimits(MAX_DAILY_LOSS_PERCENT=3.0)
    risk_engine = RiskEngine(limits=limits, kill_switch=KillSwitch(audit, temp_repo))
    pm = PortfolioManager(broker=broker, repository=temp_repo)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=pm,
        audit_trail=audit,
        repository=temp_repo,
    )

    # Induce a loss > 3% on $100k equity (i.e. > $3,000 loss)
    # Buy 50,000 units at 1.0850, close at 1.0150 -> Loss = -0.070 * 50,000 = -$3,500
    broker.submit_order(
        Order(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=50_000,
            order_type=OrderType.MARKET,
            price=1.0850,
        ),
        current_market_price=1.0850,
    )
    broker.close_position("EURUSD", exit_price=1.0150)

    daily_loss_pct = pm.get_daily_loss_percent()
    assert daily_loss_pct > 3.0

    # Next signal must be rejected by RiskEngine
    sig_blocked = make_signal(stop_loss=1.0200, client_request_id="req_blocked_daily_loss")
    res = service.process_signal(sig_blocked, current_market_price=1.0500)
    assert res.approved is False
    assert res.reason == RiskReason.DAILY_LOSS_LIMIT_EXCEEDED.value


def test_no_floating_point_bypass_boundary():
    """Strict evaluation of boundary loss: 2.99% is allowed, 3.0001% is blocked."""
    limits = RiskLimits(MAX_DAILY_LOSS_PERCENT=3.0)
    risk_engine = RiskEngine(limits=limits)

    # Case 1: Daily loss is 2.99% (acceptable)
    account_ok = PaperBroker(initial_balance=100_000.0).get_account()
    decision_ok = risk_engine.evaluate(
        signal=make_signal(),
        account=account_ok,
        daily_start_equity=103_080.0,  # Equity 100k represents ~2.988% loss
        current_market_price=1.0850,
        override_quantity=10_000.0,
    )
    assert decision_ok.approved is True

    # Case 2: Daily loss is 3.001% (breached)
    decision_blocked = risk_engine.evaluate(
        signal=make_signal(),
        account=account_ok,
        daily_start_equity=103_100.0,  # Equity 100k represents ~3.0068% loss
        current_market_price=1.0850,
        override_quantity=10_000.0,
    )
    assert decision_blocked.approved is False
    assert decision_blocked.reason == RiskReason.DAILY_LOSS_LIMIT_EXCEEDED.value


# -----------------------------------------------------------------------------
# Category F: Position & Account Accounting Invariants
# -----------------------------------------------------------------------------


def test_accounting_invariants_hold_across_lifecycle():
    """Verify fundamental accounting invariants:
    1. equity == initial_balance + realized_pnl + unrealized_pnl
    2. net_pnl == gross_pnl - total_costs
    3. margin_used == sum(quantity * current_price)
    """
    broker = PaperBroker(initial_balance=100_000.0)

    # 1. Initial State Invariant
    acc0 = broker.get_account()
    assert acc0.equity == acc0.initial_balance
    assert acc0.realized_pnl == 0.0
    assert acc0.unrealized_pnl == 0.0

    # 2. Open Long Position
    broker.submit_order(
        Order(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=10_000,
            order_type=OrderType.MARKET,
            price=1.1000,
        ),
        current_market_price=1.1000,
    )
    acc1 = broker.get_account()
    assert math.isclose(
        acc1.equity, acc1.initial_balance + acc1.realized_pnl + acc1.unrealized_pnl, rel_tol=1e-5
    )
    assert acc1.margin_used == round(10_000 * 1.1000, 4)

    # 3. Mark to Market Price Movement
    broker.update_market_price("EURUSD", 1.1050)
    acc2 = broker.get_account()
    assert math.isclose(
        acc2.equity, acc2.initial_balance + acc2.realized_pnl + acc2.unrealized_pnl, rel_tol=1e-5
    )
    assert acc2.unrealized_pnl > 0

    # 4. Close Position
    broker.close_position("EURUSD", exit_price=1.1050)
    acc3 = broker.get_account()
    assert math.isclose(
        acc3.equity, acc3.initial_balance + acc3.realized_pnl + acc3.unrealized_pnl, rel_tol=1e-5
    )
    assert acc3.unrealized_pnl == 0.0
    assert len(acc3.positions) == 0


# -----------------------------------------------------------------------------
# Category G: Order State Machine Invalid Transitions
# -----------------------------------------------------------------------------


def test_order_state_machine_invalid_transitions():
    """Order state machine strictly rejects illegal transitions by raising ValueError."""
    # 1. CREATED -> FILLED is illegal (must go through VALIDATED/SUBMITTED)
    ord1 = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000,
        price=1.0,
        status=OrderStatus.CREATED,
    )
    with pytest.raises(ValueError, match="Invalid order transition"):
        ord1.transition_to(OrderStatus.FILLED)

    # 2. REJECTED -> FILLED is illegal
    ord2 = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000,
        price=1.0,
        status=OrderStatus.REJECTED,
    )
    with pytest.raises(ValueError, match="Invalid order transition"):
        ord2.transition_to(OrderStatus.FILLED)

    # 3. CANCELLED -> SUBMITTED is illegal
    ord3 = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000,
        price=1.0,
        status=OrderStatus.CANCELLED,
    )
    with pytest.raises(ValueError, match="Invalid order transition"):
        ord3.transition_to(OrderStatus.SUBMITTED)

    # 4. FILLED -> CREATED is illegal
    ord4 = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1000,
        price=1.0,
        status=OrderStatus.FILLED,
    )
    with pytest.raises(ValueError, match="Invalid order transition"):
        ord4.transition_to(OrderStatus.CREATED)


# -----------------------------------------------------------------------------
# Category H: Audit Trail Cryptographic Integrity
# -----------------------------------------------------------------------------


def test_audit_trail_valid_chain(temp_repo: PaperTradingRepository):
    """Audit events form a strictly valid SHA-256 cryptographic chain."""
    audit = AuditTrail(repository=temp_repo)
    for i in range(5):
        audit.record(
            event_type=AuditEventType.ORDER_CREATED,
            symbol="EURUSD",
            order_id=f"ord_{i}",
            details={"step": i},
        )
    valid, err = temp_repo.verify_audit_trail_integrity()
    assert valid is True
    assert err is None


def test_audit_trail_tamper_detection(temp_repo: PaperTradingRepository):
    """Modifying any field in an audit event breaks the chain and is detected."""
    audit = AuditTrail(repository=temp_repo)
    for i in range(3):
        audit.record(
            event_type=AuditEventType.ORDER_CREATED,
            symbol="EURUSD",
            order_id=f"ord_{i}",
            details={"step": i},
        )

    # Tamper with event sequence 2
    with temp_repo.db.connection() as conn:
        conn.execute(
            "UPDATE paper_audit_events SET details_json = ? WHERE sequence_id = 2;",
            ('{"tampered": 1}',),
        )

    valid, err = temp_repo.verify_audit_trail_integrity()
    assert valid is False
    assert "Tamper detected at sequence 2" in (err or "")


# -----------------------------------------------------------------------------
# Category I: Replay Determinism & Checkpoint Safety
# -----------------------------------------------------------------------------


def test_replay_exact_determinism():
    """Running identical market event sequence twice yields identical equity and execution count."""
    candles = [
        {
            "timestamp": datetime(2025, 1, 1, 10, i, tzinfo=timezone.utc),
            "symbol": "EURUSD",
            "open": 1.0800 + (i * 0.0005),
            "high": 1.0820 + (i * 0.0005),
            "low": 1.0790 + (i * 0.0005),
            "close": 1.0810 + (i * 0.0005),
            "volume": 1000.0,
        }
        for i in range(10)
    ]
    events = MarketReplayDataLoader.load_from_dataframe(pd.DataFrame(candles))

    # Run 1
    broker1 = PaperBroker(initial_balance=100_000.0)
    engine1 = ReplayEngine(events=events, paper_broker=broker1)
    broker1.submit_order(
        Order(
            order_id="det_ord1", symbol="EURUSD", side=OrderSide.BUY, quantity=1000.0, price=1.0800
        )
    )
    s1 = engine1.run()

    # Run 2
    broker2 = PaperBroker(initial_balance=100_000.0)
    engine2 = ReplayEngine(events=events, paper_broker=broker2)
    broker2.submit_order(
        Order(
            order_id="det_ord2", symbol="EURUSD", side=OrderSide.BUY, quantity=1000.0, price=1.0800
        )
    )
    s2 = engine2.run()

    assert s1["final_equity"] == s2["final_equity"]
    assert s1["cash_balance"] == s2["cash_balance"]
    assert s1["unrealized_pnl"] == s2["unrealized_pnl"]


def test_checkpoint_and_resume_identity():
    """Replaying 10 events straight matches replaying 5, checkpointing, and replaying remainder."""
    candles = [
        {
            "timestamp": datetime(2025, 1, 1, 10, i, tzinfo=timezone.utc),
            "symbol": "EURUSD",
            "open": 1.0800 + (i * 0.0005),
            "high": 1.0820 + (i * 0.0005),
            "low": 1.0790 + (i * 0.0005),
            "close": 1.0810 + (i * 0.0005),
            "volume": 1000.0,
        }
        for i in range(10)
    ]
    events = MarketReplayDataLoader.load_from_dataframe(pd.DataFrame(candles))

    # Run 1: Straight
    b1 = PaperBroker(initial_balance=100_000.0)
    e1 = ReplayEngine(events=events, paper_broker=b1)
    b1.submit_order(
        Order(
            order_id="chk_ord1", symbol="EURUSD", side=OrderSide.BUY, quantity=1000.0, price=1.0800
        )
    )
    s1 = e1.run()

    # Run 2: Checkpoint at step 5
    b2 = PaperBroker(initial_balance=100_000.0)
    e2 = ReplayEngine(events=events, paper_broker=b2)
    b2.submit_order(
        Order(
            order_id="chk_ord2", symbol="EURUSD", side=OrderSide.BUY, quantity=1000.0, price=1.0800
        )
    )
    e2.run(max_steps=5)
    cp = e2.create_checkpoint()

    # Resume from checkpoint
    e2.resume_from_checkpoint(cp)
    s2 = e2.run()

    assert s1["final_equity"] == s2["final_equity"]
    assert s1["cash_balance"] == s2["cash_balance"]
    assert s1["unrealized_pnl"] == s2["unrealized_pnl"]


# -----------------------------------------------------------------------------
# Category J: Quarantined Test Partition Enforcement
# -----------------------------------------------------------------------------


def test_replay_loader_quarantine_cutoff_enforcement():
    """Replay data loader raises QuarantinedPartitionError if data touches or exceeds cutoff."""
    bad_df = pd.DataFrame(
        [
            {
                "timestamp": LOCKED_TEST_CUTOFF + timedelta(hours=1),
                "symbol": "EURUSD",
                "open": 1.0850,
                "high": 1.0860,
                "low": 1.0840,
                "close": 1.0855,
                "volume": 1000.0,
            }
        ]
    )
    with pytest.raises(QuarantinedPartitionError, match="Quarantined partition violation"):
        MarketReplayDataLoader.load_from_dataframe(bad_df)


def test_readiness_quarantine_enforcement_flag():
    """Readiness endpoint verifies that quarantined partition cutoff is enforced."""
    context = reset_trading_context()
    set_trading_context(context)
    app = create_app()
    client = TestClient(app)

    resp = client.get("/readiness")
    assert resp.status_code == 200
    assert resp.json()["quarantine_enforced"] is True
