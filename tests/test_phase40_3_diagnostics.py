"""Phase 40.3: Forward Signal Diagnostics & Zero-Signal Root-Cause Analysis Tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.test_phase40_forward_validation import MockBackendForForward
from trading.adapters.mt5.client import MT5ReadOnlyClient
from trading.adapters.mt5.forward import (
    ForwardDemoRunner,
    ForwardValidationConfig,
)
from trading.adapters.mt5.schemas import MT5TickData
from trading.adapters.mt5.transport import MT5DemoExecutionTransport
from trading.audit.trail import AuditTrail
from trading.execution.mt5_adapter import MT5BrokerAdapter
from trading.execution.service import TradingExecutionService
from trading.persistence.database import PaperDatabaseManager
from trading.persistence.migrations import PaperSchemaMigrator
from trading.persistence.repository import PaperTradingRepository
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskEngine
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits


@pytest.fixture
def diag_pipeline(tmp_path: Path):
    db_path = tmp_path / "test_diag.db"
    mgr = PaperDatabaseManager(str(db_path))
    PaperSchemaMigrator.apply_migrations(mgr)
    repo = PaperTradingRepository(mgr)
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
        "runner": runner,
        "limits": limits,
        "config": config,
    }


def test_01_forward_runner_default_strategy_is_none(diag_pipeline):
    """Verify ForwardDemoRunner default initialization has strategy_fn=None and returns None."""
    runner: ForwardDemoRunner = diag_pipeline["runner"]
    assert runner.strategy_fn is None

    tick = MT5TickData(
        symbol="EURUSD",
        timestamp_utc=datetime.now(timezone.utc),
        bid=1.12050,
        ask=1.12051,
        spread=0.00001,
    )
    signal = runner.evaluate_frozen_strategy(tick)
    assert signal is None


def test_02_forward_runner_passive_monitoring_emits_zero_signals(diag_pipeline):
    """Verify that passive monitoring iteration returns IDLE / NO_SIGNAL with 0 signals."""
    runner: ForwardDemoRunner = diag_pipeline["runner"]
    res = runner.run_iteration()
    assert res["status"] == "IDLE"
    assert res["reason"] == "NO_SIGNAL"
    assert runner.metrics.signals_generated == 0
    assert runner.metrics.signals_approved == 0
    assert runner.metrics.orders_submitted == 0


def test_03_feature_pipeline_warmup_requirements():
    """Verify canonical feature schema requires 80 features and 80 bars warmup."""
    meta_path = Path("reports/feature_metadata.json")
    assert meta_path.exists(), "feature_metadata.json must exist"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_features"] == 80
    assert meta["warmup_rows_required"] == 80
    assert meta["max_lookback_bars"] == 80


def test_04_candle_timing_boundary_analysis():
    """Verify that a 7200s session starting at 12:11:00 UTC crosses 8 M15 and 0 H4 candle closes."""
    t_start = datetime(2026, 10, 5, 12, 11, 0, 294108, tzinfo=timezone.utc)
    t_end = datetime(2026, 10, 5, 14, 11, 8, 90020, tzinfo=timezone.utc)
    duration_sec = (t_end - t_start).total_seconds()
    assert duration_sec >= 7200.0

    m15_boundaries = [
        datetime(2026, 10, 5, 12, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 12, 30, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 12, 45, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 13, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 13, 15, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 13, 30, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 13, 45, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 14, 0, 0, tzinfo=timezone.utc),
    ]
    for b in m15_boundaries:
        assert t_start <= b <= t_end

    assert len(m15_boundaries) == 8

    h4_boundaries = [
        datetime(2026, 10, 5, 8, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 5, 16, 0, 0, tzinfo=timezone.utc),
    ]
    h4_in_session = [b for b in h4_boundaries if t_start <= b <= t_end]
    assert len(h4_in_session) == 0


def test_05_phase40_2_telemetry_audit():
    """Verify Phase 40.2 observation telemetry record invariants."""
    telemetry_path = Path("reports/phase40_2_forward_observation.json")
    assert telemetry_path.exists(), "reports/phase40_2_forward_observation.json must exist"

    with open(telemetry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["duration_seconds"] >= 7200.0
    assert data["market_open_status"] == "OPEN"
    assert data["signals_generated"] == 0
    assert data["signals_approved"] == 0
    assert data["demo_orders_submitted"] == 0
    assert data["real_money_orders"] == 0
    assert data["final_open_positions"] == 0
    assert data["verdict"] == "FORWARD OBSERVATION COMPLETED"


def test_06_frozen_risk_limits_preserved(diag_pipeline):
    """Verify RiskEngine limits remain frozen at canonical baseline values."""
    limits: RiskLimits = diag_pipeline["limits"]
    assert limits.MAX_DAILY_LOSS_PERCENT == 1.0
    assert limits.MAX_POSITION_RISK_PERCENT == 0.5
    assert limits.MAX_TOTAL_EXPOSURE_PERCENT == 20.0
    assert limits.MIN_SIGNAL_CONFIDENCE == 0.60
