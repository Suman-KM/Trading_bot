"""Unit tests for the emergency Kill Switch."""

from datetime import datetime, timezone

from trading.models.portfolio import AccountInfo
from trading.models.signal import Signal, SignalAction
from trading.risk.engine import RiskEngine, RiskReason
from trading.risk.kill_switch import KillSwitch


def make_test_signal() -> Signal:
    return Signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.80,
        timestamp=datetime.now(timezone.utc),
        model_version="v1",
        timeframe="1h",
        expected_return=0.01,
        feature_version="f1",
        suggested_entry_price=1.1000,
        suggested_stop_loss=1.0950,
    )


def test_kill_switch_state_transitions():
    """Test 9a: Kill switch activate and deactivate cycle."""
    ks = KillSwitch()
    assert not ks.is_active()
    assert ks.reason is None

    ks.activate(reason="MANUAL_EMERGENCY")
    assert ks.is_active()
    assert ks.reason == "MANUAL_EMERGENCY"
    assert ks.activated_at is not None

    ks.deactivate()
    assert not ks.is_active()
    assert ks.reason is None


def test_kill_switch_audit_history():
    """Test 9b: Kill switch records full activation audit history."""
    ks = KillSwitch()
    ks.activate(reason="DAILY_LOSS_LIMIT")
    ks.deactivate()
    ks.activate(reason="SYSTEM_ERROR")

    history = ks.history
    assert len(history) == 3
    assert history[0]["event"] == "ACTIVATED"
    assert history[0]["reason"] == "DAILY_LOSS_LIMIT"
    assert history[1]["event"] == "DEACTIVATED"
    assert history[2]["event"] == "ACTIVATED"
    assert history[2]["reason"] == "SYSTEM_ERROR"


def test_risk_engine_rejects_when_kill_switch_active():
    """Test 9c: Risk Engine immediately rejects all signals when kill switch is active."""
    ks = KillSwitch()
    ks.activate("TEST_HALT")
    engine = RiskEngine(kill_switch=ks)

    account = AccountInfo(
        initial_balance=100_000.0,
        cash_balance=100_000.0,
        equity=100_000.0,
    )

    signal = make_test_signal()
    decision = engine.evaluate(signal=signal, account=account, daily_start_equity=100_000.0)

    assert not decision.approved
    assert decision.reason == RiskReason.KILL_SWITCH_ACTIVE.value
    assert decision.order is None
    assert decision.metadata["kill_switch_reason"] == "TEST_HALT"
