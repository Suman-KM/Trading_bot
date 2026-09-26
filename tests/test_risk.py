"""Unit tests for deterministic Risk Engine rules."""

from datetime import datetime, timezone

from trading.models.order import OrderSide, OrderStatus
from trading.models.portfolio import AccountInfo
from trading.models.position import Position
from trading.models.signal import Signal, SignalAction
from trading.risk.engine import RiskEngine, RiskReason
from trading.risk.limits import RiskLimits


def make_signal(
    symbol: str = "EURUSD",
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.85,
    entry: float = 100.0,
    stop_loss: float = 95.0,  # 5.0 dist -> 100 qty -> $10,000 exposure = 10%
) -> Signal:
    return Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
        model_version="v1",
        timeframe="1h",
        expected_return=0.015,
        feature_version="f1",
        suggested_entry_price=entry,
        suggested_stop_loss=stop_loss,
    )


def test_valid_signal_approved():
    """Test 1 & 5: Valid signal meeting all conditions produces approved Order."""
    engine = RiskEngine()
    account = AccountInfo(
        initial_balance=100_000.0,
        cash_balance=100_000.0,
        equity=100_000.0,
    )
    sig = make_signal()
    decision = engine.evaluate(signal=sig, account=account, daily_start_equity=100_000.0)

    assert decision.approved is True
    assert decision.reason is None
    assert decision.order is not None
    assert decision.order.symbol == "EURUSD"
    assert decision.order.side == OrderSide.BUY
    assert decision.order.status == OrderStatus.PENDING
    assert decision.order.quantity > 0


def test_hold_signal_rejected():
    """Test 3: HOLD signal must be rejected."""
    engine = RiskEngine()
    account = AccountInfo(initial_balance=100_000.0, cash_balance=100_000.0, equity=100_000.0)
    sig = make_signal(action=SignalAction.HOLD)
    decision = engine.evaluate(signal=sig, account=account, daily_start_equity=100_000.0)

    assert decision.approved is False
    assert decision.reason == RiskReason.SIGNAL_ACTION_HOLD.value
    assert decision.order is None


def test_low_confidence_rejected():
    """Test 4: Confidence below minimum (0.60) must be rejected."""
    engine = RiskEngine(limits=RiskLimits(MIN_SIGNAL_CONFIDENCE=0.60))
    account = AccountInfo(initial_balance=100_000.0, cash_balance=100_000.0, equity=100_000.0)
    sig = make_signal(confidence=0.55)
    decision = engine.evaluate(signal=sig, account=account, daily_start_equity=100_000.0)

    assert decision.approved is False
    assert decision.reason == RiskReason.CONFIDENCE_TOO_LOW.value
    assert decision.order is None


def test_daily_loss_limit_exceeded():
    """Test 8: Daily loss >= 1.0% must reject and trip the kill switch."""
    engine = RiskEngine(limits=RiskLimits(MAX_DAILY_LOSS_PERCENT=1.0))
    # Daily start = 100,000, current equity = 98,800 -> loss is 1.2%
    account = AccountInfo(initial_balance=100_000.0, cash_balance=98_800.0, equity=98_800.0)
    sig = make_signal()
    decision = engine.evaluate(signal=sig, account=account, daily_start_equity=100_000.0)

    assert decision.approved is False
    assert decision.reason == RiskReason.DAILY_LOSS_LIMIT_EXCEEDED.value
    assert decision.order is None
    # Verify kill switch automatically engaged
    assert engine.kill_switch.is_active() is True
    assert engine.kill_switch.reason == "DAILY_LOSS_LIMIT"


def test_max_open_positions_limit():
    """Test 7: Attempting to open more than MAX_OPEN_POSITIONS (3) must be rejected."""
    engine = RiskEngine(limits=RiskLimits(MAX_OPEN_POSITIONS=3))
    positions = {
        "PAIR1": Position(
            symbol="PAIR1",
            side=OrderSide.BUY,
            quantity=100.0,
            entry_price=1.0,
            current_price=1.0,
        ),
        "PAIR2": Position(
            symbol="PAIR2",
            side=OrderSide.BUY,
            quantity=100.0,
            entry_price=1.0,
            current_price=1.0,
        ),
        "PAIR3": Position(
            symbol="PAIR3",
            side=OrderSide.BUY,
            quantity=100.0,
            entry_price=1.0,
            current_price=1.0,
        ),
    }
    account = AccountInfo(
        initial_balance=100_000.0,
        cash_balance=99_700.0,
        equity=100_000.0,
        positions=positions,
    )

    sig = make_signal(symbol="PAIR4")
    decision = engine.evaluate(signal=sig, account=account, daily_start_equity=100_000.0)

    assert decision.approved is False
    assert decision.reason == RiskReason.MAX_OPEN_POSITIONS_REACHED.value
    assert decision.order is None


def test_excessive_exposure_rejected():
    """Test 6: Trade that exceeds MAX_TOTAL_EXPOSURE_PERCENT (20%) must be rejected."""
    engine = RiskEngine(limits=RiskLimits(MAX_TOTAL_EXPOSURE_PERCENT=20.0))
    account = AccountInfo(
        initial_balance=100_000.0,
        cash_balance=100_000.0,
        equity=100_000.0,
    )
    # Entry=100, stop=99.99 -> stop distance = 0.01
    # Quantity = 50,000 -> exposure = 5,000,000 (5,000% of equity)
    sig = make_signal(entry=100.0, stop_loss=99.99)
    decision = engine.evaluate(signal=sig, account=account, daily_start_equity=100_000.0)

    assert decision.approved is False
    assert decision.reason == RiskReason.EXCESSIVE_EXPOSURE.value
    assert decision.order is None


def test_invalid_quantity_rejected():
    """Test 5: Explicit non-positive quantity must be rejected."""
    engine = RiskEngine()
    account = AccountInfo(initial_balance=100_000.0, cash_balance=100_000.0, equity=100_000.0)
    sig = make_signal()
    decision = engine.evaluate(
        signal=sig,
        account=account,
        daily_start_equity=100_000.0,
        override_quantity=-5.0,
    )

    assert decision.approved is False
    assert decision.reason == RiskReason.INVALID_QUANTITY.value
