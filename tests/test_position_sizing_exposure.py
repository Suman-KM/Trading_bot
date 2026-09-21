"""Dedicated tests verifying position sizing risk vs. gross exposure limits (Requirement 15)."""

from datetime import datetime, timezone

from trading.models.order import OrderSide, OrderStatus
from trading.models.portfolio import AccountInfo
from trading.models.position import Position
from trading.models.signal import Signal, SignalAction
from trading.risk.engine import RiskEngine, RiskReason
from trading.risk.limits import RiskLimits, calculate_position_size


def make_signal(
    symbol: str,
    entry: float,
    stop_loss: float,
    confidence: float = 0.85,
) -> Signal:
    return Signal(
        symbol=symbol,
        action=SignalAction.BUY,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
        model_version="ml_v1",
        timeframe="15m",
        expected_return=0.015,
        feature_version="feat_v1",
        suggested_entry_price=entry,
        suggested_stop_loss=stop_loss,
    )


def test_tight_stop_loss_excessive_exposure_rejected():
    """Verify that a tight stop-loss generating large quantity is rejected by exposure limits.

    Scenario:
    - Equity: $100,000
    - Max position risk: 0.5% ($500 risk budget)
    - Entry price: $100.0
    - Tight stop-loss: $99.80 (stop distance = $0.20)
    - Risk-based sizing: $500 / $0.20 = 2,500 units
    - Resulting Gross Exposure: 2,500 * $100.0 = $250,000 (250% of equity)
    - Max allowed exposure: 20.0% ($20,000)

    Invariant: RiskEngine MUST reject with EXCESSIVE_EXPOSURE and emit no order.
    """
    limits = RiskLimits(
        MAX_POSITION_RISK_PERCENT=0.5,
        MAX_TOTAL_EXPOSURE_PERCENT=20.0,
    )
    engine = RiskEngine(limits=limits)
    account = AccountInfo(
        initial_balance=100_000.0,
        cash_balance=100_000.0,
        equity=100_000.0,
    )

    # 1. Verify that position sizing formula produces 2,500 units
    qty = calculate_position_size(
        account_equity=100_000.0,
        entry_price=100.0,
        stop_loss_price=99.80,
        risk_percent=0.5,
    )
    assert qty == 2_500.0
    exposure = qty * 100.0
    assert exposure == 250_000.0
    assert (exposure / 100_000.0) * 100 == 250.0

    # 2. Verify RiskEngine evaluation strictly catches and blocks this trade
    signal = make_signal(symbol="TIGHT_STOP_ASSET", entry=100.0, stop_loss=99.80)
    decision = engine.evaluate(signal=signal, account=account, daily_start_equity=100_000.0)

    assert decision.approved is False
    assert decision.reason == RiskReason.EXCESSIVE_EXPOSURE.value
    assert decision.order is None
    assert decision.metadata["total_exposure_percent"] == 250.0
    assert decision.metadata["max_allowed"] == 20.0


def test_valid_position_sizing_within_exposure_limit_approved():
    """Verify that a trade with proportional stop-loss within exposure limits is approved.

    Scenario:
    - Equity: $100,000
    - Max position risk: 0.5% ($500 risk budget)
    - Entry price: $100.0
    - Wider stop-loss: $96.0 (stop distance = $4.0)
    - Risk-based sizing: $500 / $4.0 = 125 units
    - Resulting Gross Exposure: 125 * $100.0 = $12,500 (12.5% of equity)
    - Max allowed exposure: 20.0% ($20,000)

    Invariant: RiskEngine MUST approve with PENDING Order.
    """
    limits = RiskLimits(
        MAX_POSITION_RISK_PERCENT=0.5,
        MAX_TOTAL_EXPOSURE_PERCENT=20.0,
    )
    engine = RiskEngine(limits=limits)
    account = AccountInfo(
        initial_balance=100_000.0,
        cash_balance=100_000.0,
        equity=100_000.0,
    )

    qty = calculate_position_size(
        account_equity=100_000.0,
        entry_price=100.0,
        stop_loss_price=96.0,
        risk_percent=0.5,
    )
    assert qty == 125.0
    exposure = qty * 100.0
    assert exposure == 12_500.0
    assert (exposure / 100_000.0) * 100 == 12.5

    signal = make_signal(symbol="SAFE_ASSET", entry=100.0, stop_loss=96.0)
    decision = engine.evaluate(signal=signal, account=account, daily_start_equity=100_000.0)

    assert decision.approved is True
    assert decision.reason is None
    assert decision.order is not None
    assert decision.order.quantity == 125.0
    assert decision.order.status == OrderStatus.PENDING
    assert decision.metadata["exposure_percent"] == 12.5


def test_cumulative_exposure_across_multiple_positions():
    """Verify that multiple open positions cumulative exposure blocks an otherwise safe trade."""
    limits = RiskLimits(
        MAX_POSITION_RISK_PERCENT=0.5,
        MAX_TOTAL_EXPOSURE_PERCENT=20.0,
        MAX_OPEN_POSITIONS=3,
    )
    engine = RiskEngine(limits=limits)

    # Existing position with $12,500 exposure (12.5%)
    existing_pos = Position(
        symbol="EXISTING_POS",
        side=OrderSide.BUY,
        quantity=125.0,
        entry_price=100.0,
        current_price=100.0,
    )
    account = AccountInfo(
        initial_balance=100_000.0,
        cash_balance=87_500.0,
        equity=100_000.0,
        positions={"EXISTING_POS": existing_pos},
    )

    # New signal with 10.0% exposure ($10,000)
    # Total would be 12.5% + 10.0% = 22.5% > 20.0%
    signal = make_signal(symbol="NEW_ASSET", entry=100.0, stop_loss=95.0)
    decision = engine.evaluate(signal=signal, account=account, daily_start_equity=100_000.0)

    assert decision.approved is False
    assert decision.reason == RiskReason.EXCESSIVE_EXPOSURE.value
    assert decision.metadata["total_exposure_percent"] == 22.5
