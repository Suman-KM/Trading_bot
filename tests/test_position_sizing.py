"""Unit tests for deterministic position sizing."""

import pytest

from trading.risk.limits import calculate_position_size


def test_valid_position_sizing():
    """Test 10: Valid position sizing formula calculation.

    Account equity = $100,000
    Entry price = 1.1000
    Stop loss = 1.0950 (stop distance = 0.0050)
    Risk percent = 0.5% -> $500 risk capital
    Quantity = $500 / 0.0050 = 100,000 units
    """
    qty = calculate_position_size(
        account_equity=100_000.0,
        entry_price=1.1000,
        stop_loss_price=1.0950,
        risk_percent=0.5,
        max_risk_percent=0.5,
    )
    assert qty == 100_000.0


def test_valid_position_sizing_short():
    """Valid position sizing for SELL with stop above entry."""
    qty = calculate_position_size(
        account_equity=50_000.0,
        entry_price=150.0,
        stop_loss_price=155.0,  # stop distance = 5.0
        risk_percent=0.2,  # $100 risk capital
        max_risk_percent=0.5,
    )
    # $100 / 5.0 = 20 units
    assert qty == 20.0


def test_invalid_position_sizing_zero_or_negative_equity():
    """Test 11a: Zero or negative equity must be rejected."""
    with pytest.raises(ValueError, match="Account equity must be a positive finite number"):
        calculate_position_size(
            account_equity=0.0,
            entry_price=100.0,
            stop_loss_price=95.0,
            risk_percent=0.5,
        )

    with pytest.raises(ValueError, match="Account equity must be a positive finite number"):
        calculate_position_size(
            account_equity=-10_000.0,
            entry_price=100.0,
            stop_loss_price=95.0,
            risk_percent=0.5,
        )


def test_invalid_position_sizing_zero_stop_distance():
    """Test 11b: Zero stop distance (entry == stop) must be rejected."""
    with pytest.raises(ValueError, match="Zero stop distance"):
        calculate_position_size(
            account_equity=100_000.0,
            entry_price=1.1000,
            stop_loss_price=1.1000,
            risk_percent=0.5,
        )


def test_invalid_position_sizing_prices():
    """Test 11c: Negative or non-finite prices must be rejected."""
    with pytest.raises(ValueError, match="Entry price must be a positive finite number"):
        calculate_position_size(
            account_equity=100_000.0,
            entry_price=-1.1000,
            stop_loss_price=1.0950,
            risk_percent=0.5,
        )

    with pytest.raises(ValueError, match="Stop loss price must be a positive finite number"):
        calculate_position_size(
            account_equity=100_000.0,
            entry_price=1.1000,
            stop_loss_price=0.0,
            risk_percent=0.5,
        )


def test_invalid_position_sizing_risk_percentage_out_of_range():
    """Test 11d: Risk percent <= 0 or > max_risk_percent must be rejected."""
    with pytest.raises(ValueError, match="Risk percent must be a positive finite number"):
        calculate_position_size(
            account_equity=100_000.0,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            risk_percent=0.0,
        )

    with pytest.raises(ValueError, match="exceeds max position risk"):
        calculate_position_size(
            account_equity=100_000.0,
            entry_price=1.1000,
            stop_loss_price=1.0950,
            risk_percent=1.0,  # Max allowed is 0.5%
            max_risk_percent=0.5,
        )
