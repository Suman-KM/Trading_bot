"""Unit tests for Signal validation and parsing."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from trading.models.signal import Signal, SignalAction


def make_valid_signal_dict() -> dict:
    return {
        "symbol": "EURUSD",
        "action": "BUY",
        "confidence": 0.85,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "v1.0.0",
        "timeframe": "1h",
        "expected_return": 0.015,
        "feature_version": "f_v2",
        "suggested_entry_price": 1.1000,
        "suggested_stop_loss": 1.0950,
        "suggested_take_profit": 1.1100,
    }


def test_valid_signal_creation():
    """Test 1: Valid signal instantiates cleanly."""
    data = make_valid_signal_dict()
    sig = Signal.model_validate(data)
    assert sig.symbol == "EURUSD"
    assert sig.action == SignalAction.BUY
    assert sig.confidence == 0.85
    assert sig.expected_return == 0.015
    assert sig.suggested_stop_loss == 1.0950


def test_invalid_confidence_bounds():
    """Test 2: Invalid confidence (< 0 or > 1) raises ValidationError."""
    data = make_valid_signal_dict()

    data["confidence"] = 1.05
    with pytest.raises(ValidationError) as exc:
        Signal.model_validate(data)
    assert "Confidence must be between 0.0 and 1.0" in str(exc.value)

    data["confidence"] = -0.1
    with pytest.raises(ValidationError) as exc:
        Signal.model_validate(data)
    assert "Confidence must be between 0.0 and 1.0" in str(exc.value)

    data["confidence"] = float("nan")
    with pytest.raises(ValidationError) as exc:
        Signal.model_validate(data)
    assert "Confidence must be a finite number" in str(exc.value)


def test_hold_signal_action():
    """Test 3: HOLD signal is valid in model schema."""
    data = make_valid_signal_dict()
    data["action"] = "HOLD"
    sig = Signal.model_validate(data)
    assert sig.action == SignalAction.HOLD


def test_empty_symbol():
    """Empty or whitespace symbol raises ValidationError."""
    data = make_valid_signal_dict()
    data["symbol"] = "   "
    with pytest.raises(ValidationError):
        Signal.model_validate(data)


def test_non_finite_expected_return():
    """Non-finite expected return raises ValidationError."""
    data = make_valid_signal_dict()
    data["expected_return"] = float("inf")
    with pytest.raises(ValidationError):
        Signal.model_validate(data)

    data["expected_return"] = float("nan")
    with pytest.raises(ValidationError):
        Signal.model_validate(data)


def test_forbid_extra_fields():
    """Unexpected extra fields must be rejected."""
    data = make_valid_signal_dict()
    data["malicious_payload"] = "drop_database()"
    with pytest.raises(ValidationError):
        Signal.model_validate(data)
