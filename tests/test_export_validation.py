"""Deterministic unit tests for EURUSD M15 export validation logic."""

from scripts.export_eurusd_m15 import validate_bar_records


def _make_valid_bar(
    time: int = 1663320600,
    open: float = 1.0000,
    high: float = 1.0050,
    low: float = 0.9950,
    close: float = 1.0020,
    tick_volume: int = 150,
    spread: int = 2,
    real_volume: int = 0,
) -> dict:
    return {
        "time": time,
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": tick_volume,
        "spread": spread,
        "real_volume": real_volume,
    }


def test_validation_passes_on_valid_sequence():
    bars = [_make_valid_bar(time=1663320600 + i * 900) for i in range(10)]
    res = validate_bar_records(bars)
    assert res["valid"] is True
    assert res["total_rows"] == 10
    assert res["duplicate_timestamps"] == 0
    assert res["strictly_chronological"] is True
    assert res["ohlc_violations"] == 0
    assert res["min_interval_seconds"] == 900


def test_validation_detects_duplicates():
    bars = [
        _make_valid_bar(time=1663320600),
        _make_valid_bar(time=1663320600),  # Duplicate
    ]
    res = validate_bar_records(bars)
    assert res["valid"] is False
    assert res["duplicate_timestamps"] == 1


def test_validation_detects_out_of_order_timestamps():
    bars = [
        _make_valid_bar(time=1663320600),
        _make_valid_bar(time=1663319700),  # Decreasing timestamp
    ]
    res = validate_bar_records(bars)
    assert res["valid"] is False
    assert res["strictly_chronological"] is False


def test_validation_detects_ohlc_violations():
    # High lower than open
    b1 = _make_valid_bar(time=1000, open=1.05, high=1.04, low=1.00, close=1.02)
    res1 = validate_bar_records([b1])
    assert res1["valid"] is False
    assert res1["ohlc_violations"] == 1

    # Low higher than close
    b2 = _make_valid_bar(time=1000, open=1.05, high=1.06, low=1.03, close=1.02)
    res2 = validate_bar_records([b2])
    assert res2["valid"] is False
    assert res2["ohlc_violations"] == 1


def test_validation_detects_negative_or_nan_prices():
    b_neg = _make_valid_bar(time=1000, open=-1.0, high=1.0, low=-1.1, close=0.5)
    assert validate_bar_records([b_neg])["valid"] is False

    b_nan = _make_valid_bar(time=1000, open=float("nan"), high=1.0, low=0.9, close=0.95)
    res_nan = validate_bar_records([b_nan])
    assert res_nan["valid"] is False
    assert res_nan["nan_count"] == 1
