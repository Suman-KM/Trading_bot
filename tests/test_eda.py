"""Unit tests for the Phase 4 Exploratory Data Analysis (EDA) module.

Tests compute_price_statistics, compute_temporal_statistics,
compute_volume_spread_statistics, compute_gap_statistics, and run_full_eda
using deterministic, synthetic market data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.data.eda import (
    compute_gap_statistics,
    compute_price_statistics,
    compute_temporal_statistics,
    compute_volume_spread_statistics,
    run_full_eda,
)


@pytest.fixture
def synthetic_m15_series() -> pd.DataFrame:
    """Create a deterministic synthetic EURUSD M15 DataFrame.

    Covers 16 consecutive M15 candles (4 hours) starting Monday 2026-09-21 00:00:00 UTC.
    """
    base_time = 1790035200  # 2026-09-21 00:00:00 UTC
    n = 16
    times = [base_time + (i * 900) for i in range(n)]

    # Deterministic price series with known bullish, bearish, and doji candles
    opens = [1.08500 + (i * 0.00010) for i in range(n)]
    closes = [
        opens[i] + 0.00020 if i % 3 == 0 else (opens[i] - 0.00015 if i % 3 == 1 else opens[i])
        for i in range(n)
    ]
    highs = [max(opens[i], closes[i]) + 0.00010 for i in range(n)]
    lows = [min(opens[i], closes[i]) - 0.00010 for i in range(n)]
    tick_volumes = [100 + (i * 20) for i in range(n)]
    spreads = [10 + (i % 5) for i in range(n)]
    real_volumes = [0 for _ in range(n)]

    return pd.DataFrame(
        {
            "time": np.array(times, dtype=np.int64),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": np.array(tick_volumes, dtype=np.uint64),
            "spread": np.array(spreads, dtype=np.int32),
            "real_volume": np.array(real_volumes, dtype=np.uint64),
        }
    )


def test_compute_price_statistics(synthetic_m15_series: pd.DataFrame) -> None:
    """Verify price and return metrics computation."""
    res = compute_price_statistics(synthetic_m15_series)

    assert res["count"] == 16
    assert "close" in res
    assert res["close"]["min"] > 0
    assert res["close"]["max"] >= res["close"]["min"]
    assert "p50" in res["close"]["quantiles"]

    # Candle types classification
    ct = res["candle_types"]
    assert ct["bullish"] + ct["bearish"] + ct["doji"] == 16
    assert ct["bullish_pct"] > 0
    assert ct["bearish_pct"] > 0
    assert ct["doji_pct"] > 0

    # Returns and volatility
    sr = res["simple_returns"]
    lr = res["log_returns"]
    vol = res["volatility"]

    assert isinstance(sr["mean"], float)
    assert isinstance(lr["mean"], float)
    assert vol["bar_std"] > 0
    assert vol["annualized_vol"] > 0
    assert vol["atr_14_mean"] > 0


def test_compute_temporal_statistics(synthetic_m15_series: pd.DataFrame) -> None:
    """Verify temporal breakdown across hours, weekdays, months, and sessions."""
    res = compute_temporal_statistics(synthetic_m15_series)

    assert "hourly" in res
    assert "weekday" in res
    assert "monthly" in res
    assert "yearly" in res
    assert "sessions" in res

    # 16 candles over 4 hours (hours 0, 1, 2, 3)
    hourly = res["hourly"]
    assert len(hourly) == 4
    for h in hourly:
        assert h["candle_count"] == 4
        assert "return_mean" in h
        assert "spread_mean" in h
        assert "tick_volume_mean" in h

    # All candles are on Tuesday (day_of_week == 1)
    weekday = res["weekday"]
    assert len(weekday) == 1
    assert weekday[0]["day_name"] == "Tuesday"
    assert weekday[0]["candle_count"] == 16


def test_compute_volume_spread_statistics(synthetic_m15_series: pd.DataFrame) -> None:
    """Verify volume and spread statistics calculations."""
    res = compute_volume_spread_statistics(synthetic_m15_series)

    assert "tick_volume" in res
    assert "spread_points" in res
    assert "real_volume" in res

    tv = res["tick_volume"]
    assert tv["min"] == 100
    assert tv["max"] == 100 + (15 * 20)
    assert tv["zero_pct"] == 0.0

    sp = res["spread_points"]
    assert sp["min"] == 10
    assert sp["max"] == 14
    assert sp["zero_pct"] == 0.0

    rv = res["real_volume"]
    assert rv["zero_count"] == 16
    assert rv["zero_pct"] == 100.0


def test_compute_gap_statistics_no_gaps(synthetic_m15_series: pd.DataFrame) -> None:
    """Verify gap statistics when all intervals are normal 900s."""
    res = compute_gap_statistics(synthetic_m15_series)

    assert res["total_intervals"] == 15
    assert res["normal_intervals_900s"] == 15
    assert res["normal_interval_pct"] == 100.0
    assert res["total_gaps"] == 0
    assert res["max_gap_seconds"] == 900


def test_compute_gap_statistics_with_weekend() -> None:
    """Verify gap statistics correctly detects weekend interval."""
    # Friday 2026-09-18 20:00:00 UTC (1789848000) to Sunday 2026-09-20 21:00:00 UTC (1790024400)
    # Duration = 176400 seconds (~49 hours)
    df = pd.DataFrame(
        {
            "time": [1789848000, 1790024400],
            "open": [1.08, 1.08],
            "high": [1.09, 1.09],
            "low": [1.07, 1.07],
            "close": [1.085, 1.085],
            "tick_volume": [100, 100],
            "spread": [10, 10],
            "real_volume": [0, 0],
        }
    )
    res = compute_gap_statistics(df)
    assert res["total_intervals"] == 1
    assert res["total_gaps"] == 1
    assert res["expected_weekend_count"] == 1
    assert res["normal_intervals_900s"] == 0
    assert res["max_gap_seconds"] == 176400
    assert res["max_gap_hours"] == 49.0


def test_run_full_eda(synthetic_m15_series: pd.DataFrame, tmp_path: Path) -> None:
    """Verify run_full_eda orchestrates all calculations and saves plots."""
    out_dir = tmp_path / "figures"
    results = run_full_eda(synthetic_m15_series, figures_dir=out_dir)

    assert "price" in results
    assert "temporal" in results
    assert "volume_spread" in results
    assert "gaps" in results
    assert "figures" in results

    # Check that 8 plot files were generated
    assert len(results["figures"]) == 8
    for fig_path in results["figures"]:
        assert Path(fig_path).is_file()
