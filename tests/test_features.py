"""Unit and leakage tests for Phase 5 feature engineering.

Verifies mathematical correctness, NaN warm-up characteristics, and strictly enforces
point-in-time constraints through mandatory leakage tests (future-shift, truncation,
ordering, no-target, and gap preservation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.features import (
    MAX_LOOKBACK_BARS,
    build_feature_pipeline,
    compute_activity_features,
    compute_gap_features,
    compute_momentum_features,
    compute_price_features,
    compute_time_features,
    compute_trend_features,
    compute_volatility_features,
    generate_feature_metadata,
    get_feature_registry,
)


@pytest.fixture
def synthetic_m15_history() -> pd.DataFrame:
    """Generate 120 deterministic synthetic M15 bars with an embedded weekend gap.

    Starts Monday 2026-09-21 00:00:00 UTC (epoch 1789948800).
    At bar index 60, a weekend gap of 172,800 seconds (48h) is embedded.
    """
    n = 120
    base_epoch = 1789948800
    times: list[int] = []
    curr = base_epoch
    for i in range(n):
        if i == 60:
            # 48-hour weekend gap
            curr += 172800
        else:
            curr += 900
        times.append(curr)

    # Deterministic price series with realistic drift and oscillation
    np.random.seed(42)
    steps = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.0005 + 0.00005
    prices = 1.0850 + np.cumsum(steps)

    opens = prices - 0.0001
    closes = prices + 0.0001
    highs = np.maximum(opens, closes) + 0.0002
    lows = np.minimum(opens, closes) - 0.0002
    tick_volumes = [500 + int(200 * np.sin(i / 5.0)) for i in range(n)]
    spreads = [12 + (i % 4) for i in range(n)]
    real_volumes = [0 for _ in range(n)]

    timestamps = pd.to_datetime(times, unit="s", utc=True)

    return pd.DataFrame(
        {
            "time": np.array(times, dtype=np.int64),
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": np.array(tick_volumes, dtype=np.uint64),
            "spread": np.array(spreads, dtype=np.int32),
            "real_volume": np.array(real_volumes, dtype=np.uint64),
        }
    )


# -----------------------------------------------------------------------------
# 1. Feature Group Functionality Tests
# -----------------------------------------------------------------------------


def test_price_features(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify price and candle geometry feature computation."""
    res = compute_price_features(synthetic_m15_history)

    expected_cols = [
        "return_1",
        "log_return_1",
        "open_to_close_return",
        "hl_range",
        "hl_range_norm",
        "candle_body",
        "candle_body_norm",
        "upper_wick",
        "upper_wick_ratio",
        "lower_wick",
        "lower_wick_ratio",
        "candle_direction",
        "return_mean_5",
        "return_mean_20",
    ]
    for c in expected_cols:
        assert c in res.columns

    # Non-negativity constraints
    assert (res["hl_range"] >= 0).all()
    assert (res["candle_body"] >= 0).all()
    assert (res["upper_wick"] >= 0).all()
    assert (res["lower_wick"] >= 0).all()
    assert res["candle_direction"].isin([-1.0, 0.0, 1.0]).all()


def test_momentum_features(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify momentum, ROC, and oscillator features."""
    res = compute_momentum_features(synthetic_m15_history)

    for w in [5, 10, 20, 40, 80]:
        assert f"roc_{w}" in res.columns
    for w in [10, 20, 40, 80]:
        assert f"dist_sma_{w}" in res.columns
    assert "rsi_14" in res.columns

    # RSI bounded in [0, 100] after warm-up
    clean_rsi = res["rsi_14"].dropna()
    assert (clean_rsi >= 0.0).all() and (clean_rsi <= 100.0).all()


def test_volatility_features(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify volatility, ATR, and dispersion features."""
    res = compute_volatility_features(synthetic_m15_history)

    for w in [10, 20, 40, 80]:
        assert f"vol_std_{w}" in res.columns
    assert "vol_ann_20" in res.columns
    assert "vol_ann_80" in res.columns
    assert "atr_14" in res.columns
    assert "atr_norm_14" in res.columns
    assert "atr_40" in res.columns
    assert "atr_norm_40" in res.columns
    assert "vol_ratio_10_40" in res.columns
    assert "vol_ratio_20_80" in res.columns
    assert "rolling_hl_ratio_20" in res.columns

    # Non-negativity of volatility and ATR
    assert (res["vol_std_10"].dropna() >= 0).all()
    assert (res["atr_14"].dropna() >= 0).all()
    assert (res["atr_norm_14"].dropna() >= 0).all()


def test_trend_features(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify trend, SMA, EMA, and slope features."""
    res = compute_trend_features(synthetic_m15_history)

    for w in [10, 20, 40, 80]:
        assert f"sma_{w}" in res.columns
        assert f"ema_{w}" in res.columns
        assert f"dist_ema_{w}" in res.columns
    assert "ema_spread_10_40" in res.columns
    assert "ema_spread_20_80" in res.columns
    assert "ema_slope_10" in res.columns
    assert "ema_slope_20" in res.columns


def test_activity_features(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify activity features and verify real_volume exclusion."""
    res = compute_activity_features(synthetic_m15_history)

    expected_cols = [
        "log_tick_volume",
        "tick_vol_sma_20",
        "tick_vol_ratio_20",
        "tick_vol_zscore_20",
        "tick_vol_zscore_80",
        "spread_norm",
        "spread_sma_20",
        "spread_ratio_20",
        "spread_zscore_20",
    ]
    for c in expected_cols:
        assert c in res.columns

    # Explicit requirement: real_volume is 100% zero and must NOT be in feature set
    assert "real_volume" not in res.columns
    assert "real_volume_ratio" not in res.columns


def test_time_features(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify calendar and harmonic cyclical time features."""
    res = compute_time_features(synthetic_m15_history)

    assert "hour" in res.columns
    assert "minute" in res.columns
    assert "day_of_week" in res.columns
    assert "sin_hour" in res.columns
    assert "cos_hour" in res.columns
    assert "sin_minute" in res.columns
    assert "cos_minute" in res.columns
    assert "sin_day_of_week" in res.columns
    assert "cos_day_of_week" in res.columns
    assert "is_asian_session" in res.columns
    assert "is_london_session" in res.columns
    assert "is_ny_session" in res.columns
    assert "is_rollover_window" in res.columns

    # Harmonic identity: sin^2 + cos^2 == 1
    hour_sq_sum = res["sin_hour"] ** 2 + res["cos_hour"] ** 2
    np.testing.assert_allclose(hour_sq_sum, 1.0, atol=1e-6)

    min_sq_sum = res["sin_minute"] ** 2 + res["cos_minute"] ** 2
    np.testing.assert_allclose(min_sq_sum, 1.0, atol=1e-6)


def test_gap_features(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify gap detection and continuity features."""
    res = compute_gap_features(synthetic_m15_history)

    assert "delta_seconds" in res.columns
    assert "is_normal_m15" in res.columns
    assert "is_post_gap" in res.columns
    assert "is_weekend_reopen" in res.columns
    assert "bars_since_last_gap" in res.columns

    # Bar 60 has the 48-hour gap
    assert res["is_post_gap"].iloc[60] == 1.0
    assert res["delta_seconds"].iloc[60] == 172800


# -----------------------------------------------------------------------------
# 2. Pipeline Integration and NaN Warm-up Tests
# -----------------------------------------------------------------------------


def test_build_feature_pipeline_dimensions(
    synthetic_m15_history: pd.DataFrame,
) -> None:
    """Verify total feature count and pipeline output dimensions."""
    df_feat, defs = build_feature_pipeline(
        synthetic_m15_history, drop_warmup=False, include_raw_columns=True
    )

    assert len(defs) == 80
    # 9 input raw columns + 80 features = 89 columns
    assert df_feat.shape == (120, 89)

    registry = get_feature_registry()
    assert len(registry) == 80


def test_nan_warmup_behavior(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify that after the 80-bar warm-up period, zero NaNs exist."""
    df_feat, defs = build_feature_pipeline(
        synthetic_m15_history, drop_warmup=False, include_raw_columns=False
    )

    # After bar index 80 (the maximum lookback window), all features must be populated
    post_warmup = df_feat.iloc[MAX_LOOKBACK_BARS:]
    nan_counts = post_warmup.isna().sum()
    assert (nan_counts == 0).all(), f"Found NaNs after warm-up: {nan_counts[nan_counts > 0]}"


def test_drop_warmup(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify drop_warmup drops exactly MAX_LOOKBACK_BARS rows."""
    df_feat, _ = build_feature_pipeline(
        synthetic_m15_history, drop_warmup=True, include_raw_columns=True
    )
    assert len(df_feat) == len(synthetic_m15_history) - MAX_LOOKBACK_BARS
    # Ensure zero NaNs exist across all feature columns in the dropped version
    feat_cols = [c for c in df_feat.columns if c not in synthetic_m15_history.columns]
    assert (df_feat[feat_cols].isna().sum() == 0).all()


def test_metadata_generation(synthetic_m15_history: pd.DataFrame) -> None:
    """Verify metadata generation produces complete and valid statistics."""
    df_feat, defs = build_feature_pipeline(synthetic_m15_history, drop_warmup=False)
    metadata = generate_feature_metadata(df_feat, defs)

    assert metadata["total_features"] == 80
    assert metadata["total_rows"] == 120
    assert metadata["max_lookback_bars"] == MAX_LOOKBACK_BARS
    assert len(metadata["features"]) == 80
    for f in metadata["features"]:
        assert "name" in f
        assert "group" in f
        assert "lookback_bars" in f
        assert "nan_count" in f
        assert "leakage_tested" in f


# -----------------------------------------------------------------------------
# 3. Mandatory Point-in-Time Leakage Tests
# -----------------------------------------------------------------------------


def test_leakage_future_shift(synthetic_m15_history: pd.DataFrame) -> None:
    """Test A (Future-shift): Modifying candle t+1 must NOT alter features at t."""
    df_orig = synthetic_m15_history.copy()
    features_orig, _ = build_feature_pipeline(df_orig, drop_warmup=False, include_raw_columns=False)

    t = 90
    # Perturb future bar t+1 dramatically
    df_perturbed = synthetic_m15_history.copy()
    df_perturbed.loc[t + 1, "open"] *= 2.5
    df_perturbed.loc[t + 1, "high"] *= 3.0
    df_perturbed.loc[t + 1, "low"] *= 0.5
    df_perturbed.loc[t + 1, "close"] *= 2.8
    df_perturbed.loc[t + 1, "tick_volume"] += 10000
    df_perturbed.loc[t + 1, "spread"] += 50

    features_perturbed, _ = build_feature_pipeline(
        df_perturbed, drop_warmup=False, include_raw_columns=False
    )

    # Features at bar t and all bars prior to t must be strictly identical
    orig_up_to_t = features_orig.iloc[: t + 1]
    perturbed_up_to_t = features_perturbed.iloc[: t + 1]

    pd.testing.assert_frame_equal(orig_up_to_t, perturbed_up_to_t)


def test_leakage_truncation(synthetic_m15_history: pd.DataFrame) -> None:
    """Test B (Truncation): Features computed on data ending at t must match full data at t."""
    t = 95
    df_full = synthetic_m15_history.copy()
    df_truncated = synthetic_m15_history.iloc[: t + 1].copy()

    features_full, _ = build_feature_pipeline(df_full, drop_warmup=False, include_raw_columns=False)
    features_trunc, _ = build_feature_pipeline(
        df_truncated, drop_warmup=False, include_raw_columns=False
    )

    # Compare row t
    row_full = features_full.iloc[t]
    row_trunc = features_trunc.iloc[t]

    pd.testing.assert_series_equal(row_full, row_trunc)


def test_leakage_timestamp_ordering(synthetic_m15_history: pd.DataFrame) -> None:
    """Test C (Timestamp ordering): Output features must strictly align with input timestamps."""
    df_feat, _ = build_feature_pipeline(
        synthetic_m15_history, drop_warmup=False, include_raw_columns=True
    )

    # Monotonicity check
    assert (df_feat["time"].diff().dropna() > 0).all()
    # Alignment check
    assert (df_feat["time"] == synthetic_m15_history["time"]).all()
    assert (df_feat["timestamp"] == synthetic_m15_history["timestamp"]).all()


def test_leakage_no_future_targets(synthetic_m15_history: pd.DataFrame) -> None:
    """Test D (No-target): Feature dataframe must not include any future targets or labels."""
    df_feat, defs = build_feature_pipeline(
        synthetic_m15_history, drop_warmup=False, include_raw_columns=False
    )

    forbidden_patterns = [
        "target",
        "label",
        "future",
        "next",
        "lead",
        "forward",
    ]
    for col in df_feat.columns:
        col_lower = col.lower()
        assert col_lower != "y", f"Forbidden target column 'y' found: '{col}'"
        for pat in forbidden_patterns:
            assert pat not in col_lower, (
                f"Forbidden target pattern '{pat}' found in feature '{col}'"
            )


def test_leakage_gap_preservation(synthetic_m15_history: pd.DataFrame) -> None:
    """Test E (Gap preservation): Market closures must NOT be filled with fake candles."""
    df_feat, _ = build_feature_pipeline(
        synthetic_m15_history, drop_warmup=False, include_raw_columns=True
    )

    # Length must be precisely identical to input (no synthetic row generation)
    assert len(df_feat) == len(synthetic_m15_history)

    # Weekend gap of 172,800s at bar 60 must be preserved exactly
    gap_delta = df_feat["time"].iloc[60] - df_feat["time"].iloc[59]
    assert gap_delta == 172800
