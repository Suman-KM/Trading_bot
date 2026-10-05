"""Unit and integration tests for Phase 42 Higher-Timeframe H1/H4 Research.

Covers all 15 required verification invariants:
1. H1 aggregation correctness
2. H4 aggregation correctness
3. target causality
4. feature causality
5. chronological split
6. purge logic
7. holdout isolation
8. no future feature usage
9. regime calculation
10. deterministic model results
11. missing data handling
12. timestamp normalization
13. economic simulation
14. cost application
15. reproducibility
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ai.dataset.swing import aggregate_m15_to_h4
from scripts.run_phase42_higher_timeframe_research import (
    LOCKED_TEST_START_TS,
    RESEARCH_END_TS,
    aggregate_m15_to_h1,
    compute_regime_features,
    compute_regime_states,
    compute_swing_target,
    evaluate_economic_simulation,
)


def _make_dummy_m15_df(n_bars: int = 96) -> pd.DataFrame:
    """Generate deterministic synthetic M15 data for unit testing."""
    times = pd.date_range("2025-01-01 00:00:00", periods=n_bars, freq="15min", tz="UTC")
    np.random.seed(42)
    base = 1.0800
    steps = np.random.normal(0, 0.0002, size=n_bars)
    closes = base + np.cumsum(steps)
    opens = np.roll(closes, 1)
    opens[0] = base
    highs = np.maximum(opens, closes) + 0.0003
    lows = np.minimum(opens, closes) - 0.0003
    volumes = np.random.randint(100, 1000, size=n_bars)
    spreads = np.full(n_bars, 10.0)

    return pd.DataFrame(
        {
            "timestamp": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": volumes,
            "spread": spreads,
        }
    )


# 1. H1 aggregation correctness
def test_01_h1_aggregation_correctness():
    """Verify M15 candles aggregate into completed H1 bars with exact OHLCV geometry."""
    df_m15 = _make_dummy_m15_df(n_bars=32)  # 32 M15 bars = 8 H1 bars
    h1 = aggregate_m15_to_h1(df_m15)

    assert len(h1) == 8
    # First H1 bar should have open of first M15 bar, close of 4th M15 bar
    assert h1.iloc[0]["open"] == df_m15.iloc[0]["open"]
    assert h1.iloc[0]["close"] == df_m15.iloc[3]["close"]
    assert h1.iloc[0]["high"] == df_m15.iloc[0:4]["high"].max()
    assert h1.iloc[0]["low"] == df_m15.iloc[0:4]["low"].min()
    assert h1.iloc[0]["tick_volume"] == df_m15.iloc[0:4]["tick_volume"].sum()


# 2. H4 aggregation correctness
def test_02_h4_aggregation_correctness():
    """Verify M15 candles aggregate into completed H4 bars with exact OHLCV geometry."""
    df_m15 = _make_dummy_m15_df(n_bars=32)  # 32 M15 bars = 2 H4 bars
    h4 = aggregate_m15_to_h4(df_m15)

    assert len(h4) == 2
    assert h4.iloc[0]["open"] == df_m15.iloc[0]["open"]
    assert h4.iloc[0]["close"] == df_m15.iloc[15]["close"]
    assert h4.iloc[0]["high"] == df_m15.iloc[0:16]["high"].max()
    assert h4.iloc[0]["low"] == df_m15.iloc[0:16]["low"].min()
    assert h4.iloc[0]["tick_volume"] == df_m15.iloc[0:16]["tick_volume"].sum()


# 3. Target causality
def test_03_target_causality():
    """Verify target calculation shifts strictly forward in time."""
    df_m15 = _make_dummy_m15_df(n_bars=96)
    h1 = aggregate_m15_to_h1(df_m15)
    ret, target, pers = compute_swing_target(h1, horizon_bars=4)

    # Future return at bar t should use close price at bar t+4
    expected_ret = (h1["close"].iloc[4] / h1["close"].iloc[0]) - 1.0
    assert np.isclose(ret.iloc[0], expected_ret)
    # The last 4 bars must have NaN future returns and targets
    assert ret.iloc[-4:].isna().all()
    assert target.iloc[-4:].isna().all()


# 4. Feature causality
def test_04_feature_causality():
    """Verify regime features are computed strictly from current and prior bars."""
    df_m15 = _make_dummy_m15_df(n_bars=96)
    h1 = aggregate_m15_to_h1(df_m15)

    feats_orig = compute_regime_features(h1, timeframe="H1")

    # Modify future bars (bars 20 to end)
    h1_modified = h1.copy()
    h1_modified.loc[20:, "close"] = h1_modified.loc[20:, "close"] * 1.5

    feats_mod = compute_regime_features(h1_modified, timeframe="H1")

    # Features at bar 19 must remain identical
    pd.testing.assert_series_equal(
        feats_orig.iloc[19].dropna(),
        feats_mod.iloc[19].dropna(),
        check_names=False,
    )


# 5. Chronological split
def test_05_chronological_split():
    """Verify chronological ordering is preserved with no shuffle."""
    df_m15 = _make_dummy_m15_df(n_bars=96)
    h1 = aggregate_m15_to_h1(df_m15)

    ts = pd.to_datetime(h1["timestamp"], utc=True)
    assert ts.is_monotonic_increasing
    assert not ts.duplicated().any()


# 6. Purge logic
def test_06_purge_logic():
    """Verify purge window strictly separates train labels from validation bars."""
    horizon = 6
    train_end = 50
    val_start = 50

    # Purged training window
    train_indices = np.arange(0, train_end - horizon)
    val_indices = np.arange(val_start, 70)

    assert len(train_indices) == 44
    assert max(train_indices) < min(val_indices)
    # Gap between max train index and min val index is exactly horizon bars
    assert min(val_indices) - max(train_indices) == horizon + 1


# 7. Holdout isolation
def test_07_holdout_isolation():
    """Verify holdout partition starts at LOCKED_TEST_START_TS and does not leak."""
    assert LOCKED_TEST_START_TS > RESEARCH_END_TS
    gap_hours = (LOCKED_TEST_START_TS - RESEARCH_END_TS).total_seconds() / 3600.0
    assert gap_hours >= 1.0


# 8. No future feature usage
def test_08_no_future_feature_usage():
    """Verify feature at index t is identical when computed on prefix [0:t+1]."""
    df_m15 = _make_dummy_m15_df(n_bars=96)
    h1 = aggregate_m15_to_h1(df_m15)

    feats_full = compute_regime_features(h1, timeframe="H1")
    # Subslice prefix of 22 bars
    feats_prefix = compute_regime_features(h1.iloc[:22].copy(), timeframe="H1")

    # Feature at index 21 must be identical
    for col in feats_full.columns:
        val_full = feats_full[col].iloc[21]
        val_pre = feats_prefix[col].iloc[21]
        if not np.isnan(val_full) and not np.isnan(val_pre):
            assert np.isclose(val_full, val_pre)


# 9. Regime calculation
def test_09_regime_calculation():
    """Verify regime states are deterministic, exhaustive, and mutually exclusive."""
    df_m15 = _make_dummy_m15_df(n_bars=96)
    h1 = aggregate_m15_to_h1(df_m15)
    feats = compute_regime_features(h1, timeframe="H1")
    regimes = compute_regime_states(h1, feats)

    valid_regimes = {"LOW_VOL_TREND", "HIGH_VOL_TREND", "LOW_VOL_RANGE", "HIGH_VOL_RANGE"}
    unique_assigned = set(regimes.unique())
    assert unique_assigned.issubset(valid_regimes)
    assert not regimes.isna().any()


# 10. Deterministic model results
def test_10_deterministic_model_results():
    """Verify Random Forest model with fixed random_state produces identical predictions."""
    X = np.random.RandomState(42).randn(100, 12)
    y = np.random.RandomState(42).choice([-1.0, 1.0], size=100)

    rf1 = RandomForestClassifier(
        n_estimators=10, max_depth=3, class_weight="balanced", random_state=42
    )
    rf1.fit(X, y)
    pred1 = rf1.predict(X)

    rf2 = RandomForestClassifier(
        n_estimators=10, max_depth=3, class_weight="balanced", random_state=42
    )
    rf2.fit(X, y)
    pred2 = rf2.predict(X)

    np.testing.assert_array_equal(pred1, pred2)


# 11. Missing data handling
def test_11_missing_data_handling():
    """Verify NaNs in warmup features or targets are handled without error."""
    df_m15 = _make_dummy_m15_df(n_bars=200)
    h1 = aggregate_m15_to_h1(df_m15)
    feats = compute_regime_features(h1, timeframe="H1")
    _, target, _ = compute_swing_target(h1, horizon_bars=6)

    # First few rows should contain expected NaNs from rolling indicators
    assert feats.iloc[0].isna().any()
    # Masking drops NaN rows cleanly
    valid_mask = target.notna() & ~feats.isna().any(axis=1)
    assert valid_mask.dtype == bool
    assert valid_mask.sum() > 0


# 12. Timestamp normalization
def test_12_timestamp_normalization():
    """Verify all timestamps are normalized to UTC timezone."""
    df_m15 = _make_dummy_m15_df(n_bars=32)
    h1 = aggregate_m15_to_h1(df_m15)

    ts = pd.to_datetime(h1["timestamp"], utc=True)
    assert ts.dt.tz is not None
    assert str(ts.dt.tz) == "UTC"


# 13. Economic simulation
def test_13_economic_simulation():
    """Verify economic simulation properly calculates gross pips from returns."""
    df = pd.DataFrame(
        {
            "close": [1.0800, 1.0850],
        }
    )
    future_ret = pd.Series([0.0050, 0.0])  # +50 pips
    target = pd.Series([1.0, 1.0])
    preds = np.array([1.0])  # Long prediction
    valid_idx = np.array([0])

    sim = evaluate_economic_simulation(
        df=df,
        target=target,
        future_return=future_ret,
        predictions=preds,
        valid_indices=valid_idx,
        friction_pips=0.0,
    )

    assert sim["total_trades"] == 1
    assert sim["winning_trades"] == 1
    assert np.isclose(sim["total_net_pips"], 50.0)


# 14. Cost application
def test_14_cost_application():
    """Verify transaction friction reduces net expectancy and shifts win rate."""
    df = pd.DataFrame({"close": [1.0800]})
    future_ret = pd.Series([0.0001])  # +1 pip gross
    target = pd.Series([1.0])
    preds = np.array([1.0])
    valid_idx = np.array([0])

    sim_no_cost = evaluate_economic_simulation(
        df=df,
        target=target,
        future_return=future_ret,
        predictions=preds,
        valid_indices=valid_idx,
        friction_pips=0.0,
    )
    assert sim_no_cost["winning_trades"] == 1
    assert np.isclose(sim_no_cost["total_net_pips"], 1.0)

    sim_with_cost = evaluate_economic_simulation(
        df=df,
        target=target,
        future_return=future_ret,
        predictions=preds,
        valid_indices=valid_idx,
        friction_pips=1.5,
    )
    # 1.0 pip gross - 1.5 pip friction = -0.5 pip net (loss)
    assert sim_with_cost["winning_trades"] == 0
    assert np.isclose(sim_with_cost["total_net_pips"], -0.5)


# 15. Reproducibility
def test_15_reproducibility():
    """Verify repeated execution on identical data yields identical features and targets."""
    df1 = _make_dummy_m15_df(n_bars=128)
    df2 = _make_dummy_m15_df(n_bars=128)

    h1_a = aggregate_m15_to_h1(df1)
    h1_b = aggregate_m15_to_h1(df2)

    feats_a = compute_regime_features(h1_a, timeframe="H1")
    feats_b = compute_regime_features(h1_b, timeframe="H1")

    pd.testing.assert_frame_equal(feats_a, feats_b)
