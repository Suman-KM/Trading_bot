"""Unit and leakage test suite for Phase 6 label/target engineering.

Verifies mathematical forward return definitions, multi-horizon directional class boundaries,
boundary NaN handling at dataset end, and strict separation between feature matrix X(t)
and target matrix y(t+H).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.features import build_feature_pipeline
from ai.labels import (
    DEFAULT_FIXED_THRESHOLDS,
    DEFAULT_HORIZONS,
    build_label_pipeline,
    compute_direction_labels,
    compute_future_returns,
    generate_label_metadata,
    get_label_registry,
)


@pytest.fixture
def synthetic_m15_series() -> pd.DataFrame:
    """Create a deterministic synthetic EURUSD M15 DataFrame.

    Covers 60 consecutive M15 candles starting Monday 2026-09-21 00:00:00 UTC (epoch 1789948800).
    """
    n = 60
    base_epoch = 1789948800
    times = [base_epoch + (i * 900) for i in range(n)]

    # Deterministic price oscillation
    np.random.seed(42)
    drift = np.linspace(0, 0.0050, n)
    osc = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.0010
    prices = 1.0850 + drift + osc

    opens = prices - 0.0001
    closes = prices + 0.0001
    highs = np.maximum(opens, closes) + 0.0002
    lows = np.minimum(opens, closes) - 0.0002
    tick_volumes = [500 + (i * 10) for i in range(n)]
    spreads = [12 + (i % 3) for i in range(n)]
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
# 1. Target Pipeline Integration Tests
# -----------------------------------------------------------------------------


def test_compute_future_returns(synthetic_m15_series: pd.DataFrame) -> None:
    """Verify forward return calculations across all horizons."""
    res = compute_future_returns(synthetic_m15_series, horizons=[1, 4, 8, 16])

    for h in [1, 4, 8, 16]:
        assert f"future_return_{h}" in res.columns
        assert f"future_log_return_{h}" in res.columns
        assert f"future_vol_adj_return_{h}" in res.columns

        # Verify mathematical identity: future_return = close[t+h] / close[t] - 1
        expected_ret = (
            synthetic_m15_series["close"].shift(-h) / synthetic_m15_series["close"]
        ) - 1.0
        pd.testing.assert_series_equal(res[f"future_return_{h}"], expected_ret, check_names=False)

        # Verify end-of-data NaNs: exactly final h rows must be NaN
        assert res[f"future_return_{h}"].iloc[-h:].isna().all()
        assert res[f"future_return_{h}"].iloc[:-h].notna().all()


def test_compute_direction_labels(synthetic_m15_series: pd.DataFrame) -> None:
    """Verify multiclass directional label assignment."""
    returns_df = compute_future_returns(synthetic_m15_series, horizons=[1, 4, 8, 16])
    labels = compute_direction_labels(synthetic_m15_series, returns_df, horizons=[1, 4, 8, 16])

    for h in [1, 4, 8, 16]:
        assert f"direction_{h}" in labels.columns
        assert f"direction_vol_{h}" in labels.columns

        # All non-NaN labels must strictly be -1.0, 0.0, or +1.0
        clean_labels = labels[f"direction_{h}"].dropna()
        assert clean_labels.isin([-1.0, 0.0, 1.0]).all()

        # Final h rows must be NaN
        assert labels[f"direction_{h}"].iloc[-h:].isna().all()


def test_build_label_pipeline(synthetic_m15_series: pd.DataFrame) -> None:
    """Verify pipeline orchestration, dimensions, and metadata generation."""
    df_labels, label_defs = build_label_pipeline(
        synthetic_m15_series, horizons=DEFAULT_HORIZONS, include_timestamp=True
    )

    assert len(label_defs) == 20
    # 2 timestamp columns ('time', 'timestamp') + 20 target columns = 22 columns
    assert df_labels.shape == (60, 22)

    metadata = generate_label_metadata(
        df_labels, label_defs, spread_series=synthetic_m15_series["spread"]
    )
    assert metadata["total_labels"] == 20
    assert metadata["total_rows"] == 60
    assert len(metadata["labels"]) == 20


def test_label_registry() -> None:
    """Verify default label registry defines 20 targets across candidate horizons."""
    registry = get_label_registry(DEFAULT_HORIZONS)
    assert len(registry) == 20

    horizons_in_reg = set(d.horizon_bars for d in registry)
    assert horizons_in_reg == {1, 4, 8, 16}


# -----------------------------------------------------------------------------
# 2. Mandatory Label Leakage Tests (Section 12)
# -----------------------------------------------------------------------------


def test_leakage_test_1_label_future_dependency(
    synthetic_m15_series: pd.DataFrame,
) -> None:
    """TEST 1: Changing close[t+H] MUST change label[t]; changing past close[t-1] must not."""
    df_orig = synthetic_m15_series.copy()
    labels_orig, _ = build_label_pipeline(df_orig, horizons=[4], include_timestamp=False)

    t = 25
    h = 4
    # Modifying close at t + h (future candle for t)
    df_future_mod = synthetic_m15_series.copy()
    df_future_mod.loc[t + h, "close"] *= 1.05

    labels_future_mod, _ = build_label_pipeline(
        df_future_mod, horizons=[4], include_timestamp=False
    )

    # 1. Label at t MUST have changed because close[t+h] changed
    assert (
        labels_orig.loc[t, f"future_return_{h}"] != labels_future_mod.loc[t, f"future_return_{h}"]
    )

    # 2. Modifying close at t - 1 should NOT change future_return at t
    df_past_mod = synthetic_m15_series.copy()
    df_past_mod.loc[t - 1, "close"] *= 1.05

    labels_past_mod, _ = build_label_pipeline(df_past_mod, horizons=[4], include_timestamp=False)
    assert labels_orig.loc[t, f"future_return_{h}"] == labels_past_mod.loc[t, f"future_return_{h}"]


def test_leakage_test_2_feature_immutability(
    synthetic_m15_series: pd.DataFrame,
) -> None:
    """TEST 2: Adding future candles must not alter existing feature values."""
    t = 30
    df_base = synthetic_m15_series.iloc[: t + 1].copy()
    df_extended = synthetic_m15_series.copy()

    features_base, _ = build_feature_pipeline(df_base, drop_warmup=False, include_raw_columns=False)
    features_extended, _ = build_feature_pipeline(
        df_extended, drop_warmup=False, include_raw_columns=False
    )

    # Feature matrix at or before t must be completely unaffected by future rows
    pd.testing.assert_frame_equal(features_base, features_extended.iloc[: t + 1])


def test_leakage_test_3_horizon_alignment(
    synthetic_m15_series: pd.DataFrame,
) -> None:
    """TEST 3: Verify future_return_H[t] == close[t+H] / close[t] - 1 for all H."""
    df_labels, _ = build_label_pipeline(
        synthetic_m15_series, horizons=[1, 4, 8, 16], include_timestamp=False
    )
    close = synthetic_m15_series["close"]

    for h in [1, 4, 8, 16]:
        for t in range(len(synthetic_m15_series) - h):
            expected = (close.iloc[t + h] / close.iloc[t]) - 1.0
            actual = df_labels.loc[t, f"future_return_{h}"]
            np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_leakage_test_4_end_of_data_handling(
    synthetic_m15_series: pd.DataFrame,
) -> None:
    """TEST 4: Final H rows must have unavailable labels (NaN). No fabricated classes."""
    df_labels, _ = build_label_pipeline(
        synthetic_m15_series, horizons=[1, 4, 8, 16], include_timestamp=False
    )

    for h in [1, 4, 8, 16]:
        ret_col = f"future_return_{h}"
        dir_col = f"direction_{h}"

        # Final H rows must be strictly NaN
        assert df_labels[ret_col].iloc[-h:].isna().all()
        assert df_labels[dir_col].iloc[-h:].isna().all()

        # Preceding rows must be populated
        assert df_labels[ret_col].iloc[:-h].notna().all()
        assert df_labels[dir_col].iloc[:-h].notna().all()


def test_leakage_test_5_class_label_correctness(
    synthetic_m15_series: pd.DataFrame,
) -> None:
    """TEST 5: Verify LONG / NEUTRAL / SHORT boundaries exactly match documented threshold."""
    df_labels, _ = build_label_pipeline(
        synthetic_m15_series, horizons=[1, 4, 8, 16], include_timestamp=False
    )

    for h in [1, 4, 8, 16]:
        th = DEFAULT_FIXED_THRESHOLDS[h]
        ret = df_labels[f"future_return_{h}"].dropna()
        dir_lbl = df_labels[f"direction_{h}"].dropna()

        # If ret > th => direction must be +1.0 (LONG)
        long_mask = ret > th
        assert (dir_lbl[long_mask] == 1.0).all()

        # If ret < -th => direction must be -1.0 (SHORT)
        short_mask = ret < -th
        assert (dir_lbl[short_mask] == -1.0).all()

        # Otherwise => direction must be 0.0 (NEUTRAL)
        neutral_mask = (ret >= -th) & (ret <= th)
        assert (dir_lbl[neutral_mask] == 0.0).all()


def test_leakage_test_6_no_label_in_feature_matrix(
    synthetic_m15_series: pd.DataFrame,
) -> None:
    """TEST 6: Future-return and directional labels must be absent from feature columns."""
    df_features, _ = build_feature_pipeline(
        synthetic_m15_series, drop_warmup=False, include_raw_columns=False
    )

    forbidden_label_names = [
        "future_return_1",
        "future_return_4",
        "future_return_8",
        "future_return_16",
        "future_log_return_1",
        "future_log_return_4",
        "direction_1",
        "direction_4",
        "direction_8",
        "direction_16",
    ]
    for col in df_features.columns:
        assert col not in forbidden_label_names, f"Label '{col}' found in feature matrix!"
        assert not col.startswith("future_"), f"Future prefix found in feature '{col}'!"
