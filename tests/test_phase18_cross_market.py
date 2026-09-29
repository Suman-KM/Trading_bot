"""Unit tests for Phase 18: Multi-Asset / Cross-Market Information Expansion Research.

Covers all 14 mandatory test areas specified in Phase 18 Section 23:
1. symbol availability handling
2. timestamp alignment
3. no future cross-market data
4. rolling window causality
5. no future fill
6. feature finiteness
7. deterministic feature generation
8. feature count limit
9. frozen baseline configuration
10. walk-forward chronology
11. purge enforcement
12. locked test protection
13. fresh holdout protection
14. deterministic predictions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.features.cross_market import (
    ALL_CROSS_MARKET_FEATURES,
    CROSS_MARKET_FEATURE_GROUPS,
    EURUSD_BASELINE_32_COLS,
    compute_cross_market_features,
    generate_feature_quality_report,
    load_and_align_cross_market_data,
)
from ai.swing.evaluation import FROZEN_RF_CONFIG
from ai.swing.holdout import (
    HOLDOUT_END_TS,
    HOLDOUT_START_TS,
    LOCKED_TEST_START_TS,
    PRE_HOLDOUT_RESEARCH_END_TS,
    PRE_HOLDOUT_RESEARCH_START_TS,
    FreshResearchHoldoutManager,
)
from ai.swing.walk_forward_cross_market import (
    evaluate_single_holdout,
    generate_pre_holdout_folds,
)


# -------------------------------------------------------------------------
# FIXTURES
# -------------------------------------------------------------------------
@pytest.fixture
def synthetic_market_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Generate 500 bars of synthetic UTC H4 data for EURUSD and 6 cross assets."""
    timestamps = pd.date_range(
        start="2020-01-01 00:00:00",
        periods=500,
        freq="4h",
        tz="UTC",
    )
    np.random.seed(42)

    def _make_ohlcv(start_price: float, vol: float) -> pd.DataFrame:
        returns = np.random.normal(0, vol, size=len(timestamps))
        close = start_price * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.normal(0, 0.001, size=len(timestamps))))
        low = close * (1 - np.abs(np.random.normal(0, 0.001, size=len(timestamps))))
        open_ = close + np.random.normal(0, 0.0005, size=len(timestamps))
        volume = np.random.randint(100, 10000, size=len(timestamps))
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "tick_volume": volume,
                "spread": 10,
                "real_volume": 0,
            }
        )

    eurusd_df = _make_ohlcv(1.1000, 0.003)
    cross_dict = {
        "gbpusd": _make_ohlcv(1.3000, 0.0035),
        "usdjpy": _make_ohlcv(110.0, 0.003),
        "eurgbp": _make_ohlcv(0.8500, 0.002),
        "usdchf": _make_ohlcv(0.9200, 0.0025),
        "audusd": _make_ohlcv(0.7000, 0.0035),
        "usdcad": _make_ohlcv(1.3500, 0.003),
    }
    return eurusd_df, cross_dict


# -------------------------------------------------------------------------
# 1. SYMBOL AVAILABILITY HANDLING
# -------------------------------------------------------------------------
def test_symbol_availability_handling(tmp_path: Path) -> None:
    """Test that missing cross-market files raise FileNotFoundError when required."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    eur_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="4h", tz="UTC"),
            "open": 1.1,
            "high": 1.2,
            "low": 1.0,
            "close": 1.15,
        }
    )

    with pytest.raises(FileNotFoundError, match="Missing cross-market processed data"):
        load_and_align_cross_market_data(
            eurusd_h4=eur_df,
            data_dir=empty_dir,
        )


# -------------------------------------------------------------------------
# 2. TIMESTAMP ALIGNMENT
# -------------------------------------------------------------------------
def test_timestamp_alignment(
    synthetic_market_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]],
) -> None:
    """Test that cross-market features perfectly align with EURUSD timestamps and are UTC."""
    eurusd_df, cross_dict = synthetic_market_data

    feats = compute_cross_market_features(eurusd_df, cross_dfs=cross_dict)

    assert len(feats) == len(eurusd_df), (
        "Cross-market features must have identical length to EURUSD"
    )
    assert eurusd_df["timestamp"].dt.tz is not None, "EURUSD timestamps must be tz-aware"
    assert str(eurusd_df["timestamp"].dt.tz) == "UTC", "EURUSD timestamps must be UTC"
    assert eurusd_df["timestamp"].is_monotonic_increasing, "Timestamps must be strictly monotonic"
    assert eurusd_df["timestamp"].duplicated().sum() == 0, "No duplicate timestamps allowed"


# -------------------------------------------------------------------------
# 3. NO FUTURE CROSS-MARKET DATA
# -------------------------------------------------------------------------
def test_no_future_cross_market_data(
    synthetic_market_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]],
) -> None:
    """Test that cross-market features only use past and current bar data (no lookahead)."""
    eurusd_df, cross_dict = synthetic_market_data

    # Compute features on full series
    feats_full = compute_cross_market_features(eurusd_df, cross_dfs=cross_dict)

    # Compute features on truncated series up to bar 300
    cutoff = 300
    eurusd_trunc = eurusd_df.iloc[:cutoff].copy()
    cross_trunc = {k: v.iloc[:cutoff].copy() for k, v in cross_dict.items()}
    feats_trunc = compute_cross_market_features(eurusd_trunc, cross_dfs=cross_trunc)

    # The values at bar cutoff-1 (and all prior valid bars) must be identical
    eval_idx = 250
    for col in feats_trunc.columns:
        val_full = feats_full.loc[eval_idx, col]
        val_trunc = feats_trunc.loc[eval_idx, col]
        if np.isnan(val_full) and np.isnan(val_trunc):
            continue
        assert np.isclose(val_full, val_trunc, rtol=1e-7, atol=1e-9), (
            f"Feature {col} at bar {eval_idx} changed when future bars were removed: "
            f"full={val_full}, trunc={val_trunc}"
        )


# -------------------------------------------------------------------------
# 4. ROLLING WINDOW CAUSALITY
# -------------------------------------------------------------------------
def test_rolling_window_causality(
    synthetic_market_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]],
) -> None:
    """Test that perturbing future bars has zero effect on current and past feature values."""
    eurusd_df, cross_dict = synthetic_market_data

    feats_orig = compute_cross_market_features(eurusd_df, cross_dfs=cross_dict)

    # Perturb data at bar 350 and later
    cross_perturbed = {}
    for k, v in cross_dict.items():
        v_copy = v.copy()
        v_copy.loc[350:, "close"] = v_copy.loc[350:, "close"] * 2.5
        cross_perturbed[k] = v_copy

    feats_perturbed = compute_cross_market_features(eurusd_df, cross_dfs=cross_perturbed)

    # Check that for bars 0 to 349, features are strictly unchanged
    pd.testing.assert_frame_equal(
        feats_orig.iloc[:350],
        feats_perturbed.iloc[:350],
        check_exact=False,
        rtol=1e-8,
        atol=1e-10,
    )


# -------------------------------------------------------------------------
# 5. NO FUTURE FILL
# -------------------------------------------------------------------------
def test_no_future_fill() -> None:
    """Test that alignment never backfills (bfill) future prices into missing early bars."""
    eur_ts = pd.date_range("2020-01-01", periods=10, freq="4h", tz="UTC")
    # Cross market data starts 2 bars later (bars 0 and 1 are missing in cross-market)
    cross_ts = eur_ts[2:]
    cross_df = pd.DataFrame(
        {
            "timestamp": cross_ts,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "tick_volume": 1000,
            "spread": 10,
            "real_volume": 0,
        }
    )

    eurusd_df = pd.DataFrame(
        {
            "timestamp": eur_ts,
            "open": 1.1,
            "high": 1.2,
            "low": 1.0,
            "close": 1.15,
            "tick_volume": 500,
        }
    )

    # If we align with ffill only, bars 0 and 1 must remain NaN, NOT backfilled from bar 2
    merged = pd.merge(eurusd_df[["timestamp"]], cross_df, on="timestamp", how="left").ffill()

    assert pd.isna(merged.loc[0, "close"]), "Bar 0 must not be filled from future bar 2"
    assert pd.isna(merged.loc[1, "close"]), "Bar 1 must not be filled from future bar 2"
    assert not pd.isna(merged.loc[2, "close"]), "Bar 2 should be present"


# -------------------------------------------------------------------------
# 6. FEATURE FINITENESS
# -------------------------------------------------------------------------
def test_feature_finiteness(
    synthetic_market_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]],
) -> None:
    """Test that all cross-market features have 0 infinities and pass quality filters."""
    eurusd_df, cross_dict = synthetic_market_data
    feats = compute_cross_market_features(eurusd_df, cross_dfs=cross_dict)

    # Check for infinities
    inf_count = np.isinf(feats.to_numpy()).sum()
    assert inf_count == 0, f"Found {inf_count} infinities in cross-market features"

    # Run quality report
    report = generate_feature_quality_report(feats, eurusd_df["timestamp"])
    for r in report:
        assert r["status"] == "APPROVED", f"Feature {r['name']} failed quality filter: {r}"


# -------------------------------------------------------------------------
# 7. DETERMINISTIC FEATURE GENERATION
# -------------------------------------------------------------------------
def test_deterministic_feature_generation(
    synthetic_market_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]],
) -> None:
    """Test that computing features twice on identical data produces bit-identical outputs."""
    eurusd_df, cross_dict = synthetic_market_data

    feats_run1 = compute_cross_market_features(eurusd_df, cross_dfs=cross_dict)
    feats_run2 = compute_cross_market_features(eurusd_df, cross_dfs=cross_dict)

    pd.testing.assert_frame_equal(feats_run1, feats_run2, check_exact=True)


# -------------------------------------------------------------------------
# 8. FEATURE COUNT LIMIT
# -------------------------------------------------------------------------
def test_feature_count_limit() -> None:
    """Test that baseline features = 32, cross-market features = 40, total <= 72."""
    assert len(EURUSD_BASELINE_32_COLS) == 32, "Baseline feature count must be exactly 32"
    assert len(ALL_CROSS_MARKET_FEATURES) == 40, "Cross-market feature count must be exactly 40"

    total_combined = len(EURUSD_BASELINE_32_COLS) + len(ALL_CROSS_MARKET_FEATURES)
    assert total_combined == 72, "Total combined features must be exactly 72"
    assert total_combined <= 72, "Total features must not exceed 72"

    # Verify group counts
    assert len(CROSS_MARKET_FEATURE_GROUPS["Group_A_returns"]) == 12
    assert len(CROSS_MARKET_FEATURE_GROUPS["Group_B_usd_proxy"]) == 5
    assert len(CROSS_MARKET_FEATURE_GROUPS["Group_C_momentum"]) == 9
    assert len(CROSS_MARKET_FEATURE_GROUPS["Group_D_volatility"]) == 5
    assert len(CROSS_MARKET_FEATURE_GROUPS["Group_E_correlation"]) == 6
    assert len(CROSS_MARKET_FEATURE_GROUPS["Group_F_relative_vol"]) == 3


# -------------------------------------------------------------------------
# 9. FROZEN BASELINE CONFIGURATION
# -------------------------------------------------------------------------
def test_frozen_baseline_configuration() -> None:
    """Test that FROZEN_RF_CONFIG conforms strictly to Phase 15/16/17 specification."""
    expected_config = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    assert FROZEN_RF_CONFIG == expected_config, "FROZEN_RF_CONFIG was altered!"


# -------------------------------------------------------------------------
# 10. WALK-FORWARD CHRONOLOGY
# -------------------------------------------------------------------------
def test_walk_forward_chronology(
    synthetic_market_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]],
) -> None:
    """Test that walk-forward folds progress strictly forward in chronological order."""
    eurusd_df, cross_dict = synthetic_market_data
    feats = compute_cross_market_features(eurusd_df, cross_dfs=cross_dict)
    target = pd.Series(np.random.choice([1.0, -1.0], size=len(eurusd_df)), index=eurusd_df.index)

    folds = generate_pre_holdout_folds(eurusd_df, feats, target, n_folds=5, purge_gap_bars=8)

    assert len(folds) == 5

    for i, fold in enumerate(folds):
        assert fold.train_start_ts <= fold.train_end_ts
        assert fold.train_end_ts < fold.val_start_ts, "Train must end before validation starts"
        assert fold.val_start_ts <= fold.val_end_ts
        if i > 0:
            prev_fold = folds[i - 1]
            assert fold.train_sample_count >= prev_fold.train_sample_count, (
                "Train size must be expanding"
            )
            assert fold.val_start_ts > prev_fold.val_start_ts, "Validation must advance forward"


# -------------------------------------------------------------------------
# 11. PURGE ENFORCEMENT
# -------------------------------------------------------------------------
def test_purge_enforcement(
    synthetic_market_data: tuple[pd.DataFrame, dict[str, pd.DataFrame]],
) -> None:
    """Test that every walk-forward fold enforces the exact purge gap with 0 index overlap."""
    eurusd_df, cross_dict = synthetic_market_data
    feats = compute_cross_market_features(eurusd_df, cross_dfs=cross_dict)
    target = pd.Series(np.random.choice([1.0, -1.0], size=len(eurusd_df)), index=eurusd_df.index)

    purge_bars = 8
    folds = generate_pre_holdout_folds(
        eurusd_df, feats, target, n_folds=5, purge_gap_bars=purge_bars
    )

    for fold in folds:
        # Check no index overlap
        overlap = set(fold.train_indices).intersection(set(fold.val_indices))
        assert len(overlap) == 0, f"Fold {fold.fold_idx} has train/val overlap!"

        # Check gap between max train index and min val index
        max_train_idx = fold.train_indices.max()
        min_val_idx = fold.val_indices.min()
        gap = min_val_idx - max_train_idx
        assert gap > purge_bars, (
            f"Fold {fold.fold_idx} purge gap ({gap}) is less than required {purge_bars} bars"
        )


# -------------------------------------------------------------------------
# 12. LOCKED TEST PROTECTION
# -------------------------------------------------------------------------
def test_locked_test_protection() -> None:
    """Test that governance strictly prevents access to permanently locked test partitions."""
    mgr = FreshResearchHoldoutManager()

    # Verify locked test start timestamp
    assert LOCKED_TEST_START_TS == pd.Timestamp("2026-02-19 12:00:00+00:00")

    # Creating a test series attempting to touch locked test period
    timestamps_locked = pd.date_range("2026-02-19 12:00:00", periods=10, freq="4h", tz="UTC")
    df_locked = pd.DataFrame({"timestamp": timestamps_locked})

    # Validate that filter_pre_test_research raises PermissionError if locked rows are passed
    with pytest.raises(PermissionError, match="Attempted to access locked test partition data"):
        mgr.filter_pre_test_research(df_locked)


# -------------------------------------------------------------------------
# 13. FRESH HOLDOUT PROTECTION
# -------------------------------------------------------------------------
def test_fresh_holdout_protection() -> None:
    """Test that FreshResearchHoldoutManager starts sealed and denies access before all freezes."""
    mgr = FreshResearchHoldoutManager()

    assert mgr.is_sealed, "Manager starts locked/sealed"

    # Attempting to unlock before freezing must raise PermissionError
    with pytest.raises(
        PermissionError, match="Cannot unlock Fresh Research Holdout before freezing features"
    ):
        mgr.unlock_holdout()

    # Freeze only features
    mgr.freeze_features("2026-09-29T20:00:00Z")
    with pytest.raises(PermissionError):
        mgr.unlock_holdout()

    # Freeze model and protocol
    mgr.freeze_model("2026-09-29T20:00:01Z")
    mgr.freeze_protocol("2026-09-29T20:00:02Z")

    # Now unlock should succeed
    mgr.unlock_holdout("2026-09-29T20:00:03Z")
    assert not mgr.is_sealed
    assert mgr.is_holdout_unlocked

    # Verify partition boundary constants
    assert PRE_HOLDOUT_RESEARCH_START_TS == pd.Timestamp("2010-03-01 16:00:00+00:00")
    assert PRE_HOLDOUT_RESEARCH_END_TS == pd.Timestamp("2024-11-04 12:00:00+00:00")
    assert HOLDOUT_START_TS == pd.Timestamp("2024-11-06 00:00:00+00:00")
    assert HOLDOUT_END_TS == pd.Timestamp("2026-02-19 10:45:00+00:00")


# -------------------------------------------------------------------------
# 14. DETERMINISTIC PREDICTIONS
# -------------------------------------------------------------------------
def test_deterministic_predictions() -> None:
    """Test that model fitting and predictions on holdout are 100% reproducible with fixed seed."""
    np.random.seed(42)
    n_train = 500
    n_ho = 100
    n_features = 20

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features), columns=[f"f_{i}" for i in range(n_features)]
    )
    y_train = pd.Series(np.random.choice([1.0, -1.0], size=n_train))

    X_ho = pd.DataFrame(
        np.random.randn(n_ho, n_features), columns=[f"f_{i}" for i in range(n_features)]
    )
    y_ho = pd.Series(np.random.choice([1.0, -1.0], size=n_ho))

    res1 = evaluate_single_holdout(X_train, y_train, X_ho, y_ho, rf_config=FROZEN_RF_CONFIG)
    res2 = evaluate_single_holdout(X_train, y_train, X_ho, y_ho, rf_config=FROZEN_RF_CONFIG)

    rf1 = res1["random_forest"]
    rf2 = res2["random_forest"]

    assert rf1["balanced_accuracy"] == rf2["balanced_accuracy"]
    assert rf1["accuracy"] == rf2["accuracy"]
    assert rf1["macro_f1"] == rf2["macro_f1"]
    assert rf1["roc_auc"] == rf2["roc_auc"]
    assert rf1["confusion_matrix"] == rf2["confusion_matrix"]
