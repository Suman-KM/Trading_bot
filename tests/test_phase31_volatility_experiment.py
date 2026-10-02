"""Unit tests for Phase 31: Pre-Registered Realized Volatility Regime Forecasting Experiment.

Verifies:
- Mathematical accuracy of Parkinson, Garman-Klass, and Rogers-Satchell estimators.
- Exactly 24 autoregressive feature dimensions and feature naming conventions.
- Information leakage controls (causal lookback, purge=6, embargo=4, training median).
- Locked test partition quarantine preservation.
- Experiment JSON report schema and scientific verdict verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai.research.phase31_volatility_experiment import (
    EMBARGO_BARS,
    HORIZON_BARS,
    PURGE_BARS,
    REPORT_OUTPUT_PATH,
    compute_autoregressive_volatility_features,
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
    compute_rogers_satchell_volatility,
    generate_volatility_walk_forward_folds,
)
from ai.swing.holdout import (
    LOCKED_TEST_START_TS,
    PRE_HOLDOUT_RESEARCH_END_TS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Test Mathematical Accuracy of Volatility Estimators
# ---------------------------------------------------------------------------


def test_parkinson_volatility_flat_prices():
    """Flat prices (High == Low) must yield zero Parkinson volatility."""
    high = np.array([1.1000, 1.2000, 1.3000])
    low = np.array([1.1000, 1.2000, 1.3000])
    vol = compute_parkinson_volatility(high, low)
    assert np.allclose(vol, 0.0, atol=1e-8)


def test_parkinson_volatility_known_ratio():
    """Verify Parkinson formula with known values: ln(H/L)=0.01 -> sqrt(0.01^2 / (4*ln(2)))."""
    high = np.array([1.01])
    low = np.array([1.00])
    expected = np.sqrt((np.log(1.01)) ** 2 / (4.0 * np.log(2.0)))
    vol = compute_parkinson_volatility(high, low)
    assert np.isclose(vol[0], expected, rtol=1e-6)


def test_garman_klass_flat_prices():
    """Flat prices (Open == High == Low == Close) must yield zero GK volatility."""
    prices = np.array([1.1000, 1.2000, 1.3000])
    vol = compute_garman_klass_volatility(prices, prices, prices, prices)
    assert np.allclose(vol, 0.0, atol=1e-8)


def test_garman_klass_known_values():
    """Verify Garman-Klass formula calculation."""
    o_val, h_val, l_val, c_val = 1.00, 1.02, 0.99, 1.01
    term1 = 0.5 * (np.log(h_val / l_val)) ** 2
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(c_val / o_val)) ** 2
    expected = np.sqrt(max(0.0, term1 - term2))

    vol = compute_garman_klass_volatility(
        np.array([o_val]), np.array([h_val]), np.array([l_val]), np.array([c_val])
    )
    assert np.isclose(vol[0], expected, rtol=1e-6)


def test_rogers_satchell_flat_prices():
    """Flat prices must yield zero Rogers-Satchell volatility."""
    prices = np.array([1.1000, 1.2000, 1.3000])
    vol = compute_rogers_satchell_volatility(prices, prices, prices, prices)
    assert np.allclose(vol, 0.0, atol=1e-8)


def test_rogers_satchell_known_values():
    """Verify Rogers-Satchell formula with arbitrary positive bar."""
    o_val, h_val, l_val, c_val = 1.00, 1.02, 0.99, 1.01
    hc = np.log(h_val / c_val)
    ho = np.log(h_val / o_val)
    lc = np.log(l_val / c_val)
    lo = np.log(l_val / o_val)
    expected = np.sqrt(max(0.0, (hc * ho) + (lc * lo)))

    vol = compute_rogers_satchell_volatility(
        np.array([o_val]), np.array([h_val]), np.array([l_val]), np.array([c_val])
    )
    assert np.isclose(vol[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test Feature Matrix Construction & Leakage Controls
# ---------------------------------------------------------------------------


def test_feature_construction_exact_dimension_and_columns():
    """Verify that exactly 24 features are constructed with authorized names."""
    dates = pd.date_range("2020-01-01", periods=100, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(1.10, 1.15, 100),
            "high": np.linspace(1.11, 1.16, 100),
            "low": np.linspace(1.09, 1.14, 100),
            "close": np.linspace(1.10, 1.15, 100),
        },
        index=dates,
    )

    feat_df = compute_autoregressive_volatility_features(df)
    assert feat_df.shape[1] == 24
    assert len(feat_df) == 100

    estimators = ["parkinson", "garman_klass", "rogers_satchell"]
    subfeatures = ["1", "6", "12", "24", "72", "chg_1", "chg_6", "ratio_6_24"]

    expected_cols = [f"vol_{est}_{sub}" for est in estimators for sub in subfeatures]
    assert list(feat_df.columns) == expected_cols


def test_feature_causality_no_future_leakage():
    """Features at index t must be strictly invariant to modifications at index t+1."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    base_data = {
        "open": 1.10 + np.random.randn(n) * 0.005,
        "high": 1.11 + np.random.randn(n) * 0.005,
        "low": 1.09 + np.random.randn(n) * 0.005,
        "close": 1.10 + np.random.randn(n) * 0.005,
    }
    df1 = pd.DataFrame(base_data, index=dates)
    df1["high"] = np.maximum(df1["high"], np.maximum(df1["open"], df1["close"]) + 0.0001)
    df1["low"] = np.minimum(df1["low"], np.minimum(df1["open"], df1["close"]) - 0.0001)

    df2 = df1.copy()
    # Shock data at index 60 and onward
    df2.iloc[60:, df2.columns.get_loc("high")] += 0.50

    feat1 = compute_autoregressive_volatility_features(df1)
    feat2 = compute_autoregressive_volatility_features(df2)

    # All features up to index 59 must be completely identical
    pd.testing.assert_frame_equal(feat1.iloc[:60], feat2.iloc[:60])


# ---------------------------------------------------------------------------
# Test Walk-Forward Splits, Purge, and Embargo
# ---------------------------------------------------------------------------


def test_walk_forward_splits_purge_embargo():
    """Verify that walk-forward folds maintain strict purge, embargo, and causal ordering."""
    n_samples = 3000
    dates = pd.date_range("2015-01-01", periods=n_samples, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "close": np.linspace(1.10, 1.20, n_samples),
        },
        index=dates,
    )
    features_df = pd.DataFrame(np.random.randn(n_samples, 24), index=dates)
    fwd_rv = np.random.uniform(0.001, 0.005, n_samples)
    trail_rv = np.random.uniform(0.001, 0.005, n_samples)

    folds = generate_volatility_walk_forward_folds(
        df=df,
        features_df=features_df,
        fwd_rv=fwd_rv,
        trail_rv=trail_rv,
        initial_train_bars=1000,
        n_folds=5,
        purge_bars=PURGE_BARS,
        embargo_bars=EMBARGO_BARS,
    )

    assert len(folds) == 5

    for fold in folds:
        # Check index separation
        max_train_idx = np.max(fold.train_indices)
        min_val_idx = np.min(fold.val_indices)

        # Gap must be at least purge + embargo + 1
        gap = min_val_idx - max_train_idx
        assert gap >= (PURGE_BARS + EMBARGO_BARS + 1)

        # Timestamps must be strictly ordered
        assert fold.train_end_ts < fold.val_start_ts

        # Training threshold must be finite and positive
        assert fold.threshold_train > 0.0
        assert np.isfinite(fold.threshold_train)

        # Check training class distribution balance (approx 50/50 median split)
        n_high = fold.train_class_dist["high_volatility"]
        n_low = fold.train_class_dist["low_volatility"]
        assert abs(n_high - n_low) <= 2


# ---------------------------------------------------------------------------
# Test Locked Partition Quarantine
# ---------------------------------------------------------------------------


def test_locked_test_partition_quarantine():
    """Confirm pre-holdout end timestamp strictly precedes locked test start timestamp."""
    pre_end = pd.Timestamp(PRE_HOLDOUT_RESEARCH_END_TS)
    locked_start = pd.Timestamp(LOCKED_TEST_START_TS)

    assert pre_end < locked_start
    # Verify gap between research end and locked test
    gap_days = (locked_start - pre_end).days
    assert gap_days >= 400  # More than a year gap


# ---------------------------------------------------------------------------
# Test Phase 31 Report Schema & Scientific Verdict
# ---------------------------------------------------------------------------


def test_phase31_report_exists_and_valid():
    """Verify that Phase 31 report exists, passes schema validation, and confirms the verdict."""
    assert REPORT_OUTPUT_PATH.exists(), f"Report missing at {REPORT_OUTPUT_PATH}"

    with open(REPORT_OUTPUT_PATH, encoding="utf-8") as f:
        report = json.load(f)

    # Core metadata
    assert report["phase"] == 31
    assert "governance" in report
    assert report["governance"]["target_horizon_bars"] == HORIZON_BARS
    assert report["governance"]["purge_bars"] == PURGE_BARS
    assert report["governance"]["embargo_bars"] == EMBARGO_BARS
    assert report["governance"]["locked_test_partition_status"] == ("QUARANTINED_AND_UNTOUCHED")
    assert report["governance"]["trading_backtests_executed"] == 0
    assert report["governance"]["live_or_demo_trades"] == 0

    # 10 Folds
    assert len(report["fold_results"]) == 10

    # Aggregate metrics
    agg = report["aggregate_metrics"]
    comp = agg["comparison_statistics"]

    mean_base_bal_acc = agg["persistence_baseline"]["balanced_accuracy"]["mean"]
    mean_model_bal_acc = agg["volatility_model"]["balanced_accuracy"]["mean"]
    mean_diff = comp["mean_difference"]

    assert 0.55 < mean_base_bal_acc < 0.60
    assert 0.60 <= mean_model_bal_acc < 0.65
    assert mean_diff > 0.0

    # Gates verification
    gates = report["gates"]
    assert gates["success_condition_1_mean_ge_60"] is True
    assert gates["success_condition_2_pval_lt_01"] is False
    assert gates["success_gate_passed"] is False
    assert gates["failure_gate_triggered"] is True

    # Scientific verdict
    assert report["scientific_verdict"] == "NOT SUPPORTED — VOLATILITY FORECASTING FAILED"
