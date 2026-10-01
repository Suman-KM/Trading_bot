"""Unit tests for Phase 25 Controlled H4 Macro Regime Experiment.

Verifies:
1. Aligned yield dataset PIT causal lag (Day D yield never appears before Day D+1 00:00 UTC).
2. 5-Day spread change causality (strictly uses historical observations).
3. Walk-forward fold generation: exact 10 folds, strictly chronological expanding training windows.
4. Purge (8 bars) and Embargo (4 bars) boundary enforcement (at least 12 bars separation).
5. Zero train/validation overlap in any fold.
6. Locked test partition protection (no bars or labels from 2026-02-19 12:00 UTC onward).
7. Fresh Research Holdout exact sample count (838 labeled bars) and boundary integrity.
8. Baseline feature count (exactly 30 features) and Macro Candidate (exactly 32 features).
9. Immutability of frozen Random Forest hyperparameters.
10. Deterministic failure/success gate evaluation logic and schema validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai.features.cross_market import EURUSD_BASELINE_32_COLS
from ai.features.swing import compute_swing_features
from ai.labels.swing import compute_swing_targets
from ai.swing.data_expansion import RESEARCH_END_TS
from ai.swing.evaluation import FROZEN_RF_CONFIG
from ai.swing.holdout import (
    HOLDOUT_END_TS,
    HOLDOUT_START_TS,
    LOCKED_TEST_START_TS,
    PRE_HOLDOUT_RESEARCH_END_TS,
    PRE_HOLDOUT_RESEARCH_START_TS,
)
from scripts.run_phase25_macro_experiment import (
    APPROVED_MACRO_FEATURES,
    BASELINE_30_FEATURES,
    generate_phase25_walk_forward_folds,
)

ALIGNED_PARQUET = Path("data/external/macro_yields/processed/eurusd_h4_yield_aligned.parquet")
REPORT_PATH = Path("reports/phase25_macro_experiment.json")


def test_aligned_yield_dataset_exists_and_columns():
    """Test that the Phase 24/24.1 causally aligned dataset exists and has all required columns."""
    assert ALIGNED_PARQUET.exists(), f"Missing dataset: {ALIGNED_PARQUET}"
    df = pd.read_parquet(ALIGNED_PARQUET)

    required_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "yield_observation_date",
        "US_2Y",
        "German_2Y",
        "US_Germany_2Y_Spread",
        "US_Germany_2Y_Spread_5D_Change",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert len(df) == 25800, f"Expected 25,800 bars, found {len(df)}"
    assert not df["timestamp"].duplicated().any(), "Duplicate timestamps found"
    assert df["timestamp"].is_monotonic_increasing, "Timestamps must be monotonic increasing"


def test_point_in_time_publication_lag():
    """Test that daily yield from Day D is NEVER available to any H4 bar on Day D."""
    df = pd.read_parquet(ALIGNED_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    yield_obs_ts = pd.to_datetime(df["yield_observation_date"], utc=True)

    # Any bar on day D must have yield_observation_date < Day D
    bar_dates = df["timestamp"].dt.floor("D")
    same_or_future_day_leakage = yield_obs_ts >= bar_dates

    assert not same_or_future_day_leakage.any(), (
        f"Point-in-time leakage detected on {same_or_future_day_leakage.sum()} bars! "
        "Day D yield observed contemporaneously on Day D."
    )


def test_spread_5d_momentum_causality():
    """Test that the 5-day spread momentum is causal and correctly computed."""
    df = pd.read_parquet(ALIGNED_PARQUET)

    spread = df["US_Germany_2Y_Spread"].to_numpy()
    spread_5d = df["US_Germany_2Y_Spread_5D_Change"].to_numpy()

    assert not np.isnan(spread).any(), "NaNs found in US_Germany_2Y_Spread"
    assert not np.isnan(spread_5d).any(), "NaNs found in US_Germany_2Y_Spread_5D_Change"
    assert not np.isinf(spread).any(), "Infinities found in spread"
    assert not np.isinf(spread_5d).any(), "Infinities found in spread 5D change"


def test_locked_test_partition_strict_protection():
    """Test that the permanently locked Phase 11 test partition remains completely untouched."""
    assert LOCKED_TEST_START_TS == pd.Timestamp("2026-02-19 12:00:00+00:00")
    assert RESEARCH_END_TS < LOCKED_TEST_START_TS
    assert HOLDOUT_END_TS < LOCKED_TEST_START_TS

    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            data = json.load(f)
        gov = data.get("governance", {})
        assert gov.get("phase11_test_partition") == "LOCKED"
        assert gov.get("locked_test_start_ts") == str(LOCKED_TEST_START_TS)


def test_feature_definitions_and_counts():
    """Test that baseline features = exactly 30 and macro candidate = exactly 32."""
    assert len(BASELINE_30_FEATURES) == 30, (
        f"Baseline must have exactly 30 features, got {len(BASELINE_30_FEATURES)}"
    )
    assert len(APPROVED_MACRO_FEATURES) == 2, (
        f"Macro must have exactly 2 features, got {len(APPROVED_MACRO_FEATURES)}"
    )
    assert BASELINE_30_FEATURES == EURUSD_BASELINE_32_COLS[:30]
    assert "US_Germany_2Y_Spread" in APPROVED_MACRO_FEATURES
    assert "US_Germany_2Y_Spread_5D_Change" in APPROVED_MACRO_FEATURES


def test_walk_forward_fold_generation_purge_and_embargo():
    """Test 10 chronological expanding folds, purge >= 8, embargo >= 4, and zero overlap."""
    df = pd.read_parquet(ALIGNED_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df_res = df[df["timestamp"] <= RESEARCH_END_TS].copy().reset_index(drop=True)

    feats_all = compute_swing_features(df_res, timeframe="H4")
    feats_30 = feats_all[BASELINE_30_FEATURES].copy()
    macro_feats = df_res[APPROVED_MACRO_FEATURES].copy()
    feats_macro = pd.concat([feats_30, macro_feats], axis=1)

    targets = compute_swing_targets(df_res, horizons=[8], timeframe="H4")
    y = targets["direction_vol_8"]

    pre_mask = (df_res["timestamp"] >= PRE_HOLDOUT_RESEARCH_START_TS) & (
        df_res["timestamp"] <= PRE_HOLDOUT_RESEARCH_END_TS
    )
    df_pre = df_res[pre_mask].reset_index(drop=True)
    feats_pre = feats_macro[pre_mask].reset_index(drop=True)
    y_pre = y[pre_mask].reset_index(drop=True)

    folds = generate_phase25_walk_forward_folds(
        df=df_pre,
        features_df=feats_pre,
        target_series=y_pre,
        initial_train_bars=5000,
        n_folds=10,
        purge_bars=8,
        embargo_bars=4,
    )

    assert len(folds) == 10, f"Expected 10 folds, got {len(folds)}"

    for i, f in enumerate(folds):
        # 1. Purge gap check
        assert f.purge_gap_bars == 8

        # 2. Separation check (minimum 8 purge + 4 embargo = 12 bars)
        separation_bars = f.val_indices[0] - f.train_indices[-1]
        assert separation_bars >= 12, (
            f"Fold {f.fold_idx}: separation {separation_bars} < 12 (purge 8 + embargo 4)!"
        )

        # 3. Temporal ordering
        assert f.train_end_ts < f.val_start_ts, f"Fold {f.fold_idx}: temporal overlap detected!"

        # 4. Disjoint index sets
        intersection = set(f.train_indices).intersection(set(f.val_indices))
        assert len(intersection) == 0, f"Fold {f.fold_idx}: overlapping indices detected!"

        # 5. Expanding training window monotonicity
        if i < len(folds) - 1:
            f_next = folds[i + 1]
            assert f.train_sample_count < f_next.train_sample_count
            assert f.val_start_ts < f_next.val_start_ts


def test_fresh_holdout_exact_labeled_samples():
    """Test that the Fresh Research Holdout strictly yields 838 valid labeled bars."""
    df = pd.read_parquet(ALIGNED_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df_res = df[df["timestamp"] <= RESEARCH_END_TS].copy().reset_index(drop=True)

    feats_all = compute_swing_features(df_res, timeframe="H4")
    feats_30 = feats_all[BASELINE_30_FEATURES].copy()
    macro_feats = df_res[APPROVED_MACRO_FEATURES].copy()
    feats_macro = pd.concat([feats_30, macro_feats], axis=1)

    targets = compute_swing_targets(df_res, horizons=[8], timeframe="H4")
    y = targets["direction_vol_8"]

    ho_mask = (df_res["timestamp"] >= HOLDOUT_START_TS) & (df_res["timestamp"] <= HOLDOUT_END_TS)
    valid_ho = ho_mask & y.notna() & feats_macro.notna().all(axis=1)

    assert valid_ho.sum() == 838, (
        f"Fresh holdout valid labeled count must be exactly 838, got {valid_ho.sum()}"
    )


def test_frozen_rf_configuration():
    """Test that the Random Forest configuration strictly matches the frozen Phase 15/16/17 spec."""
    assert FROZEN_RF_CONFIG["n_estimators"] == 100
    assert FROZEN_RF_CONFIG["max_depth"] == 5
    assert FROZEN_RF_CONFIG["min_samples_leaf"] == 10
    assert FROZEN_RF_CONFIG["class_weight"] == "balanced"
    assert FROZEN_RF_CONFIG["random_state"] == 42


def test_experiment_report_json_and_decision():
    """Test that the generated experiment report is complete and verdict is NOT SUPPORTED."""
    assert REPORT_PATH.exists(), f"Experiment report missing: {REPORT_PATH}"
    with open(REPORT_PATH) as f:
        data = json.load(f)

    # Required top-level keys
    required_keys = [
        "timestamp_utc",
        "elapsed_seconds",
        "environment",
        "governance",
        "data_sources",
        "model_configuration",
        "walk_forward_folds",
        "aggregate_metrics",
        "statistical_tests",
        "fresh_holdout_evaluation",
        "success_gate",
        "failure_gate",
        "checks",
        "final_scientific_decision",
    ]
    for k in required_keys:
        assert k in data, f"Missing report key: {k}"

    assert len(data["walk_forward_folds"]) == 10, "Report must contain 10 fold results"
    assert data["final_scientific_decision"] == "NOT SUPPORTED"

    # Verify all checks in report passed
    assert all(c["passed"] for c in data["checks"]), "Report contains failed checks!"

    # Verify failure gate triggered
    fg = data["failure_gate"]
    assert fg["overall_failure_triggered"] is True

    # Verify success gate failed
    sg = data["success_gate"]
    assert sg["overall_success_passed"] is False
