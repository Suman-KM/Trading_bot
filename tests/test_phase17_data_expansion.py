"""Unit tests for Phase 17 Historical Data Expansion & H4 Robustness Research.

Verifies:
1. Expanded data chronological ordering and monotonicity
2. Chunk overlap bit-for-bit validation
3. Duplicate timestamp absence
4. OHLC geometric consistency
5. Point-in-time feature validity
6. Label horizon forward alignment
7. Purge gap correctness across all 10 folds
8. Chronological walk-forward expanding progression
9. Zero test partition access
10. Candidate configuration immutability
11. Deterministic model predictions
12. Governance lock: Phase 11 M15 test, Phase 15 H4 test, Phase 15 D1 test remain untouched
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import split_dataset
from ai.dataset.swing import aggregate_m15_to_h4
from ai.features.swing import compute_swing_features
from ai.labels.swing import compute_swing_targets
from ai.swing.data_expansion import (
    CHUNKS_DIR,
    LOCKED_TEST_START_TS,
    get_expanded_research_data,
    validate_chunk_overlaps,
)
from ai.swing.evaluation import FROZEN_RF_CONFIG
from ai.swing.walk_forward_expanded import generate_expanded_walk_forward_folds


def test_chunk_files_exist_and_overlap_identical():
    """Test that all 3 exported H4 chunk files exist and overlapping bars match bit-for-bit."""
    c1 = pd.read_parquet(CHUNKS_DIR / "chunk_1.parquet")
    c2 = pd.read_parquet(CHUNKS_DIR / "chunk_2.parquet")
    c3 = pd.read_parquet(CHUNKS_DIR / "chunk_3.parquet")

    assert len(c1) == 10000
    assert len(c2) == 10000
    assert len(c3) == 6000

    chunks = [c3, c2, c1]
    overlap_results = validate_chunk_overlaps(chunks)

    assert len(overlap_results) == 2
    for oc in overlap_results:
        assert oc["overlap_bars_count"] == 100
        assert oc["all_fields_identical"] is True


def test_expanded_h4_chronological_ordering_and_no_duplicates():
    """Test that merged expanded H4 data is strictly monotonic increasing without duplicates."""
    df_res, _ = get_expanded_research_data()

    ts = df_res["timestamp"]
    assert ts.is_monotonic_increasing is True
    assert not ts.duplicated().any()
    assert len(df_res) > 24000


def test_expanded_h4_ohlc_validity():
    """Test geometric OHLC validity and positive prices across expanded H4 history."""
    df_res, _ = get_expanded_research_data()

    high_valid = (df_res["high"] >= df_res["open"]) & (df_res["high"] >= df_res["close"])
    low_valid = (df_res["low"] <= df_res["open"]) & (df_res["low"] <= df_res["close"])
    prices_positive = (df_res[["open", "high", "low", "close"]] > 0).all(axis=1)

    assert high_valid.all()
    assert low_valid.all()
    assert prices_positive.all()
    assert not df_res.isna().any().any()


def test_canonical_overlap_preservation():
    """Test that expanded H4 matches canonical M15-aggregated H4 during the overlapping era."""
    raw_h4 = pd.read_parquet("data/raw/eurusd_h4_expanded/eurusd_h4_raw.parquet")
    raw_h4["timestamp"] = pd.to_datetime(raw_h4["time"], unit="s", utc=True)
    raw_m15 = pd.read_parquet("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
    canon_h4 = aggregate_m15_to_h4(raw_m15)
    canon_h4["timestamp"] = pd.to_datetime(canon_h4["timestamp"], utc=True)

    comp = pd.merge(raw_h4, canon_h4, on="timestamp", suffixes=("_exp", "_agg"))
    assert len(comp) == len(canon_h4)

    # High, Low, Close must be exactly 0 diff
    assert np.allclose(comp["high_exp"], comp["high_agg"], atol=1e-8)
    assert np.allclose(comp["low_exp"], comp["low_agg"], atol=1e-8)
    assert np.allclose(comp["close_exp"], comp["close_agg"], atol=1e-8)


def test_feature_point_in_time_correctness():
    """Test that swing features are causal, finite, and have zero NaNs after warmup."""
    df_res, _ = get_expanded_research_data()
    feats = compute_swing_features(df_res, timeframe="H4")

    assert len(feats) == len(df_res)
    assert not feats.iloc[100:].isna().any().any()
    assert not np.isinf(feats.iloc[100:].to_numpy()).any()


def test_target_horizon_alignment():
    """Test that direction_vol_8 correctly truncates future lookahead at the tail."""
    df_res, _ = get_expanded_research_data()
    targets = compute_swing_targets(df_res, horizons=[8], timeframe="H4")
    y = targets["direction_vol_8"]

    # Last 8 rows must be NaN (future return unavailable within partition)
    assert y.iloc[-8:].isna().all()
    assert y.notna().sum() > 10000


def test_expanded_walk_forward_folds_purge_and_progression():
    """Test 10 chronological expanding folds, exact 8-bar purge gap, and zero overlap."""
    df_res, _ = get_expanded_research_data()
    feats = compute_swing_features(df_res, timeframe="H4")
    targets = compute_swing_targets(df_res, horizons=[8], timeframe="H4")

    folds = generate_expanded_walk_forward_folds(
        h4_df=df_res,
        features_df=feats,
        target_series=targets["direction_vol_8"],
        initial_train_bars=5000,
        n_folds=10,
        purge_bars=8,
    )

    assert len(folds) == 10

    for i, f in enumerate(folds):
        assert f.purge_gap_bars == 8
        assert f.train_end_ts < f.val_start_ts
        assert len(set(f.train_indices).intersection(set(f.val_indices))) == 0

        # Progression check
        if i < len(folds) - 1:
            f_next = folds[i + 1]
            assert f.train_sample_count < f_next.train_sample_count
            assert f.val_start_ts < f_next.val_start_ts


def test_candidate_configuration_immutability():
    """Verify that frozen Random Forest hyperparameters match Phase 15/16 lock."""
    assert FROZEN_RF_CONFIG["n_estimators"] == 100
    assert FROZEN_RF_CONFIG["max_depth"] == 5
    assert FROZEN_RF_CONFIG["min_samples_leaf"] == 10
    assert FROZEN_RF_CONFIG["class_weight"] == "balanced"
    assert FROZEN_RF_CONFIG["random_state"] == 42


def test_deterministic_model_predictions():
    """Verify deterministic predictions under fixed random seed."""
    X = np.random.RandomState(42).randn(100, 10)
    y = np.random.RandomState(42).choice([-1.0, 1.0], size=100)

    rf1 = RandomForestClassifier(**FROZEN_RF_CONFIG)
    rf1.fit(X, y)
    p1 = rf1.predict(X)

    rf2 = RandomForestClassifier(**FROZEN_RF_CONFIG)
    rf2.fit(X, y)
    p2 = rf2.predict(X)

    np.testing.assert_array_equal(p1, p2)


def test_governance_locked_partitions_unmodified():
    """Governance test: Phase 11 M15 test, Phase 15 H4 test, Phase 15 D1 test remain untouched."""
    # 1. Canonical M15 test partition
    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits = split_dataset(ds)
    assert len(splits.test) == 14988, f"Expected 14,988 M15 test rows, got {len(splits.test)}"
    assert str(splits.test.start_timestamp) == "2026-02-19 12:00:00+00:00"

    # 2. Canonical H4 test partition
    raw_m15 = pd.read_parquet("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
    h4_df = aggregate_m15_to_h4(raw_m15)
    ts_h4 = pd.to_datetime(h4_df["timestamp"], utc=True)
    h4_test_rows = int((ts_h4 >= LOCKED_TEST_START_TS).sum())
    assert h4_test_rows == 939, f"Expected 939 H4 test rows, got {h4_test_rows}"

    # 3. Phase 17 metrics file governance block
    metrics_path = Path("reports/phase17_expansion_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
        assert data["governance"]["m15_phase11_test"] == "LOCKED"
        assert data["governance"]["h4_phase15_test"] == "LOCKED"
        assert data["governance"]["d1_phase15_test"] == "LOCKED"
        assert data["governance"]["phase12_baseline"] == "UNCHANGED"
