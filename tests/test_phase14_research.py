"""Unit tests, leakage checks, and governance verification for Phase 14 target redesign research."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import split_dataset
from ai.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from scripts.run_phase14_research import (
    build_research_targets,
    categorize_feature,
    compute_confidence_buckets,
    compute_target_quality_stats,
    compute_temporal_blocks,
    evaluate_binary_model,
)


def test_feature_categorization_phase14() -> None:
    """Verify that all 80 features map into the seven formal research categories without 'other'."""
    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    valid_categories = {
        "price/candle geometry",
        "momentum",
        "volatility",
        "trend",
        "activity",
        "time",
        "gap/session",
    }
    for fname in ds.feature_names:
        cat = categorize_feature(fname)
        assert cat in valid_categories, f"Feature '{fname}' mapped to invalid category '{cat}'"
        assert cat != "other", f"Feature '{fname}' remained uncategorized ('other')"


def test_build_research_targets_correctness() -> None:
    """Verify definitions, threshold boundaries, and exclusion rules for Targets A, B, C, D."""
    # Synthetic label inputs
    labels_df = pd.DataFrame(
        {
            "future_return_4": np.array([0.00060, -0.00070, 0.00010, -0.00010, 0.0, 0.00150]),
            "future_vol_adj_return_4": np.array([1.2, -1.5, 0.4, -0.3, 0.0, 2.5]),
            "direction_vol_4": np.array([1.0, -1.0, 0.0, 0.0, 0.0, 1.0]),
            "direction_4": np.array([1.0, -1.0, 0.0, 0.0, 0.0, 1.0]),
        }
    )

    targets = build_research_targets(labels_df)

    # Target A: Binary Direction (exact 0 excluded)
    # Row 0: >0 (1); Row 1: <0 (-1); Row 2: >0 (1); Row 3: <0 (-1); Row 4: ==0 (NaN); Row 5: >0 (1)
    expected_a = [1.0, -1.0, 1.0, -1.0, np.nan, 1.0]
    np.testing.assert_array_equal(targets["Target_A"].to_numpy(), expected_a)

    # Target B: Volatility-Adjusted Direction (direction_vol_4 != 0)
    # Row 0: 1.0; Row 1: -1.0; Row 2: NaN; Row 3: NaN; Row 4: NaN; Row 5: 1.0
    expected_b = [1.0, -1.0, np.nan, np.nan, np.nan, 1.0]
    np.testing.assert_array_equal(targets["Target_B"].to_numpy(), expected_b)

    # Target C: Fixed Economic Move (direction_4 != 0)
    # Row 0: 1.0; Row 1: -1.0; Row 2: NaN; Row 3: NaN; Row 4: NaN; Row 5: 1.0
    expected_c = [1.0, -1.0, np.nan, np.nan, np.nan, 1.0]
    np.testing.assert_array_equal(targets["Target_C"].to_numpy(), expected_c)

    # Target D: Extreme Volatility Move (|future_vol_adj_return_4| > 2.0)
    # Row 0-4: <= 2.0 -> NaN; Row 5: 2.5 > 2.0 -> 1.0
    expected_d = [np.nan, np.nan, np.nan, np.nan, np.nan, 1.0]
    np.testing.assert_array_equal(targets["Target_D"].to_numpy(), expected_d)


def test_horizon_alignment_and_end_of_data_nans() -> None:
    """Verify that all targets strictly require H=4 bars forward and end with NaNs."""
    labels_path = "data/labels/eurusd_m15/eurusd_m15_labels.parquet"
    raw_labels = pd.read_parquet(labels_path)

    # Check future_return_4 end NaNs
    assert raw_labels["future_return_4"].iloc[-4:].isna().all()
    assert raw_labels["future_return_4"].iloc[:-4].notna().all()

    # Targets built on raw_labels must also have exactly final 4 rows as NaN
    targets = build_research_targets(raw_labels)
    for t_name, s in targets.items():
        assert s.iloc[-4:].isna().all(), f"{t_name} did not have NaNs in final 4 rows"


def test_target_quality_stats_computation() -> None:
    """Verify Target Quality statistics calculation on synthetic data."""
    y = pd.Series([1.0, -1.0, 1.0, np.nan, np.nan])
    ret = pd.Series([0.0010, -0.0010, 0.0020, 0.0001, -0.0002])

    stats = compute_target_quality_stats(y, ret, total_rows=5)
    assert stats["sample_size"] == 3
    assert stats["excluded_count"] == 2
    assert pytest.approx(stats["pct_excluded"]) == 40.0
    assert stats["long_count"] == 2
    assert stats["short_count"] == 1
    assert pytest.approx(stats["long_pct"]) == 66.666666
    assert pytest.approx(stats["short_pct"]) == 33.333333
    assert pytest.approx(stats["mean_future_return"]) == (0.0010 - 0.0010 + 0.0020) / 3.0
    assert pytest.approx(stats["mean_abs_pips"]) == (10.0 + 10.0 + 20.0) / 3.0


def test_evaluate_binary_model_and_confidence_buckets() -> None:
    """Verify classification metrics, confusion matrix, and bucket segmentation."""
    y_true = np.array([-1.0, -1.0, 1.0, 1.0])
    y_pred = np.array([-1.0, 1.0, 1.0, 1.0])
    probs = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]])
    classes = [-1.0, 1.0]

    metrics = evaluate_binary_model(y_true, y_pred, probs, classes)
    assert pytest.approx(metrics["accuracy"]) == 0.75
    # TN=1, FP=1, FN=0, TP=2 -> BalAcc = (1/2 + 2/2) / 2 = 0.75
    assert pytest.approx(metrics["balanced_accuracy"]) == 0.75
    assert metrics["confusion_matrix"]["tn"] == 1
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 0
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confidence"]["mean"] == pytest.approx((0.8 + 0.6 + 0.7 + 0.9) / 4.0)
    assert metrics["confidence"]["max"] == pytest.approx(0.9)

    # Test confidence buckets
    p_dir = np.maximum(probs[:, 0], probs[:, 1])
    buckets = [(0.50, 0.70), (0.70, 0.85), (0.85, 1.01)]
    b_records = compute_confidence_buckets(p_dir, y_pred, y_true, buckets)
    assert len(b_records) == 3
    # Bucket [0.50, 0.70): sample 1 (0.60) -> pred=1, true=-1 (incorrect)
    assert b_records[0]["count"] == 1
    assert b_records[0]["observed_accuracy_pct"] == 0.0
    # Bucket [0.70, 0.85): samples 0 and 2 -> both correct
    assert b_records[1]["count"] == 2
    assert b_records[1]["observed_accuracy_pct"] == 100.0
    # Bucket [0.85, 1.01): sample 3 (0.90) -> pred=1, true=1 (correct)
    assert b_records[2]["count"] == 1
    assert b_records[2]["observed_accuracy_pct"] == 100.0


def test_temporal_block_partition_integrity() -> None:
    """Verify that temporal blocks partition candidate sets without gaps or overlaps."""
    y_true = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    y_pred = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    probs = np.tile([0.5, 0.5], (6, 1))
    ts = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
    block_slices = [
        ("Block 1", slice(0, 2)),
        ("Block 2", slice(2, 4)),
        ("Block 3", slice(4, 6)),
    ]

    records = compute_temporal_blocks(
        y_true, y_pred, probs, pd.Series(ts), [-1.0, 1.0], block_slices
    )
    assert len(records) == 3
    assert sum(r["row_count"] for r in records) == 6
    assert records[0]["start_time"] == str(ts[0])
    assert records[0]["end_time"] == str(ts[1])
    assert records[2]["start_time"] == str(ts[4])
    assert records[2]["end_time"] == str(ts[5])


def test_leakage_and_holdout_protection() -> None:
    """Verify strict partition governance: Test partition is completely locked and untouched."""
    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits = split_dataset(ds)

    # 1. Exact partition sizes
    assert len(splits.train) == 69937, f"Train size altered: {len(splits.train)}"
    assert len(splits.val) == 14983, f"Validation size altered: {len(splits.val)}"
    assert len(splits.test) == 14988, f"Test size altered: {len(splits.test)}"

    # 2. Strict chronological boundaries
    assert splits.train.end_timestamp < splits.val.start_timestamp
    assert splits.val.end_timestamp < splits.test.start_timestamp
    assert str(splits.test.start_timestamp) == "2026-02-19 12:00:00+00:00"

    # 3. No target columns in features X
    for col in ds.feature_names:
        assert not col.startswith("direction"), f"Target column '{col}' detected in features X"
        assert not col.startswith("future"), f"Future return column '{col}' detected in features X"

    assert len(ds.feature_names) == 80


def test_deterministic_reproducibility() -> None:
    """Verify that models initialized with random_state=42 produce identical predictions."""
    X = np.random.RandomState(42).randn(100, 10)
    y = np.where(X[:, 0] > 0, 1.0, -1.0)

    lr1 = LogisticRegressionBaseline(class_weight="balanced", random_state=42)
    lr2 = LogisticRegressionBaseline(class_weight="balanced", random_state=42)
    lr1.fit(X, y)
    lr2.fit(X, y)
    np.testing.assert_array_equal(lr1.predict(X), lr2.predict(X))
    np.testing.assert_array_almost_equal(lr1.predict_proba(X), lr2.predict_proba(X))

    rf1 = RandomForestBaseline(n_estimators=10, max_depth=4, random_state=42)
    rf2 = RandomForestBaseline(n_estimators=10, max_depth=4, random_state=42)
    rf1.fit(X, y)
    rf2.fit(X, y)
    np.testing.assert_array_equal(rf1.predict(X), rf2.predict(X))
    np.testing.assert_array_almost_equal(rf1.predict_proba(X), rf2.predict_proba(X))
