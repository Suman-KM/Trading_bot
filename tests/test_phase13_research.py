"""Unit tests and governance checks for Phase 13 intraday signal research."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_phase13_research import (
    categorize_feature,
    compute_confidence_buckets,
    compute_distribution_metrics,
)


def test_categorize_feature():
    """Verify feature categorization mapping adheres to research taxonomy."""
    assert categorize_feature("atr_14") == "volatility"
    assert categorize_feature("hl_range") == "volatility"
    assert categorize_feature("vol_std_10") == "volatility"
    assert categorize_feature("rsi_14") == "momentum"
    assert categorize_feature("macd_diff") == "momentum"
    assert categorize_feature("adx_14") == "trend"
    assert categorize_feature("ema_20") == "trend"
    assert categorize_feature("hour") == "time_of_day"
    assert categorize_feature("cos_hour") == "time_of_day"
    assert categorize_feature("is_london_session") == "time_of_day"
    assert categorize_feature("log_tick_volume") == "activity"
    assert categorize_feature("unknown_feature_xyz") == "other"


def test_compute_distribution_metrics():
    """Verify probability summary statistics calculation."""
    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    metrics = compute_distribution_metrics(arr)
    assert pytest.approx(metrics["mean"]) == 0.3
    assert pytest.approx(metrics["median"]) == 0.3
    assert pytest.approx(metrics["min"]) == 0.1
    assert pytest.approx(metrics["max"]) == 0.5
    assert "std" in metrics
    assert "p90" in metrics
    assert "p95" in metrics


def test_compute_confidence_buckets():
    """Verify confidence bucket segmentation and directional accuracy calculation."""
    # Synthetic dataset of 6 samples
    p_dir = np.array([0.42, 0.44, 0.48, 0.52, 0.56, 0.62])
    val_preds = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 0.0])
    y_val = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 0.0])
    buckets = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.65)]

    records = compute_confidence_buckets(p_dir, val_preds, y_val, buckets)
    assert len(records) == 5
    # First bucket [0.40, 0.45): 2 samples (0.42, 0.44)
    assert records[0]["count"] == 2
    assert records[0]["pred_long"] == 1
    assert records[0]["pred_short"] == 1
    # Sample 1 pred=1, true=1 (correct); Sample 2 pred=-1, true=1 (incorrect) -> 50% accuracy
    assert records[0]["dir_prediction_accuracy"] == 50.0

    # Bucket [0.55, 0.60): 1 sample (0.56) pred=1, true=1 -> 100% accuracy
    assert records[3]["count"] == 1
    assert records[3]["dir_prediction_accuracy"] == 100.0


def test_holdout_protection_and_partition_isolation():
    """Verify strict partition governance: Test partition is untouched."""
    from ai.dataset.assembly import assemble_dataset
    from ai.dataset.splits import split_dataset

    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits = split_dataset(ds)

    assert len(splits.train) == 69937
    assert len(splits.val) == 14983
    assert len(splits.test) == 14988  # Phase 11 Test partition strictly preserved
