"""Unit tests for Phase 16 information timing, candidate immutability, and governance locks.

Verifies:
- FEATURE TIME < DECISION TIME < OUTCOME TIME causality chain
- Deterministic model execution
- Candidate configuration immutability
- Governance test: Phase 11 M15 test, Phase 15 H4 test, Phase 15 D1 test remain untouched
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ai.dataset.assembly import assemble_dataset
from ai.dataset.swing import aggregate_m15_to_h4
from ai.labels.swing import compute_swing_targets
from ai.swing.confirmation import align_m15_with_h4_decisions
from ai.swing.evaluation import FROZEN_RF_CONFIG
from ai.swing.timing_audit import generate_timing_audit_report

LOCKED_TEST_START = pd.Timestamp("2026-02-19 12:00:00+00:00")


def test_timing_audit_causality():
    """Verify FEATURE TIME < DECISION TIME < OUTCOME TIME for sampled decisions."""
    raw_m15 = pd.read_parquet("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
    h4_df = aggregate_m15_to_h4(raw_m15)
    ds_m15 = assemble_dataset()
    aligned = align_m15_with_h4_decisions(h4_df, ds_m15.df)
    targets = compute_swing_targets(h4_df, horizons=[8], timeframe="H4")

    records = generate_timing_audit_report(
        h4_df=h4_df,
        aligned_m15=aligned,
        target_series=targets["direction_vol_8"],
        sample_indices=[200, 500, 1000, 1500, 2000],
        horizon_bars=8,
    )

    assert len(records) == 5
    for r in records:
        assert r["causality_verified"] is True
        t_m15 = pd.Timestamp(r["actual_matched_m15_timestamp"])
        t_decision = pd.Timestamp(r["h4_decision_time"])
        t_outcome = pd.Timestamp(r["target_outcome_time"])

        assert t_m15 < t_decision
        assert t_decision < t_outcome


def test_candidate_configuration_immutability():
    """Verify that frozen Random Forest hyperparameters match Phase 15 candidate lock."""
    assert FROZEN_RF_CONFIG["n_estimators"] == 100
    assert FROZEN_RF_CONFIG["max_depth"] == 5
    assert FROZEN_RF_CONFIG["min_samples_leaf"] == 10
    assert FROZEN_RF_CONFIG["class_weight"] == "balanced"
    assert FROZEN_RF_CONFIG["random_state"] == 42


def test_deterministic_model_predictions():
    """Verify Random Forest with frozen configuration produces deterministic predictions."""
    X = np.random.RandomState(42).randn(100, 10)
    y = np.random.RandomState(42).choice([-1.0, 1.0], size=100)

    rf1 = RandomForestClassifier(**FROZEN_RF_CONFIG)
    rf1.fit(X, y)
    p1 = rf1.predict(X)

    rf2 = RandomForestClassifier(**FROZEN_RF_CONFIG)
    rf2.fit(X, y)
    p2 = rf2.predict(X)

    np.testing.assert_array_equal(p1, p2)


def test_governance_locked_partitions():
    """Governance test: Phase 11 M15, Phase 15 H4, Phase 15 D1 test partitions remain untouched."""
    # 1. Canonical M15 assembled dataset has exactly 14,988 test rows
    from ai.dataset.splits import split_dataset

    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits = split_dataset(ds)
    assert len(splits.test) == 14988, f"Expected 14,988 M15 test rows, got {len(splits.test)}"
    assert str(splits.test.start_timestamp) == "2026-02-19 12:00:00+00:00"

    # 2. H4 test rows count
    raw_m15 = pd.read_parquet("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
    h4_df = aggregate_m15_to_h4(raw_m15)
    ts_h4 = pd.to_datetime(h4_df["timestamp"], utc=True)
    h4_test_rows = int((ts_h4 >= LOCKED_TEST_START).sum())
    assert h4_test_rows == 939, f"Expected 939 H4 test rows, got {h4_test_rows}"

    # 3. Phase 16 metrics file verification
    metrics_path = "reports/phase16_validation_metrics.json"
    import json
    from pathlib import Path

    if Path(metrics_path).exists():
        with open(metrics_path) as f:
            data = json.load(f)
        assert data["governance"]["m15_phase11_test"] == "LOCKED"
        assert data["governance"]["h4_phase15_test"] == "LOCKED"
        assert data["governance"]["d1_phase15_test"] == "LOCKED"
        assert data["governance"]["phase12_baseline"] == "UNCHANGED"
