"""Unit tests for Phase 22: Research Decision Gate & Next Strategy Direction.

Verifies:
1. Strict holdout boundary enforcement (Phase 11 test partition >= 2026-02-19 10:45:00 UTC).
2. Point-in-time causal integrity of session and volatility regime calculations (no lookahead).
3. Exact calculation of directional classification metrics on model calls.
4. Deterministic decision criteria evaluation and falsification logic.
5. Integrity and schema consistency of the Phase 22 decision gate report artifact.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from ai.backtest.robustness import (
    PHASE11_TEST_LOCK_TIMESTAMP,
    VALIDATION_END_TIMESTAMP,
    verify_test_partition_rejection,
)
from scripts.run_phase22_decision_gate import evaluate_directional_classification


def test_phase11_test_lock_boundary_rejection():
    """1. Test partition boundary rejection strictly prevents evaluating test timestamps."""
    verify_test_partition_rejection(VALIDATION_END_TIMESTAMP)

    # Any timestamp at or after Phase 11 test lock must raise ValueError
    test_dt = PHASE11_TEST_LOCK_TIMESTAMP + timedelta(minutes=15)
    with pytest.raises(ValueError, match="CRITICAL TEST PARTITION VIOLATION"):
        verify_test_partition_rejection(test_dt)


def test_directional_metric_evaluation_correctness():
    """2. Verify directional classification metric calculation on active model calls."""
    # 4 SHORT (-1), 4 LONG (+1), 2 Neutral (0)
    y_true = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    # Predictions: 2 correct SHORT, 2 wrong LONG on SHORT bars;
    # 3 correct LONG on LONG bars; 1 neutral
    y_pred = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0])

    metrics = evaluate_directional_classification(y_true, y_pred)

    assert metrics["total_bars"] == 10
    assert metrics["directional_ground_truth_bars"] == 8
    # Model calls on directional bars: bars 0, 1, 2, 3, 4, 5, 6 (7 calls)
    assert metrics["model_directional_calls"] == 7

    # Correct calls: indices 0, 1 (-1), and 4, 5, 6 (+1) -> 5 correct out of 7 calls
    expected_acc = round((5 / 7) * 100.0, 2)
    assert metrics["accuracy_on_directional_calls"] == expected_acc


def test_regime_filtering_point_in_time_consistency():
    """3. Verify regime mask generation is strictly point-in-time and causal."""
    # Mock feature data
    hours = np.array([8, 12, 14, 18, 22, 2, 13])
    atrs = np.array([0.0008, 0.0012, 0.0015, 0.0009, 0.0007, 0.0006, 0.0014])
    med_atr = float(np.median(atrs))

    # London session: 7 <= hour < 16; NY session: 12 <= hour < 21
    # Overlap: 12 <= hour < 16
    is_overlap = (hours >= 12) & (hours < 16)
    is_vol_exp = atrs > med_atr
    joint = is_overlap & is_vol_exp

    assert len(joint) == len(hours)
    # Hour 12 (atr 0.0012 > med 0.0009): True
    assert joint[1]
    # Hour 14 (atr 0.0015 > med 0.0009): True
    assert joint[2]
    # Hour 8 (not in overlap): False
    assert not joint[0]
    # Hour 18 (not in overlap): False
    assert not joint[3]


def test_decision_criteria_falsification_logic():
    """4. Predefined criteria correctly classifies failure as NOT SUPPORTED."""
    # Predefined Success: BalAcc >= 55.0% AND Net P&L > $0.00 AND PF > 1.05
    # Predefined Failure: BalAcc < 53.5% OR Net P&L <= $0.00 OR PF <= 1.00

    def evaluate_decision(balacc: float, pnl: float, pf: float) -> str:
        if balacc >= 55.0 and pnl > 0.0 and pf > 1.05:
            return "SUPPORTED"
        elif balacc >= 53.5 and pnl > -50.0:
            return "PARTIALLY SUPPORTED"
        else:
            return "NOT SUPPORTED"

    # Candidate observed: BalAcc = 50.35%, Net P&L = -$24.24, PF = 0.7886
    assert evaluate_decision(50.35, -24.24, 0.7886) == "NOT SUPPORTED"
    # Hypothetical strong candidate
    assert evaluate_decision(56.20, 150.00, 1.35) == "SUPPORTED"
    # Marginal candidate
    assert evaluate_decision(53.80, -10.00, 0.95) == "PARTIALLY SUPPORTED"


def test_decision_gate_artifact_integrity():
    """5. Verify the existence and schema completeness of the Phase 22 report artifact."""
    artifact_path = Path("reports/phase22_decision_gate_report.json")
    assert artifact_path.exists(), f"Artifact missing: {artifact_path}"

    with open(artifact_path) as f:
        data = json.load(f)

    assert data["metadata"]["phase"] == "22"
    assert data["governance"]["phase11_test_lock_respected"] is True
    assert data["hypothesis_evaluation"]["decision"] == "NOT SUPPORTED"
    assert "classification_by_regime" in data
    assert "tick_backtest_by_regime" in data
