"""Unit tests for Phase 16 M15 entry confirmation filtering and causality.

Verifies:
- Point-in-time alignment between H4 decisions and completed M15 candles
- Incomplete M15 candle exclusion
- Correct logic for Confirmation Rules A, B, C, D
- Filter percentage and directional accuracy integrity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.dataset.assembly import assemble_dataset
from ai.dataset.swing import aggregate_m15_to_h4
from ai.swing.confirmation import align_m15_with_h4_decisions, evaluate_m15_confirmations


@pytest.fixture(scope="module")
def aligned_m15_sample():
    """Load canonical data and align H4 with completed M15 bars."""
    raw_m15 = pd.read_parquet("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
    h4_df = aggregate_m15_to_h4(raw_m15)
    ds_m15 = assemble_dataset()
    aligned = align_m15_with_h4_decisions(h4_df, ds_m15.df)
    return h4_df, aligned


def test_m15_alignment_causality_and_completion(aligned_m15_sample):
    """Test that all matched M15 candles strictly closed before H4 decision timestamp."""
    h4_df, aligned = aligned_m15_sample

    # Check non-null matched bars
    valid_rows = aligned[aligned["matched_m15_timestamp"].notna()]
    assert len(valid_rows) > 6000

    # Strict causality assertions:
    # 1. Matched M15 bar timestamp <= cutoff_m15 (which is t_h4_close - 15m)
    assert (valid_rows["matched_m15_timestamp"] <= valid_rows["cutoff_m15"]).all()

    # 2. Matched M15 bar timestamp < H4 decision time
    assert (valid_rows["matched_m15_timestamp"] < valid_rows["h4_decision_time"]).all()

    # 3. Exactly 15 minutes before decision time for regular bars
    time_diff = valid_rows["h4_decision_time"] - valid_rows["matched_m15_timestamp"]
    assert (time_diff >= pd.Timedelta(minutes=15)).all()


def test_incomplete_m15_candle_exclusion(aligned_m15_sample):
    """Test that the M15 candle starting at H4 decision time is never used."""
    _, aligned = aligned_m15_sample
    valid_rows = aligned[aligned["matched_m15_timestamp"].notna()]

    # At decision time T, the M15 bar with timestamp T is currently forming and incomplete.
    # Therefore matched_m15_timestamp must NEVER equal h4_decision_time.
    violating = valid_rows[valid_rows["matched_m15_timestamp"] == valid_rows["h4_decision_time"]]
    assert len(violating) == 0


def test_evaluate_m15_confirmations_rules():
    """Test evaluation logic of Confirmation Rules A, B, C, D on synthetic vectors."""
    y_true = np.array([1.0] * 50 + [-1.0] * 50)
    h4_preds = np.array([1.0] * 50 + [-1.0] * 50)

    # Synthetic M15 features:
    # 25 agree on long, 25 disagree on long, 25 agree on short, 25 disagree on short
    df_m15 = pd.DataFrame(
        {
            "m15_candle_direction": [1.0] * 25 + [-1.0] * 25 + [-1.0] * 25 + [1.0] * 25,
            "m15_rsi_14": [60.0] * 25 + [40.0] * 25 + [40.0] * 25 + [60.0] * 25,
            "m15_dist_ema_20": [0.001] * 25 + [-0.001] * 25 + [-0.001] * 25 + [0.001] * 25,
        }
    )

    results = evaluate_m15_confirmations(y_true, h4_preds, df_m15)

    assert "Base (H4 Signal Alone)" in results
    assert results["Base (H4 Signal Alone)"]["signals_count"] == 100
    assert results["Base (H4 Signal Alone)"]["pct_filtered"] == 0.0

    assert results["Confirmation A (M15 Direction)"]["signals_count"] == 50
    assert results["Confirmation A (M15 Direction)"]["pct_filtered"] == 50.0

    assert results["Confirmation B (M15 Momentum)"]["signals_count"] == 50
    assert results["Confirmation C (M15 Trend)"]["signals_count"] == 50
    assert results["Confirmation D (Two-of-Three)"]["signals_count"] == 50
