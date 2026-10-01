"""Unit tests and temporal leakage audits for Phase 29 Pre-Registered COT Experiment.

Verifies:
1. COT observation date is never confused with publication date (obs_ts < pub_ts < eff_ts).
2. Friday release cannot influence earlier EURUSD bars (zero publication or effective leakage).
3. DST transitions correctly alter UTC publication timestamps (19:30 vs 20:30 UTC).
4. Holiday delays are respected (delayed release dates enforced).
5. Missing reports do not create future leakage (strictly causal forward-fill).
6. Rolling 3-year Z-score uses only historical observations (rolling window, min_periods=52).
7. Four-week delta uses only previous reports (shift 4).
8. Target uses only future EURUSD prices (shift -h, tail bars are NaN).
9. Train data strictly precedes validation data in all 10 folds (purge & embargo enforced).
10. Locked test partitions remain untouched (quarantine preserved).
11. Feature matrix dimensions: baseline exactly 30 features, candidate exactly 33 features.
12. Pre-registration document exists and contains all 20 required sections.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ai.data.exogenous.cftc_alignment import align_cftc_to_eurusd, compute_cftc_timestamps
from ai.features.swing import compute_swing_features
from ai.labels.swing import compute_swing_targets
from ai.research.phase29_cot_experiment import (
    APPROVED_COT_FEATURES,
    BASELINE_30_FEATURES,
    CFTC_PARQUET_PATH,
    INITIAL_TRAIN_BARS,
    N_FOLDS,
    PRE_HOLDOUT_CUTOFF,
    REPORT_OUTPUT_PATH,
    generate_phase29_walk_forward_folds,
    load_and_prepare_d1_data,
)
from ai.swing.holdout import LOCKED_TEST_START_TS

PREREG_PATH = Path("docs/phase-29-cot-experiment-preregistration.md")
NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def d1_pre_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and prepare pre-holdout D1 dataset."""
    return load_and_prepare_d1_data()


def test_cot_observation_vs_publication_dates():
    """1. COT observation date is never confused with publication date."""
    cftc_df = pd.read_parquet(CFTC_PARQUET_PATH)
    obs_ts = pd.to_datetime(cftc_df["observation_timestamp_utc"], utc=True)
    pub_ts = pd.to_datetime(cftc_df["publication_time_utc"], utc=True)
    eff_ts = pd.to_datetime(cftc_df["effective_time_utc"], utc=True)

    # Observation is Tuesday (21:00 UTC), Publication is Friday (or later), Effective is Monday
    assert (obs_ts < pub_ts).all(), "Observation timestamp must precede publication timestamp"
    assert (pub_ts < eff_ts).all(), "Publication timestamp must precede effective timestamp"
    assert (cftc_df["report_date"] != cftc_df["publication_date"]).all()


def test_no_earlier_eurusd_bar_influenced(d1_pre_data: tuple[pd.DataFrame, dict[str, Any]]):
    """2. Friday release cannot influence earlier EURUSD bars."""
    df_d1, audit = d1_pre_data
    assert audit["causally_valid"] is True
    assert audit["leakage_violations_publication"] == 0
    assert audit["leakage_violations_effective"] == 0

    bar_ts = pd.to_datetime(df_d1["timestamp"], utc=True)
    eff_ts = pd.to_datetime(df_d1["effective_time_utc"], utc=True)
    pub_ts = pd.to_datetime(df_d1["publication_time_utc"], utc=True)

    # No daily bar on date T can observe an effective time > T
    assert (bar_ts >= eff_ts).all()
    assert (eff_ts > pub_ts).all()


def test_dst_transitions_correctness():
    """3. US DST transitions shift publication UTC hour from 20:30 (EST) to 19:30 (EDT)."""
    ts_edt = compute_cftc_timestamps("2024-07-16")
    pub_edt = datetime.fromisoformat(ts_edt["publication_time_utc"])
    assert pub_edt.hour == 19
    assert pub_edt.minute == 30

    ts_est = compute_cftc_timestamps("2024-01-16")
    pub_est = datetime.fromisoformat(ts_est["publication_time_utc"])
    assert pub_est.hour == 20
    assert pub_est.minute == 30


def test_holiday_delays_respected():
    """4. Holiday delays are respected."""
    # Good Friday 2024: March 29 -> delayed to Monday April 1
    ts_gf = compute_cftc_timestamps("2024-03-26")
    assert ts_gf["publication_date"] == "2024-04-01"
    pub_gf = datetime.fromisoformat(ts_gf["publication_time_utc"])
    assert pub_gf.weekday() == 0  # Monday


def test_missing_reports_causal_forward_fill():
    """5. Missing reports do not create future leakage."""
    cftc_df = pd.read_parquet(CFTC_PARQUET_PATH)
    # Filter out one report in the middle
    cftc_subset = cftc_df[cftc_df["report_date"] != "2024-01-16"].copy()

    eurusd_mock = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-10", "2024-01-30", freq="1D", tz="UTC"),
            "open": 1.10,
            "high": 1.11,
            "low": 1.09,
            "close": 1.105,
            "tick_volume": 1000,
        }
    )
    aligned, audit = align_cftc_to_eurusd(
        cftc_subset, eurusd_mock, eurusd_timestamp_col="timestamp"
    )
    assert audit["causally_valid"] is True
    assert audit["leakage_violations_publication"] == 0


def test_rolling_3y_zscore_causality():
    """6. Rolling 3-year Z-score uses only historical observations."""
    cftc_df = pd.read_parquet(CFTC_PARQUET_PATH)
    # The first 51 reports must be NaN because min_periods=52
    assert cftc_df["net_spec_zscore_3y"].iloc[:51].isna().all()
    assert cftc_df["net_comm_zscore_3y"].iloc[:51].isna().all()

    # Report 52 onward must be valid numbers
    assert cftc_df["net_spec_zscore_3y"].iloc[52:].notna().all()
    assert cftc_df["net_comm_zscore_3y"].iloc[52:].notna().all()


def test_speculative_4w_delta_causality():
    """7. Four-week delta uses only previous reports."""
    cftc_df = pd.read_parquet(CFTC_PARQUET_PATH)
    # The first 4 reports must be NaN for 4-week change
    assert cftc_df["net_spec_4w_change"].iloc[:4].isna().all()
    # Check that row 4 equals net_spec(4) - net_spec(0)
    expected_delta_4 = (
        cftc_df["net_speculative_pos"].iloc[4] - cftc_df["net_speculative_pos"].iloc[0]
    )
    assert cftc_df["net_spec_4w_change"].iloc[4] == expected_delta_4


def test_target_uses_only_future_prices(d1_pre_data: tuple[pd.DataFrame, dict[str, Any]]):
    """8. Target uses only future EURUSD prices, with tail bars set to NaN."""
    df_d1, _ = d1_pre_data
    targets = compute_swing_targets(df_d1, horizons=[10, 20], timeframe="D1")

    # Tail H bars must be NaN for horizon H
    assert targets["future_return_10"].iloc[-10:].isna().all()
    assert targets["direction_vol_10"].iloc[-10:].isna().all()
    assert targets["future_return_20"].iloc[-20:].isna().all()
    assert targets["direction_vol_20"].iloc[-20:].isna().all()

    # Direction vol values must be strictly in {-1.0, 1.0, nan}
    valid_10 = targets["direction_vol_10"].dropna()
    assert set(valid_10.unique()).issubset({-1.0, 1.0})
    valid_20 = targets["direction_vol_20"].dropna()
    assert set(valid_20.unique()).issubset({-1.0, 1.0})


def test_walk_forward_fold_separation_and_integrity(
    d1_pre_data: tuple[pd.DataFrame, dict[str, Any]],
):
    """9. Train data strictly precedes validation data in all 10 folds with purge & embargo."""
    df_d1, _ = d1_pre_data
    raw_feats = compute_swing_features(df_d1, timeframe="D1")
    feats_cand = pd.concat([raw_feats[BASELINE_30_FEATURES], df_d1[APPROVED_COT_FEATURES]], axis=1)
    targets = compute_swing_targets(df_d1, horizons=[10, 20], timeframe="D1")

    for h in [10, 20]:
        y_h = targets[f"direction_vol_{h}"]
        folds = generate_phase29_walk_forward_folds(
            df=df_d1,
            features_df=feats_cand,
            target_series=y_h,
            horizon_bars=h,
            initial_train_bars=INITIAL_TRAIN_BARS,
            n_folds=N_FOLDS,
            embargo_bars=5,
        )
        assert len(folds) == 10

        prev_train_end = None
        for fold in folds:
            # 1. Train precedes validation
            assert fold.train_end_ts < fold.val_start_ts
            # 2. Separation satisfies purge (h) + embargo (5)
            sep = fold.val_indices[0] - fold.train_indices[-1]
            assert sep >= h + 5
            # 3. Expanding train window
            if prev_train_end is not None:
                assert fold.train_end_ts >= prev_train_end
            prev_train_end = fold.train_end_ts
            # 4. Valid samples in train and val
            assert fold.train_sample_count > 100
            assert fold.val_sample_count >= 30


def test_locked_test_partitions_untouched(d1_pre_data: tuple[pd.DataFrame, dict[str, Any]]):
    """10. Locked test partition (2026-02-19 12:00 UTC onward) is untouched."""
    df_d1, _ = d1_pre_data
    ts = pd.to_datetime(df_d1["timestamp"], utc=True)
    assert (ts < LOCKED_TEST_START_TS).all()
    assert (ts <= PRE_HOLDOUT_CUTOFF).all()


def test_feature_matrix_dimensions(d1_pre_data: tuple[pd.DataFrame, dict[str, Any]]):
    """11. Baseline is exactly 30 features, Candidate is exactly 33 features."""
    df_d1, _ = d1_pre_data
    raw_feats = compute_swing_features(df_d1, timeframe="D1")
    feats_base = raw_feats[BASELINE_30_FEATURES]
    feats_cand = pd.concat([feats_base, df_d1[APPROVED_COT_FEATURES]], axis=1)

    assert feats_base.shape[1] == 30
    assert feats_cand.shape[1] == 33
    assert APPROVED_COT_FEATURES == [
        "speculative_net_zscore_3y",
        "commercial_position_zscore_3y",
        "speculative_net_4w_change",
    ]


def test_preregistration_document_exists_and_sections():
    """12. Pre-registration document exists and contains all required sections."""
    assert PREREG_PATH.exists(), f"Missing pre-registration document: {PREREG_PATH}"
    content = PREREG_PATH.read_text(encoding="utf-8")

    required_sections = [
        "1. Hypothesis",
        "2. Null Hypothesis",
        "3. Data Source",
        "4. COT Contract",
        "5. Candidate Features",
        "6. Target Definition",
        "7. Pre-Registered Horizons",
        "8. Baseline Benchmark",
        "9. Model Family",
        "10. Fixed Hyperparameters",
        "11. Walk-Forward Methodology",
        "12. Purge Window",
        "13. Embargo Window",
        "14. Publication Lag",
        "15. Leakage Prevention Verification",
        "16. Primary Evaluation Metric",
        "17. Statistical Testing",
        "18. Pre-Registered Success Gate",
        "19. Pre-Registered Failure Gate",
        "20. Locked Data Policy",
    ]
    for sec in required_sections:
        assert sec in content, f"Missing section '{sec}' in pre-registration document!"


def test_phase29_report_schema_and_verdict():
    """13. Report artifact exists, validates against schema, and records scientific decision."""
    assert REPORT_OUTPUT_PATH.exists(), f"Missing report: {REPORT_OUTPUT_PATH}"
    report = json.loads(REPORT_OUTPUT_PATH.read_text(encoding="utf-8"))

    assert report["phase"] == 29
    assert report["scientific_verdict"] in [
        "SUPPORTED",
        "PARTIALLY SUPPORTED",
        "NOT SUPPORTED",
        "INCONCLUSIVE",
    ]
    assert report["governance"]["locked_test_partition_status"] == "QUARANTINED_AND_UNTOUCHED"
    assert report["governance"]["live_or_demo_trades"] == 0

    assert "H_10" in report["horizon_results"]
    assert "H_20" in report["horizon_results"]
    assert len(report["horizon_results"]["H_10"]["folds"]) == 10
    assert len(report["horizon_results"]["H_20"]["folds"]) == 10
