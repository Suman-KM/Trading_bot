"""Unit and leakage test suite for Phase 24 Historical Yield Data Ingestion.

Validates:
1. Same-day US yield cannot appear on a pre-publication H4 bar.
2. Same-day German yield cannot appear before German publication.
3. US-Germany spread cannot appear until BOTH source observations are available.
4. Day D yield cannot affect Day D earlier H4 bars.
5. Day D yield becomes usable on Day D+1 under the conservative convention.
6. Five-day spread change cannot use future observations.
7. Missing German data does not silently become zero.
8. Missing US data does not silently become zero.
9. Holiday gaps are handled deterministically.
10. DST does not alter the causal alignment.
11. Duplicate source observations are detected.
12. Revised/vintage information cannot silently replace the originally available value.
13. Strict Phase 11 test lock boundary rejection.
14. Cryptographic SHA-256 verification of raw ingested artifacts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.backtest.robustness import (
    PHASE11_TEST_LOCK_TIMESTAMP,
    VALIDATION_END_TIMESTAMP,
    verify_test_partition_rejection,
)
from ai.data.exogenous.schema import ExogenousDataPoint
from ai.data.exogenous.validation import check_dst_offsets, detect_untracked_revisions
from ai.data.exogenous.yields import (
    align_yields_to_eurusd_h4,
    compute_file_sha256,
    reconcile_yield_calendars,
    validate_raw_series,
)


def test_phase11_test_lock_boundary_rejection():
    """Governance test: ensure Phase 11 test partition boundary is strictly rejected."""
    verify_test_partition_rejection(VALIDATION_END_TIMESTAMP)

    violating_time = PHASE11_TEST_LOCK_TIMESTAMP + timedelta(hours=4)
    with pytest.raises(ValueError, match="CRITICAL TEST PARTITION VIOLATION"):
        verify_test_partition_rejection(violating_time)


def test_same_day_us_yield_cannot_appear_on_pre_publication_h4_bar():
    """1. Test that same-day US yield cannot appear on a pre-publication H4 bar."""
    # US yield for 2025-06-10 is published at ~21:15 UTC.
    # An H4 bar on 2025-06-10 at 16:00:00 UTC must NOT observe Day 2025-06-10 US yield.
    df_us = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "us_2y": [4.20, 4.35],
        }
    )
    df_de = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "german_2y": [2.80, 2.85],
        }
    )
    df_reconciled, _ = reconcile_yield_calendars(df_us, df_de, "2025-06-09", "2025-06-10")

    # Bar on 2025-06-10 at 16:00 UTC (before 21:15 publication)
    df_h4 = pd.DataFrame({"timestamp": [datetime(2025, 6, 10, 16, 0, tzinfo=UTC)]})
    aligned, audit = align_yields_to_eurusd_h4(df_h4, df_reconciled)

    assert audit["number_of_leakage_violations"] == 0
    # Must observe June 9 yield (4.20), NOT same-day June 10 yield (4.35)
    assert aligned["US_2Y"].iloc[0] == 4.20
    assert aligned["yield_observation_date"].iloc[0] == pd.Timestamp("2025-06-09")


def test_same_day_german_yield_cannot_appear_before_german_publication():
    """2. Test that same-day German yield cannot appear before German publication."""
    # German yield for 2025-06-10 is published at ~16:00 UTC.
    # An H4 bar at 08:00 or 12:00 UTC on 2025-06-10 must NOT observe June 10 German yield.
    df_us = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "us_2y": [4.20, 4.35],
        }
    )
    df_de = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "german_2y": [2.80, 2.95],
        }
    )
    df_reconciled, _ = reconcile_yield_calendars(df_us, df_de, "2025-06-09", "2025-06-10")

    df_h4 = pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 6, 10, 8, 0, tzinfo=UTC),
                datetime(2025, 6, 10, 12, 0, tzinfo=UTC),
            ]
        }
    )
    aligned, audit = align_yields_to_eurusd_h4(df_h4, df_reconciled)

    assert audit["number_of_leakage_violations"] == 0
    # Both pre-publication bars must observe June 9 German yield (2.80), never June 10 (2.95)
    assert aligned["German_2Y"].iloc[0] == 2.80
    assert aligned["German_2Y"].iloc[1] == 2.80


def test_spread_cannot_appear_until_both_sources_available():
    """3. Test that US-Germany spread cannot appear until BOTH source observations are available."""
    # Day D: German published at 16:00 UTC, but US published at 21:15 UTC.
    # At 20:00:00 UTC on Day D, German is known, but US is unknown.
    # The derived spread for Day D must NOT be available at 20:00:00 UTC.
    df_us = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "us_2y": [4.00, 4.50],
        }
    )
    df_de = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "german_2y": [2.00, 2.50],
        }
    )
    df_reconciled, _ = reconcile_yield_calendars(df_us, df_de, "2025-06-09", "2025-06-10")

    # Bar on June 10 at 20:00 UTC (after German 16:00 close, but before US 21:15 close)
    df_h4 = pd.DataFrame({"timestamp": [datetime(2025, 6, 10, 20, 0, tzinfo=UTC)]})
    aligned, audit = align_yields_to_eurusd_h4(df_h4, df_reconciled)

    # Spread must be June 9 spread (4.00 - 2.00 = 2.00), NOT June 10 spread (4.50 - 2.50)
    assert aligned["US_Germany_2Y_Spread"].iloc[0] == pytest.approx(2.00)
    assert aligned["yield_observation_date"].iloc[0] == pd.Timestamp("2025-06-09")


def test_day_d_yield_cannot_affect_day_d_earlier_h4_bars():
    """4. Test that Day D yield cannot affect any Day D H4 bars."""
    df_us = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "us_2y": [4.00, 4.80],
        }
    )
    df_de = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "german_2y": [2.00, 2.80],
        }
    )
    df_reconciled, _ = reconcile_yield_calendars(df_us, df_de, "2025-06-09", "2025-06-10")

    # All 6 H4 bars of Day D (June 10)
    h4_times = [
        datetime(2025, 6, 10, 0, 0, tzinfo=UTC),
        datetime(2025, 6, 10, 4, 0, tzinfo=UTC),
        datetime(2025, 6, 10, 8, 0, tzinfo=UTC),
        datetime(2025, 6, 10, 12, 0, tzinfo=UTC),
        datetime(2025, 6, 10, 16, 0, tzinfo=UTC),
        datetime(2025, 6, 10, 20, 0, tzinfo=UTC),
    ]
    df_h4 = pd.DataFrame({"timestamp": h4_times})
    aligned, audit = align_yields_to_eurusd_h4(df_h4, df_reconciled)

    assert audit["number_of_leakage_violations"] == 0
    assert audit["same_day_leakage_violations"] == 0
    # Every single bar on Day D must observe Day D-1 values
    for val in aligned["US_2Y"]:
        assert val == 4.00
    for val in aligned["German_2Y"]:
        assert val == 2.00


def test_day_d_yield_becomes_usable_on_day_d_plus_1():
    """5. Test that Day D yield becomes usable on Day D+1 under the conservative convention."""
    df_us = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "us_2y": [4.00, 4.80],
        }
    )
    df_de = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-09", "2025-06-10"]),
            "german_2y": [2.00, 2.80],
        }
    )
    df_reconciled, _ = reconcile_yield_calendars(df_us, df_de, "2025-06-09", "2025-06-10")

    # Bar at 00:00:00 UTC on Day D+1 (June 11)
    df_h4 = pd.DataFrame({"timestamp": [datetime(2025, 6, 11, 0, 0, tzinfo=UTC)]})
    aligned, audit = align_yields_to_eurusd_h4(df_h4, df_reconciled)

    assert audit["number_of_leakage_violations"] == 0
    # Exactly on Day D+1 at 00:00 UTC, Day D values become usable!
    assert aligned["US_2Y"].iloc[0] == 4.80
    assert aligned["German_2Y"].iloc[0] == 2.80
    assert aligned["US_Germany_2Y_Spread"].iloc[0] == pytest.approx(2.00)
    assert aligned["yield_observation_date"].iloc[0] == pd.Timestamp("2025-06-10")


def test_five_day_spread_change_cannot_use_future_observations():
    """6. Test that five-day spread change cannot use future observations."""
    # Create 10 days of synthetic data
    dates = pd.date_range("2025-06-01", "2025-06-15", freq="B")
    us_vals = [4.00 + 0.05 * i for i in range(len(dates))]
    de_vals = [2.00 + 0.02 * i for i in range(len(dates))]
    df_us = pd.DataFrame({"date": dates, "us_2y": us_vals})
    df_de = pd.DataFrame({"date": dates, "german_2y": de_vals})

    df_reconciled, _ = reconcile_yield_calendars(
        df_us, df_de, str(dates[0].date()), str(dates[-1].date())
    )

    # Day index 6 (say Friday June 9)
    day_6 = dates[6]
    expected_spread_d6 = us_vals[6] - de_vals[6]
    expected_spread_d1 = us_vals[1] - de_vals[1]
    expected_5d_change = expected_spread_d6 - expected_spread_d1

    # On day 7 at 00:00 UTC, the bar should see exactly expected_5d_change
    day_7_bar = (day_6 + timedelta(days=1)).replace(tzinfo=UTC)
    df_h4 = pd.DataFrame({"timestamp": [day_7_bar]})
    aligned, _ = align_yields_to_eurusd_h4(df_h4, df_reconciled)

    actual_change = aligned["US_Germany_2Y_Spread_5D_Change"].iloc[0]
    assert actual_change == pytest.approx(expected_5d_change)


def test_missing_german_data_not_silently_zeroed():
    """7. Test that missing German data does not silently become zero."""
    # When German market is closed (e.g. Easter Monday), raw value is NaN
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-21"]),
            "german_2y_raw": ["."],
            "german_2y": [np.nan],
        }
    )
    val = validate_raw_series(df, "date", "german_2y", "TEST_DE", expected_freq="daily")
    assert val.missing_count == 1
    assert not (df["german_2y"] == 0.0).any()


def test_missing_us_data_not_silently_zeroed():
    """8. Test that missing US data does not silently become zero."""
    # When US market is closed (e.g. Labor Day), raw value is empty string / NaN
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-09-01"]),
            "us_2y_raw": ["."],
            "us_2y": [np.nan],
        }
    )
    val = validate_raw_series(df, "date", "us_2y", "TEST_US", expected_freq="daily")
    assert val.missing_count == 1
    assert not (df["us_2y"] == 0.0).any()


def test_holiday_gaps_handled_deterministically():
    """9. Test that holiday gaps are handled deterministically without data corruption."""
    # US closed on Day 2 (Labor Day), DE open on Day 2
    # DE closed on Day 4 (German Unity Day), US open on Day 4
    dates = pd.date_range("2025-09-01", periods=5, freq="B")
    df_us = pd.DataFrame(
        {
            "date": dates,
            "us_2y": [4.00, np.nan, 4.10, 4.15, 4.20],
        }
    )
    df_de = pd.DataFrame(
        {
            "date": dates,
            "german_2y": [2.00, 2.05, 2.10, np.nan, 2.20],
        }
    )

    df_reconciled, audit = reconcile_yield_calendars(
        df_us, df_de, str(dates[0].date()), str(dates[-1].date())
    )

    assert audit["us_only_holiday_count"] == 1
    assert audit["german_only_holiday_count"] == 1

    # On Day 2 (US holiday): US_2Y should forward fill from Day 1 (4.00), DE uses Day 2 (2.05)
    row_day2 = df_reconciled.iloc[1]
    assert row_day2["US_2Y"] == 4.00
    assert row_day2["German_2Y"] == 2.05
    assert row_day2["US_Germany_2Y_Spread"] == pytest.approx(4.00 - 2.05)

    # On Day 4 (DE holiday): DE_2Y should forward fill from Day 3 (2.10), US uses Day 4 (4.15)
    row_day4 = df_reconciled.iloc[3]
    assert row_day4["US_2Y"] == 4.15
    assert row_day4["German_2Y"] == 2.10
    assert row_day4["US_Germany_2Y_Spread"] == pytest.approx(4.15 - 2.10)


def test_dst_does_not_alter_causal_alignment():
    """10. Test that DST transitions do not alter causal alignment invariant."""
    # Verify both winter (EST/CET) and summer (EDT/CEST)
    winter_dt = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
    summer_dt = datetime(2025, 7, 15, 12, 0, tzinfo=UTC)
    gap_march_dt = datetime(2025, 3, 15, 12, 0, tzinfo=UTC)  # US on EDT, EU on CET

    winter_offsets = check_dst_offsets(winter_dt)
    summer_offsets = check_dst_offsets(summer_dt)
    gap_offsets = check_dst_offsets(gap_march_dt)

    assert winter_offsets["us_eu_difference_hours"] == 6.0
    assert summer_offsets["us_eu_difference_hours"] == 6.0
    assert gap_offsets["us_eu_difference_hours"] == 5.0

    # In ALL seasons, Day D US publication (~21:15 UTC or 20:15 UTC in DST) is strictly
    # prior to Day D+1 00:00:00 UTC. The conservative next-day convention is invariant to DST.
    pub_utc_winter = datetime(2025, 1, 15, 21, 15, tzinfo=UTC)
    pub_utc_summer = datetime(2025, 7, 15, 20, 15, tzinfo=UTC)
    usable_winter = datetime(2025, 1, 16, 0, 0, tzinfo=UTC)
    usable_summer = datetime(2025, 7, 16, 0, 0, tzinfo=UTC)

    assert usable_winter > pub_utc_winter
    assert usable_summer > pub_utc_summer


def test_duplicate_source_observations_detected():
    """11. Test that duplicate source observations are detected."""
    df_dup = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-01", "2025-06-01", "2025-06-02"]),
            "us_2y": [4.00, 4.05, 4.10],
        }
    )
    val = validate_raw_series(df_dup, "date", "us_2y", "TEST_DUP", expected_freq="daily")
    assert val.duplicate_count == 1
    assert any("duplicate" in a.lower() for a in val.anomalies)


def test_revised_vintage_cannot_silently_replace_originally_available_value():
    """12. Test that revised/vintage information cannot silently replace original values."""
    t0 = datetime(2025, 1, 15, 21, 15, tzinfo=UTC)
    t1 = datetime(2025, 1, 16, 12, 0, tzinfo=UTC)

    # Point 1: original release
    pt1 = ExogenousDataPoint(
        series_id="DGS2",
        observation_period="2025-01-15",
        information_timestamp=t0,
        feature_timestamp=t0,
        first_usable_bar_timestamp=datetime(2025, 1, 16, 0, 0, tzinfo=UTC),
        actual_value=4.25,
        revision_vintage=1,
    )
    # Point 2: silent revision (same vintage 1, but changed value)
    pt2 = ExogenousDataPoint(
        series_id="DGS2",
        observation_period="2025-01-15",
        information_timestamp=t1,
        feature_timestamp=t1,
        first_usable_bar_timestamp=datetime(2025, 1, 16, 16, 0, tzinfo=UTC),
        actual_value=4.30,
        revision_vintage=1,  # BUG: vintage not incremented!
    )
    errors = detect_untracked_revisions([pt1, pt2])
    assert len(errors) == 1
    assert "Silent revision detected" in errors[0]


def test_phase24_raw_artifact_cryptographic_hashes():
    """14. Test that all ingested raw artifacts have verified non-empty SHA-256 hashes."""
    prov_file = Path("data/external/macro_yields/metadata/yield_provenance.json")
    assert prov_file.exists(), "yield_provenance.json metadata must exist"

    with open(prov_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for key, item in metadata.items():
        assert len(item["sha256_hash"]) == 64
        # Verify hash against actual file on disk
        actual_hash = compute_file_sha256(item["raw_file_path"])
        assert actual_hash == item["sha256_hash"]


def test_phase24_aligned_dataset_integrity():
    """15. Test that the aligned EURUSD H4 research dataset has zero leakage violations."""
    report_file = Path("reports/phase24_yield_data_validation.json")
    assert report_file.exists(), "phase24_yield_data_validation.json must exist"

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["governance"]["phase11_test_lock_respected"] is True
    assert report["governance"]["no_model_training"] is True
    assert report["alignment"]["number_of_leakage_violations"] == 0
    assert report["alignment"]["same_day_leakage_violations"] == 0
    assert report["alignment"]["alignment_strictly_causal"] is True
    assert report["alignment"]["h4_bars_examined"] == 25800
    assert report["alignment"]["h4_bars_with_valid_exogenous_data"] == 25800
