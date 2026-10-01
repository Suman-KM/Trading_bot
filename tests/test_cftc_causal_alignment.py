"""Deterministic unit tests for CFTC COT data acquisition, validation, and causal alignment.

Verifies:
1. No COT observation is available to EURUSD bars prior to official publication time.
2. No future COT report leaks backward into historical bars.
3. US Daylight Saving Time (DST) transitions adjust publication UTC hours (19:30 vs 20:30 UTC).
4. US Federal holidays (e.g. Good Friday, July 4th) correctly delay publication.
5. Historical delayed publications (e.g. 2018-2019 government shutdown) are strictly enforced.
6. Duplicate report handling and validation.
7. Missing-report and gap handling via causal forward-fill.
8. Monotonicity of publication and effective timestamps.
9. Contract-code filtering strictly isolates CME Euro FX (code 099741).
10. Alignment reproducibility is bit-for-bit deterministic.
11. Open Interest accounting identities hold across all historical rows.
12. Provenance, validation, and coverage metadata artifacts exist and are consistent.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ai.data.exogenous.cftc_alignment import (
    align_cftc_to_eurusd,
    compute_cftc_timestamps,
)

PARQUET_PATH = Path("data/exogenous/cftc/cftc_eurofx_cot.parquet")
PROVENANCE_PATH = Path("reports/cftc_cot_provenance.json")
RAW_VAL_PATH = Path("reports/cftc_cot_raw_validation.json")
COVERAGE_PATH = Path("reports/cftc_cot_coverage.json")
EURUSD_H4_PATH = Path("data/raw/eurusd_h4_expanded/eurusd_h4_raw.parquet")

NY_TZ = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def cftc_df() -> pd.DataFrame:
    """Load normalized CFTC dataset."""
    assert PARQUET_PATH.exists(), f"Missing canonical dataset: {PARQUET_PATH}"
    return pd.read_parquet(PARQUET_PATH)


@pytest.fixture(scope="module")
def eurusd_h4_df() -> pd.DataFrame:
    """Load expanded EURUSD H4 dataset."""
    assert EURUSD_H4_PATH.exists(), f"Missing H4 dataset: {EURUSD_H4_PATH}"
    return pd.read_parquet(EURUSD_H4_PATH)


def test_contract_code_and_market_name(cftc_df: pd.DataFrame):
    """9. Contract code strictly isolates CME Euro FX (code 099741)."""
    assert len(cftc_df) >= 800, f"Expected at least 800 weekly reports, got {len(cftc_df)}"
    assert (cftc_df["contract_code"] == "099741").all(), "Non-099741 contract code detected"
    assert cftc_df["market_name"].str.contains("EURO FX").all(), "Market name mismatch"


def test_cftc_accounting_identities(cftc_df: pd.DataFrame):
    """11. Verify exact Open Interest accounting identities across all historical rows."""
    long_diffs = (
        cftc_df["open_interest"]
        - (cftc_df["total_reportable_long"] + cftc_df["non_reportable_long"])
    ).abs()
    short_diffs = (
        cftc_df["open_interest"]
        - (cftc_df["total_reportable_short"] + cftc_df["non_reportable_short"])
    ).abs()

    max_l = long_diffs.max()
    max_s = short_diffs.max()
    assert (long_diffs == 0).all(), f"Long-side accounting failure (max diff={max_l})"
    assert (short_diffs == 0).all(), f"Short-side accounting failure (max diff={max_s})"
    assert (cftc_df["open_interest"] > 0).all(), "Zero or negative open interest detected"


def test_dst_transition_correctness():
    """3. US DST transitions shift publication UTC hour from 20:30 (EST) to 19:30 (EDT)."""
    # Summer (EDT: UTC-4): 2024-07-16 -> Release 2024-07-19 Friday 15:30 EDT = 19:30 UTC
    ts_summer = compute_cftc_timestamps("2024-07-16")
    pub_summer_utc = datetime.fromisoformat(ts_summer["publication_time_utc"])
    assert pub_summer_utc.hour == 19
    assert pub_summer_utc.minute == 30

    # Winter (EST: UTC-5): 2024-01-16 -> Release 2024-01-19 Friday 15:30 EST = 20:30 UTC
    ts_winter = compute_cftc_timestamps("2024-01-16")
    pub_winter_utc = datetime.fromisoformat(ts_winter["publication_time_utc"])
    assert pub_winter_utc.hour == 20
    assert pub_winter_utc.minute == 30


def test_holiday_handling():
    """4. US Federal holiday falling on Friday correctly delays publication to Monday."""
    # Good Friday 2024: March 29, 2024. Tuesday obs: March 26, 2024.
    # Friday March 29 is holiday -> publication should be Monday April 1, 2024.
    ts_gf = compute_cftc_timestamps("2024-03-26")
    assert ts_gf["publication_date"] == "2024-04-01"
    pub_dt = datetime.fromisoformat(ts_gf["publication_time_utc"])
    assert pub_dt.weekday() == 0  # Monday release


def test_delayed_publication_handling():
    """5. Enforce historical delayed release schedule during government shutdown."""
    # 2018-2019 US Government Shutdown: Dec 24, 2018 report was delayed to Feb 1, 2019
    ts_shut = compute_cftc_timestamps("2018-12-24")
    assert ts_shut["publication_date"] == "2019-02-01"
    eff_dt = datetime.fromisoformat(ts_shut["effective_time_utc"])
    assert eff_dt >= datetime(2019, 2, 4, 0, 0, tzinfo=UTC)


def test_effective_timestamp_monotonicity(cftc_df: pd.DataFrame):
    """8. Effective timestamps must be strictly non-decreasing."""
    eff_ts = pd.to_datetime(cftc_df["effective_time_utc"], utc=True)
    diffs = eff_ts.diff().dropna()
    assert (diffs >= timedelta(0)).all(), "Effective timestamps are not monotonic"


def test_no_cot_observation_available_before_official_publication(
    cftc_df: pd.DataFrame,
    eurusd_h4_df: pd.DataFrame,
):
    """1. No EURUSD bar ever accesses a COT report prior to its official publication timestamp."""
    aligned_df, audit = align_cftc_to_eurusd(
        cftc_df, eurusd_h4_df, eurusd_timestamp_col="timestamp"
    )

    assert audit["causally_valid"] is True
    assert audit["leakage_violations_publication"] == 0
    assert audit["leakage_violations_effective"] == 0


def test_no_future_cot_report_leaks_backward(
    cftc_df: pd.DataFrame,
    eurusd_h4_df: pd.DataFrame,
):
    """2. Strict causal ordering: bar_ts >= effective_ts > pub_ts > obs_ts."""
    aligned_df, _ = align_cftc_to_eurusd(cftc_df, eurusd_h4_df, eurusd_timestamp_col="timestamp")

    aligned_subset = aligned_df.dropna(subset=["effective_time_utc"]).copy()
    bar_ts = pd.to_datetime(aligned_subset["timestamp"], utc=True)
    eff_ts = pd.to_datetime(aligned_subset["effective_time_utc"], utc=True)
    pub_ts = pd.to_datetime(aligned_subset["publication_time_utc"], utc=True)
    obs_ts = pd.to_datetime(aligned_subset["observation_timestamp_utc"], utc=True)

    # 1. Bar >= Effective
    assert (bar_ts >= eff_ts).all(), "Bar timestamp precedes effective timestamp"
    # 2. Effective > Publication
    assert (eff_ts > pub_ts).all(), "Effective timestamp does not succeed publication timestamp"
    # 3. Publication > Observation
    assert (pub_ts > obs_ts).all(), "Publication timestamp does not succeed observation timestamp"


def test_missing_report_handling():
    """7. Test that simulated gap in COT reports forward-fills safely without fabricating data."""
    mock_cftc = pd.DataFrame(
        [
            {
                "observation_date": "2024-01-09",
                "effective_time_utc": "2024-01-15T00:00:00+00:00",
                "publication_time_utc": "2024-01-12T20:30:00+00:00",
                "net_speculative_pos": 10000,
            },
            # Week of 2024-01-16 is intentionally missing
            {
                "observation_date": "2024-01-23",
                "effective_time_utc": "2024-01-29T00:00:00+00:00",
                "publication_time_utc": "2024-01-26T20:30:00+00:00",
                "net_speculative_pos": 15000,
            },
        ]
    )

    mock_eurusd = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-14", "2024-02-02", freq="D", tz="UTC"),
        }
    )

    aligned, audit = align_cftc_to_eurusd(mock_cftc, mock_eurusd, eurusd_timestamp_col="timestamp")
    assert audit["causally_valid"] is True

    # Bars between Jan 15 and Jan 28 should hold 10,000 (forward filled)
    in_gap = (aligned["timestamp"] >= "2024-01-15") & (aligned["timestamp"] < "2024-01-29")
    mid_bars = aligned[in_gap]
    assert (mid_bars["net_speculative_pos"] == 10000).all()

    # Bars on or after Jan 29 should update to 15,000
    late_bars = aligned[aligned["timestamp"] >= "2024-01-29"]
    assert (late_bars["net_speculative_pos"] == 15000).all()


def test_alignment_reproducibility(cftc_df: pd.DataFrame, eurusd_h4_df: pd.DataFrame):
    """10. Alignment execution is 100% deterministic and bit-for-bit reproducible."""
    aligned1, audit1 = align_cftc_to_eurusd(cftc_df, eurusd_h4_df)
    aligned2, audit2 = align_cftc_to_eurusd(cftc_df, eurusd_h4_df)

    assert audit1 == audit2
    assert aligned1.equals(aligned2)


def test_provenance_and_validation_reports_exist():
    """12. Verify that provenance, validation, and coverage JSON reports exist and passed."""
    assert PROVENANCE_PATH.exists(), f"Missing: {PROVENANCE_PATH}"
    assert RAW_VAL_PATH.exists(), f"Missing: {RAW_VAL_PATH}"
    assert COVERAGE_PATH.exists(), f"Missing: {COVERAGE_PATH}"

    prov = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert prov["status"] == "verified"
    assert prov["downloaded_files_count"] == 34
    assert len(prov["files"]) == 34

    val = json.loads(RAW_VAL_PATH.read_text(encoding="utf-8"))
    assert val["status"] == "PASS"
    assert val["accounting_identities_valid"] is True
    assert len(val["validation_errors"]) == 0

    cov = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    assert cov["total_weekly_reports"] >= 800
    assert cov["duplicate_dates"] == 0
    assert cov["coverage_status"] == "continuous_weekly_16_years"
