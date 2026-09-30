"""Unit test suite for Phase 23 Exogenous Information Feasibility & Research Design.

Verifies strict non-anticipative causality, publication timestamp enforcement,
revision vintage protection, deterministic timezone and DST conversions, and
absence of lookahead or silent zero-imputation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from ai.backtest.robustness import (
    PHASE11_TEST_LOCK_TIMESTAMP,
    VALIDATION_END_TIMESTAMP,
    verify_test_partition_rejection,
)
from ai.data.exogenous.alignment import (
    align_exogenous_events_to_bars,
    compute_first_usable_bar,
)
from ai.data.exogenous.schema import (
    AlignmentConfig,
    ExogenousDataPoint,
    FeasibilityClassification,
)
from ai.data.exogenous.validation import (
    check_dst_offsets,
    detect_untracked_revisions,
    verify_no_silent_zero_imputation,
)


def test_phase11_test_lock_boundary_rejection():
    """Governance test: ensure test partition boundaries are strictly enforced."""
    verify_test_partition_rejection(VALIDATION_END_TIMESTAMP)

    test_dt = PHASE11_TEST_LOCK_TIMESTAMP + timedelta(minutes=15)
    with pytest.raises(ValueError, match="CRITICAL TEST PARTITION VIOLATION"):
        verify_test_partition_rejection(test_dt)


def test_future_event_cannot_affect_earlier_bar():
    """1. Test that a future exogenous event cannot affect an earlier market bar."""
    bar_times = [
        datetime(2025, 1, 15, 13, 0, tzinfo=UTC),
        datetime(2025, 1, 15, 13, 15, tzinfo=UTC),
        datetime(2025, 1, 15, 13, 30, tzinfo=UTC),
    ]
    df_bars = pd.DataFrame({"timestamp": bar_times})

    # Event published at 13:30:00 UTC (during the 13:30 bar)
    event_t = datetime(2025, 1, 15, 13, 30, tzinfo=UTC)
    config = AlignmentConfig(
        bar_timeframe_minutes=15,
        bar_timestamp_is_open=True,
        latency_buffer_seconds=60,
    )
    usable_bar = compute_first_usable_bar(event_t, config)

    event = ExogenousDataPoint(
        series_id="TEST_EVENT",
        observation_period="2025-01",
        information_timestamp=event_t,
        feature_timestamp=event_t,
        first_usable_bar_timestamp=usable_bar,
        actual_value=100.0,
        surprise_value=1.5,
    )

    series, result = align_exogenous_events_to_bars(
        df_bars=df_bars,
        events=[event],
        value_attribute="surprise_value",
        config=config,
    )

    assert result.is_valid
    assert result.lookahead_violations == 0
    # Bars at 13:00 and 13:15 MUST NOT know the event
    assert np.isnan(series.iloc[0])  # 13:00 bar
    assert np.isnan(series.iloc[1])  # 13:15 bar
    # Bar at 13:30 is allowed to incorporate the event
    assert series.iloc[2] == 1.5


def test_macro_release_publication_timestamp_enforcement():
    """2. Test that a macro release becomes available strictly at its publication timestamp."""
    # Publication at 08:30:00 ET (13:30:00 UTC) with 60-second transmission buffer
    pub_time = datetime(2025, 6, 6, 12, 30, 0, tzinfo=UTC)  # Summer EDT: 08:30 ET = 12:30 UTC
    config = AlignmentConfig(
        bar_timeframe_minutes=15,
        bar_timestamp_is_open=True,
        latency_buffer_seconds=60,
    )

    first_usable = compute_first_usable_bar(pub_time, config)

    # Effective arrival is 12:31:00 UTC.
    # The bar opening at 12:15 closes at 12:30:00 (before 12:31:00).
    # The bar opening at 12:30 closes at 12:45:00 (after 12:31:00).
    # Thus, the first usable bar open timestamp is 12:30:00 UTC.
    assert first_usable == datetime(2025, 6, 6, 12, 30, 0, tzinfo=UTC)

    # Creating an event where first_usable precedes publication MUST raise ValueError
    with pytest.raises(ValueError, match="Lookahead violation"):
        ExogenousDataPoint(
            series_id="ILLEGAL_LEAK",
            observation_period="2025-05",
            information_timestamp=pub_time,
            feature_timestamp=pub_time,
            first_usable_bar_timestamp=pub_time - timedelta(minutes=15),
            actual_value=250.0,
        )


def test_revision_tracking_prevents_silent_replacement():
    """3. Test that revised historical values cannot silently replace point-in-time figures."""
    t1 = datetime(2025, 1, 15, 13, 30, tzinfo=UTC)
    t2 = datetime(2025, 2, 15, 13, 30, tzinfo=UTC)

    # Point 1: Initial release for 2024-12
    pt1 = ExogenousDataPoint(
        series_id="GDP_GROWTH",
        observation_period="2024-Q4",
        information_timestamp=t1,
        feature_timestamp=t1,
        first_usable_bar_timestamp=t1,
        actual_value=2.4,
        revision_vintage=1,
    )

    # Point 2: Revised release for 2024-Q4 without incrementing vintage (silent overwrite)
    pt2_corrupt = ExogenousDataPoint(
        series_id="GDP_GROWTH",
        observation_period="2024-Q4",
        information_timestamp=t2,
        feature_timestamp=t2,
        first_usable_bar_timestamp=t2,
        actual_value=2.8,  # Revised from 2.4 to 2.8!
        revision_vintage=1,  # Corrupt: Failed to increment vintage
    )

    errors = detect_untracked_revisions([pt1, pt2_corrupt])
    assert len(errors) == 1
    assert "Silent revision detected" in errors[0]

    # Valid revision with incremented vintage (vintage 1 -> 2)
    pt2_valid = ExogenousDataPoint(
        series_id="GDP_GROWTH",
        observation_period="2024-Q4",
        information_timestamp=t2,
        feature_timestamp=t2,
        first_usable_bar_timestamp=t2,
        actual_value=2.8,
        revision_vintage=2,
    )
    assert len(detect_untracked_revisions([pt1, pt2_valid])) == 0


def test_timezone_conversion_deterministic():
    """4. Test that timezone conversion to UTC is strictly deterministic."""
    # US Eastern Standard Time (EST is UTC-5)
    est_zone = ZoneInfo("America/New_York")
    local_time = datetime(2025, 1, 15, 8, 30, 0, tzinfo=est_zone)
    utc_time = local_time.astimezone(UTC)

    assert utc_time.hour == 13
    assert utc_time.minute == 30
    assert utc_time.tzinfo == UTC

    # Central European Time (CET is UTC+1)
    cet_zone = ZoneInfo("Europe/Berlin")
    ecb_local = datetime(2025, 1, 15, 14, 15, 0, tzinfo=cet_zone)
    ecb_utc = ecb_local.astimezone(UTC)

    assert ecb_utc.hour == 13
    assert ecb_utc.minute == 15
    assert ecb_utc.tzinfo == UTC


def test_dst_transition_deterministic():
    """5. Test deterministic handling of US and EU DST transition gap."""
    # In March 2025:
    # US DST starts Sunday, March 9, 2025 (clocks forward: UTC-5 -> UTC-4).
    # EU DST starts Sunday, March 30, 2025 (clocks forward: UTC+1 -> UTC+2).
    # Between March 9 and March 30, 2025, the gap between NY and Berlin is 5 hours instead of 6!

    # 1. Standard winter (January 15)
    winter_offsets = check_dst_offsets(datetime(2025, 1, 15, 12, 0, tzinfo=UTC))
    assert winter_offsets["us_offset_hours"] == -5.0
    assert winter_offsets["eu_offset_hours"] == 1.0
    assert winter_offsets["us_eu_difference_hours"] == 6.0

    # 2. Desynchronization gap (March 15)
    march_offsets = check_dst_offsets(datetime(2025, 3, 15, 12, 0, tzinfo=UTC))
    assert march_offsets["us_offset_hours"] == -4.0  # US EDT
    assert march_offsets["eu_offset_hours"] == 1.0  # Still EU CET!
    assert march_offsets["us_eu_difference_hours"] == 5.0  # 5 hours gap!

    # 3. Summer synchronized (June 15)
    summer_offsets = check_dst_offsets(datetime(2025, 6, 15, 12, 0, tzinfo=UTC))
    assert summer_offsets["us_offset_hours"] == -4.0  # US EDT
    assert summer_offsets["eu_offset_hours"] == 2.0  # EU CEST
    assert summer_offsets["us_eu_difference_hours"] == 6.0  # Back to 6 hours gap!


def test_missing_data_not_silently_zeroed():
    """6. Test that missing external data does not silently become zero."""
    bar_times = [
        datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
        datetime(2025, 1, 15, 10, 15, tzinfo=UTC),
        datetime(2025, 1, 15, 10, 30, tzinfo=UTC),
    ]
    df_bars = pd.DataFrame({"timestamp": bar_times})

    # Empty events list
    series, result = align_exogenous_events_to_bars(
        df_bars=df_bars,
        events=[],
        value_attribute="surprise_value",
    )

    assert result.aligned_bars == 0
    assert result.missing_bars == 3
    # Check that missing values are NaN and NEVER 0.0
    for val in series:
        assert np.isnan(val)
        assert val != 0.0

    assert verify_no_silent_zero_imputation(series, [0, 1, 2])


def test_missing_timestamps_do_not_cause_lookahead():
    """7. Test that missing or irregular timestamps do not cause forward-looking fills."""
    # Weekend gap between Friday close and Sunday open
    bar_times = [
        datetime(2025, 1, 17, 21, 45, tzinfo=UTC),  # Friday close
        datetime(2025, 1, 19, 22, 0, tzinfo=UTC),  # Sunday open (gap of >48h)
        datetime(2025, 1, 19, 22, 15, tzinfo=UTC),
    ]
    df_bars = pd.DataFrame({"timestamp": bar_times})

    # Event on Friday afternoon
    ev_time = datetime(2025, 1, 17, 18, 0, tzinfo=UTC)
    config = AlignmentConfig(
        bar_timeframe_minutes=15,
        max_lookback_bars=4,  # Max 4 bars lookback (1 hour)
    )

    event = ExogenousDataPoint(
        series_id="FRIDAY_SURPRISE",
        observation_period="2025-01-17",
        information_timestamp=ev_time,
        feature_timestamp=ev_time,
        first_usable_bar_timestamp=compute_first_usable_bar(ev_time, config),
        actual_value=50.0,
        surprise_value=2.0,
    )

    series, result = align_exogenous_events_to_bars(
        df_bars=df_bars,
        events=[event],
        value_attribute="surprise_value",
        config=config,
    )

    assert result.is_valid
    # On Friday 21:45 (3.75 hours later), elapsed time > 1h max lookback, so it must expire to NaN
    assert np.isnan(series.iloc[0])
    # Sunday bar must NOT be populated with Friday event
    assert np.isnan(series.iloc[1])
    assert np.isnan(series.iloc[2])


def test_data_gaps_do_not_fabricate_information():
    """8. Test that data gaps cannot produce fabricated or linearly interpolated data."""
    # Series of 5 bars
    bar_times = [
        datetime(2025, 1, 15, 12, 0, tzinfo=UTC) + timedelta(minutes=15 * i) for i in range(5)
    ]
    df_bars = pd.DataFrame({"timestamp": bar_times})

    # Only one point at bar 2 (12:30)
    ev_time = datetime(2025, 1, 15, 12, 30, tzinfo=UTC)
    config = AlignmentConfig(bar_timeframe_minutes=15, latency_buffer_seconds=0)

    event = ExogenousDataPoint(
        series_id="SINGLE_PULSE",
        observation_period="2025-01-15",
        information_timestamp=ev_time,
        feature_timestamp=ev_time,
        first_usable_bar_timestamp=ev_time,
        actual_value=10.0,
        surprise_value=5.0,
    )

    # Exact bar only policy (pulse signal)
    series, result = align_exogenous_events_to_bars(
        df_bars=df_bars,
        events=[event],
        value_attribute="surprise_value",
        fill_policy="exact_bar_only",
        config=config,
    )

    assert result.is_valid
    assert np.isnan(series.iloc[0])
    assert np.isnan(series.iloc[1])
    assert series.iloc[2] == 5.0
    assert np.isnan(series.iloc[3])  # Must not carry over or interpolate!
    assert np.isnan(series.iloc[4])


def test_phase23_feasibility_report_artifact_integrity():
    """9. Verify the structural integrity of the generated Phase 23 JSON report."""
    report_path = Path("reports/phase23_exogenous_data_feasibility.json")
    assert report_path.exists(), f"Report artifact {report_path} must exist."

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["metadata"]["phase"] == "23"
    assert data["metadata"]["scientific_decision"] in [c.value for c in FeasibilityClassification]
    assert data["governance"]["phase11_test_lock_respected"] is True
    assert data["governance"]["no_live_trading"] is True
    assert len(data["availability_matrix"]) >= 14
    assert len(data["source_audits"]) >= 14
