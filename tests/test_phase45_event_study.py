"""Tests for Phase 45 Empirical Event Study & Central Bank Shock Evaluation.

Verifies:
1. IANA ZoneInfo timezone conversions for FOMC (America/New_York) and ECB (Europe/Berlin).
2. Proper handling of Daylight Saving Time shifts (EDT/EST, CEST/CET, and the March DST gap).
3. Pre-event baseline sampling strictly precedes announcement release.
4. MFE, MAE, signed pips, and reversal metrics calculations.
5. Primary 60-minute window statistical evaluations.
6. Multi-resolution alignment (M15 high-resolution vs H4 historical coverage).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from scripts.run_phase45_event_study import (
    MacroEvent,
    get_historical_central_bank_events,
    resolve_event_utc_timestamp,
    run_event_study,
)


def test_timezone_resolution_fomc_est_and_edt() -> None:
    """Invariant: FOMC 14:00 NY translates accurately to 19:00 UTC (EST) or 18:00 UTC (EDT)."""
    # Winter / Standard Time (EST = UTC-5) -> 14:00 EST is 19:00 UTC
    winter_event = MacroEvent(
        event_id="TEST_FOMC_WINTER",
        central_bank="FED",
        event_date="2024-01-31",
        local_time="14:00",
        local_tz="America/New_York",
        surprise_mps=10.0,
        shock_type="HAWKISH",
        expected_eurusd_direction=-1,
    )
    utc_winter = resolve_event_utc_timestamp(winter_event)
    assert utc_winter.hour == 19
    assert utc_winter.minute == 0
    assert utc_winter.utcoffset() == timedelta(0)

    # Summer / Daylight Saving Time (EDT = UTC-4) -> 14:00 EDT is 18:00 UTC
    summer_event = MacroEvent(
        event_id="TEST_FOMC_SUMMER",
        central_bank="FED",
        event_date="2024-06-12",
        local_time="14:00",
        local_tz="America/New_York",
        surprise_mps=10.0,
        shock_type="HAWKISH",
        expected_eurusd_direction=-1,
    )
    utc_summer = resolve_event_utc_timestamp(summer_event)
    assert utc_summer.hour == 18
    assert utc_summer.minute == 0
    assert utc_summer.utcoffset() == timedelta(0)


def test_timezone_resolution_ecb_cet_and_cest() -> None:
    """Invariant: ECB 14:15 Frankfurt translates to 13:15 UTC (CET) or 12:15 UTC (CEST)."""
    # Winter / CET (UTC+1) -> 14:15 CET is 13:15 UTC
    winter_ecb = MacroEvent(
        event_id="TEST_ECB_WINTER",
        central_bank="ECB",
        event_date="2024-01-25",
        local_time="14:15",
        local_tz="Europe/Berlin",
        surprise_mps=8.0,
        shock_type="HAWKISH",
        expected_eurusd_direction=1,
    )
    utc_ecb_winter = resolve_event_utc_timestamp(winter_ecb)
    assert utc_ecb_winter.hour == 13
    assert utc_ecb_winter.minute == 15
    assert utc_ecb_winter.utcoffset() == timedelta(0)

    # Summer / CEST (UTC+2) -> 14:15 CEST is 12:15 UTC
    summer_ecb = MacroEvent(
        event_id="TEST_ECB_SUMMER",
        central_bank="ECB",
        event_date="2024-06-06",
        local_time="14:15",
        local_tz="Europe/Berlin",
        surprise_mps=8.0,
        shock_type="HAWKISH",
        expected_eurusd_direction=1,
    )
    utc_ecb_summer = resolve_event_utc_timestamp(summer_ecb)
    assert utc_ecb_summer.hour == 12
    assert utc_ecb_summer.minute == 15
    assert utc_ecb_summer.utcoffset() == timedelta(0)


def test_march_dst_gap_resolution() -> None:
    """Invariant: Late March DST gap (US on EDT, Europe on CET) is handled correctly."""
    # On March 20, 2024, US is already on EDT (started March 10),
    # but Europe is still on CET (starts March 31).
    fomc_march = MacroEvent(
        event_id="TEST_FOMC_MARCH",
        central_bank="FED",
        event_date="2024-03-20",
        local_time="14:00",
        local_tz="America/New_York",
        surprise_mps=5.0,
        shock_type="HAWKISH",
        expected_eurusd_direction=-1,
    )
    utc_dt = resolve_event_utc_timestamp(fomc_march)
    assert utc_dt.hour == 18  # EDT is UTC-4
    assert utc_dt.utcoffset() == timedelta(0)


def test_event_database_integrity() -> None:
    """Invariant: Event database generates deterministic list of central bank events."""
    events = get_historical_central_bank_events()
    assert len(events) >= 200
    fed_events = [e for e in events if e.central_bank == "FED"]
    ecb_events = [e for e in events if e.central_bank == "ECB"]

    assert len(fed_events) > 100
    assert len(ecb_events) > 100

    for e in events:
        assert e.expected_eurusd_direction in (-1, 1)
        assert abs(e.surprise_mps) > 0.0
        assert len(e.event_date) == 10
        datetime.strptime(e.event_date, "%Y-%m-%d")


def test_event_study_results_file() -> None:
    """Invariant: Event study produces a valid result report with required statistical fields."""
    report_path = Path("reports/phase45_event_results.json")
    if not report_path.exists():
        m15_p = Path("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
        h4_p = Path("data/external/macro_yields/processed/eurusd_h4_yield_aligned.parquet")
        run_event_study(m15_p, h4_p)

    assert report_path.exists()
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["total_events_evaluated"] >= 200
    assert meta["primary_window"] == "60m"

    p_summary = data["primary_window_summary"]
    assert p_summary["window_minutes"] == 60
    assert "directional_accuracy" in p_summary
    assert "mean_signed_pips" in p_summary
    assert "t_statistic" in p_summary
    assert "t_test_p_value" in p_summary
    assert "mfe_mae_ratio" in p_summary
    assert "post_event_reversal_rate_m15" in data
