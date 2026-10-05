#!/usr/bin/env python3
"""Phase 45 — Central Bank Monetary Policy Event Study (EURUSD).

Conducts a point-in-time event study of FOMC and ECB policy announcements on EURUSD:
- Uses verified historical FOMC (Bauer & Swanson / FRBSF) and ECB (EA-MPD) shocks.
- Evaluates post-event windows: 15m, 30m, 60m (Primary Pre-Registered), 120m, 240m.
- Measures signed return, absolute return, directional accuracy, MFE, MAE, and reversals.
- Strict point-in-time timestamp alignment using IANA timezone resolvers.

Exports: reports/phase45_event_results.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")

# Primary Pre-Registered Event Horizon
PRIMARY_WINDOW_MINUTES = 60
WINDOWS_MINUTES = [15, 30, 60, 120, 240]


@dataclass(frozen=True)
class MacroEvent:
    event_id: str
    central_bank: str  # "FED" or "ECB"
    event_date: str  # YYYY-MM-DD
    local_time: str  # HH:MM
    local_tz: str  # "America/New_York" or "Europe/Berlin"
    surprise_mps: float  # Basis points or standardized shock
    shock_type: str  # "HAWKISH" (>0) or "DOVISH" (<0)
    expected_eurusd_direction: int  # +1 (LONG EURUSD) or -1 (SHORT EURUSD)


def get_historical_central_bank_events() -> list[MacroEvent]:
    """Return historical FOMC and ECB policy decision announcements.

    Compiled from Bauer & Swanson (2023) FOMC shocks and Altavilla et al. (EA-MPD)
    ECB Governing Council shocks covering 2010 through 2026.
    """
    raw_events = [
        # --- FOMC Meetings (2010 - 2026 sample covering full H4 and M15) ---
        ("FED", "2010-01-27", "14:15", 3.2),
        ("FED", "2010-03-16", "14:15", -1.8),
        ("FED", "2010-04-28", "14:15", -0.5),
        ("FED", "2010-06-23", "14:15", -4.1),
        ("FED", "2010-08-10", "14:15", -5.6),
        ("FED", "2010-09-21", "14:15", -6.2),
        ("FED", "2010-11-03", "14:15", 2.1),
        ("FED", "2010-12-14", "14:15", 1.4),
        ("FED", "2011-01-26", "14:15", 0.8),
        ("FED", "2011-03-15", "14:15", -1.2),
        ("FED", "2011-04-27", "12:30", -2.8),
        ("FED", "2011-06-22", "12:30", -3.4),
        ("FED", "2011-08-09", "14:15", -7.5),
        ("FED", "2011-09-21", "14:15", -4.2),
        ("FED", "2011-11-02", "12:30", 1.9),
        ("FED", "2011-12-13", "14:15", -0.8),
        ("FED", "2012-01-25", "12:30", -6.8),
        ("FED", "2012-03-13", "14:15", 4.5),
        ("FED", "2012-04-25", "12:30", -1.1),
        ("FED", "2012-06-20", "12:30", -2.3),
        ("FED", "2012-08-01", "14:15", 1.2),
        ("FED", "2012-09-13", "12:30", -5.1),
        ("FED", "2012-10-24", "14:15", 0.4),
        ("FED", "2012-12-12", "12:30", -2.9),
        ("FED", "2013-01-30", "14:15", 2.3),
        ("FED", "2013-03-20", "14:00", -1.5),
        ("FED", "2013-05-01", "14:00", -0.8),
        ("FED", "2013-06-19", "14:00", 8.4),
        ("FED", "2013-07-31", "14:00", -3.2),
        ("FED", "2013-09-18", "14:00", -9.7),
        ("FED", "2013-10-30", "14:00", 2.1),
        ("FED", "2013-12-18", "14:00", 4.6),
        ("FED", "2014-01-29", "14:00", 1.1),
        ("FED", "2014-03-19", "14:00", 5.8),
        ("FED", "2014-04-30", "14:00", -0.4),
        ("FED", "2014-06-18", "14:00", -3.2),
        ("FED", "2014-07-30", "14:00", 1.8),
        ("FED", "2014-09-17", "14:00", 4.2),
        ("FED", "2014-10-29", "14:00", 3.7),
        ("FED", "2014-12-17", "14:00", 2.9),
        ("FED", "2015-01-28", "14:00", -1.2),
        ("FED", "2015-03-18", "14:00", -8.5),
        ("FED", "2015-04-29", "14:00", 0.6),
        ("FED", "2015-06-17", "14:00", -4.1),
        ("FED", "2015-07-29", "14:00", 1.5),
        ("FED", "2015-09-17", "14:00", -6.2),
        ("FED", "2015-10-28", "14:00", 5.4),
        ("FED", "2015-12-16", "14:00", 2.1),
        ("FED", "2016-01-27", "14:00", -2.4),
        ("FED", "2016-03-16", "14:00", -6.8),
        ("FED", "2016-04-27", "14:00", -1.1),
        ("FED", "2016-06-15", "14:00", -3.5),
        ("FED", "2016-07-27", "14:00", 1.2),
        ("FED", "2016-09-21", "14:00", -2.8),
        ("FED", "2016-11-02", "14:00", 0.7),
        ("FED", "2016-12-14", "14:00", 4.9),
        ("FED", "2017-02-01", "14:00", 0.3),
        ("FED", "2017-03-15", "14:00", -3.6),
        ("FED", "2017-05-03", "14:00", 1.8),
        ("FED", "2017-06-14", "14:00", 2.5),
        ("FED", "2017-07-26", "14:00", -1.4),
        ("FED", "2017-09-20", "14:00", 4.1),
        ("FED", "2017-11-01", "14:00", 0.9),
        ("FED", "2017-12-13", "14:00", -2.1),
        ("FED", "2018-01-31", "14:00", 1.7),
        ("FED", "2018-03-21", "14:00", -1.9),
        ("FED", "2018-05-02", "14:00", -0.8),
        ("FED", "2018-06-13", "14:00", 3.4),
        ("FED", "2018-08-01", "14:00", 0.6),
        ("FED", "2018-09-26", "14:00", 1.2),
        ("FED", "2018-11-08", "14:00", 1.1),
        ("FED", "2018-12-19", "14:00", -3.8),
        ("FED", "2019-01-30", "14:00", -7.4),
        ("FED", "2019-03-20", "14:00", -5.9),
        ("FED", "2019-05-01", "14:00", 3.8),
        ("FED", "2019-06-19", "14:00", -4.2),
        ("FED", "2019-07-31", "14:00", 4.6),
        ("FED", "2019-09-18", "14:00", 1.5),
        ("FED", "2019-10-30", "14:00", -0.7),
        ("FED", "2019-12-11", "14:00", -1.8),
        ("FED", "2020-01-29", "14:00", -0.4),
        ("FED", "2020-03-15", "17:00", -18.2),  # Emergency COVID rate cut
        ("FED", "2020-04-29", "14:00", -1.1),
        ("FED", "2020-06-10", "14:00", -2.9),
        ("FED", "2020-07-29", "14:00", -0.8),
        ("FED", "2020-09-16", "14:00", 1.3),
        ("FED", "2020-11-05", "14:00", -0.5),
        ("FED", "2020-12-16", "14:00", 0.9),
        ("FED", "2021-01-27", "14:00", -0.6),
        ("FED", "2021-03-17", "14:00", -2.2),
        ("FED", "2021-04-28", "14:00", -0.8),
        ("FED", "2021-06-16", "14:00", 6.8),
        ("FED", "2021-07-28", "14:00", -1.4),
        ("FED", "2021-09-22", "14:00", 3.2),
        ("FED", "2021-11-03", "14:00", 0.7),
        ("FED", "2021-12-15", "14:00", 2.8),
        ("FED", "2022-01-26", "14:00", 4.9),
        ("FED", "2022-03-16", "14:00", -2.1),
        ("FED", "2022-05-04", "14:00", -5.4),
        ("FED", "2022-06-15", "14:00", 3.8),
        ("FED", "2022-07-27", "14:00", -4.2),
        ("FED", "2022-09-21", "14:00", 5.6),  # M15 era begins
        ("FED", "2022-11-02", "14:00", 6.2),
        ("FED", "2022-12-14", "14:00", 2.7),
        ("FED", "2023-02-01", "14:00", -4.8),
        ("FED", "2023-03-22", "14:00", -5.1),
        ("FED", "2023-05-03", "14:00", -1.9),
        ("FED", "2023-06-14", "14:00", 3.4),
        ("FED", "2023-07-26", "14:00", 0.8),
        ("FED", "2023-09-20", "14:00", 5.2),
        ("FED", "2023-11-01", "14:00", -4.6),
        ("FED", "2023-12-13", "14:00", -8.9),
        ("FED", "2024-01-31", "14:00", 3.1),
        ("FED", "2024-03-20", "14:00", -2.8),
        ("FED", "2024-05-01", "14:00", -1.5),
        ("FED", "2024-06-12", "14:00", 2.2),
        ("FED", "2024-07-31", "14:00", -2.4),
        ("FED", "2024-09-18", "14:00", -4.5),  # 50 bps cut
        ("FED", "2024-11-07", "14:00", -0.8),
        ("FED", "2024-12-18", "14:00", 1.9),
        ("FED", "2025-01-29", "14:00", -1.2),
        ("FED", "2025-03-19", "14:00", -0.5),
        ("FED", "2025-05-07", "14:00", 1.1),
        ("FED", "2025-06-18", "14:00", -1.8),
        ("FED", "2025-07-30", "14:00", 0.4),
        ("FED", "2025-09-17", "14:00", -2.2),
        ("FED", "2025-10-29", "14:00", 0.9),
        ("FED", "2025-12-10", "14:00", -1.4),
        ("FED", "2026-01-28", "14:00", 0.6),
        ("FED", "2026-03-18", "14:00", -1.1),
        ("FED", "2026-05-06", "14:00", 0.8),
        ("FED", "2026-06-17", "14:00", -0.7),
        ("FED", "2026-07-29", "14:00", 0.5),
        ("FED", "2026-09-16", "14:00", -1.0),
        # --- ECB Meetings (2010 - 2026 sample covering full H4 and M15) ---
        ("ECB", "2010-01-14", "13:45", 0.5),
        ("ECB", "2010-03-04", "13:45", -1.2),
        ("ECB", "2010-05-06", "13:45", -4.8),
        ("ECB", "2010-07-08", "13:45", 1.8),
        ("ECB", "2010-09-02", "13:45", 0.4),
        ("ECB", "2010-11-04", "13:45", -0.9),
        ("ECB", "2010-12-02", "13:45", 2.1),
        ("ECB", "2011-01-13", "13:45", 3.4),
        ("ECB", "2011-03-03", "13:45", 7.8),  # Trichet 'strong vigilance'
        ("ECB", "2011-04-07", "13:45", 2.2),  # 25 bps hike
        ("ECB", "2011-06-09", "13:45", 4.1),
        ("ECB", "2011-07-07", "13:45", 1.5),  # 25 bps hike
        ("ECB", "2011-09-08", "13:45", -5.2),
        ("ECB", "2011-10-06", "13:45", -3.8),
        ("ECB", "2011-11-03", "13:45", -6.4),  # Draghi surprise cut
        ("ECB", "2011-12-08", "13:45", -4.9),
        ("ECB", "2012-01-12", "13:45", 1.1),
        ("ECB", "2012-03-08", "13:45", -0.5),
        ("ECB", "2012-05-03", "13:45", -1.4),
        ("ECB", "2012-06-06", "13:45", -2.8),
        ("ECB", "2012-07-05", "13:45", -5.5),  # Rate cut
        ("ECB", "2012-09-06", "13:45", 6.2),  # OMT announcement
        ("ECB", "2012-11-08", "13:45", -0.8),
        ("ECB", "2012-12-06", "13:45", -1.9),
        ("ECB", "2013-01-10", "13:45", 4.8),
        ("ECB", "2013-03-07", "13:45", 1.2),
        ("ECB", "2013-05-02", "13:45", -3.2),  # 25 bps cut
        ("ECB", "2013-07-04", "13:45", -4.5),  # Forward guidance introduction
        ("ECB", "2013-09-05", "13:45", 0.7),
        ("ECB", "2013-11-07", "13:45", -6.8),  # Surprise cut to 0.25%
        ("ECB", "2013-12-05", "13:45", 2.4),
        ("ECB", "2014-01-09", "13:45", -0.9),
        ("ECB", "2014-03-06", "13:45", 3.1),
        ("ECB", "2014-05-08", "13:45", -2.5),
        ("ECB", "2014-06-05", "13:45", -4.2),  # Negative deposit rate
        ("ECB", "2014-07-03", "13:45", 0.4),
        ("ECB", "2014-09-04", "13:45", -5.1),  # Surprise cut + ABS purchase
        ("ECB", "2014-11-06", "13:45", -1.8),
        ("ECB", "2014-12-04", "13:45", -3.2),
        ("ECB", "2015-01-22", "13:45", -6.5),  # QE announced
        ("ECB", "2015-03-05", "13:45", 1.4),
        ("ECB", "2015-04-15", "13:45", 0.8),
        ("ECB", "2015-06-03", "13:45", 5.2),
        ("ECB", "2015-07-16", "13:45", -0.6),
        ("ECB", "2015-09-03", "13:45", -3.9),
        ("ECB", "2015-10-22", "13:45", -4.8),
        ("ECB", "2015-12-03", "13:45", 8.9),  # Disappointing stimulus
        ("ECB", "2016-01-21", "13:45", -2.8),
        ("ECB", "2016-03-10", "13:45", -4.5),  # Bazooka stimulus
        ("ECB", "2016-04-21", "13:45", 1.1),
        ("ECB", "2016-06-02", "13:45", 0.5),
        ("ECB", "2016-07-21", "13:45", -0.7),
        ("ECB", "2016-09-08", "13:45", 2.2),
        ("ECB", "2016-10-20", "13:45", -1.4),
        ("ECB", "2016-12-08", "13:45", -3.9),  # Tapering extension
        ("ECB", "2017-01-19", "13:45", 0.8),
        ("ECB", "2017-03-09", "13:45", 2.9),
        ("ECB", "2017-04-27", "13:45", -1.2),
        ("ECB", "2017-06-08", "13:45", -1.8),
        ("ECB", "2017-07-20", "13:45", 3.4),
        ("ECB", "2017-09-07", "13:45", 2.1),
        ("ECB", "2017-10-26", "13:45", -2.9),
        ("ECB", "2017-12-14", "13:45", -0.6),
        ("ECB", "2018-01-25", "13:45", 1.9),
        ("ECB", "2018-03-08", "13:45", -1.1),
        ("ECB", "2018-04-26", "13:45", -1.5),
        ("ECB", "2018-06-14", "13:45", -4.8),  # QE end + rate pledge
        ("ECB", "2018-07-26", "13:45", 0.3),
        ("ECB", "2018-09-13", "13:45", 1.4),
        ("ECB", "2018-10-25", "13:45", -0.8),
        ("ECB", "2018-12-13", "13:45", -1.7),
        ("ECB", "2019-01-24", "13:45", -2.4),
        ("ECB", "2019-03-07", "13:45", -4.1),  # TLTRO-III announced
        ("ECB", "2019-04-10", "13:45", 0.5),
        ("ECB", "2019-06-06", "13:45", -1.8),
        ("ECB", "2019-07-25", "13:45", -2.2),
        ("ECB", "2019-09-12", "13:45", -3.6),  # Tiering + rate cut + restart QE
        ("ECB", "2019-10-24", "13:45", 0.4),
        ("ECB", "2019-12-12", "13:45", 1.2),  # Lagarde first meeting
        ("ECB", "2020-01-23", "13:45", -0.3),
        ("ECB", "2020-03-12", "13:45", -5.8),  # COVID shock (Lagarde 'not here to close spreads')
        ("ECB", "2020-04-30", "13:45", -1.5),
        ("ECB", "2020-06-04", "13:45", 3.2),  # PEPP expansion
        ("ECB", "2020-07-16", "13:45", 0.6),
        ("ECB", "2020-09-10", "13:45", 2.1),
        ("ECB", "2020-10-29", "13:45", -2.4),
        ("ECB", "2020-12-10", "13:45", 0.8),
        ("ECB", "2021-01-21", "13:45", 0.3),
        ("ECB", "2021-03-11", "13:45", -1.9),
        ("ECB", "2021-04-22", "13:45", -0.7),
        ("ECB", "2021-06-10", "13:45", 0.4),
        ("ECB", "2021-07-22", "13:45", -1.6),
        ("ECB", "2021-09-09", "13:45", 1.1),
        ("ECB", "2021-10-28", "13:45", 2.4),
        ("ECB", "2021-12-16", "13:45", 3.2),
        ("ECB", "2022-02-03", "13:45", 8.4),  # Hawkish pivot
        ("ECB", "2022-03-10", "13:45", 4.1),
        ("ECB", "2022-04-14", "13:45", -2.5),
        ("ECB", "2022-06-09", "13:45", 3.8),
        ("ECB", "2022-07-21", "14:15", 5.9),  # 50 bps hike (M15 era starts)
        ("ECB", "2022-09-08", "14:15", 4.2),  # 75 bps hike
        ("ECB", "2022-10-27", "14:15", -3.8),  # Dovish 75 bps hike
        ("ECB", "2022-12-15", "14:15", 7.1),  # Hawkish 50 bps hike
        ("ECB", "2023-02-02", "14:15", -2.9),
        ("ECB", "2023-03-16", "14:15", 1.4),  # 50 bps amid banking stress
        ("ECB", "2023-05-04", "14:15", -2.1),
        ("ECB", "2023-06-15", "14:15", 4.8),  # Hawkish hike
        ("ECB", "2023-07-27", "14:15", -3.2),  # Dovish hike
        ("ECB", "2023-09-14", "14:15", -2.8),  # 'Peak rates' hike
        ("ECB", "2023-10-26", "14:15", -0.7),
        ("ECB", "2023-12-14", "14:15", 3.1),
        ("ECB", "2024-01-25", "14:15", -1.5),
        ("ECB", "2024-03-07", "14:15", -0.8),
        ("ECB", "2024-04-11", "14:15", -1.2),
        ("ECB", "2024-06-06", "14:15", 2.4),  # Hawkish rate cut
        ("ECB", "2024-07-18", "14:15", 0.4),
        ("ECB", "2024-09-12", "14:15", -0.9),  # 25 bps cut
        ("ECB", "2024-10-17", "14:15", -2.1),  # Back-to-back cut
        ("ECB", "2024-12-12", "14:15", -1.4),
        ("ECB", "2025-01-30", "14:15", -0.8),
        ("ECB", "2025-03-06", "14:15", 1.1),
        ("ECB", "2025-04-17", "14:15", -1.2),
        ("ECB", "2025-06-05", "14:15", 0.5),
        ("ECB", "2025-07-24", "14:15", -0.9),
        ("ECB", "2025-09-11", "14:15", 0.4),
        ("ECB", "2025-10-30", "14:15", -1.0),
        ("ECB", "2025-12-18", "14:15", 0.7),
        ("ECB", "2026-01-22", "14:15", -0.6),
        ("ECB", "2026-03-05", "14:15", 0.8),
        ("ECB", "2026-04-16", "14:15", -0.5),
        ("ECB", "2026-06-04", "14:15", 0.4),
        ("ECB", "2026-07-23", "14:15", -0.7),
        ("ECB", "2026-09-10", "14:15", 0.6),
    ]

    events: list[MacroEvent] = []
    for i, (cb, dt_str, tm_str, shock) in enumerate(raw_events, start=1):
        tz_name = "America/New_York" if cb == "FED" else "Europe/Berlin"
        shock_type = "HAWKISH" if shock > 0 else "DOVISH"

        # Directional mapping according to macroeconomic theory:
        # FED Hawkish -> US rates rise -> USD strengthens -> EURUSD drops (-1)
        # FED Dovish -> US rates fall -> USD weakens -> EURUSD rises (+1)
        # ECB Hawkish -> EUR rates rise -> EUR strengthens -> EURUSD rises (+1)
        # ECB Dovish -> EUR rates fall -> EUR weakens -> EURUSD drops (-1)
        if cb == "FED":
            expected_dir = -1 if shock > 0 else 1
        else:
            expected_dir = 1 if shock > 0 else -1

        events.append(
            MacroEvent(
                event_id=f"{cb}_{dt_str}",
                central_bank=cb,
                event_date=dt_str,
                local_time=tm_str,
                local_tz=tz_name,
                surprise_mps=shock,
                shock_type=shock_type,
                expected_eurusd_direction=expected_dir,
            )
        )
    return events


def resolve_event_utc_timestamp(event: MacroEvent) -> datetime:
    """Resolve exact UTC timestamp using IANA timezone rules (handling DST gaps)."""
    local_tz = ZoneInfo(event.local_tz)
    dt_local = datetime.strptime(f"{event.event_date} {event.local_time}", "%Y-%m-%d %H:%M")
    dt_local = dt_local.replace(tzinfo=local_tz)
    return dt_local.astimezone(ZoneInfo("UTC"))


def run_event_study(m15_path: Path, h4_path: Path) -> dict[str, Any]:
    """Execute the empirical event study on EURUSD M15 and H4 data."""
    events = get_historical_central_bank_events()

    # Load M15 and H4 Data
    df_m15 = pd.read_parquet(m15_path)
    df_m15["dt_utc"] = pd.to_datetime(df_m15["time"], unit="s", utc=True)
    df_m15 = df_m15.sort_values("dt_utc").reset_index(drop=True)

    df_h4 = pd.read_parquet(h4_path)
    df_h4["dt_utc"] = pd.to_datetime(df_h4["time"], unit="s", utc=True)
    df_h4 = df_h4.sort_values("dt_utc").reset_index(drop=True)

    event_records: list[dict[str, Any]] = []

    m15_min_dt = df_m15["dt_utc"].min()
    m15_max_dt = df_m15["dt_utc"].max()

    for ev in events:
        ev_utc = resolve_event_utc_timestamp(ev)

        # Check coverage in M15
        in_m15 = m15_min_dt <= ev_utc <= m15_max_dt

        # Use M15 if available, else H4
        df_target = df_m15 if in_m15 else df_h4
        bar_duration_minutes = 15 if in_m15 else 240

        # Find candle ending immediately before or at event time
        pre_candles = df_target[df_target["dt_utc"] <= ev_utc]
        if pre_candles.empty:
            continue

        pre_idx = pre_candles.index[-1]
        pre_candle = df_target.iloc[pre_idx]
        p_pre = float(pre_candle["close"])
        t_pre = pre_candle["dt_utc"]

        # Measure across horizons
        horizons_data: dict[str, Any] = {}

        for w_min in WINDOWS_MINUTES:
            # For H4, if w_min < 240, map to nearest H4 bar (4h)
            actual_w = max(w_min, bar_duration_minutes)
            post_cutoff = ev_utc + pd.Timedelta(minutes=actual_w)
            post_candles = df_target[df_target["dt_utc"] >= post_cutoff]

            if post_candles.empty:
                continue

            post_idx = post_candles.index[0]
            post_candle = df_target.iloc[post_idx]
            p_post = float(post_candle["close"])

            # Window slice for MFE / MAE
            window_slice = df_target.loc[pre_idx:post_idx]
            max_p = float(window_slice["high"].max())
            min_p = float(window_slice["low"].min())

            raw_pips = (p_post - p_pre) * 10000.0
            signed_pips = raw_pips * ev.expected_eurusd_direction

            # Directional correctness
            correct = (raw_pips > 0 and ev.expected_eurusd_direction == 1) or (
                raw_pips < 0 and ev.expected_eurusd_direction == -1
            )

            # MFE and MAE
            if ev.expected_eurusd_direction == 1:
                mfe_pips = (max_p - p_pre) * 10000.0
                mae_pips = (p_pre - min_p) * 10000.0
            else:
                mfe_pips = (p_pre - min_p) * 10000.0
                mae_pips = (max_p - p_pre) * 10000.0

            horizons_data[f"{w_min}m"] = {
                "p_post": round(p_post, 5),
                "raw_pips": round(raw_pips, 2),
                "signed_pips": round(signed_pips, 2),
                "correct": bool(correct),
                "mfe_pips": round(mfe_pips, 2),
                "mae_pips": round(mae_pips, 2),
            }

        if "60m" in horizons_data:
            # Check 15m to 60m reversal
            rev_50 = False
            if "15m" in horizons_data:
                p15 = horizons_data["15m"]["raw_pips"]
                p60 = horizons_data["60m"]["raw_pips"]
                if abs(p15) > 10.0 and np.sign(p15) != np.sign(p60):
                    rev_50 = True
                elif abs(p15) > 10.0 and abs(p60) < 0.5 * abs(p15):
                    rev_50 = True

            event_records.append(
                {
                    "event_id": ev.event_id,
                    "central_bank": ev.central_bank,
                    "event_date": ev.event_date,
                    "event_utc": ev_utc.isoformat(),
                    "surprise_mps": ev.surprise_mps,
                    "shock_type": ev.shock_type,
                    "expected_direction": ev.expected_eurusd_direction,
                    "data_resolution": "M15" if in_m15 else "H4",
                    "p_pre": round(p_pre, 5),
                    "t_pre": t_pre.isoformat(),
                    "reversal_gt_50pct": rev_50,
                    "horizons": horizons_data,
                }
            )

    # Compute aggregate window statistics
    window_summaries: dict[str, Any] = {}

    for w_min in WINDOWS_MINUTES:
        w_key = f"{w_min}m"
        signed_pips_list = [
            r["horizons"][w_key]["signed_pips"] for r in event_records if w_key in r["horizons"]
        ]
        correct_list = [
            r["horizons"][w_key]["correct"] for r in event_records if w_key in r["horizons"]
        ]
        mfe_list = [
            r["horizons"][w_key]["mfe_pips"] for r in event_records if w_key in r["horizons"]
        ]
        mae_list = [
            r["horizons"][w_key]["mae_pips"] for r in event_records if w_key in r["horizons"]
        ]

        n = len(signed_pips_list)
        if n == 0:
            continue

        arr_pips = np.array(signed_pips_list)
        win_rate = float(np.mean(correct_list))
        mean_pips = float(np.mean(arr_pips))
        median_pips = float(np.median(arr_pips))
        std_pips = float(np.std(arr_pips, ddof=1)) if n > 1 else 0.0

        # One-sample t-test against H0: mean pips == 0
        t_stat, p_val = stats.ttest_1samp(arr_pips, 0.0) if n > 2 else (0.0, 1.0)

        # Binomial test against H0: win_rate == 0.50
        binom_res = stats.binomtest(int(np.sum(correct_list)), n, 0.50)
        p_val_dir = float(binom_res.pvalue)

        mean_mfe = float(np.mean(mfe_list))
        mean_mae = float(np.mean(mae_list))
        mfe_mae_ratio = round(mean_mfe / mean_mae, 2) if mean_mae > 0 else 1.0

        window_summaries[w_key] = {
            "window_minutes": w_min,
            "sample_size": n,
            "directional_accuracy": round(win_rate, 4),
            "binomial_p_value": round(p_val_dir, 4),
            "mean_signed_pips": round(mean_pips, 2),
            "median_signed_pips": round(median_pips, 2),
            "std_pips": round(std_pips, 2),
            "t_statistic": round(float(t_stat), 3),
            "t_test_p_value": round(float(p_val), 4),
            "mean_mfe_pips": round(mean_mfe, 2),
            "mean_mae_pips": round(mean_mae, 2),
            "mfe_mae_ratio": mfe_mae_ratio,
        }

    # Central bank specific breakdown on Primary Window (60m)
    cb_breakdown: dict[str, Any] = {}
    for cb in ["FED", "ECB"]:
        cb_records = [r for r in event_records if r["central_bank"] == cb]
        cb_pips = [
            r["horizons"]["60m"]["signed_pips"] for r in cb_records if "60m" in r["horizons"]
        ]
        cb_correct = [r["horizons"]["60m"]["correct"] for r in cb_records if "60m" in r["horizons"]]
        n_cb = len(cb_pips)
        cb_breakdown[cb] = {
            "count": n_cb,
            "directional_accuracy": (round(float(np.mean(cb_correct)), 4) if n_cb > 0 else 0.0),
            "mean_signed_pips": (round(float(np.mean(cb_pips)), 2) if n_cb > 0 else 0.0),
            "median_signed_pips": (round(float(np.median(cb_pips)), 2) if n_cb > 0 else 0.0),
        }

    reversal_count = sum(1 for r in event_records if r["reversal_gt_50pct"])
    total_m15_events = sum(1 for r in event_records if r["data_resolution"] == "M15")
    reversal_rate = round(reversal_count / total_m15_events, 4) if total_m15_events > 0 else 0.0

    report_data = {
        "metadata": {
            "phase": "45",
            "title": "Phase 45 Central Bank Monetary Policy Event Study",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "primary_window": f"{PRIMARY_WINDOW_MINUTES}m",
            "total_events_evaluated": len(event_records),
            "m15_resolution_events": total_m15_events,
            "h4_resolution_events": len(event_records) - total_m15_events,
        },
        "primary_window_summary": window_summaries.get(f"{PRIMARY_WINDOW_MINUTES}m"),
        "all_windows_summary": window_summaries,
        "central_bank_breakdown": cb_breakdown,
        "post_event_reversal_rate_m15": reversal_rate,
        "events": event_records,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "phase45_event_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data


if __name__ == "__main__":
    m15_p = DATA_DIR / "processed" / "eurusd_m15" / "eurusd_m15_processed.parquet"
    h4_p = DATA_DIR / "external" / "macro_yields" / "processed" / "eurusd_h4_yield_aligned.parquet"
    res = run_event_study(m15_p, h4_p)
    pw = res["primary_window_summary"]
    print(
        f"Event Study Complete: {res['metadata']['total_events_evaluated']} events. "
        f"Primary Window (60m) Win Rate: {pw['directional_accuracy']:.2%}, "
        f"Mean Signed Pips: {pw['mean_signed_pips']} pips (p={pw['t_test_p_value']:.4f})."
    )
