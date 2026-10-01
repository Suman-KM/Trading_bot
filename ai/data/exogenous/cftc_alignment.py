"""Causal point-in-time alignment module for CFTC Commitments of Traders (COT) data.

Enforces strict causal alignment between weekly CFTC COT reports and EURUSD price bars:
1. Tuesday position observations are NEVER available to EURUSD bars on Tuesday-Thursday.
2. Official Friday publication (15:30 US Eastern) timestamp is converted via America/New_York.
3. Information becomes actionable ONLY at effective timestamp (Monday 00:00:00 UTC post-release).
4. Effective timestamp strictly satisfies:
   effective_time_utc >= publication_time_utc + safety_buffer_hours.
5. All forward filling is strictly causal with zero future lookahead.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

NY_TZ = ZoneInfo("America/New_York")
MIN_SAFETY_BUFFER_HOURS = 4

# Known historical CFTC delayed release schedules (e.g., US Federal Government Shutdowns)
# Maps observation Tuesday date -> actual historical release date YYYY-MM-DD
HISTORICAL_DELAYED_RELEASES: dict[str, str] = {
    # 2013 US Federal Government Shutdown (Oct 1 - Oct 16, 2013)
    "2013-10-01": "2013-10-18",
    "2013-10-08": "2013-10-22",
    "2013-10-15": "2013-10-25",
    # 2018-2019 US Federal Government Shutdown (Dec 22, 2018 - Jan 25, 2019)
    "2018-12-24": "2019-02-01",
    "2018-12-31": "2019-02-05",
    "2019-01-08": "2019-02-08",
    "2019-01-15": "2019-02-12",
    "2019-01-22": "2019-02-15",
    "2019-01-29": "2019-02-20",
    "2019-02-05": "2019-02-22",
    "2019-02-12": "2019-02-26",
    "2019-02-19": "2019-03-01",
}


def is_us_holiday_friday(date_val: datetime.date) -> bool:
    """Check if a given Friday date is a standard US Federal holiday or observed closure."""
    year = date_val.year
    month = date_val.month
    day = date_val.day

    # New Year's Day (Jan 1) or Dec 31 observed when Jan 1 is Saturday
    if month == 1 and day == 1:
        return True
    if month == 12 and day == 31:
        return True

    # Juneteenth (June 19) or June 18 observed when June 19 is Saturday
    if month == 6 and day in (18, 19):
        return True

    # Independence Day (July 4) or July 3 observed when July 4 is Saturday
    if month == 7 and day in (3, 4):
        return True

    # Veterans Day (Nov 11) or Nov 10 observed when Nov 11 is Saturday
    if month == 11 and day in (10, 11):
        return True

    # Christmas Day (Dec 25) or Dec 24 observed when Dec 25 is Saturday
    if month == 12 and day in (24, 25):
        return True

    # Good Friday (financial markets and federal agencies frequently closed)
    # Approximate or known Good Friday dates for 2010-2026
    good_fridays = {
        2010: (4, 2),
        2011: (4, 22),
        2012: (4, 6),
        2013: (3, 29),
        2014: (4, 18),
        2015: (4, 3),
        2016: (3, 25),
        2017: (4, 14),
        2018: (3, 30),
        2019: (4, 19),
        2020: (4, 10),
        2021: (4, 2),
        2022: (4, 15),
        2023: (4, 7),
        2024: (3, 29),
        2025: (4, 18),
        2026: (4, 3),
    }
    if year in good_fridays and (month, day) == good_fridays[year]:
        return True

    return False


def compute_cftc_timestamps(
    observation_date_str: str,
    safety_buffer_hours: int = MIN_SAFETY_BUFFER_HOURS,
) -> dict[str, Any]:
    """Compute deterministic point-in-time publication and effective timestamps for a COT report.

    Parameters
    ----------
    observation_date_str : str
        Tuesday observation date formatted as 'YYYY-MM-DD'.
    safety_buffer_hours : int
        Minimum hours after official release before information is actionable. Default 4.

    Returns
    -------
    dict[str, Any]
        Dictionary containing observation, publication, and effective timestamps in UTC.
    """
    obs_date = datetime.strptime(observation_date_str, "%Y-%m-%d").date()
    obs_dt_utc = datetime(obs_date.year, obs_date.month, obs_date.day, 21, 0, tzinfo=UTC)

    # 1. Determine publication date
    if observation_date_str in HISTORICAL_DELAYED_RELEASES:
        pub_date_str = HISTORICAL_DELAYED_RELEASES[observation_date_str]
        pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d").date()
    else:
        # Standard release is Friday (+3 days from Tuesday)
        # In case observation date was shifted (e.g. Wednesday due to Monday holiday),
        # target the corresponding Friday of that report week.
        days_to_friday = (4 - obs_date.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7  # If observed on Friday, release next Friday
        pub_date = obs_date + timedelta(days=days_to_friday)

        # If Friday is a federal holiday, release is delayed to Monday (+3 days)
        if is_us_holiday_friday(pub_date):
            pub_date = pub_date + timedelta(days=3)

    # 2. Publication is officially at 15:30 US Eastern Time
    pub_dt_local = datetime(pub_date.year, pub_date.month, pub_date.day, 15, 30, tzinfo=NY_TZ)
    pub_dt_utc = pub_dt_local.astimezone(UTC)

    # 3. Actionable effective timestamp:
    # Forex markets trade Sunday ~21:00 UTC through Friday ~21:00 UTC.
    # Reports published Friday 15:30 ET (19:30 UTC during EDT, 20:30 UTC during EST)
    # are first actionable at Monday 00:00:00 UTC open.
    # If publication occurs on Monday or Tuesday (delayed schedule),
    # effective time is midnight UTC following publication.
    if pub_date.weekday() == 4:  # Friday release
        effective_date = pub_date + timedelta(days=3)  # Following Monday
        effective_dt_utc = datetime(
            effective_date.year, effective_date.month, effective_date.day, 0, 0, tzinfo=UTC
        )
    elif pub_date.weekday() == 0:  # Monday release
        effective_date = pub_date + timedelta(days=1)  # Tuesday
        effective_dt_utc = datetime(
            effective_date.year, effective_date.month, effective_date.day, 0, 0, tzinfo=UTC
        )
    elif pub_date.weekday() == 1:  # Tuesday release
        effective_date = pub_date + timedelta(days=1)  # Wednesday
        effective_dt_utc = datetime(
            effective_date.year, effective_date.month, effective_date.day, 0, 0, tzinfo=UTC
        )
    else:
        # Fallback: next calendar day 00:00 UTC
        effective_date = pub_date + timedelta(days=1)
        effective_dt_utc = datetime(
            effective_date.year, effective_date.month, effective_date.day, 0, 0, tzinfo=UTC
        )

    # Enforce safety buffer
    min_effective = pub_dt_utc + timedelta(hours=safety_buffer_hours)
    if effective_dt_utc < min_effective:
        effective_dt_utc = min_effective

    return {
        "observation_date": observation_date_str,
        "observation_timestamp_utc": obs_dt_utc.isoformat(),
        "publication_date": pub_date.isoformat(),
        "publication_time_local": pub_dt_local.isoformat(),
        "publication_time_utc": pub_dt_utc.isoformat(),
        "effective_time_utc": effective_dt_utc.isoformat(),
        "safety_buffer_hours": safety_buffer_hours,
    }


def align_cftc_to_eurusd(
    df_cftc: pd.DataFrame,
    df_eurusd: pd.DataFrame,
    eurusd_timestamp_col: str = "timestamp",
    cftc_effective_col: str = "effective_time_utc",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align weekly CFTC COT positioning data causally to EURUSD bars with zero lookahead.

    Uses an as-of backward join on 'effective_time_utc <= bar_timestamp'.
    Any EURUSD bar occurring before the first effective COT release date receives NaN.
    Position values are forward-filled until the next effective release becomes active.

    Parameters
    ----------
    df_cftc : pd.DataFrame
        Normalized CFTC dataset containing 'effective_time_utc' and positioning metrics.
    df_eurusd : pd.DataFrame
        EURUSD OHLCV price DataFrame containing a datetime column.
    eurusd_timestamp_col : str
        Column name of the EURUSD bar timestamp.
    cftc_effective_col : str
        Column name of the CFTC effective timestamp.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Aligned EURUSD DataFrame with CFTC columns and alignment audit metrics.
    """
    df_cftc_sorted = df_cftc.copy()
    df_eurusd_sorted = df_eurusd.copy()

    # Ensure datetimes are timezone-aware UTC with matching dtype resolution
    target_dtype = "datetime64[us, UTC]"
    df_cftc_sorted[cftc_effective_col] = pd.to_datetime(
        df_cftc_sorted[cftc_effective_col], utc=True
    ).astype(target_dtype)
    df_eurusd_sorted[eurusd_timestamp_col] = pd.to_datetime(
        df_eurusd_sorted[eurusd_timestamp_col], utc=True
    ).astype(target_dtype)

    df_cftc_sorted = df_cftc_sorted.sort_values(cftc_effective_col).reset_index(drop=True)
    df_eurusd_sorted = df_eurusd_sorted.sort_values(eurusd_timestamp_col).reset_index(drop=True)

    # Perform backward merge_asof (each EURUSD bar gets the latest CFTC where effective <= bar_ts)
    aligned_df = pd.merge_asof(
        df_eurusd_sorted,
        df_cftc_sorted,
        left_on=eurusd_timestamp_col,
        right_on=cftc_effective_col,
        direction="backward",
    )

    # Audit & Verification
    total_bars = len(aligned_df)
    aligned_bars = int(aligned_df[cftc_effective_col].notna().sum())
    unaligned_bars = total_bars - aligned_bars

    # Strict point-in-time leak check
    leakage_violations = 0
    bar_ts = pd.to_datetime(aligned_df[eurusd_timestamp_col], utc=True).astype(target_dtype)
    if "publication_time_utc" in aligned_df.columns:
        pub_ts = pd.to_datetime(aligned_df["publication_time_utc"], utc=True).astype(target_dtype)
        # Bar using COT before publication timestamp is a strict leakage violation
        violations = (aligned_df["publication_time_utc"].notna()) & (bar_ts < pub_ts)
        leakage_violations = int(violations.sum())

    # Effective time leak check
    effective_violations = 0
    eff_ts = pd.to_datetime(aligned_df[cftc_effective_col], utc=True).astype(target_dtype)
    eff_viols = (aligned_df[cftc_effective_col].notna()) & (bar_ts < eff_ts)
    effective_violations = int(eff_viols.sum())

    unique_reports_used = (
        int(aligned_df["observation_date"].nunique())
        if "observation_date" in aligned_df.columns
        else 0
    )

    audit_summary = {
        "total_eurusd_bars": total_bars,
        "aligned_eurusd_bars": aligned_bars,
        "unaligned_bars_prior_to_first_cot": unaligned_bars,
        "unique_cot_reports_utilized": unique_reports_used,
        "leakage_violations_publication": leakage_violations,
        "leakage_violations_effective": effective_violations,
        "causally_valid": (leakage_violations == 0) and (effective_violations == 0),
    }

    return aligned_df, audit_summary
