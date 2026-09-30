"""Validation and integrity audit routines for exogenous data pipelines.

Verifies point-in-time causality, revision vintage integrity, DST transition
consistency, and prevents silent zero-imputation or lookahead leakage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ai.data.exogenous.schema import ExogenousDataPoint


def validate_point_in_time_safety(points: list[ExogenousDataPoint]) -> list[str]:
    """Validate that exogenous observations respect strict chronological causality.

    Parameters
    ----------
    points : list[ExogenousDataPoint]
        List of data observations to inspect.

    Returns
    -------
    list[str]
        List of identified causality or integrity violation messages.
    """
    errors: list[str] = []
    seen_periods: dict[str, list[ExogenousDataPoint]] = {}

    for idx, pt in enumerate(points):
        # 1. Timezone verification
        if pt.information_timestamp.tzinfo is None:
            errors.append(f"Point {idx} ({pt.series_id}): information_timestamp lacks tzinfo.")
        if pt.first_usable_bar_timestamp.tzinfo is None:
            errors.append(f"Point {idx} ({pt.series_id}): first_usable_bar_timestamp lacks tzinfo.")

        # 2. Chronological causality check
        if pt.first_usable_bar_timestamp < pt.information_timestamp:
            errors.append(
                f"Point {idx} ({pt.series_id}): first_usable_bar ({pt.first_usable_bar_timestamp}) "
                f"< information_timestamp ({pt.information_timestamp})."
            )

        # 3. Track observation periods for revision audit
        if pt.observation_period not in seen_periods:
            seen_periods[pt.observation_period] = []
        seen_periods[pt.observation_period].append(pt)

    return errors


def detect_untracked_revisions(points: list[ExogenousDataPoint]) -> list[str]:
    """Detect if historical values were revised without vintage increments.

    Parameters
    ----------
    points : list[ExogenousDataPoint]
        List of data points.

    Returns
    -------
    list[str]
        List of detected silent revision anomalies.
    """
    errors: list[str] = []
    periods: dict[str, list[ExogenousDataPoint]] = {}

    for pt in points:
        periods.setdefault(pt.observation_period, []).append(pt)

    for period, pts in periods.items():
        if len(pts) > 1:
            # Sort by publication timestamp
            sorted_pts = sorted(pts, key=lambda p: p.information_timestamp)
            for j in range(1, len(sorted_pts)):
                prev = sorted_pts[j - 1]
                curr = sorted_pts[j]
                if (
                    prev.actual_value != curr.actual_value
                    and curr.revision_vintage <= prev.revision_vintage
                ):
                    errors.append(
                        f"Silent revision detected for period '{period}': actual changed from "
                        f"{prev.actual_value} to {curr.actual_value} without incrementing "
                        f"vintage ({prev.revision_vintage} -> {curr.revision_vintage})."
                    )

    return errors


def check_dst_offsets(
    dt_utc: datetime,
    us_tz: str = "America/New_York",
    eu_tz: str = "Europe/Berlin",
) -> dict[str, int]:
    """Calculate exact UTC offsets in seconds for US and EU financial centers.

    Parameters
    ----------
    dt_utc : datetime
        Reference datetime in UTC.
    us_tz : str, default "America/New_York"
        US financial center timezone.
    eu_tz : str, default "Europe/Berlin"
        European financial center timezone.

    Returns
    -------
    dict[str, int]
        Dictionary of UTC offset seconds and time gap between the two centers.
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)

    us_zone = ZoneInfo(us_tz)
    eu_zone = ZoneInfo(eu_tz)

    us_dt = dt_utc.astimezone(us_zone)
    eu_dt = dt_utc.astimezone(eu_zone)

    us_offset_sec = int(us_dt.utcoffset().total_seconds()) if us_dt.utcoffset() else 0
    eu_offset_sec = int(eu_dt.utcoffset().total_seconds()) if eu_dt.utcoffset() else 0

    return {
        "us_offset_seconds": us_offset_sec,
        "eu_offset_seconds": eu_offset_sec,
        "us_offset_hours": us_offset_sec / 3600.0,
        "eu_offset_hours": eu_offset_sec / 3600.0,
        "us_eu_difference_hours": (eu_offset_sec - us_offset_sec) / 3600.0,
    }


def verify_no_silent_zero_imputation(
    series: pd.Series,
    known_missing_indices: list[int] | None = None,
) -> bool:
    """Verify that unpopulated or missing intervals remain NaN and are not coerced to 0.0.

    Parameters
    ----------
    series : pd.Series
        Aligned feature series.
    known_missing_indices : list[int], optional
        Indices where data is known to be missing.

    Returns
    -------
    bool
        True if all verified missing values are NaN (and not 0.0).
    """
    if known_missing_indices is None:
        # If no specific indices specified, check if series has NaN when not completely populated
        return bool(series.isna().any()) if len(series) > 0 else True

    for idx in known_missing_indices:
        val = series.iloc[idx]
        if not np.isnan(val) and val == 0.0:
            return False  # Coerced to zero!
        if not np.isnan(val):
            return False

    return True
