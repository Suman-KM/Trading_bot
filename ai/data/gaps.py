"""Market session gap detection and classification.

Analyzes time intervals between consecutive M15 candles, distinguishing legitimate
weekend closures and recognized financial holidays from unexpected intraday data dropouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

EXPECTED_M15_INTERVAL_SECONDS: int = 900  # 15 minutes


@dataclass(frozen=True)
class GapItem:
    """Individual market gap record."""

    gap_start_utc: str
    gap_end_utc: str
    previous_timestamp: int
    next_timestamp: int
    gap_duration_seconds: int
    gap_duration_hours: float
    missing_bars: int
    classification: str

    def to_dict(self) -> dict[str, Any]:
        """Convert gap item to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class GapAnalysisResult:
    """Aggregated gap analysis summary."""

    total_gaps: int
    expected_weekend_count: int
    expected_holiday_count: int
    unexpected_gap_count: int
    unclassified_gap_count: int
    gaps: list[GapItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "total_gaps": self.total_gaps,
            "expected_weekend_count": self.expected_weekend_count,
            "expected_holiday_count": self.expected_holiday_count,
            "unexpected_gap_count": self.unexpected_gap_count,
            "unclassified_gap_count": self.unclassified_gap_count,
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


def _is_known_holiday(start_dt: datetime, end_dt: datetime) -> bool:
    """Check if the gap coincides with recognized global FX holiday closures."""
    # Christmas period: Dec 24, 25, 26
    # New Year period: Dec 31, Jan 1, Jan 2
    for dt in (start_dt, end_dt):
        month, day = dt.month, dt.day
        if month == 12 and day in (24, 25, 26):
            return True
        if month == 1 and day in (1, 2):
            return True
        if month == 12 and day == 31:
            return True

    return False


def classify_gap(
    start_epoch: int,
    end_epoch: int,
    duration_seconds: int,
) -> str:
    """Classify a market data gap by analyzing session boundaries and calendar context.

    Parameters
    ----------
    start_epoch : int
        Epoch seconds of the last bar before the gap.
    end_epoch : int
        Epoch seconds of the first bar following the gap.
    duration_seconds : int
        Total gap duration in seconds.

    Returns
    -------
    str
        Classification: 'EXPECTED_WEEKEND', 'EXPECTED_HOLIDAY',
        'UNEXPECTED_GAP', or 'UNCLASSIFIED_GAP'.
    """
    start_dt = datetime.fromtimestamp(start_epoch, tz=UTC)
    end_dt = datetime.fromtimestamp(end_epoch, tz=UTC)

    start_weekday = start_dt.weekday()  # Monday is 0, Sunday is 6
    end_weekday = end_dt.weekday()

    duration_hours = duration_seconds / 3600.0

    # 1. Weekend detection:
    # Friday market close (typically between 20:00 and 23:59 UTC)
    # Sunday market open (typically between 20:00 and 23:59 UTC) or early Monday
    # Typical duration is ~45 to ~52 hours
    is_friday_start = start_weekday == 4
    is_weekend_end = end_weekday in (6, 0)

    if is_friday_start and is_weekend_end and (36.0 <= duration_hours <= 65.0):
        return "EXPECTED_WEEKEND"

    # Also handle weekend with extended holiday attached
    if start_weekday in (4, 5) and end_weekday in (6, 0, 1) and (36.0 <= duration_hours <= 80.0):
        if _is_known_holiday(start_dt, end_dt):
            return "EXPECTED_HOLIDAY"
        return "EXPECTED_WEEKEND"

    # 2. Known holiday detection
    if _is_known_holiday(start_dt, end_dt) and duration_hours <= 72.0:
        return "EXPECTED_HOLIDAY"

    # 3. If gap occurs within normal weekday trading hours (Mon-Thu)
    if start_weekday in (0, 1, 2, 3) and end_weekday in (0, 1, 2, 3):
        return "UNEXPECTED_GAP"

    # 4. Otherwise unclassified
    return "UNCLASSIFIED_GAP"


def analyze_gaps(
    df: pd.DataFrame,
    time_col: str = "time",
    expected_interval_seconds: int = EXPECTED_M15_INTERVAL_SECONDS,
) -> GapAnalysisResult:
    """Identify and classify all interval gaps exceeding the expected candle spacing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with strictly increasing epoch timestamps.
    time_col : str, default "time"
        Epoch timestamp column name.
    expected_interval_seconds : int, default 900
        Nominal interval spacing in seconds.

    Returns
    -------
    GapAnalysisResult
        Summary statistics and granular details for all observed gaps.
    """
    if df.empty or len(df) < 2 or time_col not in df.columns:
        return GapAnalysisResult(
            total_gaps=0,
            expected_weekend_count=0,
            expected_holiday_count=0,
            unexpected_gap_count=0,
            unclassified_gap_count=0,
            gaps=[],
        )

    times = df[time_col].to_numpy()
    diffs = times[1:] - times[:-1]

    gap_indices = (diffs > expected_interval_seconds).nonzero()[0]

    gaps: list[GapItem] = []
    weekend_count = 0
    holiday_count = 0
    unexpected_count = 0
    unclassified_count = 0

    for idx in gap_indices:
        prev_t = int(times[idx])
        next_t = int(times[idx + 1])
        dur_sec = int(next_t - prev_t)
        dur_hrs = round(dur_sec / 3600.0, 2)
        missing_bars = int((dur_sec // expected_interval_seconds) - 1)

        classification = classify_gap(prev_t, next_t, dur_sec)

        if classification == "EXPECTED_WEEKEND":
            weekend_count += 1
        elif classification == "EXPECTED_HOLIDAY":
            holiday_count += 1
        elif classification == "UNEXPECTED_GAP":
            unexpected_count += 1
        else:
            unclassified_count += 1

        gap_start_utc = datetime.fromtimestamp(prev_t, tz=UTC).isoformat()
        gap_end_utc = datetime.fromtimestamp(next_t, tz=UTC).isoformat()

        gaps.append(
            GapItem(
                gap_start_utc=gap_start_utc,
                gap_end_utc=gap_end_utc,
                previous_timestamp=prev_t,
                next_timestamp=next_t,
                gap_duration_seconds=dur_sec,
                gap_duration_hours=dur_hrs,
                missing_bars=missing_bars,
                classification=classification,
            )
        )

    return GapAnalysisResult(
        total_gaps=len(gaps),
        expected_weekend_count=weekend_count,
        expected_holiday_count=holiday_count,
        unexpected_gap_count=unexpected_count,
        unclassified_gap_count=unclassified_count,
        gaps=gaps,
    )
