"""Historical tick data repository and validation structures for backtesting.

Provides high-performance, leakage-safe access to chronological tick feeds with
microsecond/millisecond precision, validating timestamp monotonicity, price sanity,
and strict holdout isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Strict holdout governance boundary: Phase 11 Test partition starts 2026-02-19 12:00:00 UTC
# Validation partition ends 2026-02-19 10:45:00 UTC
PHASE11_TEST_LOCK_TIMESTAMP = datetime(2026, 2, 19, 10, 45, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class TickCoverageInterval:
    """Continuous temporal interval of verified tick coverage."""

    start_time_msc: int
    end_time_msc: int
    start_dt: datetime
    end_dt: datetime
    tick_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time_msc": self.start_time_msc,
            "end_time_msc": self.end_time_msc,
            "start_dt": self.start_dt.isoformat(),
            "end_dt": self.end_dt.isoformat(),
            "tick_count": self.tick_count,
        }


@dataclass(frozen=True, slots=True)
class TickDataGap:
    """Detected unpopulated gap between tick coverage intervals."""

    gap_start_msc: int
    gap_end_msc: int
    gap_start_dt: datetime
    gap_end_dt: datetime
    duration_seconds: float
    duration_hours: float
    ticks_before: int
    ticks_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_start_msc": self.gap_start_msc,
            "gap_end_msc": self.gap_end_msc,
            "gap_start_dt": self.gap_start_dt.isoformat(),
            "gap_end_dt": self.gap_end_dt.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "duration_hours": round(self.duration_hours, 2),
            "ticks_before": self.ticks_before,
            "ticks_after": self.ticks_after,
        }


@dataclass(frozen=True, slots=True)
class TickQuote:
    """Immutable representation of a single top-of-book market quote tick."""

    time_msc: int
    timestamp: datetime
    bid: float
    ask: float
    spread_points: float
    flags: int

    @property
    def mid(self) -> float:
        """Midpoint quote price."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread_price(self) -> float:
        """Spread in price units."""
        return self.ask - self.bid


class TickDataRepository:
    """In-memory binary repository providing sub-millisecond tick slices via searchsorted.

    Enforces strict gap detection, interval coverage verification, and holdout protection.
    """

    def __init__(
        self,
        time_msc: np.ndarray,
        bids: np.ndarray,
        asks: np.ndarray,
        flags: np.ndarray | None = None,
        point_value: float = 1e-05,
        max_allowed_gap_ms: int = 300_000,
        max_entry_delay_ms: int = 60_000,
    ) -> None:
        if len(time_msc) != len(bids) or len(bids) != len(asks):
            raise ValueError("time_msc, bids, and asks arrays must have identical length.")

        self.point_value = point_value
        self.max_allowed_gap_ms = max_allowed_gap_ms
        self.max_entry_delay_ms = max_entry_delay_ms
        self.time_msc = np.ascontiguousarray(time_msc, dtype=np.int64)
        self.bids = np.ascontiguousarray(bids, dtype=np.float64)
        self.asks = np.ascontiguousarray(asks, dtype=np.float64)
        self.flags = (
            np.ascontiguousarray(flags, dtype=np.uint32)
            if flags is not None
            else np.zeros(len(time_msc), dtype=np.uint32)
        )

        self._validate_invariants()
        self._detect_coverage_and_gaps()

    def _detect_coverage_and_gaps(self) -> None:
        """Partition repository into continuous coverage intervals and detect gaps."""
        if len(self.time_msc) == 0:
            self.coverage_intervals: list[TickCoverageInterval] = []
            self.gaps: list[TickDataGap] = []
            return

        diffs = np.diff(self.time_msc)
        gap_indices = np.where(diffs > self.max_allowed_gap_ms)[0]

        intervals: list[TickCoverageInterval] = []
        gaps: list[TickDataGap] = []
        seg_start = 0

        for g_idx in gap_indices:
            seg_end = g_idx
            s_msc = int(self.time_msc[seg_start])
            e_msc = int(self.time_msc[seg_end])
            cnt = int(seg_end - seg_start + 1)
            intervals.append(
                TickCoverageInterval(
                    start_time_msc=s_msc,
                    end_time_msc=e_msc,
                    start_dt=datetime.fromtimestamp(s_msc / 1000.0, tz=timezone.utc),
                    end_dt=datetime.fromtimestamp(e_msc / 1000.0, tz=timezone.utc),
                    tick_count=cnt,
                )
            )

            gap_start = int(self.time_msc[g_idx])
            gap_end = int(self.time_msc[g_idx + 1])
            dur_sec = (gap_end - gap_start) / 1000.0
            dur_hr = dur_sec / 3600.0
            gaps.append(
                TickDataGap(
                    gap_start_msc=gap_start,
                    gap_end_msc=gap_end,
                    gap_start_dt=datetime.fromtimestamp(gap_start / 1000.0, tz=timezone.utc),
                    gap_end_dt=datetime.fromtimestamp(gap_end / 1000.0, tz=timezone.utc),
                    duration_seconds=dur_sec,
                    duration_hours=dur_hr,
                    ticks_before=int(g_idx + 1),
                    ticks_after=int(len(self.time_msc) - (g_idx + 1)),
                )
            )
            seg_start = g_idx + 1

        # Final interval
        s_msc = int(self.time_msc[seg_start])
        e_msc = int(self.time_msc[-1])
        cnt = int(len(self.time_msc) - seg_start)
        intervals.append(
            TickCoverageInterval(
                start_time_msc=s_msc,
                end_time_msc=e_msc,
                start_dt=datetime.fromtimestamp(s_msc / 1000.0, tz=timezone.utc),
                end_dt=datetime.fromtimestamp(e_msc / 1000.0, tz=timezone.utc),
                tick_count=cnt,
            )
        )

        self.coverage_intervals = intervals
        self.gaps = gaps

    def _validate_invariants(self) -> None:
        """Validate core physical and governance invariants."""
        if len(self.time_msc) == 0:
            return

        # 1. Monotonicity
        diffs = np.diff(self.time_msc)
        if np.any(diffs < 0):
            bad_idx = int(np.where(diffs < 0)[0][0])
            raise ValueError(
                f"Timestamp monotonicity violation at index {bad_idx}: "
                f"{self.time_msc[bad_idx]} -> {self.time_msc[bad_idx + 1]}"
            )

        # 2. Price sanity
        if not np.all(np.isfinite(self.bids)) or not np.all(np.isfinite(self.asks)):
            raise ValueError("Non-finite bid or ask quotes detected in tick feed.")
        if np.any(self.bids <= 0.0) or np.any(self.asks <= 0.0):
            raise ValueError("Non-positive bid or ask quotes detected in tick feed.")

        # 3. Nonnegative spread
        spreads = self.asks - self.bids
        if np.any(spreads < -1e-9):
            bad_idx = int(np.where(spreads < -1e-9)[0][0])
            raise ValueError(
                f"Negative spread (crossed quote) at index {bad_idx}: "
                f"bid={self.bids[bad_idx]}, ask={self.asks[bad_idx]}"
            )

        # 4. Strict Test Lock Governance
        latest_ms = self.time_msc[-1]
        lock_ms = int(PHASE11_TEST_LOCK_TIMESTAMP.timestamp() * 1000)
        if latest_ms > lock_ms:
            raise ValueError(
                f"CRITICAL GOVERNANCE VIOLATION: Tick timestamp {latest_ms} ms crosses "
                f"Phase 11 test partition boundary ({lock_ms} ms / {PHASE11_TEST_LOCK_TIMESTAMP})."
            )

    @classmethod
    def from_parquet(
        cls,
        path: Path | str,
        point_value: float = 1e-05,
        max_allowed_gap_ms: int = 300_000,
        max_entry_delay_ms: int = 60_000,
    ) -> TickDataRepository:
        """Load and validate tick feed from Parquet archive with coverage gap detection."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Tick parquet file not found: {p}")

        df = pd.read_parquet(p)
        required_cols = {"time_msc", "bid", "ask"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns in tick parquet: {missing}")

        time_msc = df["time_msc"].to_numpy(dtype=np.int64)
        bids = df["bid"].to_numpy(dtype=np.float64)
        asks = df["ask"].to_numpy(dtype=np.float64)
        flags = df["flags"].to_numpy(dtype=np.uint32) if "flags" in df.columns else None

        return cls(
            time_msc=time_msc,
            bids=bids,
            asks=asks,
            flags=flags,
            point_value=point_value,
            max_allowed_gap_ms=max_allowed_gap_ms,
            max_entry_delay_ms=max_entry_delay_ms,
        )

    def __len__(self) -> int:
        return len(self.time_msc)

    def get_slice_indices(self, start_time: datetime, end_time: datetime) -> tuple[int, int]:
        """Find integer array slice boundaries [idx_start, idx_end) using binary search."""
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        idx_start = int(np.searchsorted(self.time_msc, start_ms, side="left"))
        idx_end = int(np.searchsorted(self.time_msc, end_ms, side="right"))
        return idx_start, idx_end

    def get_first_tick_at_or_after(self, timestamp: datetime) -> TickQuote | None:
        """Retrieve earliest tick occurring at or after the timestamp within continuous coverage.

        Enforces strict gap and future-data substitution prevention:
        - If requested timestamp is before earliest tick or after latest tick: returns None.
        - If requested timestamp falls inside a known data gap: returns None.
        - If the nearest available tick is separated by > max_entry_delay_ms: returns None.
        - If target timestamp does not fall within a continuous coverage interval: returns None.
        """
        if len(self.time_msc) == 0:
            return None

        target_ms = int(timestamp.timestamp() * 1000)
        earliest_ms = int(self.time_msc[0])
        latest_ms = int(self.time_msc[-1])

        # 1. Reject if strictly before earliest tick or strictly after latest tick
        if target_ms < (earliest_ms - self.max_entry_delay_ms) or target_ms > latest_ms:
            return None

        # 2. Reject if inside any known data gap
        for g in self.gaps:
            if g.gap_start_msc <= target_ms < g.gap_end_msc:
                return None

        # 3. Find nearest tick via searchsorted
        idx = int(np.searchsorted(self.time_msc, target_ms, side="left"))
        if idx >= len(self.time_msc):
            return None

        t_ms = int(self.time_msc[idx])

        # 4. Reject if nearest available tick is separated by a large delay (future substitution)
        if (t_ms - target_ms) > self.max_entry_delay_ms:
            return None

        # 5. Verify target_ms and t_ms belong to the same continuous coverage interval
        matching_interval = None
        for interval in self.coverage_intervals:
            if interval.start_time_msc <= t_ms <= interval.end_time_msc:
                matching_interval = interval
                break

        if matching_interval is None:
            return None

        if target_ms < (matching_interval.start_time_msc - self.max_entry_delay_ms):
            return None

        dt = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
        bid = float(self.bids[idx])
        ask = float(self.asks[idx])
        spread_pts = (ask - bid) / self.point_value
        return TickQuote(
            time_msc=t_ms,
            timestamp=dt,
            bid=bid,
            ask=ask,
            spread_points=spread_pts,
            flags=int(self.flags[idx]),
        )

    def is_window_covered(self, start_time: datetime, end_time: datetime) -> bool:
        """Verify whether [start_time, end_time] has unbroken coverage in a single interval."""
        if len(self.time_msc) == 0:
            return False

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        if start_ms >= end_ms:
            return False

        # If window crosses any gap, coverage is broken
        for g in self.gaps:
            if max(start_ms, g.gap_start_msc) < min(end_ms, g.gap_end_msc):
                return False

        # Must be contained within a single coverage interval
        for interval in self.coverage_intervals:
            if (interval.start_time_msc - self.max_entry_delay_ms <= start_ms) and (
                end_ms <= interval.end_time_msc + self.max_entry_delay_ms
            ):
                return True
        return False

    def get_coverage_summary(self) -> dict[str, Any]:
        """Audit summary of repository coverage intervals and detected gaps."""
        if len(self.time_msc) == 0:
            return {"tick_count": 0, "intervals_count": 0, "gaps_count": 0}

        return {
            "tick_count": len(self.time_msc),
            "earliest_timestamp": datetime.fromtimestamp(
                self.time_msc[0] / 1000.0, tz=timezone.utc
            ).isoformat(),
            "latest_timestamp": datetime.fromtimestamp(
                self.time_msc[-1] / 1000.0, tz=timezone.utc
            ).isoformat(),
            "coverage_intervals_count": len(self.coverage_intervals),
            "gaps_count": len(self.gaps),
            "max_allowed_gap_seconds": self.max_allowed_gap_ms / 1000.0,
            "max_entry_delay_seconds": self.max_entry_delay_ms / 1000.0,
            "gaps": [g.to_dict() for g in self.gaps],
            "coverage_intervals": [i.to_dict() for i in self.coverage_intervals],
        }

    def get_ticks_in_window(self, start_time: datetime, end_time: datetime) -> list[TickQuote]:
        """Retrieve chronological sequence of TickQuote objects within [start_time, end_time]."""
        idx_0, idx_1 = self.get_slice_indices(start_time, end_time)
        if idx_0 >= idx_1:
            return []

        quotes: list[TickQuote] = []
        for i in range(idx_0, idx_1):
            t_ms = int(self.time_msc[i])
            dt = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
            bid = float(self.bids[i])
            ask = float(self.asks[i])
            spread_pts = (ask - bid) / self.point_value
            quotes.append(
                TickQuote(
                    time_msc=t_ms,
                    timestamp=dt,
                    bid=bid,
                    ask=ask,
                    spread_points=spread_pts,
                    flags=int(self.flags[i]),
                )
            )
        return quotes

    def get_raw_slice(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return zero-copy raw numpy slices (time_msc, bids, asks) for vectorized processing."""
        idx_0, idx_1 = self.get_slice_indices(start_time, end_time)
        return self.time_msc[idx_0:idx_1], self.bids[idx_0:idx_1], self.asks[idx_0:idx_1]


def validate_tick_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Audit tick dataset quality metrics and governance compliance."""
    if len(df) == 0:
        return {"status": "EMPTY", "tick_count": 0}

    time_msc = df["time_msc"].to_numpy()
    bids = df["bid"].to_numpy()
    asks = df["ask"].to_numpy()

    diffs = np.diff(time_msc)
    is_monotonic = bool(np.all(diffs >= 0))
    duplicate_count = int(np.sum(diffs == 0))

    spread_points = (asks - bids) / 1e-05
    min_spread = float(np.min(spread_points))
    max_spread = float(np.max(spread_points))
    mean_spread = float(np.mean(spread_points))
    median_spread = float(np.median(spread_points))
    p95_spread = float(np.percentile(spread_points, 95))
    p99_spread = float(np.percentile(spread_points, 99))

    negative_spreads = int(np.sum(spread_points < 0))
    non_finite = ~np.isfinite(bids) | ~np.isfinite(asks)
    non_positive = (bids <= 0) | (asks <= 0)
    invalid_prices = int(np.sum(non_finite | non_positive))

    earliest_ts = datetime.fromtimestamp(time_msc[0] / 1000.0, tz=timezone.utc).isoformat()
    latest_ts = datetime.fromtimestamp(time_msc[-1] / 1000.0, tz=timezone.utc).isoformat()

    lock_timestamp_ms = int(PHASE11_TEST_LOCK_TIMESTAMP.timestamp() * 1000)
    test_leakage = bool(time_msc[-1] > lock_timestamp_ms)
    pass_audit = is_monotonic and negative_spreads == 0 and not test_leakage

    return {
        "status": "PASS" if pass_audit else "FAIL",
        "tick_count": len(df),
        "earliest_timestamp": earliest_ts,
        "latest_timestamp": latest_ts,
        "is_monotonic": is_monotonic,
        "duplicate_timestamps_count": duplicate_count,
        "duplicate_pct": round((duplicate_count / len(df)) * 100.0, 4),
        "invalid_prices": invalid_prices,
        "negative_spreads": negative_spreads,
        "spread_metrics": {
            "mean_points": round(mean_spread, 2),
            "median_points": round(median_spread, 2),
            "p95_points": round(p95_spread, 2),
            "p99_points": round(p99_spread, 2),
            "min_points": round(min_spread, 2),
            "max_points": round(max_spread, 2),
        },
        "phase11_test_lock_respected": not test_leakage,
    }
