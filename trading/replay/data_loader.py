"""Data loader and sanitizer for deterministic market-data replay."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from trading.replay.models import MarketEvent

# Strictly quarantined test partition start timestamp (Phase 11-34 Governance)
LOCKED_TEST_CUTOFF = datetime(2026, 2, 19, 12, 0, 0, tzinfo=timezone.utc)


class QuarantinedPartitionError(ValueError):
    """Raised when replay data attempts to access the permanently locked test partition."""

    pass


class ReplayDataValidationError(ValueError):
    """Raised when replay data fails ordering, duplicate, or consistency checks."""

    pass


class DataIntegrityError(ReplayDataValidationError):
    """Raised when replay data fails chronological ordering, monotonicity, or consistency."""

    pass


class MarketReplayDataLoader:
    """Loads, validates, and streams historical market observations for replay."""

    @classmethod
    def load_from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str = "EURUSD",
        enforce_quarantine: bool = True,
    ) -> List[MarketEvent]:
        """Validate and convert a DataFrame of candles or ticks into MarketEvents."""
        if df.empty:
            return []

        # Ensure datetime timestamp column
        work_df = df.copy()
        if "timestamp" in work_df.columns:
            ts_col = "timestamp"
        elif "time" in work_df.columns:
            ts_col = "time"
        else:
            raise ReplayDataValidationError("Missing 'timestamp' or 'time' column in replay data.")

        # Convert to UTC datetime series
        if pd.api.types.is_numeric_dtype(work_df[ts_col]):
            work_df["_dt"] = pd.to_datetime(work_df[ts_col], unit="s", utc=True)
        else:
            work_df["_dt"] = pd.to_datetime(work_df[ts_col], utc=True)

        events: List[MarketEvent] = []
        prev_ts: Optional[datetime] = None

        for idx, row in work_df.iterrows():
            ts = row["_dt"].to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            # 1. Enforce Locked Test Quarantine
            if enforce_quarantine and ts >= LOCKED_TEST_CUTOFF:
                raise QuarantinedPartitionError(
                    f"Quarantined partition violation at index {idx}: timestamp {ts} "
                    f"is at or beyond locked test cutoff {LOCKED_TEST_CUTOFF}."
                )

            # 2. Enforce strict monotonic timestamp ordering
            if prev_ts is not None:
                if ts < prev_ts:
                    raise DataIntegrityError(
                        f"Non-monotonic timestamp at index {idx}: {ts} < {prev_ts}"
                    )
                if ts == prev_ts:
                    raise DataIntegrityError(f"Duplicate timestamp at index {idx}: {ts}")
            prev_ts = ts

            # 3. Extract and validate prices
            op = float(row["open"]) if "open" in row and pd.notna(row["open"]) else None
            hi = float(row["high"]) if "high" in row and pd.notna(row["high"]) else None
            lo = float(row["low"]) if "low" in row and pd.notna(row["low"]) else None
            cl = float(row["close"]) if "close" in row and pd.notna(row["close"]) else None
            bid = float(row["bid"]) if "bid" in row and pd.notna(row["bid"]) else None
            ask = float(row["ask"]) if "ask" in row and pd.notna(row["ask"]) else None
            mid = float(row["mid"]) if "mid" in row and pd.notna(row["mid"]) else None
            last = float(row["last"]) if "last" in row and pd.notna(row["last"]) else None
            vol = (
                float(row.get("tick_volume", row.get("volume", 0.0)))
                if ("tick_volume" in row or "volume" in row)
                else None
            )

            # Check for NaN / Inf / non-positive prices
            for name, val in [
                ("open", op),
                ("high", hi),
                ("low", lo),
                ("close", cl),
                ("bid", bid),
                ("ask", ask),
                ("mid", mid),
                ("last", last),
            ]:
                if val is not None and (not math.isfinite(val) or val <= 0):
                    raise ReplayDataValidationError(
                        f"Invalid {name} price at index {idx} ({ts}): {val}"
                    )

            # 4. Check OHLC consistency if full candle is present
            if op is not None and hi is not None and lo is not None and cl is not None:
                if not (lo <= op <= hi and lo <= cl <= hi):
                    raise ReplayDataValidationError(
                        f"OHLC inconsistency at index {idx} ({ts}): open={op}, high={hi}, "
                        f"low={lo}, close={cl}"
                    )

            # Derive mid or last if not explicitly present
            if mid is None:
                if bid is not None and ask is not None:
                    mid = (bid + ask) / 2.0
                elif cl is not None:
                    mid = cl

            events.append(
                MarketEvent(
                    timestamp=ts,
                    symbol=symbol,
                    open=op,
                    high=hi,
                    low=lo,
                    close=cl,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    last=last or cl,
                    volume=vol,
                )
            )

        return events

    @classmethod
    def load_from_parquet(
        cls,
        file_path: Union[str, Path],
        symbol: str = "EURUSD",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        max_rows: Optional[int] = None,
        enforce_quarantine: bool = True,
    ) -> List[MarketEvent]:
        """Load and filter parquet file records for replay."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        df = pd.read_parquet(path)

        # Apply row limit or date slices if specified
        if start_date is not None:
            st = pd.to_datetime(start_date, utc=True)
            ts_col = "timestamp" if "timestamp" in df.columns else "time"
            if pd.api.types.is_numeric_dtype(df[ts_col]):
                df["_dt_filter"] = pd.to_datetime(df[ts_col], unit="s", utc=True)
            else:
                df["_dt_filter"] = pd.to_datetime(df[ts_col], utc=True)
            df = df[df["_dt_filter"] >= st]

        if end_date is not None:
            et = pd.to_datetime(end_date, utc=True)
            if "_dt_filter" not in df.columns:
                ts_col = "timestamp" if "timestamp" in df.columns else "time"
                if pd.api.types.is_numeric_dtype(df[ts_col]):
                    df["_dt_filter"] = pd.to_datetime(df[ts_col], unit="s", utc=True)
                else:
                    df["_dt_filter"] = pd.to_datetime(df[ts_col], utc=True)
            df = df[df["_dt_filter"] <= et]

        if "_dt_filter" in df.columns:
            df = df.drop(columns=["_dt_filter"])

        if max_rows is not None and len(df) > max_rows:
            df = df.iloc[:max_rows]

        return cls.load_from_dataframe(
            df=df,
            symbol=symbol,
            enforce_quarantine=enforce_quarantine,
        )
