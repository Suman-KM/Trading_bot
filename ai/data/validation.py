"""Comprehensive market data validation engine.

Performs OHLC relationship verification, price finiteness checks, spread and volume
distribution analyses, duplicate detection, and gap classification to evaluate overall
dataset trustworthiness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai.data.gaps import GapAnalysisResult, analyze_gaps
from ai.data.schema import SchemaValidationResult, validate_schema
from ai.data.timestamps import TimestampValidationResult, validate_timestamps

POINT_SIZE_EURUSD: float = 0.00001  # 1 MT5 point = 0.00001 (0.1 pip) for 5-digit EURUSD


@dataclass(frozen=True)
class OHLCValidationResult:
    """Result of candle price relationship and sanity verification."""

    is_valid: bool
    high_lt_open_count: int
    high_lt_close_count: int
    low_gt_open_count: int
    low_gt_close_count: int
    high_lt_low_count: int
    non_positive_prices_count: int
    nan_prices_count: int
    inf_prices_count: int
    total_violations: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class SpreadValidationResult:
    """Statistical and sanity verification of the spread column."""

    is_valid: bool
    unit: str  # "MT5 points (0.00001 for EURUSD)"
    negative_spread_count: int
    zero_spread_count: int
    missing_spread_count: int
    min_spread: float | None
    max_spread: float | None
    mean_spread: float | None
    median_spread: float | None
    p25_spread: float | None
    p75_spread: float | None
    p95_spread: float | None
    p99_spread: float | None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class VolumeValidationResult:
    """Statistical and sanity verification of tick and real volume columns."""

    is_valid: bool
    missing_tick_volume: int
    zero_tick_volume_count: int
    min_tick_volume: int | None
    max_tick_volume: int | None
    mean_tick_volume: float | None
    median_tick_volume: float | None
    real_volume_all_zero: bool
    real_volume_zero_percentage: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class MarketDataValidationReport:
    """Comprehensive master validation report for a market dataset."""

    symbol: str
    timeframe: str
    source: str
    row_count: int
    earliest_timestamp_utc: str | None
    latest_timestamp_utc: str | None
    schema: SchemaValidationResult
    timestamps: TimestampValidationResult
    ohlc: OHLCValidationResult
    spread: SpreadValidationResult
    volume: VolumeValidationResult
    gaps: GapAnalysisResult
    duplicate_rows_count: int
    missing_values_count: int
    validation_status: str  # "PASS" or "FAIL"
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to nested dictionary for JSON serialization."""
        return {
            "dataset": {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "source": self.source,
                "row_count": self.row_count,
                "earliest_timestamp": self.earliest_timestamp_utc,
                "latest_timestamp": self.latest_timestamp_utc,
            },
            "schema": self.schema.to_dict(),
            "quality": {
                "duplicate_timestamps": self.timestamps.duplicate_count,
                "duplicate_rows": self.duplicate_rows_count,
                "chronological": self.timestamps.is_strictly_monotonic,
                "ohlc_violations": self.ohlc.total_violations,
                "invalid_prices": (
                    self.ohlc.non_positive_prices_count
                    + self.ohlc.nan_prices_count
                    + self.ohlc.inf_prices_count
                ),
                "missing_values": self.missing_values_count,
                "zero_tick_volume": self.volume.zero_tick_volume_count,
                "negative_spread": self.spread.negative_spread_count,
                "real_volume_zero_percentage": self.volume.real_volume_zero_percentage,
            },
            "gaps": self.gaps.to_dict(),
            "validation_status": self.validation_status,
            "critical_errors": self.critical_errors,
            "warnings": self.warnings,
        }


def validate_ohlc(df: pd.DataFrame) -> OHLCValidationResult:
    """Validate that every candle obeys OHLC physical price constraints.

    Rules verified:
    - high >= max(open, close)
    - low <= min(open, close)
    - high >= low
    - open > 0, high > 0, low > 0, close > 0
    - No NaN, no +inf, no -inf
    """
    errors: list[str] = []

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return OHLCValidationResult(
                is_valid=False,
                high_lt_open_count=0,
                high_lt_close_count=0,
                low_gt_open_count=0,
                low_gt_close_count=0,
                high_lt_low_count=0,
                non_positive_prices_count=0,
                nan_prices_count=0,
                inf_prices_count=0,
                total_violations=1,
                errors=[f"Required price column '{col}' missing for OHLC validation."],
            )

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l_arr = df["low"].to_numpy()
    c = df["close"].to_numpy()

    # Check finite and positive
    all_prices = np.stack([o, h, l_arr, c])
    nan_count = int(np.isnan(all_prices).sum())
    inf_count = int(np.isinf(all_prices).sum())
    non_pos_count = int((all_prices <= 0).sum())

    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN values across price columns.")
    if inf_count > 0:
        errors.append(f"Found {inf_count} infinite values across price columns.")
    if non_pos_count > 0:
        errors.append(f"Found {non_pos_count} zero or negative values across price columns.")

    # Check candle relationships
    high_lt_open = int((h < o).sum())
    high_lt_close = int((h < c).sum())
    low_gt_open = int((l_arr > o).sum())
    low_gt_close = int((l_arr > c).sum())
    high_lt_low = int((h < l_arr).sum())

    if high_lt_open > 0:
        errors.append(f"High < Open in {high_lt_open} candles.")
    if high_lt_close > 0:
        errors.append(f"High < Close in {high_lt_close} candles.")
    if low_gt_open > 0:
        errors.append(f"Low > Open in {low_gt_open} candles.")
    if low_gt_close > 0:
        errors.append(f"Low > Close in {low_gt_close} candles.")
    if high_lt_low > 0:
        errors.append(f"High < Low in {high_lt_low} candles.")

    total_violations = (
        high_lt_open
        + high_lt_close
        + low_gt_open
        + low_gt_close
        + high_lt_low
        + nan_count
        + inf_count
        + non_pos_count
    )

    is_valid = total_violations == 0

    return OHLCValidationResult(
        is_valid=is_valid,
        high_lt_open_count=high_lt_open,
        high_lt_close_count=high_lt_close,
        low_gt_open_count=low_gt_open,
        low_gt_close_count=low_gt_close,
        high_lt_low_count=high_lt_low,
        non_positive_prices_count=non_pos_count,
        nan_prices_count=nan_count,
        inf_prices_count=inf_count,
        total_violations=total_violations,
        errors=errors,
    )


def validate_spread(df: pd.DataFrame, spread_col: str = "spread") -> SpreadValidationResult:
    """Validate spread values in MT5 points."""
    errors: list[str] = []

    if spread_col not in df.columns:
        return SpreadValidationResult(
            is_valid=False,
            unit="MT5 points (0.00001 for EURUSD)",
            negative_spread_count=0,
            zero_spread_count=0,
            missing_spread_count=len(df),
            min_spread=None,
            max_spread=None,
            mean_spread=None,
            median_spread=None,
            p25_spread=None,
            p75_spread=None,
            p95_spread=None,
            p99_spread=None,
            errors=[f"Column '{spread_col}' missing from DataFrame."],
        )

    s = df[spread_col]
    missing_count = int(s.isna().sum())
    valid_s = s.dropna()

    if missing_count > 0:
        errors.append(f"Found {missing_count} missing spread values.")

    negative_count = int((valid_s < 0).sum())
    if negative_count > 0:
        errors.append(f"Found {negative_count} negative spread values (strictly invalid).")

    zero_count = int((valid_s == 0).sum())

    if len(valid_s) > 0:
        min_s = float(valid_s.min())
        max_s = float(valid_s.max())
        mean_s = round(float(valid_s.mean()), 2)
        med_s = float(valid_s.median())
        p25 = float(np.percentile(valid_s, 25))
        p75 = float(np.percentile(valid_s, 75))
        p95 = float(np.percentile(valid_s, 95))
        p99 = float(np.percentile(valid_s, 99))
    else:
        min_s = max_s = mean_s = med_s = p25 = p75 = p95 = p99 = None

    is_valid = (negative_count == 0) and (missing_count == 0)

    return SpreadValidationResult(
        is_valid=is_valid,
        unit="MT5 points (0.00001 for EURUSD)",
        negative_spread_count=negative_count,
        zero_spread_count=zero_count,
        missing_spread_count=missing_count,
        min_spread=min_s,
        max_spread=max_s,
        mean_spread=mean_s,
        median_spread=med_s,
        p25_spread=p25,
        p75_spread=p75,
        p95_spread=p95,
        p99_spread=p99,
        errors=errors,
    )


def validate_volume(
    df: pd.DataFrame,
    tick_vol_col: str = "tick_volume",
    real_vol_col: str = "real_volume",
) -> VolumeValidationResult:
    """Validate tick volume and real volume distributions."""
    errors: list[str] = []

    # Check tick_volume
    if tick_vol_col not in df.columns:
        return VolumeValidationResult(
            is_valid=False,
            missing_tick_volume=len(df),
            zero_tick_volume_count=0,
            min_tick_volume=None,
            max_tick_volume=None,
            mean_tick_volume=None,
            median_tick_volume=None,
            real_volume_all_zero=False,
            real_volume_zero_percentage=0.0,
            errors=[f"Column '{tick_vol_col}' missing from DataFrame."],
        )

    tv = df[tick_vol_col]
    missing_tv = int(tv.isna().sum())
    if missing_tv > 0:
        errors.append(f"Found {missing_tv} missing tick volume values.")

    valid_tv = tv.dropna()
    zero_tv = int((valid_tv <= 0).sum())

    if len(valid_tv) > 0:
        min_tv = int(valid_tv.min())
        max_tv = int(valid_tv.max())
        mean_tv = round(float(valid_tv.mean()), 2)
        med_tv = float(valid_tv.median())
    else:
        min_tv = max_tv = mean_tv = med_tv = None

    # Check real_volume (expected 0 for retail OTC FX)
    real_all_zero = False
    real_zero_pct = 0.0
    if real_vol_col in df.columns:
        rv = df[real_vol_col]
        zero_rv = int((rv == 0).sum())
        real_zero_pct = round((zero_rv / len(df)) * 100.0, 2)
        real_all_zero = zero_rv == len(df)
    else:
        errors.append(f"Column '{real_vol_col}' missing from DataFrame.")

    is_valid = (missing_tv == 0) and (len(errors) == 0)

    return VolumeValidationResult(
        is_valid=is_valid,
        missing_tick_volume=missing_tv,
        zero_tick_volume_count=zero_tv,
        min_tick_volume=min_tv,
        max_tick_volume=max_tv,
        mean_tick_volume=mean_tv,
        median_tick_volume=med_tv,
        real_volume_all_zero=real_all_zero,
        real_volume_zero_percentage=real_zero_pct,
        errors=errors,
    )


def validate_market_data(
    df: pd.DataFrame,
    symbol: str = "EURUSD",
    timeframe: str = "M15",
    source: str = "MT5",
) -> MarketDataValidationReport:
    """Execute complete validation suite on the market dataset."""
    critical_errors: list[str] = []
    warnings: list[str] = []

    # 1. Schema check
    schema_res = validate_schema(df, strict=True)
    if not schema_res.is_valid:
        critical_errors.extend(schema_res.errors)

    # 2. Timestamp check
    ts_res = validate_timestamps(df, time_col="time")
    if not ts_res.is_valid:
        critical_errors.extend(ts_res.errors)

    # 3. Duplicate row check
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        critical_errors.append(f"Found {duplicate_rows} duplicate complete rows.")

    # 4. OHLC check
    ohlc_res = validate_ohlc(df)
    if not ohlc_res.is_valid:
        critical_errors.extend(ohlc_res.errors)

    # 5. Spread check
    spread_res = validate_spread(df, spread_col="spread")
    if not spread_res.is_valid:
        critical_errors.extend(spread_res.errors)
    if spread_res.zero_spread_count > 0:
        warnings.append(f"Observed {spread_res.zero_spread_count} candles with zero spread.")

    # 6. Volume check
    vol_res = validate_volume(df, tick_vol_col="tick_volume", real_vol_col="real_volume")
    if not vol_res.is_valid:
        critical_errors.extend(vol_res.errors)
    if vol_res.zero_tick_volume_count > 0:
        warnings.append(f"Observed {vol_res.zero_tick_volume_count} candles with zero tick volume.")
    if not vol_res.real_volume_all_zero:
        warnings.append(
            f"real_volume is non-zero in {round(100.0 - vol_res.real_volume_zero_percentage, 2)}% of candles."
        )

    # 7. Total missing values
    missing_count = int(df.isna().sum().sum())
    if missing_count > 0:
        critical_errors.append(f"Found {missing_count} total null/NaN values in dataset.")

    # 8. Gap analysis
    gap_res = analyze_gaps(df, time_col="time")
    if gap_res.unexpected_gap_count > 0:
        warnings.append(f"Detected {gap_res.unexpected_gap_count} unexpected intraday market gaps.")

    # Final pass/fail determination
    validation_status = "PASS" if len(critical_errors) == 0 else "FAIL"

    return MarketDataValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        row_count=len(df),
        earliest_timestamp_utc=ts_res.earliest_utc,
        latest_timestamp_utc=ts_res.latest_utc,
        schema=schema_res,
        timestamps=ts_res,
        ohlc=ohlc_res,
        spread=spread_res,
        volume=vol_res,
        gaps=gap_res,
        duplicate_rows_count=duplicate_rows,
        missing_values_count=missing_count,
        validation_status=validation_status,
        critical_errors=critical_errors,
        warnings=warnings,
    )
