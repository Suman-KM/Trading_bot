"""Comprehensive unit tests for the historical market data pipeline (Phase 3).

Uses deterministic, synthetic test fixtures to test schema validation, timestamp handling,
OHLC sanity, spread and volume statistics, gap classification, and artifact persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.data.gaps import analyze_gaps, classify_gap
from ai.data.ingestion import (
    generate_dataset_metadata,
    load_raw_dataset,
    process_and_validate,
    save_dataset_metadata,
    save_processed_dataset,
    save_quality_report,
)
from ai.data.schema import REQUIRED_RAW_COLUMNS, validate_schema
from ai.data.timestamps import add_utc_timestamp, validate_timestamps
from ai.data.validation import (
    validate_market_data,
    validate_ohlc,
    validate_spread,
    validate_volume,
)


@pytest.fixture
def valid_synthetic_m15_df() -> pd.DataFrame:
    """Create a minimal valid synthetic EURUSD M15 DataFrame.

    Starts Monday 2026-09-21 00:00:00 UTC (epoch 1790035200).
    5 consecutive 15-minute candles (spacing 900s).
    """
    base_time = 1790035200  # 2026-09-21 00:00:00 UTC
    times = [base_time + (i * 900) for i in range(5)]
    return pd.DataFrame(
        {
            "time": np.array(times, dtype=np.int64),
            "open": [1.08500, 1.08520, 1.08510, 1.08530, 1.08540],
            "high": [1.08550, 1.08560, 1.08540, 1.08580, 1.08570],
            "low": [1.08480, 1.08500, 1.08490, 1.08520, 1.08510],
            "close": [1.08520, 1.08510, 1.08530, 1.08540, 1.08550],
            "tick_volume": [250, 310, 195, 420, 305],
            "spread": [12, 11, 12, 10, 11],
            "real_volume": [0, 0, 0, 0, 0],
        }
    )


# -----------------------------------------------------------------------------
# 1. Schema Validation Tests
# -----------------------------------------------------------------------------


def test_schema_valid(valid_synthetic_m15_df: pd.DataFrame) -> None:
    result = validate_schema(valid_synthetic_m15_df, strict=True)
    assert result.is_valid is True
    assert len(result.missing_columns) == 0
    assert len(result.unexpected_columns) == 0
    assert len(result.dtype_issues) == 0


def test_schema_missing_column(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_missing = valid_synthetic_m15_df.drop(columns=["spread"])
    result = validate_schema(df_missing, strict=True)
    assert result.is_valid is False
    assert "spread" in result.missing_columns


def test_schema_unexpected_column(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_extra = valid_synthetic_m15_df.copy()
    df_extra["extra_col"] = 123
    result = validate_schema(df_extra, strict=True)
    assert "extra_col" in result.unexpected_columns


def test_schema_string_dtype_rejection(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_obj = valid_synthetic_m15_df.copy()
    df_obj["close"] = df_obj["close"].astype(str)
    result = validate_schema(df_obj, strict=True)
    assert result.is_valid is False
    assert any("non-numeric" in issue or "expected float" in issue for issue in result.dtype_issues)


def test_schema_empty_dataframe() -> None:
    df_empty = pd.DataFrame()
    result = validate_schema(df_empty)
    assert result.is_valid is False
    assert len(result.missing_columns) == len(REQUIRED_RAW_COLUMNS)


# -----------------------------------------------------------------------------
# 2. Timestamp Tests
# -----------------------------------------------------------------------------


def test_timestamps_valid(valid_synthetic_m15_df: pd.DataFrame) -> None:
    result = validate_timestamps(valid_synthetic_m15_df)
    assert result.is_valid is True
    assert result.is_strictly_monotonic is True
    assert result.duplicate_count == 0
    assert result.earliest_utc == "2026-09-22T00:00:00+00:00"


def test_timestamps_duplicates(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_dup = valid_synthetic_m15_df.copy()
    df_dup.loc[1, "time"] = df_dup.loc[0, "time"]
    result = validate_timestamps(df_dup)
    assert result.is_valid is False
    assert result.duplicate_count == 1


def test_timestamps_non_monotonic(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_shuffled = valid_synthetic_m15_df.iloc[[0, 2, 1, 3, 4]].copy()
    result = validate_timestamps(df_shuffled)
    assert result.is_valid is False
    assert result.is_strictly_monotonic is False


def test_add_utc_timestamp(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_enriched = add_utc_timestamp(valid_synthetic_m15_df, time_col="time", target_col="timestamp")
    assert "timestamp" in df_enriched.columns
    assert "time" in df_enriched.columns  # original raw preserved
    assert isinstance(df_enriched["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(df_enriched["timestamp"].dtype.tz) == "UTC"
    assert df_enriched["timestamp"].iloc[0].isoformat() == "2026-09-22T00:00:00+00:00"


# -----------------------------------------------------------------------------
# 3. OHLC Sanity Tests
# -----------------------------------------------------------------------------


def test_ohlc_valid(valid_synthetic_m15_df: pd.DataFrame) -> None:
    result = validate_ohlc(valid_synthetic_m15_df)
    assert result.is_valid is True
    assert result.total_violations == 0


def test_ohlc_high_less_than_open(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_bad = valid_synthetic_m15_df.copy()
    df_bad.loc[0, "high"] = df_bad.loc[0, "open"] - 0.001
    result = validate_ohlc(df_bad)
    assert result.is_valid is False
    assert result.high_lt_open_count == 1


def test_ohlc_low_greater_than_close(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_bad = valid_synthetic_m15_df.copy()
    df_bad.loc[1, "low"] = df_bad.loc[1, "close"] + 0.001
    result = validate_ohlc(df_bad)
    assert result.is_valid is False
    assert result.low_gt_close_count == 1


def test_ohlc_high_less_than_low(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_bad = valid_synthetic_m15_df.copy()
    df_bad.loc[2, "high"] = 1.08400
    df_bad.loc[2, "low"] = 1.08600
    result = validate_ohlc(df_bad)
    assert result.is_valid is False
    assert result.high_lt_low_count == 1


def test_ohlc_negative_or_nan_prices(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_neg = valid_synthetic_m15_df.copy()
    df_neg.loc[0, "open"] = -1.085
    result_neg = validate_ohlc(df_neg)
    assert result_neg.is_valid is False
    assert result_neg.non_positive_prices_count == 1

    df_nan = valid_synthetic_m15_df.copy()
    df_nan.loc[0, "close"] = np.nan
    result_nan = validate_ohlc(df_nan)
    assert result_nan.is_valid is False
    assert result_nan.nan_prices_count == 1


# -----------------------------------------------------------------------------
# 4. Spread and Volume Validation Tests
# -----------------------------------------------------------------------------


def test_spread_valid(valid_synthetic_m15_df: pd.DataFrame) -> None:
    result = validate_spread(valid_synthetic_m15_df)
    assert result.is_valid is True
    assert result.negative_spread_count == 0
    assert result.min_spread == 10.0
    assert result.max_spread == 12.0


def test_spread_negative_detection(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_bad = valid_synthetic_m15_df.copy()
    df_bad.loc[0, "spread"] = -5
    result = validate_spread(df_bad)
    assert result.is_valid is False
    assert result.negative_spread_count == 1


def test_volume_validation(valid_synthetic_m15_df: pd.DataFrame) -> None:
    result = validate_volume(valid_synthetic_m15_df)
    assert result.is_valid is True
    assert result.zero_tick_volume_count == 0
    assert result.real_volume_all_zero is True
    assert result.real_volume_zero_percentage == 100.0


def test_volume_zero_tick_reporting(valid_synthetic_m15_df: pd.DataFrame) -> None:
    df_zero = valid_synthetic_m15_df.copy()
    df_zero.loc[0, "tick_volume"] = 0
    result = validate_volume(df_zero)
    assert result.zero_tick_volume_count == 1


# -----------------------------------------------------------------------------
# 5. Gap Analysis Tests
# -----------------------------------------------------------------------------


def test_gap_analysis_no_gaps(valid_synthetic_m15_df: pd.DataFrame) -> None:
    result = analyze_gaps(valid_synthetic_m15_df)
    assert result.total_gaps == 0
    assert len(result.gaps) == 0


def test_gap_analysis_weekend_detection() -> None:
    # Friday 2026-09-18 21:00:00 UTC (weekday 4) -> Sunday 2026-09-20 21:00:00 UTC (weekday 6)
    fri_close = 1789851600
    sun_open = 1790024400  # 48 hours later (172800 seconds)
    df = pd.DataFrame({"time": [fri_close, sun_open]})
    result = analyze_gaps(df)
    assert result.total_gaps == 1
    assert result.expected_weekend_count == 1
    assert result.gaps[0].classification == "EXPECTED_WEEKEND"
    assert result.gaps[0].gap_duration_hours == 48.0


def test_gap_analysis_unexpected_intraday_gap() -> None:
    # Tuesday 2026-09-22 10:00:00 UTC -> Tuesday 2026-09-22 14:00:00 UTC (4 hour gap)
    tues_10 = 1790157600
    tues_14 = tues_10 + (4 * 3600)
    df = pd.DataFrame({"time": [tues_10, tues_14]})
    result = analyze_gaps(df)
    assert result.total_gaps == 1
    assert result.unexpected_gap_count == 1
    assert result.gaps[0].classification == "UNEXPECTED_GAP"


def test_classify_gap_holiday() -> None:
    # Dec 24 20:00 UTC to Dec 26 00:00 UTC (Christmas holiday closure)
    dec24_epoch = 1798142400
    dec26_epoch = dec24_epoch + (28 * 3600)
    classification = classify_gap(dec24_epoch, dec26_epoch, 28 * 3600)
    assert classification == "EXPECTED_HOLIDAY"


# -----------------------------------------------------------------------------
# 6. Master Validation & Ingestion Roundtrip Tests
# -----------------------------------------------------------------------------


def test_master_validation_pass(valid_synthetic_m15_df: pd.DataFrame) -> None:
    report = validate_market_data(valid_synthetic_m15_df)
    assert report.validation_status == "PASS"
    assert len(report.critical_errors) == 0
    report_dict = report.to_dict()
    assert report_dict["validation_status"] == "PASS"
    assert report_dict["dataset"]["row_count"] == 5


def test_master_validation_fail_on_critical_corruption(
    valid_synthetic_m15_df: pd.DataFrame,
) -> None:
    df_corrupt = valid_synthetic_m15_df.copy()
    df_corrupt.loc[0, "high"] = 0.5  # high < open violation
    df_corrupt.loc[1, "spread"] = -10  # negative spread violation
    report = validate_market_data(df_corrupt)
    assert report.validation_status == "FAIL"
    assert len(report.critical_errors) >= 2


def test_ingestion_and_artifact_roundtrip(
    valid_synthetic_m15_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    # 1. Save raw test file
    raw_file = tmp_path / "raw_test.parquet"
    valid_synthetic_m15_df.to_parquet(raw_file, index=False)

    # 2. Load raw file
    loaded_raw = load_raw_dataset(raw_file)
    assert len(loaded_raw) == 5

    # 3. Process and validate
    df_processed, report = process_and_validate(loaded_raw, symbol="EURUSD", timeframe="M15")
    assert report.validation_status == "PASS"
    assert "timestamp" in df_processed.columns

    # 4. Save processed artifacts
    processed_dir = tmp_path / "processed"
    saved_parquet = save_processed_dataset(
        df_processed,
        output_dir=processed_dir,
        symbol="EURUSD",
        timeframe="M15",
        export_csv=True,
    )
    assert saved_parquet.is_file()
    assert (processed_dir / "eurusd_m15_processed.csv").is_file()

    # 5. Save quality report & metadata
    report_file = save_quality_report(report, output_dir=processed_dir)
    assert report_file.is_file()
    with open(report_file, encoding="utf-8") as f:
        rep_json = json.load(f)
    assert rep_json["validation_status"] == "PASS"

    metadata = generate_dataset_metadata(df_processed, report)
    meta_file = save_dataset_metadata(metadata, output_dir=processed_dir)
    assert meta_file.is_file()
    with open(meta_file, encoding="utf-8") as f:
        meta_json = json.load(f)
    assert meta_json["symbol"] == "EURUSD"
    assert meta_json["row_count"] == 5
