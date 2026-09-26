"""Data ingestion, processing, and artifact persistence.

Loads raw historical market data, applies validation and timestamp enrichment,
and writes immutable processed Parquet datasets, quality reports, and dataset metadata.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ai.data.schema import REQUIRED_RAW_COLUMNS
from ai.data.timestamps import add_utc_timestamp
from ai.data.validation import MarketDataValidationReport, validate_market_data


def get_git_commit_hash() -> str | None:
    """Retrieve the current Git commit hash if in a Git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load raw MT5 historical market data from Parquet or CSV.

    Parameters
    ----------
    path : str | Path
        Path to the raw market data file.

    Returns
    -------
    pd.DataFrame
        Unmodified raw market data DataFrame.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(file_path)
    elif suffix in (".csv", ".txt"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .parquet, .csv")

    if df.empty:
        raise ValueError(f"Loaded dataset from {file_path} is empty.")

    return df


def process_and_validate(
    df_raw: pd.DataFrame,
    symbol: str = "EURUSD",
    timeframe: str = "M15",
    source: str = "MT5",
) -> tuple[pd.DataFrame, MarketDataValidationReport]:
    """Validate raw data and generate processed DataFrame with timezone-aware UTC timestamps.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw market data DataFrame.
    symbol : str, default "EURUSD"
        Trading instrument symbol.
    timeframe : str, default "M15"
        Candle timeframe.
    source : str, default "MT5"
        Data provider source name.

    Returns
    -------
    tuple[pd.DataFrame, MarketDataValidationReport]
        Processed DataFrame and comprehensive validation report.
    """
    # 1. Execute complete validation suite on the raw data
    report = validate_market_data(df_raw, symbol=symbol, timeframe=timeframe, source=source)

    # 2. Add timezone-aware UTC timestamp without modifying the original raw 'time' column
    df_processed = add_utc_timestamp(df_raw, time_col="time", target_col="timestamp")

    return df_processed, report


def save_processed_dataset(
    df_processed: pd.DataFrame,
    output_dir: str | Path,
    symbol: str = "EURUSD",
    timeframe: str = "M15",
    export_csv: bool = False,
) -> Path:
    """Persist processed dataset to Parquet and optional CSV.

    Parameters
    ----------
    df_processed : pd.DataFrame
        Validated and enriched DataFrame.
    output_dir : str | Path
        Target directory for processed artifacts.
    symbol : str, default "EURUSD"
        Instrument symbol.
    timeframe : str, default "M15"
        Timeframe identifier.
    export_csv : bool, default False
        If True, also exports a CSV copy for debugging.

    Returns
    -------
    Path
        Path to the primary Parquet processed dataset.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_name = f"{symbol.lower()}_{timeframe.lower()}_processed"
    parquet_file = out_path / f"{base_name}.parquet"

    # Write primary Parquet dataset
    df_processed.to_parquet(parquet_file, index=False)

    if export_csv:
        csv_file = out_path / f"{base_name}.csv"
        df_processed.to_csv(csv_file, index=False)

    return parquet_file


def save_quality_report(
    report: MarketDataValidationReport,
    output_dir: str | Path,
    filename: str = "data_quality_report.json",
) -> Path:
    """Save machine-readable validation quality report in JSON format."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_file = out_path / filename
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    return report_file


def generate_dataset_metadata(
    df_processed: pd.DataFrame,
    report: MarketDataValidationReport,
    broker: str = "MetaQuotes Ltd.",
    server: str = "MetaQuotes-Demo",
    pipeline_version: str = "1.0.0",
) -> dict[str, Any]:
    """Construct dataset metadata dictionary for experiment tracking and reproducibility."""
    now_utc = datetime.now(UTC).isoformat()
    git_commit = get_git_commit_hash()

    return {
        "symbol": report.symbol,
        "timeframe": report.timeframe,
        "broker": broker,
        "server": server,
        "source": report.source,
        "timestamp_convention": "Unix epoch seconds (MT5 raw)",
        "timezone": "UTC (timezone-aware datetime64[ns, UTC])",
        "source_columns": REQUIRED_RAW_COLUMNS,
        "processed_columns": list(df_processed.columns),
        "row_count": len(df_processed),
        "earliest_timestamp": report.earliest_timestamp_utc,
        "latest_timestamp": report.latest_timestamp_utc,
        "generation_timestamp": now_utc,
        "pipeline_version": pipeline_version,
        "git_commit": git_commit,
        "validation_status": report.validation_status,
    }


def save_dataset_metadata(
    metadata: dict[str, Any],
    output_dir: str | Path,
    filename: str = "dataset_metadata.json",
) -> Path:
    """Save dataset metadata to JSON format."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta_file = out_path / filename
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return meta_file
