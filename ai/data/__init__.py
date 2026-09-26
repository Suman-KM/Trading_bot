"""Historical market data acquisition, validation, and ingestion pipeline.

Member 2 quantitative data package responsible for immutable raw data handling,
schema verification, UTC timestamp normalization, OHLC integrity checks,
gap classification, and dataset metadata management.
"""

from ai.data.gaps import GapAnalysisResult, GapItem, analyze_gaps
from ai.data.ingestion import (
    generate_dataset_metadata,
    load_raw_dataset,
    process_and_validate,
    save_dataset_metadata,
    save_processed_dataset,
    save_quality_report,
)
from ai.data.schema import (
    INTEGER_COLUMNS,
    NUMERIC_PRICE_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    SchemaValidationResult,
    validate_schema,
)
from ai.data.timestamps import (
    TimestampValidationResult,
    add_utc_timestamp,
    validate_timestamps,
)
from ai.data.validation import (
    MarketDataValidationReport,
    OHLCValidationResult,
    SpreadValidationResult,
    VolumeValidationResult,
    validate_market_data,
    validate_ohlc,
    validate_spread,
    validate_volume,
)

__all__ = [
    "INTEGER_COLUMNS",
    "NUMERIC_PRICE_COLUMNS",
    "REQUIRED_RAW_COLUMNS",
    "GapAnalysisResult",
    "GapItem",
    "MarketDataValidationReport",
    "OHLCValidationResult",
    "SchemaValidationResult",
    "SpreadValidationResult",
    "TimestampValidationResult",
    "VolumeValidationResult",
    "add_utc_timestamp",
    "analyze_gaps",
    "generate_dataset_metadata",
    "load_raw_dataset",
    "process_and_validate",
    "save_dataset_metadata",
    "save_processed_dataset",
    "save_quality_report",
    "validate_market_data",
    "validate_ohlc",
    "validate_schema",
    "validate_spread",
    "validate_timestamps",
    "validate_volume",
]
