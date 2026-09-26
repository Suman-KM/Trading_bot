"""Market data schema definition and validation.

Strictly validates the MetaTrader 5 raw OHLCV schema, ensuring proper column presence,
type consistency, and absence of accidental string/object columns.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

REQUIRED_RAW_COLUMNS: list[str] = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]

NUMERIC_PRICE_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
]

INTEGER_COLUMNS: list[str] = [
    "time",
    "tick_volume",
    "spread",
    "real_volume",
]


@dataclass(frozen=True)
class SchemaValidationResult:
    """Result of schema integrity validation."""

    is_valid: bool
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    dtype_issues: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary representation."""
        return asdict(self)


def validate_schema(df: pd.DataFrame, strict: bool = True) -> SchemaValidationResult:
    """Validate that a DataFrame conforms to the expected MT5 raw schema.

    Parameters
    ----------
    df : pd.DataFrame
        Market data DataFrame to validate.
    strict : bool, default True
        If True, flags unexpected columns as schema issues.

    Returns
    -------
    SchemaValidationResult
        Detailed validation status including missing, unexpected, and type issues.
    """
    errors: list[str] = []
    missing_columns: list[str] = []
    unexpected_columns: list[str] = []
    dtype_issues: list[str] = []

    if df.empty:
        errors.append("DataFrame is empty.")
        return SchemaValidationResult(
            is_valid=False,
            missing_columns=REQUIRED_RAW_COLUMNS.copy(),
            unexpected_columns=[],
            dtype_issues=[],
            errors=errors,
        )

    # 1. Check required columns
    present_columns = set(df.columns)
    for col in REQUIRED_RAW_COLUMNS:
        if col not in present_columns:
            missing_columns.append(col)
            errors.append(f"Missing required column: '{col}'")

    # 2. Check unexpected columns
    if strict:
        for col in df.columns:
            if col not in REQUIRED_RAW_COLUMNS:
                unexpected_columns.append(str(col))

    # 3. Check data types for available columns
    for col in present_columns:
        dtype = df[col].dtype
        if col in NUMERIC_PRICE_COLUMNS:
            if not pd.api.types.is_float_dtype(dtype):
                dtype_issues.append(f"Column '{col}' expected float dtype, got {dtype}")
        elif col in INTEGER_COLUMNS and not pd.api.types.is_integer_dtype(dtype):
            dtype_issues.append(f"Column '{col}' expected integer dtype, got {dtype}")

        # Check for non-numeric types
        if not pd.api.types.is_numeric_dtype(dtype):
            dtype_issues.append(f"Column '{col}' has non-numeric dtype ({dtype}); expected numeric")

    is_valid = len(missing_columns) == 0 and len(dtype_issues) == 0 and len(errors) == 0

    return SchemaValidationResult(
        is_valid=is_valid,
        missing_columns=missing_columns,
        unexpected_columns=unexpected_columns,
        dtype_issues=dtype_issues,
        errors=errors,
    )
