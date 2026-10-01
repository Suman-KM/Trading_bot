"""Exogenous data feasibility and research infrastructure package."""

from ai.data.exogenous.schema import (
    AlignmentConfig,
    AlignmentResult,
    DataSourceAudit,
    ExogenousCategory,
    ExogenousDataPoint,
    FeasibilityClassification,
    RevisionHandlingPolicy,
)
from ai.data.exogenous.yields import (
    RawSeriesValidation,
    YieldProvenance,
    align_yields_to_eurusd_h4,
    load_bundesbank_2y,
    load_fred_german_yield,
    load_fred_us_2y,
    reconcile_yield_calendars,
    validate_raw_series,
)

__all__ = [
    "AlignmentConfig",
    "AlignmentResult",
    "DataSourceAudit",
    "ExogenousCategory",
    "ExogenousDataPoint",
    "FeasibilityClassification",
    "RawSeriesValidation",
    "RevisionHandlingPolicy",
    "YieldProvenance",
    "align_yields_to_eurusd_h4",
    "load_bundesbank_2y",
    "load_fred_german_yield",
    "load_fred_us_2y",
    "reconcile_yield_calendars",
    "validate_raw_series",
]
