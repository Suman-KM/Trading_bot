"""Schema definitions and data models for exogenous information research.

Defines structural contracts, audit models, and point-in-time metadata
for macro yields, economic event releases, and order flow metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExogenousCategory(str, Enum):
    """Broad category of exogenous market information."""

    MACRO_YIELD = "MACRO_YIELD"
    ECONOMIC_EVENT = "ECONOMIC_EVENT"
    ORDER_FLOW = "ORDER_FLOW"


class FeasibilityClassification(str, Enum):
    """Scientific feasibility classification for an exogenous source."""

    SUPPORTED_FOR_FUTURE_RESEARCH = "SUPPORTED FOR FUTURE RESEARCH"
    PARTIALLY_SUPPORTED = "PARTIALLY SUPPORTED"
    NOT_FEASIBLE = "NOT FEASIBLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class RevisionHandlingPolicy(str, Enum):
    """Policy for handling historical revisions in external time series."""

    FIRST_RELEASE_ONLY = "FIRST_RELEASE_ONLY"  # Strict point-in-time initial publication
    VINTAGE_TRACKED = "VINTAGE_TRACKED"  # Multiple timestamped revisions preserved
    REVISED_BENCHMARK = "REVISED_BENCHMARK"  # Final revised figure (NOT point-in-time safe)
    NOT_APPLICABLE = "NOT_APPLICABLE"  # Fixed market transaction (e.g. bond yield)


@dataclass(frozen=True)
class DataSourceAudit:
    """Audit profile for an individual external data source."""

    source_name: str
    category: ExogenousCategory
    data_type: str
    historical_range: str
    timestamp_resolution: str
    timezone: str
    publication_time_available: bool
    expected_value_available: bool
    revision_history_tracked: bool
    point_in_time_safe: bool
    api_access_available: bool
    license_limitations: str
    data_quality_risks: list[str]
    feasibility: FeasibilityClassification

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit record to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ExogenousDataPoint:
    """Point-in-time atomic data observation."""

    series_id: str
    observation_period: str  # e.g. "2025-01", "2025-Q1", "2025-01-15"
    information_timestamp: datetime  # Exact UTC time when first publicly known
    feature_timestamp: datetime  # Assigned timestamp in feature pipeline
    first_usable_bar_timestamp: datetime  # Earliest bar close allowed to use value
    actual_value: float | None
    expected_value: float | None = None
    surprise_value: float | None = None
    revision_vintage: int = 1
    is_preliminary: bool = False
    source_attribution: str = ""

    def __post_init__(self) -> None:
        """Validate causal chronological invariants."""
        if self.information_timestamp.tzinfo is None:
            raise ValueError(f"information_timestamp for {self.series_id} must be timezone-aware.")
        if self.feature_timestamp.tzinfo is None:
            raise ValueError(f"feature_timestamp for {self.series_id} must be timezone-aware.")
        if self.first_usable_bar_timestamp.tzinfo is None:
            raise ValueError(
                f"first_usable_bar_timestamp for {self.series_id} must be timezone-aware."
            )
        if self.first_usable_bar_timestamp < self.information_timestamp:
            raise ValueError(
                f"Lookahead violation: first_usable_bar ({self.first_usable_bar_timestamp}) "
                f"precedes publication information_timestamp ({self.information_timestamp})."
            )


@dataclass(frozen=True)
class AlignmentConfig:
    """Configuration for mapping exogenous events onto bar timeframes."""

    bar_timeframe_minutes: int = 15
    bar_timestamp_is_open: bool = True  # True if bar timestamp represents open time
    latency_buffer_seconds: int = 60  # Buffer for transmission and processing latency
    max_lookback_bars: int = 96  # Maximum bar lookback for valid propagation (1 day on M15)
    require_strict_causality: bool = True


@dataclass(frozen=True)
class AlignmentResult:
    """Result of causal point-in-time alignment."""

    total_bars: int
    aligned_bars: int
    missing_bars: int
    lookahead_violations: int
    earliest_aligned: str | None
    latest_aligned: str | None
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if alignment strictly passed without lookahead violations."""
        return self.lookahead_violations == 0 and len(self.errors) == 0
