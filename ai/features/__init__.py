"""Feature engineering package for quantitative market data.

Provides modular, point-in-time feature extraction covering price action,
momentum, volatility, trend, activity, diurnal/session timing, and gap mechanics.
"""

from __future__ import annotations

from ai.features.activity import compute_activity_features
from ai.features.gaps import compute_gap_features
from ai.features.momentum import compute_momentum_features
from ai.features.pipeline import (
    MAX_LOOKBACK_BARS,
    FeatureDefinition,
    build_feature_pipeline,
    generate_feature_metadata,
    get_feature_registry,
)
from ai.features.price import compute_price_features
from ai.features.time import compute_time_features
from ai.features.trend import compute_trend_features
from ai.features.volatility import compute_volatility_features

__all__ = [
    "MAX_LOOKBACK_BARS",
    "FeatureDefinition",
    "build_feature_pipeline",
    "compute_activity_features",
    "compute_gap_features",
    "compute_momentum_features",
    "compute_price_features",
    "compute_time_features",
    "compute_trend_features",
    "compute_volatility_features",
    "generate_feature_metadata",
    "get_feature_registry",
]
