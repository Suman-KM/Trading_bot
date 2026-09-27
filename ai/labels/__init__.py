"""Target label engineering package for supervised machine learning.

Provides forward-looking return calculations, multi-scale prediction horizons,
ternary directional classification targets, and boundary metadata.
"""

from __future__ import annotations

from ai.labels.direction import DEFAULT_FIXED_THRESHOLDS, compute_direction_labels
from ai.labels.pipeline import (
    LabelDefinition,
    build_label_pipeline,
    generate_label_metadata,
    get_label_registry,
)
from ai.labels.returns import DEFAULT_HORIZONS, compute_future_returns

__all__ = [
    "DEFAULT_FIXED_THRESHOLDS",
    "DEFAULT_HORIZONS",
    "LabelDefinition",
    "build_label_pipeline",
    "compute_direction_labels",
    "compute_future_returns",
    "generate_label_metadata",
    "get_label_registry",
]
