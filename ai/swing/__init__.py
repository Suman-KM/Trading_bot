"""Swing trading research package for EURUSD H4 and D1 timeframes.

Provides chronological walk-forward validation, market regime analysis,
multi-timeframe M15 entry confirmation, data expansion utilities, and timing audits.
"""

from __future__ import annotations

from ai.swing.confirmation import align_m15_with_h4_decisions, evaluate_m15_confirmations
from ai.swing.data_expansion import (
    get_expanded_research_data,
    merge_and_validate_h4_chunks,
    validate_chunk_overlaps,
)
from ai.swing.evaluation import evaluate_walk_forward_candidate
from ai.swing.holdout import (
    HOLDOUT_END_TS,
    HOLDOUT_START_TS,
    PRE_HOLDOUT_RESEARCH_END_TS,
    PRE_HOLDOUT_RESEARCH_START_TS,
    ResearchHoldoutManager,
)
from ai.swing.regimes import analyze_h4_regimes
from ai.swing.timing_audit import generate_timing_audit_report
from ai.swing.walk_forward import WalkForwardFold, generate_chronological_walk_forward_folds
from ai.swing.walk_forward_cross_market import (
    evaluate_feature_ablation_experiment,
    evaluate_single_holdout,
    generate_pre_holdout_folds,
)
from ai.swing.walk_forward_expanded import (
    evaluate_expanded_walk_forward,
    generate_expanded_walk_forward_folds,
)

__all__ = [
    "HOLDOUT_END_TS",
    "HOLDOUT_START_TS",
    "PRE_HOLDOUT_RESEARCH_END_TS",
    "PRE_HOLDOUT_RESEARCH_START_TS",
    "ResearchHoldoutManager",
    "WalkForwardFold",
    "align_m15_with_h4_decisions",
    "analyze_h4_regimes",
    "evaluate_expanded_walk_forward",
    "evaluate_feature_ablation_experiment",
    "evaluate_m15_confirmations",
    "evaluate_single_holdout",
    "evaluate_walk_forward_candidate",
    "generate_chronological_walk_forward_folds",
    "generate_expanded_walk_forward_folds",
    "generate_pre_holdout_folds",
    "generate_timing_audit_report",
    "get_expanded_research_data",
    "merge_and_validate_h4_chunks",
    "validate_chunk_overlaps",
]
