"""Swing trading research package for EURUSD H4 and D1 timeframes.

Provides chronological walk-forward validation, market regime analysis,
multi-timeframe M15 entry confirmation, and strict point-in-time timing audits.
"""

from __future__ import annotations

from ai.swing.confirmation import align_m15_with_h4_decisions, evaluate_m15_confirmations
from ai.swing.evaluation import evaluate_walk_forward_candidate
from ai.swing.regimes import analyze_h4_regimes
from ai.swing.timing_audit import generate_timing_audit_report
from ai.swing.walk_forward import WalkForwardFold, generate_chronological_walk_forward_folds

__all__ = [
    "WalkForwardFold",
    "align_m15_with_h4_decisions",
    "analyze_h4_regimes",
    "evaluate_m15_confirmations",
    "evaluate_walk_forward_candidate",
    "generate_chronological_walk_forward_folds",
    "generate_timing_audit_report",
]
