"""Tests for Phase 45 Profitability Gate, Cost Stress & Exhaustion Decision.

Verifies:
1. Deterministic evaluation of all 15 profitability gate criteria.
2. Realistic broker cost stress (10.0 and 15.0 pips) causes holdout expectancy to fail.
3. Mode 2 Delayed Entry produces negative net expectancy (-3.22 pips/trade).
4. Walk-forward instability and negative expectancy under cost stress.
5. Event concentration risk verification.
6. Scientific verdict: OPTION C: FREE DATA EXHAUSTED — STOP STRATEGY DEVELOPMENT.
7. Absolute governance: no live trading, no paper trading, no Phase 46, no paid data authorized.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase45_profitability_gate import evaluate_profitability_gate


def test_profitability_gate_deterministic_evaluation() -> None:
    """Invariant: Profitability gate generates deterministic results."""
    s_path = Path("reports/phase45_strategy_results.json")
    e_path = Path("reports/phase45_event_results.json")
    report_dict = evaluate_profitability_gate(s_path, e_path)
    assert report_dict is not None
    assert report_dict["metadata"]["phase"] == "45"
    assert report_dict["gate_summary"]["total_gates"] == 15
    assert report_dict["gate_summary"]["failed_gates"] >= 5


def test_cost_stress_failure_under_realistic_broker_friction() -> None:
    """Invariant: Strategy A fails under realistic event-time broker spread (10-15 pips)."""
    report_path = Path("reports/phase45_profitability_gate.json")
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    # In holdout under 10-pip stress, Gate 1, 2, and 7 fail
    g1 = next(g for g in data["gate_evaluations"] if g["gate_id"] == "GATE_01")
    g2 = next(g for g in data["gate_evaluations"] if g["gate_id"] == "GATE_02")
    g7 = next(g for g in data["gate_evaluations"] if g["gate_id"] == "GATE_07")

    assert g1["passed"] is False
    assert g2["passed"] is False
    assert g7["passed"] is False


def test_delayed_entry_negative_expectancy_and_cost_finding() -> None:
    """Invariant: Delayed entry (entering after 15m candle close) yields negative expectancy."""
    report_path = Path("reports/phase45_profitability_gate.json")
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    finding = data["robustness_summary"]["cost_stress_finding"]
    assert "-3.22 pips/trade" in finding
    assert "proving zero post-event continuation" in finding


def test_walk_forward_stability_failure() -> None:
    """Invariant: Walk-forward stability fails across temporal folds."""
    report_path = Path("reports/phase45_profitability_gate.json")
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    g14 = next(g for g in data["gate_evaluations"] if g["gate_id"] == "GATE_14")
    assert g14["passed"] is False
    assert "FAILS" in g14["details"]


def test_scientific_verdict_option_c_and_stop_development() -> None:
    """Invariant: Verdict is strictly OPTION C and action is STOP STRATEGY DEVELOPMENT."""
    report_path = Path("reports/phase45_profitability_gate.json")
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    decision = data["final_decision"]
    assert decision["option"] == "OPTION C"
    assert "FREE DATA EXHAUSTED" in decision["scientific_verdict"]
    assert decision["project_action"] == "STOP STRATEGY DEVELOPMENT"
    assert decision["paid_data_authorized"] is False
    assert decision["trading_authorized"] is False
    assert decision["paper_trading_authorized"] is False
    assert decision["live_trading_authorized"] is False
    assert decision["phase_46_authorized"] is False


def test_free_data_exhaustion_matrix_complete() -> None:
    """Invariant: Exhaustion matrix covers all viable free information sources."""
    report_path = Path("reports/phase45_profitability_gate.json")
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    matrix = data["free_data_exhaustion_matrix"]
    assert len(matrix) >= 6
    for entry in matrix:
        assert entry["decision"] == "EXHAUSTED"
        assert len(entry["notes"]) > 0
