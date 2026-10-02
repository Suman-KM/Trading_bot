"""Unit tests for Phase 32: Final Research Synthesis & System Direction Gate.

Verifies:
- Both Phase 32 markdown document and JSON report exist and are valid.
- Complete research ledger covers historical phases from Phase 8 to Phase 31.
- Phase 31 verdict is strictly preserved as NOT SUPPORTED.
- Locked test partition remains quarantined and untouched.
- Zero new experiments, models trained, or backtests executed.
- Final governance decision is one of the three authorized options.
- All 17 required sections exist in the final documentation report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORT_PATH = Path("reports/phase32_final_research_synthesis.json")
DOC_PATH = Path("docs/phase-32-final-research-synthesis.md")


@pytest.fixture(scope="module")
def report_data() -> dict:
    """Load Phase 32 research synthesis JSON report."""
    assert REPORT_PATH.exists(), f"Missing synthesis report: {REPORT_PATH}"
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def doc_content() -> str:
    """Load Phase 32 research synthesis Markdown document."""
    assert DOC_PATH.exists(), f"Missing synthesis document: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


def test_required_files_exist():
    """Ensure both JSON report and Markdown synthesis documents exist."""
    assert REPORT_PATH.exists(), f"Report file missing: {REPORT_PATH}"
    assert DOC_PATH.exists(), f"Doc file missing: {DOC_PATH}"


def test_governance_invariants(report_data: dict):
    """Confirm zero new predictive experiments, models, or backtests were executed."""
    gov = report_data["governance"]
    assert gov["models_trained"] == 0
    assert gov["backtests_executed"] == 0
    assert gov["trading_simulations_executed"] == 0
    assert gov["live_trades_executed"] == 0
    assert gov["demo_trades_executed"] == 0
    assert gov["paper_trades_executed"] == 0
    assert gov["locked_test_partition_status"] == "QUARANTINED_AND_UNTOUCHED"
    assert "2026-02-19 12:00:00 UTC onward" in gov["locked_test_period"]


def test_research_ledger_completeness(report_data: dict):
    """Research ledger must cover all historical phases from Phase 8 through Phase 31."""
    ledger = report_data["research_ledger"]
    assert len(ledger) >= 23, f"Expected at least 23 ledger entries, got {len(ledger)}"

    phase_names = [entry["phase"] for entry in ledger]
    expected_phases = [
        "Phase 8",
        "Phase 9",
        "Phase 10",
        "Phase 11",
        "Phase 12",
        "Phase 13",
        "Phase 14",
        "Phase 15",
        "Phase 16",
        "Phase 17",
        "Phase 18",
        "Phase 19",
        "Phase 20",
        "Phase 21",
        "Phase 21.1",
        "Phase 22",
        "Phase 23",
        "Phase 24 & 24.1",
        "Phase 25",
        "Phase 26",
        "Phase 28",
        "Phase 29",
        "Phase 30",
        "Phase 31",
    ]
    for exp in expected_phases:
        assert exp in phase_names, f"Missing phase '{exp}' in research ledger!"


def test_phase31_verdict_preserved(report_data: dict):
    """Phase 31 verdict must be strictly preserved as NOT SUPPORTED."""
    p31_audit = report_data["phase31_interpretation"]
    assert p31_audit["verdict"] == "NOT SUPPORTED — VOLATILITY FORECASTING FAILED"
    assert p31_audit["success_condition_1_mean_ge_60"] is True
    assert p31_audit["success_condition_2_pval_lt_01"] is False
    assert p31_audit["paired_t_pvalue"] == 0.0321
    assert p31_audit["model_mean_balanced_accuracy"] == 0.6121
    assert p31_audit["persistence_baseline_mean_balanced_accuracy"] == 0.5799


def test_final_governance_decision_valid(report_data: dict):
    """Final decision must be one of the three pre-authorized options."""
    allowed_decisions = [
        "OPTION A: PAUSE FOR NEW INFORMATION / INFRASTRUCTURE",
        "OPTION B: PIVOT TO SYSTEM ENGINEERING / RISK / EXECUTION",
        "OPTION C: END CURRENT TRADING-ALPHA RESEARCH PROGRAM",
    ]
    decision = report_data["final_governance_decision"]
    assert decision in allowed_decisions, f"Invalid decision: '{decision}'"
    assert decision == "OPTION B: PIVOT TO SYSTEM ENGINEERING / RISK / EXECUTION"


def test_information_set_audit(report_data: dict):
    """Verify information set audit covers categories A through G."""
    info_audit = report_data["information_set_audit"]
    required_keys = [
        "category_a_price_derived_directional",
        "category_b_cross_market_price",
        "category_c_macro_yield",
        "category_d_cftc_positioning",
        "category_e_microstructure_execution",
        "category_f_volatility_regime",
        "category_g_unavailable_information",
    ]
    for k in required_keys:
        assert k in info_audit, f"Missing category '{k}' in information set audit!"
        assert len(info_audit[k]["empirical_verdict"]) > 0


def test_model_capacity_audit(report_data: dict):
    """Model capacity audit confirms complexity is not the primary limitation."""
    cap_audit = report_data["model_capacity_audit"]
    assert cap_audit["is_model_capacity_primary_limitation"] == "NO"
    assert len(cap_audit["findings"]) > 50


def test_execution_and_safety_core_audits(report_data: dict):
    """Execution and Safety Core audits confirm verified status and conceptual separation."""
    assert report_data["execution_audit"]["status"] == "INFRASTRUCTURE_VERIFIED_AND_RELIABLE"
    assert (
        report_data["safety_core_audit"]["status"]
        == "VERIFIED_CONCEPTUALLY_AND_PROGRAMMATICALLY_SEPARATED"
    )
    assert (
        report_data["safety_core_audit"]["architecture_flow"]
        == "AI / Research Signal -> Risk Engine -> Execution Engine -> Broker"
    )


def test_prohibited_actions_and_conditions_exist(report_data: dict):
    """Ensure prohibited actions and conditions for future research are present."""
    prohibited = report_data["prohibited_next_actions"]
    assert len(prohibited) >= 6
    assert any("train any new" in p.lower() for p in prohibited)
    assert any("backtest" in p.lower() for p in prohibited)

    conditions = report_data["conditions_required_before_future_predictive_research"]
    assert len(conditions) >= 3


def test_doc_structure_all_17_sections(doc_content: str):
    """Document must contain all 17 required sections."""
    required_sections = [
        "1. Executive Summary",
        "2. Repository State",
        "3. Complete Phase 8–31 Evidence Ledger",
        "4. Information-Set Audit",
        "5. Model-Capacity Audit",
        "6. Target and Objective Audit",
        "7. Execution Evidence Audit",
        "8. Safety Core Audit",
        "9. Locked-Test Integrity Audit",
        "10. Phase 31 Interpretation",
        "11. Data-Dredging / Repeated-Search Audit",
        "12. Remaining Information Gaps",
        "13. Possible Future System Directions",
        "14. Final Governance Decision",
        "15. Explicit List of Prohibited Next Actions",
        "16. Conditions Required Before Future Predictive Research",
        "17. Test and Repository Verification",
    ]
    for sec in required_sections:
        assert sec in doc_content, f"Missing required section '{sec}' in doc!"
