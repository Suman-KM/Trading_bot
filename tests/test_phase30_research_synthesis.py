"""Unit tests for Phase 30 Exhaustive Research Synthesis & Final Directional-Alpha Gate.

Verifies:
1. Synthesis report and documentation files exist.
2. Complete research ledger covers all 23 historical phases (Phases 8–29).
3. Information-set audit covers Categories A through F.
4. Bottleneck audit covers all 10 potential bottlenecks with factual justification.
5. Model-capacity audit confirms capacity is NOT the primary limitation.
6. Execution audit confirms simulation infrastructure is sound and collision-free.
7. COT audit accurately reflects Phase 29 empirical failure.
8. Scientific justification gate verifies all 8 criteria for volatility hypothesis.
9. Final decision is OPTION A with directional alpha permanently halted.
10. Future experiment specification for Candidate Hypothesis 2 is complete.
11. Governance constraints strictly satisfied: zero models trained, zero backtests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORT_PATH = Path("reports/phase30_research_synthesis.json")
DOC_PATH = Path("docs/phase-30-research-synthesis.md")


@pytest.fixture(scope="module")
def report_data() -> dict:
    """Load Phase 30 research synthesis JSON report."""
    assert REPORT_PATH.exists(), f"Missing synthesis report: {REPORT_PATH}"
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def doc_content() -> str:
    """Load Phase 30 research synthesis Markdown document."""
    assert DOC_PATH.exists(), f"Missing synthesis document: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


def test_required_files_exist():
    """1. Ensure both JSON report and Markdown synthesis documents exist."""
    assert REPORT_PATH.exists(), f"Report file missing: {REPORT_PATH}"
    assert DOC_PATH.exists(), f"Doc file missing: {DOC_PATH}"


def test_research_ledger_completeness(report_data: dict):
    """2. Research ledger must cover all historical phases from Phase 8 to Phase 29."""
    ledger = report_data["research_ledger"]
    assert len(ledger) >= 20, f"Expected at least 20 ledger entries, got {len(ledger)}"

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
    ]
    for exp in expected_phases:
        assert exp in phase_names, f"Missing phase '{exp}' in research ledger!"

    # Ensure distinctions between FAILED, UNTESTED, and INFEASIBLE hypotheses exist
    classifications = {entry["hypothesis_classification"] for entry in ledger}
    assert any("FAILED" in c for c in classifications)
    assert any("INFEASIBLE" in c for c in classifications)


def test_information_set_audit(report_data: dict):
    """3. Information-set audit covers Categories A through F."""
    info_audit = report_data["information_set_audit"]
    required_cats = [
        "category_a_price_derived",
        "category_b_cross_market",
        "category_c_macro_yields",
        "category_d_institutional_positioning",
        "category_e_microstructure_execution",
        "category_f_unavailable_information",
    ]
    for cat in required_cats:
        assert cat in info_audit, f"Missing category '{cat}' in information-set audit!"
        assert len(info_audit[cat]["description"]) > 0
        assert len(info_audit[cat]["empirical_result"]) > 0


def test_bottleneck_audit(report_data: dict):
    """4. Bottleneck audit covers all 10 bottlenecks and identifies Information Deficiency."""
    bottlenecks = report_data["bottleneck_audit"]
    required_bottlenecks = [
        "data_quality",
        "data_coverage",
        "leakage",
        "model_complexity",
        "sample_size",
        "execution_realism",
        "transaction_costs",
        "target_formulation",
        "feature_quality",
        "information_deficiency",
    ]
    for bn in required_bottlenecks:
        assert bn in bottlenecks, f"Missing bottleneck '{bn}'!"
        assert "status" in bottlenecks[bn]
        assert "evidence" in bottlenecks[bn]

    assert "NOT BOTTLENECK" in bottlenecks["data_quality"]["status"]
    assert "NOT BOTTLENECK" in bottlenecks["data_coverage"]["status"]
    assert "NOT BOTTLENECK" in bottlenecks["leakage"]["status"]
    assert "PRIMARY" in bottlenecks["information_deficiency"]["status"]


def test_model_capacity_audit(report_data: dict):
    """5. Model-capacity audit confirms architecture is NOT the primary limitation."""
    cap_audit = report_data["model_capacity_audit"]
    assert cap_audit["is_model_capacity_primary_limitation"] == "NO"
    assert len(cap_audit["rationale"]) > 50


def test_execution_audit(report_data: dict):
    """6. Execution audit confirms simulation engine is verified and collision-free."""
    exec_audit = report_data["execution_audit"]
    assert exec_audit["status"] == "INFRASTRUCTURE_VERIFIED_RESOLVED"
    assert "TickRealisticExecutionEngine" in exec_audit["findings"]


def test_cot_audit(report_data: dict):
    """7. COT audit accurately records Phase 29 metrics and failure."""
    cot_audit = report_data["cot_audit"]
    assert cot_audit["status"] == "FAILED_HYPOTHESIS"
    assert cot_audit["metrics"]["H10"]["cot_bal_acc"] == 0.5300
    assert cot_audit["metrics"]["H10"]["p_value_paired_t"] == 0.5710
    assert cot_audit["metrics"]["H20"]["cot_bal_acc"] == 0.4959
    assert cot_audit["metrics"]["H20"]["p_value_paired_t"] == 0.9308


def test_scientific_justification_gate(report_data: dict):
    """8. Scientific justification gate verifies all 8 criteria passed."""
    gate = report_data["scientific_justification_gate"]
    assert gate["all_criteria_passed"] is True

    for i in range(1, 9):
        key = [k for k in gate if k.startswith(f"criterion_{i}")][0]
        assert gate[key]["passed"] is True, f"Criterion {i} failed!"


def test_final_decision_and_directional_status(report_data: dict):
    """9. Final decision is Option A, with directional alpha research permanently halted."""
    expected_dec = "OPTION A: AUTHORIZE ONE CONTROLLED VOLATILITY EXPERIMENT"
    assert report_data["final_decision"] == expected_dec
    assert (
        report_data["directional_alpha_status"]
        == "PERMANENTLY_HALTED_UNDER_CURRENT_INFORMATION_SET"
    )


def test_future_volatility_experiment_specification(report_data: dict):
    """10. Future experiment specification is fully defined."""
    spec = report_data["future_volatility_experiment_specification"]
    assert "Parkinson" in str(spec["feature_set"])
    assert spec["timeframe"] == "H4"
    assert "RandomForestClassifier" in spec["model_family"]
    assert spec["success_gate"] is not None
    assert spec["failure_gate"] is not None


def test_governance_enforcement(report_data: dict):
    """11. Governance confirms zero models trained, zero backtests executed in Phase 30."""
    gov = report_data["governance"]
    assert gov["experiment_executed"] is False
    assert gov["models_trained"] == 0
    assert gov["backtests_executed"] == 0
    assert gov["live_or_demo_trades"] == 0
    assert gov["locked_test_partition_status"] == "QUARANTINED_AND_UNTOUCHED"


def test_doc_structure_and_sections(doc_content: str):
    """12. Document contains all 12 required sections."""
    required_sections = [
        "1. Executive Summary",
        "2. Complete Research Ledger",
        "3. Information-Set Audit",
        "4. Bottleneck Audit",
        "5. Model-Capacity Audit",
        "6. Execution Audit",
        "7. CFTC COT Audit",
        "8. Remaining Hypothesis",
        "9. Scientific Justification Gate",
        "10. Final Decision",
        "11. Pre-Specification of Future Phase 31 Experiment",
        "12. Governance Statement",
    ]
    for sec in required_sections:
        assert sec in doc_content, f"Missing section '{sec}' in documentation!"
