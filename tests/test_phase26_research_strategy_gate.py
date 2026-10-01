"""Unit tests for Phase 26: Research Strategy Pivot & Evidence Gate.

Verifies:
1. Integrity and schema validity of the Phase 26 research decision gate report.
2. Complete research ledger covering all 19 research iterations (Phases 8 through 25).
3. Systematic bottleneck taxonomy categorization and severity ratings.
4. Information quality audit categorization (Available, Partially Available, Not Available).
5. Candidate hypotheses count (<= 3), schema, and factual feasibility classification.
6. Decision gate value is strictly one of the 3 authorized options
   ("PAUSE FOR NEW DATA / INFRASTRUCTURE").
7. Permanent holdout quarantine and governance integrity assertions.
8. Phase 26 documentation existence and structural section completeness.
"""

from __future__ import annotations

import json
from pathlib import Path

REPORT_PATH = Path("reports/phase26_research_strategy_gate.json")
DOC_PATH = Path("docs/phase-26-research-strategy-pivot.md")

ALLOWED_DECISIONS = {
    "PROCEED TO A CONTROLLED NEW EXPERIMENT",
    "PAUSE FOR NEW DATA / INFRASTRUCTURE",
    "STOP RESEARCH UNDER CURRENT INFORMATION SET",
}

ALLOWED_FEASIBILITY_STATUSES = {
    "FEASIBLE NOW",
    "FEASIBLE WITH NEW DATA",
    "NOT FEASIBLE",
}

EXPECTED_PHASES = [
    "Phase 8",
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
    "Phase 24",
    "Phase 24.1",
    "Phase 25",
]

EXPECTED_BOTTLENECK_KEYS = {
    "A_INFORMATION",
    "B_LABEL_TARGET",
    "C_MARKET_MICROSTRUCTURE",
    "D_EXECUTION_COST",
    "E_MODEL_CAPACITY",
    "F_DATA_QUALITY",
    "G_REGIME_NON_STATIONARITY",
    "H_SAMPLE_SIZE",
}


def test_phase26_report_file_exists():
    """Verify that reports/phase26_research_strategy_gate.json exists and is valid JSON."""
    assert REPORT_PATH.exists(), f"Missing Phase 26 report: {REPORT_PATH}"
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert data["phase"] == "Phase 26"
    assert data["title"] == "Research Strategy Pivot & Evidence Gate"


def test_research_ledger_completeness():
    """Verify that all historical research phases (8-25) are recorded with full metadata."""
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    ledger = data.get("research_ledger", [])
    expected_len = len(EXPECTED_PHASES)
    assert len(ledger) == expected_len, f"Expected {expected_len} phases, got {len(ledger)}"

    recorded_phases = [entry["phase"] for entry in ledger]
    assert recorded_phases == EXPECTED_PHASES

    required_entry_fields = {
        "phase",
        "question",
        "data_source",
        "timeframe",
        "target",
        "models",
        "validation_design",
        "primary_metric",
        "observed_result",
        "failure_mode",
        "what_learned",
        "status",
    }
    for entry in ledger:
        missing = required_entry_fields - set(entry.keys())
        assert not missing, f"Phase {entry.get('phase')} missing fields: {missing}"
        assert len(entry["question"]) > 10
        assert len(entry["what_learned"]) > 10
        assert len(entry["observed_result"]) > 10


def test_bottleneck_taxonomy():
    """Verify that bottleneck taxonomy covers all 8 structural categories."""
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    taxonomy = data.get("bottleneck_taxonomy", {})
    assert set(taxonomy.keys()) == EXPECTED_BOTTLENECK_KEYS

    for key, item in taxonomy.items():
        assert "severity" in item, f"Missing severity for {key}"
        assert "evidence" in item, f"Missing evidence for {key}"
        assert "tested_in" in item, f"Missing tested_in for {key}"
        assert len(item["evidence"]) > 20
        assert len(item["tested_in"]) > 0


def test_information_quality_audit():
    """Verify that information quality audit categorizes all asset data sources."""
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    audit = data.get("information_quality_audit", {})
    assert "AVAILABLE" in audit
    assert "PARTIALLY_AVAILABLE" in audit
    assert "NOT_AVAILABLE" in audit

    assert len(audit["AVAILABLE"]) >= 4
    assert len(audit["PARTIALLY_AVAILABLE"]) >= 2
    assert len(audit["NOT_AVAILABLE"]) >= 3


def test_candidate_hypotheses_integrity():
    """Verify candidate hypotheses count (<=3), schemas, and objective classifications."""
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    hypotheses = data.get("candidate_hypotheses", [])

    assert 1 <= len(hypotheses) <= 3, f"Expected 1 to 3 candidate hypotheses, got {len(hypotheses)}"

    required_fields = {
        "id",
        "title",
        "hypothesis",
        "why_different",
        "required_data",
        "timeframe",
        "target",
        "feasibility_status",
        "falsification_condition",
    }

    for hyp in hypotheses:
        missing = required_fields - set(hyp.keys())
        assert not missing, f"Hypothesis {hyp.get('id')} missing fields: {missing}"
        assert hyp["feasibility_status"] in ALLOWED_FEASIBILITY_STATUSES, (
            f"Invalid feasibility status {hyp['feasibility_status']} in {hyp['id']}"
        )
        # Ensure no subjective ranking fields exist
        assert "rank" not in hyp, f"Subjective ranking field 'rank' found in {hyp['id']}"
        assert "score" not in hyp, f"Subjective ranking field 'score' found in {hyp['id']}"


def test_decision_gate_strictness():
    """Verify decision gate choice conforms to allowable enum and is strictly PAUSE."""
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    decision = data.get("decision_gate")

    assert decision in ALLOWED_DECISIONS, (
        f"Decision '{decision}' not in allowed set {ALLOWED_DECISIONS}"
    )
    assert decision == "PAUSE FOR NEW DATA / INFRASTRUCTURE"


def test_governance_and_quarantine_integrity():
    """Verify permanent holdout locks and zero-training assertions in Phase 26."""
    data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    gov = data.get("governance_verification", {})

    assert "2026-02-19 12:00:00 UTC" in gov.get("phase11_test_partition", "")
    assert gov.get("phase15_holdouts") == "LOCKED"
    assert gov.get("phase18_holdout") == "LOCKED"
    assert gov.get("phase25_holdout") == "LOCKED"
    assert gov.get("zero_model_training_in_phase26") is True
    assert gov.get("zero_feature_mining_in_phase26") is True
    assert gov.get("zero_trading_execution") is True


def test_documentation_file_and_sections():
    """Verify docs/phase-26-research-strategy-pivot.md exists and contains all required sections."""
    assert DOC_PATH.exists(), f"Missing doc file: {DOC_PATH}"
    content = DOC_PATH.read_text(encoding="utf-8")

    required_sections = [
        "## 1. Executive Summary & Purpose",
        "## 2. Complete Research Ledger",
        "## 3. Systematic Identification of the Bottleneck",
        "## 4. Tested vs. Untested Matrix",
        "## 5. Information Quality Audit",
        "## 6. Candidate Next Hypotheses",
        "## 7. Factual Feasibility Classification",
        "## 8. Requirements for Any Authorized Future Experiment",
        "## 9. Decision Gate",
        "## 10. Specification of the Next Data Requirement",
        "## 11. Scientific Interpretation & Scope",
        "## 12. Absolute Stop Condition",
    ]

    for section in required_sections:
        assert section in content, f"Missing section '{section}' in {DOC_PATH}"
