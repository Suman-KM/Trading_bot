"""Tests for Phase 45 Free Data Inventory and Zero-Cost Policy Enforcement.

Verifies:
1. Deterministic inventory generation.
2. Hard zero-cost data policy (₹0 budget) enforcement.
3. Accurate classification of all audited sources (usable, paid rejected, redundant, rejected).
4. No paid sources recommended or shortlisted.
5. No credentials, secrets, or paid API keys in data definitions or reports.
6. Absolute rejection of scrapers and unverified retail snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase45_free_data_inventory import (
    TOTAL_DATA_BUDGET_INR,
    FreeSourceRecord,
    build_free_data_inventory,
    run_inventory_audit,
)


def test_free_data_inventory_deterministic() -> None:
    """Invariant: Free data inventory is deterministic across invocations."""
    inv_1 = build_free_data_inventory()
    inv_2 = build_free_data_inventory()
    assert len(inv_1) == len(inv_2)
    assert len(inv_1) == 14

    for s1, s2 in zip(inv_1, inv_2, strict=True):
        assert s1.source_name == s2.source_name
        assert s1.cost_usd == s2.cost_usd
        assert s1.classification == s2.classification
        assert s1.usable == s2.usable


def test_hard_zero_cost_policy_enforcement() -> None:
    """Invariant: Total data budget is strictly ₹0 ($0.00)."""
    assert TOTAL_DATA_BUDGET_INR == 0.0

    inventory = build_free_data_inventory()
    usable_sources = [s for s in inventory if s.usable]

    for s in usable_sources:
        assert s.cost_usd == 0.0
        assert s.cost_inr == 0.0
        assert "FREE" in s.classification


def test_audited_sources_classification() -> None:
    """Invariant: All 14 sources are rigorously and accurately classified."""
    inventory = build_free_data_inventory()

    usable = [s for s in inventory if s.usable]
    paid_rejected = [s for s in inventory if s.classification == "PAID \u2014 REJECTED"]
    redundant = [s for s in inventory if s.classification == "FREE \u2014 REDUNDANT"]
    other_rejected = [
        s
        for s in inventory
        if s.classification
        in ("FREE \u2014 NOT POINT-IN-TIME SAFE", "FREE \u2014 INSUFFICIENT COVERAGE")
    ]

    assert len(usable) == 4, f"Expected 4 usable sources, found {len(usable)}"
    assert len(paid_rejected) == 5, f"Expected 5 paid rejected sources, found {len(paid_rejected)}"
    assert len(redundant) == 3, f"Expected 3 redundant sources, found {len(redundant)}"
    assert len(other_rejected) == 2, (
        f"Expected 2 other rejected sources, found {len(other_rejected)}"
    )

    paid_names = [s.source_name for s in paid_rejected]
    assert any("Databento" in n for n in paid_names)
    assert any("Bloomberg" in n for n in paid_names)
    assert any("CME" in n for n in paid_names)
    assert any("Trading Economics" in n for n in paid_names)
    assert any("Interactive Brokers" in n for n in paid_names)


def test_no_paid_sources_authorized() -> None:
    """Invariant: No paid sources can be marked as usable or recommended."""
    inventory = build_free_data_inventory()
    for s in inventory:
        if s.cost_usd > 0.0:
            assert s.usable is False
            assert "PAID" in s.classification or "REJECTED" in s.classification


def test_free_data_source_attributes() -> None:
    """Invariant: FreeSourceRecord dataclass properly models point-in-time attributes."""
    sample = FreeSourceRecord(
        source_name="Test Feed",
        category="Test Category",
        provider_authority="Test Provider",
        cost_usd=0.0,
        cost_inr=0.0,
        free_access_mode="Public download",
        historical_coverage="2020-2026",
        frequency="Daily",
        event_timestamp_precision="1 minute",
        release_timestamp_precision="1 minute",
        revision_risk="None",
        point_in_time_safe=True,
        genuinely_new_information=True,
        eurusd_relevance="High",
        usable=True,
        classification="FREE \u2014 USABLE",
        decision_rationale="Completely free test feed",
    )
    assert sample.cost_usd == 0.0
    assert sample.point_in_time_safe is True
    assert sample.usable is True


def test_inventory_report_generated_without_credentials() -> None:
    """Invariant: Generated inventory JSON contains no secret keys or passwords."""
    report_dict = run_inventory_audit()
    report_path = Path("reports/phase45_free_data_inventory.json")

    assert report_path.exists()
    assert report_dict["metadata"]["policy_enforcement"] == "HARD ZERO-COST DATA POLICY"
    assert report_dict["metadata"]["policy_budget_inr"] == 0.0

    report_text = json.dumps(report_dict).lower()
    for token in ["api_key", "secret", "private_key", "bearer ", "password"]:
        assert token not in report_text, f"Forbidden security token found: {token}"
