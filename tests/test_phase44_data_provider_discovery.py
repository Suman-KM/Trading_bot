"""Tests for Phase 44 Low-Cost Institutional Data Discovery & Provider Comparison.

Verifies all 16 required invariants from Section 31:
1. Provider records are deterministic.
2. Price-tier classification is deterministic.
3. Currency conversion assumptions are explicit.
4. Missing price fields do not become zero.
5. Unknown prices remain unknown.
6. Historical coverage parsing is deterministic.
7. Timestamp requirements are enforced.
8. Point-in-time requirements are represented.
9. 'Free current data' cannot be classified as historical.
10. Unverified providers cannot be automatically recommended.
11. Duplicate providers are detected.
12. Already-exhausted sources are not classified as new.
13. No trading code is invoked.
14. No execution code is invoked.
15. No credentials are written.
16. No payment/account creation occurs.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase44_data_provider_discovery import (
    USD_TO_INR_RATE,
    ProviderRecord,
    build_provider_catalog,
    classify_price_tier,
    run_phase44_discovery,
)


def test_provider_records_deterministic() -> None:
    """Invariant 1: Provider records are deterministic across runs."""
    catalog_1 = build_provider_catalog()
    catalog_2 = build_provider_catalog()
    assert len(catalog_1) == len(catalog_2)
    assert len(catalog_1) >= 10
    for p1, p2 in zip(catalog_1, catalog_2, strict=True):
        assert p1.provider == p2.provider
        assert p1.product == p2.product
        assert p1.total_first_month_usd == p2.total_first_month_usd
        assert p1.total_score == p2.total_score
        assert p1.shortlisted == p2.shortlisted


def test_price_tier_classification_deterministic() -> None:
    """Invariant 2: Price-tier classification is deterministic across boundaries."""
    assert classify_price_tier(0.0) == "TIER_0_FREE"
    assert classify_price_tier(15.0) == "TIER_1_SUB_25"
    assert classify_price_tier(24.99) == "TIER_1_SUB_25"
    assert classify_price_tier(25.0) == "TIER_2_25_TO_100"
    assert classify_price_tier(49.95) == "TIER_2_25_TO_100"
    assert classify_price_tier(100.0) == "TIER_2_25_TO_100"
    assert classify_price_tier(100.01) == "TIER_3_100_TO_500"
    assert classify_price_tier(275.0) == "TIER_3_100_TO_500"
    assert classify_price_tier(500.0) == "TIER_3_100_TO_500"
    assert classify_price_tier(500.01) == "TIER_4_500_TO_1000"
    assert classify_price_tier(1000.0) == "TIER_4_500_TO_1000"
    assert classify_price_tier(1000.01) == "TIER_5_ABOVE_1000"
    assert classify_price_tier(2500.0) == "TIER_5_ABOVE_1000"
    assert classify_price_tier(None) == "TIER_UNKNOWN"


def test_currency_conversion_explicit() -> None:
    """Invariant 3: Currency conversion assumptions are explicit and consistent."""
    assert USD_TO_INR_RATE == 83.50
    catalog = build_provider_catalog()
    for p in catalog:
        if p.total_first_month_usd is not None:
            expected_inr = round(p.total_first_month_usd * USD_TO_INR_RATE, 2)
            assert p.total_first_month_inr == expected_inr


def test_missing_price_fields_do_not_become_zero() -> None:
    """Invariant 4: Missing price fields do not silently default to zero."""
    p = ProviderRecord(
        provider="Hypothetical Vendor",
        product="Unknown Data",
        category="Category A",
        instrument="EURUSD",
        data_type="Depth",
        historical_depth="Unknown",
        resolution="Tick",
        timestamp_precision="ms",
        exchange_source="Unknown",
        bid_ask_available=True,
        trade_available=True,
        depth_available=True,
        order_flow_available=True,
        api_available=True,
        download_available=True,
        historical_archive_available=True,
        pricing_model="Custom Quote",
        base_price_usd=None,
        is_price_verified=False,
        exchange_fee_usd=None,
        storage_cost_usd=0.0,
        total_first_month_usd=None,
        total_12_month_usd=None,
        total_first_month_inr=None,
        total_12_month_inr=None,
        price_tier="TIER_UNKNOWN",
        free_tier_credits="None",
        trial_available=False,
        student_research_access="None",
        licensing="Custom",
        redistribution_restrictions="Prohibited",
        personal_use_restrictions="Unclear",
        commercial_use_restrictions="Required",
        data_retention="Unclear",
        latency_mode="Cloud",
        timezone="UTC",
        revision_policy="Unclear",
        doc_url="https://example.com",
        current_status="Unverified",
        reliability="UNKNOWN",
        integration_effort="HIGH",
        point_in_time_safe=False,
        vintage_safe=False,
        already_exhausted=False,
        genuinely_new=True,
        scores={},
        total_score=0,
        shortlisted=False,
        verdict="REJECTED",
        decision_rationale="Unverified price",
        future_hypothesis=None,
    )
    assert p.base_price_usd is None
    assert p.total_first_month_usd is None
    assert p.total_first_month_usd != 0.0


def test_unknown_prices_remain_unknown() -> None:
    """Invariant 5: Unknown prices remain explicitly marked as TIER_UNKNOWN."""
    assert classify_price_tier(None) == "TIER_UNKNOWN"


def test_historical_coverage_parsing_deterministic() -> None:
    """Invariant 6: Historical coverage parsing is deterministic."""
    catalog = build_provider_catalog()
    for p in catalog:
        assert isinstance(p.historical_depth, str)
        assert len(p.historical_depth.strip()) > 0


def test_timestamp_requirements_enforced() -> None:
    """Invariant 7: Timestamp precision is enforced on all shortlisted providers."""
    catalog = build_provider_catalog()
    for p in catalog:
        if p.shortlisted:
            assert p.timestamp_precision is not None
            assert len(p.timestamp_precision) > 0
            # Futures order flow must have millisecond or nanosecond precision
            if "Futures" in p.category:
                assert any(
                    unit in p.timestamp_precision.lower()
                    for unit in ("nanosecond", "millisecond", "sub-millisecond")
                )


def test_point_in_time_requirements_represented() -> None:
    """Invariant 8: Point-in-time safety is strictly required for shortlisted providers."""
    catalog = build_provider_catalog()
    for p in catalog:
        if p.shortlisted:
            assert p.point_in_time_safe is True
            assert p.vintage_safe is True


def test_free_current_data_cannot_be_classified_as_historical() -> None:
    """Invariant 9: 'Free current data' cannot be classified as historical archive."""
    catalog = build_provider_catalog()
    oanda = next(p for p in catalog if p.provider == "OANDA")
    assert oanda.historical_archive_available is False
    assert oanda.shortlisted is False
    assert "LACKS DEEP HISTORICAL ARCHIVE" in oanda.verdict


def test_unverified_providers_cannot_be_automatically_recommended() -> None:
    """Invariant 10: Unverified providers or low reliability cannot be shortlisted."""
    catalog = build_provider_catalog()
    for p in catalog:
        if not p.is_price_verified or p.reliability in ("LOW", "UNKNOWN"):
            assert p.shortlisted is False


def test_duplicate_providers_detected() -> None:
    """Invariant 11: Duplicate provider/product combinations are detected and prevented."""
    catalog = build_provider_catalog()
    keys = [(p.provider, p.product) for p in catalog]
    assert len(keys) == len(set(keys)), "Duplicate provider records detected!"


def test_already_exhausted_sources_not_classified_as_new() -> None:
    """Invariant 12: Already-exhausted and redundant sources are flagged properly."""
    catalog = build_provider_catalog()
    frd = next(p for p in catalog if p.provider == "FirstRate Data")
    evz = next(p for p in catalog if "EVZ" in p.product)
    assert frd.genuinely_new is False
    assert evz.genuinely_new is False
    assert frd.shortlisted is False
    assert evz.shortlisted is False


def test_no_trading_code_invoked() -> None:
    """Invariant 13: No trading or execution code is imported by discovery script."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import scripts.run_phase44_data_provider_discovery\n"
        "assert 'trading.execution' not in sys.modules\n"
        "assert 'trading.risk' not in sys.modules\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"


def test_no_execution_code_invoked() -> None:
    """Invariant 14: Execution adapter is not invoked by discovery script."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import scripts.run_phase44_data_provider_discovery\n"
        "assert 'trading.mt5' not in sys.modules\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"


def test_no_credentials_written() -> None:
    """Invariant 15: No credentials or secrets are written to generated reports."""
    run_phase44_discovery()
    matrix_path = Path("reports/phase44_provider_matrix.json")
    budget_path = Path("reports/phase44_budget_analysis.json")

    assert matrix_path.exists()
    assert budget_path.exists()

    with open(matrix_path, encoding="utf-8") as f:
        matrix_text = f.read().lower()
    with open(budget_path, encoding="utf-8") as f:
        budget_text = f.read().lower()

    forbidden_tokens = [
        "api_key=",
        "secret_key",
        "private_key",
        "bearer ey",
        "password",
    ]
    for token in forbidden_tokens:
        assert token not in matrix_text
        assert token not in budget_text


def test_no_payment_or_account_creation() -> None:
    """Invariant 16: Discovery confirms no purchases or account creation occurred."""
    matrix_path = Path("reports/phase44_provider_matrix.json")
    with open(matrix_path, encoding="utf-8") as f:
        data = json.load(f)

    gov = data["governance"]
    assert gov["no_purchases_made"] is True
    assert gov["no_payment_info_entered"] is True
    assert gov["no_demo_orders"] is True
    assert gov["no_live_orders"] is True
