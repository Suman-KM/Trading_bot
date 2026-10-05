#!/usr/bin/env python3
"""Phase 45 — Free Data Source Inventory & Feasibility Classification.

Audits and classifies candidate free data sources under the strict project policy:
- HARD ZERO-COST BUDGET: Total budget is ₹0.
- All commercial, paid, and trial-with-card sources are strictly rejected.
- Evaluates provenance, timestamps, revision safety, and point-in-time causality.

Exports: reports/phase45_free_data_inventory.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Hard project constraint
TOTAL_DATA_BUDGET_INR = 0.0


@dataclass(frozen=True)
class FreeSourceRecord:
    source_name: str
    category: str
    provider_authority: str
    cost_usd: float
    cost_inr: float
    free_access_mode: str
    historical_coverage: str
    frequency: str
    event_timestamp_precision: str
    release_timestamp_precision: str
    revision_risk: str
    point_in_time_safe: bool
    genuinely_new_information: bool
    eurusd_relevance: str
    usable: bool
    classification: str
    decision_rationale: str


def build_free_data_inventory() -> list[FreeSourceRecord]:
    """Audit and classify candidate data sources under the ₹0 policy."""
    records: list[FreeSourceRecord] = []

    # 1. FRBSF Bauer & Swanson (2023) Monetary Policy Surprises
    records.append(
        FreeSourceRecord(
            source_name="FRBSF Bauer & Swanson Monetary Policy Surprises",
            category="A — Central Bank Shocks",
            provider_authority="Federal Reserve Bank of San Francisco",
            cost_usd=0.0,
            cost_inr=0.0,
            free_access_mode="Public research download (Excel/CSV on FRBSF website)",
            historical_coverage="1988–2024+ (36 years, ~290 FOMC meetings)",
            frequency="8 scheduled meetings per year + inter-meeting actions",
            event_timestamp_precision="30-minute window around 14:00 US Eastern announcement",
            release_timestamp_precision="Announcement minute (14:00 EST / 19:00 UTC)",
            revision_risk="Zero (derived from market futures prices in announcement window)",
            point_in_time_safe=True,
            genuinely_new_information=True,
            eurusd_relevance="HIGH (primary driver of USD short-rate repricing)",
            usable=True,
            classification="FREE — USABLE",
            decision_rationale=(
                "Official Federal Reserve high-frequency event study dataset. Measures the exact "
                "monetary policy surprise directly from money-market futures repricing. "
                "Point-in-time safe, zero revision risk, $0 cost, verified academic provenance."
            ),
        )
    )

    # 2. ECB Euro Area Monetary Policy Event-Study Database (EA-MPD)
    records.append(
        FreeSourceRecord(
            source_name="ECB Euro Area Monetary Policy Event-Study Database (EA-MPD)",
            category="B — Central Bank Shocks",
            provider_authority="European Central Bank (Altavilla et al. 2019/2023)",
            cost_usd=0.0,
            cost_inr=0.0,
            free_access_mode="Public official download (ECB Working Paper replication file)",
            historical_coverage="1999–2024+ (25 years, ~220 Governing Council meetings)",
            frequency="8 scheduled meetings per year",
            event_timestamp_precision=(
                "Separate 15-min decision window (14:15 CET) and 50-min press conference window"
            ),
            release_timestamp_precision="Announcement minute (14:15 CET / 13:15 UTC)",
            revision_risk="Zero (derived from market OIS/bond prices in announcement windows)",
            point_in_time_safe=True,
            genuinely_new_information=True,
            eurusd_relevance="HIGH (primary driver of EUR interest rate expectations)",
            usable=True,
            classification="FREE — USABLE",
            decision_rationale=(
                "Official ECB research database measuring intraday OIS and bond repricing around "
                "both policy rate announcements and press conference forward guidance. "
                "Essential counterweight to FOMC data for EURUSD macro pricing at ₹0 cost."
            ),
        )
    )

    # 3. Federal Reserve Bank of St. Louis — ALFRED Real-Time Macro Vintages
    records.append(
        FreeSourceRecord(
            source_name="ALFRED Real-Time Macroeconomic Data Vintages",
            category="C — Macro Vintages",
            provider_authority="Federal Reserve Bank of St. Louis (ALFRED)",
            cost_usd=0.0,
            cost_inr=0.0,
            free_access_mode="Public API (fredapi) / bulk CSV download",
            historical_coverage="1920s–Present (>100 years)",
            frequency="Monthly / Quarterly release dates",
            event_timestamp_precision="Publication date / vintage date",
            release_timestamp_precision="Release day / morning snapshot",
            revision_risk="Zero (explicitly tracks initial unrevised values vs later vintages)",
            point_in_time_safe=True,
            genuinely_new_information=True,
            eurusd_relevance=(
                "MEDIUM (contains actual release values, but lacks pre-release consensus)"
            ),
            usable=True,
            classification="FREE — CONDITIONALLY USABLE",
            decision_rationale=(
                "The gold standard for macroeconomic publication vintages. Prevents revision "
                "lookahead bias. However, it only records the actual releases, not the pre-release "
                "market consensus expectations. Usable for initial-release macro state features."
            ),
        )
    )

    # 4. Harvard Dataverse & Zenodo Published Replication Shock Archives
    records.append(
        FreeSourceRecord(
            source_name="Harvard Dataverse & Zenodo Macroeconomic News Surprise Archives",
            category="D — Academic Surprise Datasets",
            provider_authority="Peer-reviewed academic replication repositories (CC-BY 4.0)",
            cost_usd=0.0,
            cost_inr=0.0,
            free_access_mode="Open-access repository direct download (CSV/Parquet)",
            historical_coverage="1990–2022+ (varies by published study)",
            frequency="Release event calendar (NFP, CPI, GDP, FOMC)",
            event_timestamp_precision="Announcement date and minute",
            release_timestamp_precision="Release second / minute",
            revision_risk=(
                "Low (standardized surprises compiled from pre-release Bloomberg surveys)"
            ),
            point_in_time_safe=True,
            genuinely_new_information=True,
            eurusd_relevance="HIGH (contains actual vs consensus surprises)",
            usable=True,
            classification="FREE — CONDITIONALLY USABLE",
            decision_rationale=(
                "Provides legitimate historical surprise series (Actual - Consensus) compiled "
                "for peer-reviewed publications without commercial subscriptions. "
                "Conditionally usable: coverage ends with publication dates and requires "
                "explicit holdout alignment."
            ),
        )
    )

    # 5. FRED Daily Constant Maturity Treasury & Yield Spreads
    records.append(
        FreeSourceRecord(
            source_name="FRED Daily US-Germany 2Y Sovereign Yield Spread",
            category="E — Sovereign Yields",
            provider_authority="Federal Reserve Bank of St. Louis (FRED) / Bundesbank",
            cost_usd=0.0,
            cost_inr=0.0,
            free_access_mode="Public API (fredapi)",
            historical_coverage="1990–Present (>30 years)",
            frequency="Daily close",
            event_timestamp_precision="Daily snapshot (21:30 UTC)",
            release_timestamp_precision="T+1 morning",
            revision_risk="Zero",
            point_in_time_safe=True,
            genuinely_new_information=False,
            eurusd_relevance="HIGH (fundamental carry/macro factor)",
            usable=False,
            classification="FREE — REDUNDANT",
            decision_rationale=(
                "Already thoroughly evaluated in Phase 24 and Phase 25. Delivered 50.84% "
                "walk-forward balanced accuracy on H4, underperforming the technical "
                "baseline (51.62%). Contains zero unexhausted predictive alpha."
            ),
        )
    )

    # 6. CFTC Commitments of Traders (COT)
    records.append(
        FreeSourceRecord(
            source_name="CFTC COT Euro FX Net Speculative Positioning",
            category="E — Positioning",
            provider_authority="US Commodity Futures Trading Commission",
            cost_usd=0.0,
            cost_inr=0.0,
            free_access_mode="Public open data download",
            historical_coverage="2010–2026 (843 weeks)",
            frequency="Weekly (Tuesday snapshot, released Friday 20:30 UTC)",
            event_timestamp_precision="Weekly Friday release",
            release_timestamp_precision="Friday 20:30 UTC",
            revision_risk="Zero",
            point_in_time_safe=True,
            genuinely_new_information=False,
            eurusd_relevance="HIGH (institutional positioning)",
            usable=False,
            classification="FREE — REDUNDANT",
            decision_rationale=(
                "Already thoroughly evaluated in Phase 28 and Phase 29. Delivered 50.62% "
                "walk-forward balanced accuracy with negative expectancy (-0.73 pips/trade). "
                "Acts as a lagging trend follower."
            ),
        )
    )

    # 7. CBOE EuroCurrency ETF Volatility Index (EVZ)
    records.append(
        FreeSourceRecord(
            source_name="CBOE EuroCurrency ETF Volatility Index (EVZCLS)",
            category="E — Implied Volatility",
            provider_authority="CBOE / FRED",
            cost_usd=0.0,
            cost_inr=0.0,
            free_access_mode="Public FRED series",
            historical_coverage="2007–March 2025 (17.5 years, DISCONTINUED)",
            frequency="Daily close",
            event_timestamp_precision="Daily market close",
            release_timestamp_precision="Daily close",
            revision_risk="Zero",
            point_in_time_safe=True,
            genuinely_new_information=False,
            eurusd_relevance="LOW (ATM volatility magnitude only, no directional skew)",
            usable=False,
            classification="FREE — REDUNDANT",
            decision_rationale=(
                "Phase 31 proved that volatility magnitude contains zero directional sign "
                "information. EVZ lacks 25-delta risk reversals (directional skew) and was "
                "discontinued by CBOE in March 2025."
            ),
        )
    )

    # 8. Web Scraped Forex Calendars (Forex Factory / Investing.com)
    records.append(
        FreeSourceRecord(
            source_name="Unverified Web Calendar Scrapers (Forex Factory / Investing.com)",
            category="F — Web Scraped Calendars",
            provider_authority="Third-party web scraping",
            cost_usd=0.0,
            cost_inr=0.0,
            free_access_mode="Automated HTTP scraping",
            historical_coverage="Variable (2010–Present)",
            frequency="Event calendar",
            event_timestamp_precision="Minute resolution, unverified timezone shifts",
            release_timestamp_precision="Unverified / retroactively updated",
            revision_risk=(
                "Severe (retroactively overwrites initial consensus with revised numbers)"
            ),
            point_in_time_safe=False,
            genuinely_new_information=True,
            eurusd_relevance="HIGH in theory",
            usable=False,
            classification="FREE — NOT POINT-IN-TIME SAFE",
            decision_rationale=(
                "Scraping violates provider Terms of Service. Databases lack publication audit "
                "trails, suffer from DST gap misalignment, and retroactively overwrite original "
                "consensus figures with revised values. Unsafe for causal backtesting."
            ),
        )
    )

    # 9. OANDA Retail Position Book API
    records.append(
        FreeSourceRecord(
            source_name="OANDA REST v20 Retail Order & Position Book",
            category="E — Retail Positioning",
            provider_authority="OANDA Corporation",
            cost_usd=0.0,
            cost_inr=0.0,
            free_access_mode="Free with practice/demo account",
            historical_coverage="Current snapshots only (lacks multi-year historical archive)",
            frequency="20-minute snapshots",
            event_timestamp_precision="Snapshot minute",
            release_timestamp_precision="Snapshot minute",
            revision_risk="Zero",
            point_in_time_safe=True,
            genuinely_new_information=True,
            eurusd_relevance="LOW (retail client flow only, not institutional interbank)",
            usable=False,
            classification="FREE — INSUFFICIENT COVERAGE",
            decision_rationale=(
                "OANDA does not provide a multi-year historical API archive for backtesting. Only "
                "serves recent snapshots. Additionally, represents retail positioning rather than "
                "institutional order flow."
            ),
        )
    )

    # 10. Commercial Paid Sources Rejected by Project Policy
    paid_sources = [
        ("Databento CME 6E Globex MDP 3.0", 15.0, 1252.50, "CME Futures Order Flow"),
        ("Bloomberg Professional Terminal", 2500.0, 208750.00, "Enterprise Institutional Data"),
        ("CME DataMine Historical MBO/MBP", 275.0, 22962.50, "Exchange Order Book Depth"),
        ("Trading Economics API", 149.0, 12441.50, "Macroeconomic Consensus Survey"),
        ("Interactive Brokers API Market Data", 2.25, 187.88, "Broker Tick Data Feed"),
    ]
    for name, usd, inr, cat in paid_sources:
        records.append(
            FreeSourceRecord(
                source_name=name,
                category=cat,
                provider_authority="Commercial Data Vendor",
                cost_usd=usd,
                cost_inr=inr,
                free_access_mode="None (Paid subscription / commercial agreement required)",
                historical_coverage="Multi-year",
                frequency="Tick / Intraday",
                event_timestamp_precision="Sub-second",
                release_timestamp_precision="Sub-second",
                revision_risk="Low",
                point_in_time_safe=True,
                genuinely_new_information=True,
                eurusd_relevance="HIGH",
                usable=False,
                classification="PAID — REJECTED",
                decision_rationale=(
                    f"PAID SOURCE — REJECTED BY PROJECT POLICY. Cost is ${usd:.2f} USD "
                    f"(~₹{inr:.2f} INR). Phase 45 strictly enforces a HARD ZERO-COST BUDGET (₹0). "
                    "Commercial sources are permanently prohibited."
                ),
            )
        )

    return records


def run_inventory_audit() -> dict[str, Any]:
    """Execute the inventory audit and write the structured JSON report."""
    records = build_free_data_inventory()

    # Classification counts
    classification_counts: dict[str, int] = {}
    for r in records:
        classification_counts[r.classification] = classification_counts.get(r.classification, 0) + 1

    usable_free_sources = [r for r in records if r.usable]

    report = {
        "metadata": {
            "phase": "45",
            "title": "Exhaustive Zero-Cost Data Research & Profitability Gate — Inventory",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "policy_budget_inr": TOTAL_DATA_BUDGET_INR,
            "policy_enforcement": "HARD ZERO-COST DATA POLICY",
        },
        "statistics": {
            "total_sources_audited": len(records),
            "free_sources_audited": sum(1 for r in records if r.cost_usd == 0.0),
            "paid_sources_rejected": sum(1 for r in records if r.cost_usd > 0.0),
            "usable_free_sources_count": len(usable_free_sources),
            "classification_breakdown": classification_counts,
        },
        "usable_free_sources": [asdict(r) for r in usable_free_sources],
        "all_sources": [asdict(r) for r in records],
        "audit_verdict": (
            f"Found {len(usable_free_sources)} usable free sources. "
            "Proceeding to empirical event-study evaluation on EURUSD M15/H4."
        ),
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "phase45_free_data_inventory.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_inventory_audit()
    print(
        f"Phase 45 Inventory Audit Complete: "
        f"{rep['statistics']['total_sources_audited']} sources audited, "
        f"{rep['statistics']['usable_free_sources_count']} usable free sources identified."
    )
