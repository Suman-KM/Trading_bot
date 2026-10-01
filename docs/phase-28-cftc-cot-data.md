# Phase 28 — CFTC COT Data Acquisition, Verification & Point-in-Time Alignment

**Date:** 2026-10-02  
**Project:** AI Autonomous Trading System  
**Repository:** `https://github.com/Suman-KM/Trading_bot.git`  
**Branch:** `develop`  
**Base Commit:** [`206fa9c`](https://github.com/Suman-KM/Trading_bot/commit/206fa9cdd8d70f9795c3b96f5ee5ef01de0fd0d9) (`research: finalize Phase 26 evidence gate`)  
**Status:** COMPLETE — SCIENTIFIC DATA READINESS: **`READY FOR FUTURE EXPERIMENT`**  

---

## 1. Objective

The objective of Phase 28 is to acquire, cryptographically verify, validate, normalize, and causally align historical weekly Commitments of Traders (COT) institutional positioning data from the official US Commodity Futures Trading Commission (CFTC) for CME Euro FX futures (contract code `099741`).

> [!IMPORTANT]
> **Strict Operational Boundaries:**
> - Zero predictive model training.
> - Zero backtesting or strategy P&L calculations.
> - Zero parameter or threshold optimization.
> - Zero live or demo trading execution.
> - Permanent holdouts remain strictly quarantined.
> - This phase establishes the data infrastructure only.

---

## 2. Phase 26 Decision Context

In Phase 26, after evaluating the complete empirical record from Phases 8 through 25, the research gate concluded with **`PAUSE FOR NEW DATA / INFRASTRUCTURE`**. Phase 26 established that public retail price action (EURUSD OHLCV, tick quotes, cross-market FX prices) and public daily sovereign bond yields (US 2Y, German 2Y) contain near-zero stationary forward directional edge (~50%–51% out-of-sample balanced accuracy).

Phase 26 formulated Candidate Hypothesis 1:
> Extreme non-commercial (speculative) positioning or commercial (hedger) positioning in CFTC Euro FX futures predicts medium-term multi-week directional mean reversion on EURUSD D1 bars due to speculative inventory exhaustion.

Phase 27 strictly enforced the pause gate (zero models trained, zero backtests executed). Phase 28 implements the prerequisite data acquisition and causal alignment infrastructure.

---

## 3. Official CFTC Source

All data was retrieved directly from the official regulatory repository of the United States Commodity Futures Trading Commission:
- **Agency:** Commodity Futures Trading Commission (CFTC)
- **Portal Base URL:** `https://www.cftc.gov/files/dea/history/`
- **Documentation:** `https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm`
- **Public Reporting Environment:** `https://publicreporting.cftc.gov/`

Two report families were retrieved across the entire 2010–2026 historical window:
1. **COT Legacy Futures Only Reports** (`deacot{year}.zip`): Reports positions across Commercial (Hedgers), Non-Commercial (Speculators), Total Reportable, Non-Reportable, and Open Interest.
2. **Traders in Financial Futures (TFF) Futures Only Reports** (`fut_fin_txt_{year}.zip`): The financial counterpart to disaggregated reports, categorizing positions into Dealer/Intermediary, Asset Manager/Institutional, Leveraged Money (Hedge Funds), and Other Reportables.

---

## 4. Contract Identification & Mapping

The target contract was verified through official CFTC market directories:
- **CFTC Contract Market Code:** `099741`
- **Market & Exchange Name:** `EURO FX - CHICAGO MERCANTILE EXCHANGE`
- **Underlying Instrument:** CME Euro FX Futures (contract size: €125,000)
- **Economic Relevance:** Primary institutional derivative contract for EUR/USD exchange rate exposure.
- **Cross-Contract Isolation:** Contract `099741` was explicitly separated from cross-rate contracts (`299741` = Euro FX/British Pound, `399741` = Euro FX/Japanese Yen).

---

## 5. Download Methodology

Acquisition was automated via a deterministic, repeatable Python pipeline in [`scripts/download_cftc_cot.py`](file:///home/cino/projects/ai-trading-system/scripts/download_cftc_cot.py):
- Supports `--start-year` (2010), `--end-year` (2026), and `--output-dir`.
- Fetches all 17 annual zip archives for Legacy COT and 17 annual zip archives for TFF (34 archives total).
- Uses standard HTTP requests with explicit User-Agent headers.
- Caches raw archives locally to prevent redundant downloads.

---

## 6. Raw Artifacts

The immutable raw archives are stored under:
```
data/exogenous/cftc/
├── raw/
│   ├── legacy/
│   │   ├── deacot2010.zip ... deacot2026.zip (17 files, ~29.1 MB)
│   └── tff/
│       ├── fut_fin_txt_2010.zip ... fut_fin_txt_2026.zip (17 files, ~7.2 MB)
├── processed/
│   └── cftc_eurofx_cot.parquet (873 rows, 33 columns)
├── metadata/
│   ├── cftc_cot_provenance.json
│   ├── cftc_cot_raw_validation.json
│   └── cftc_cot_coverage.json
└── manifests/
```

---

## 7. Provenance

Detailed provenance metadata is recorded in [`reports/cftc_cot_provenance.json`](file:///home/cino/projects/ai-trading-system/reports/cftc_cot_provenance.json):
- Sourced directly from official CFTC servers.
- Records exact source URLs, HTTP content sizes, download UTC timestamps, and cryptographic hashes for all 34 raw zip files.
- Records the canonical output parquet path and its SHA-256 digest.

---

## 8. SHA-256 Hashes

Select raw artifact SHA-256 digests:
- `deacot2010.zip`: `ba52af024b2cc24f4bf7699950b70b0ae70ce2a123fca63e88c93f5d67561253`
- `deacot2015.zip`: `e8ce35817c13aa5e8e826aa7519a793a388f61530932c0d8fca02d60e7f8e813`
- `deacot2020.zip`: `a35eece6fa05d34da091f8fe73998bdfbc8a230e719047970dffc3dc3fa55b62`
- `deacot2025.zip`: `3134fb3f33cefa78d531efec58e6e584f3ebc69f20e4da66bc0bc457223b9d07`
- `deacot2026.zip`: `88561d36987f259062eb6821379ec63ff75c2e17e34fb7039a03b573a62ea67f`
- `fut_fin_txt_2026.zip`: `e0bf79f2feff66504a3f2b45cf355aa7374ddc3b28b776ec0d87c95e1e127602`
- Canonical `cftc_eurofx_cot.parquet`: `82f9ef02ff0b343cb64e2235c5c8fc38e55e89a5450890aa9b0f491c7848f074`

---

## 9. Historical Coverage

- **Coverage Period:** 2010-01-05 through 2026-09-22
- **Calendar Depth:** 16.72 continuous years (matches the expanded EURUSD H4/D1 history from Phase 17).
- **Total Weekly Reports:** 873 unique weekly observations.
- **Duplicate Observation Dates:** Exactly 0.
- **Coverage Continuity:** Fully continuous weekly series; maximum calendar gap between consecutive observations is 8 days (due to holiday Monday observation shifts).

---

## 10. Data Schema

The canonical normalized dataset [`data/exogenous/cftc/cftc_eurofx_cot.parquet`](file:///home/cino/projects/ai-trading-system/data/exogenous/cftc/cftc_eurofx_cot.parquet) contains 33 columns:

| Column Name | Type | Description |
| :--- | :---: | :--- |
| `report_date` | string / date | Tuesday observation date (`YYYY-MM-DD`) |
| `contract_code` | string | `099741` |
| `market_name` | string | `EURO FX - CHICAGO MERCANTILE EXCHANGE` |
| `open_interest` | int64 | Total open contracts |
| `non_commercial_long` | int64 | Speculator long contracts |
| `non_commercial_short` | int64 | Speculator short contracts |
| `non_commercial_spreading` | int64 | Speculator calendar/spread contracts |
| `commercial_long` | int64 | Hedger long contracts |
| `commercial_short` | int64 | Hedger short contracts |
| `total_reportable_long` | int64 | Total reportable long positions |
| `total_reportable_short` | int64 | Total reportable short positions |
| `non_reportable_long` | int64 | Small trader long positions |
| `non_reportable_short` | int64 | Small trader short positions |
| `dealer_long` / `short` | int64 | TFF Dealer / Intermediary positions |
| `asset_mgr_long` / `short` | int64 | TFF Institutional Asset Manager positions |
| `lev_money_long` / `short` | int64 | TFF Leveraged Funds (Hedge Funds) positions |
| `other_rept_long` / `short` | int64 | TFF Other reportable positions |
| `observation_timestamp_utc` | string / ISO | Tuesday 21:00:00 UTC (CME daily settlement) |
| `publication_date` | string / date | Friday release date (`YYYY-MM-DD`) |
| `publication_time_utc` | string / ISO | Friday 15:30 US Eastern converted to UTC |
| `effective_time_utc` | string / ISO | Monday 00:00:00 UTC following publication |
| `net_speculative_pos` | int64 | Derived: `non_commercial_long - non_commercial_short` |
| `net_commercial_pos` | int64 | Derived: `commercial_long - commercial_short` |
| `net_spec_zscore_3y` | float64 | Derived: Rolling 156-week Z-score of speculative net pos |
| `net_comm_zscore_3y` | float64 | Derived: Rolling 156-week Z-score of commercial net pos |
| `net_spec_4w_change` | float64 | Derived: 4-week change in net speculative pos |
| `net_leveraged_money_pos` | int64 | Derived: `lev_money_long - lev_money_short` |
| `net_asset_mgr_pos` | int64 | Derived: `asset_mgr_long - asset_mgr_short` |

---

## 11. Data Quality & Accounting Identities

Automated validation verified strict accounting balance across all 873 reports:
1. **Open Interest Long Identity:**
   $$\text{Open Interest} - (\text{Total Reportable Long} + \text{Non-Reportable Long}) = 0 \quad (\text{Max Diff} = 0)$$
2. **Open Interest Short Identity:**
   $$\text{Open Interest} - (\text{Total Reportable Short} + \text{Non-Reportable Short}) = 0 \quad (\text{Max Diff} = 0)$$
3. **Non-Negativity:** Zero negative open interest, zero negative position counts.
4. **Validation Verdict:** **PASS** (documented in [`reports/cftc_cot_raw_validation.json`](file:///home/cino/projects/ai-trading-system/reports/cftc_cot_raw_validation.json)).

---

## 12. Publication Timing

The CFTC Commitments of Traders release schedule operates with an inherent point-in-time publication lag:
- **Observation Day:** Tuesday market close.
- **Normal Publication Time:** Friday at 15:30 US Eastern Time (`America/New_York`).
- **Publication Lag:** Exactly 3 calendar days (72 hours) between observation and public release.

---

## 13. Timezone & Daylight Saving Time (DST) Handling

All publication timestamps are converted to UTC via Python's IANA `zoneinfo.ZoneInfo("America/New_York")`:
- **US Eastern Daylight Time (EDT, UTC-4):** 15:30 EDT = **19:30 UTC**.
- **US Eastern Standard Time (EST, UTC-5):** 15:30 EST = **20:30 UTC**.
- This dynamic adjustment prevents artificial 1-hour timestamp errors across biannual DST transition boundaries.

---

## 14. Causal Alignment to EURUSD Bars

Causal alignment is implemented in [`ai/data/exogenous/cftc_alignment.py`](file:///home/cino/projects/ai-trading-system/ai/data/exogenous/cftc_alignment.py):
- **Effective Timestamp Rule:**
  Because Forex markets trade until Friday ~21:00 UTC and reopen on Sunday evening (~21:00 UTC), a report released on Friday at 15:30 ET (19:30 or 20:30 UTC) is first actionable at the weekly market open: **Monday 00:00:00 UTC** (or next available bar if Monday is a holiday).
- **As-Of Merge:**
  EURUSD bars are joined using `merge_asof(direction="backward")` matching on `bar_timestamp >= effective_time_utc`.
- **Forward-Fill Integrity:**
  Positioning metrics remain constant throughout the week until the subsequent report's effective timestamp arrives.

---

## 15. Leakage Controls & Proof of Causality

Strict temporal ordering is guaranteed and verified:
$$\text{Observation Time (Tue 21:00 UTC)} < \text{Publication Time (Fri 15:30 ET)} < \text{Effective Time (Mon 00:00 UTC)} \le \text{EURUSD Bar Open}$$

Empirical verification against 25,800 historical EURUSD H4 bars (2010–2026):
- **Publication Leakage Violations:** Exactly **0** (0 bars used a report before its publication timestamp).
- **Effective Leakage Violations:** Exactly **0** (0 bars used a report before its Monday effective timestamp).
- **Pre-Release Quarantine:** Tuesday, Wednesday, Thursday, and Friday bars never access Tuesday's position data.

---

## 16. Coverage Gaps & Delayed Release Handling

1. **Standard Federal Holidays:**
   When Friday is a US Federal holiday (e.g. Good Friday, July 4th, Christmas), publication is delayed to Monday, and effective availability moves to Tuesday 00:00 UTC.
2. **2018–2019 US Government Shutdown (Dec 22, 2018 – Jan 25, 2019):**
   The 35-day lapse in federal appropriations suspended CFTC operations. Reports for late December 2018 and January 2019 were released in February 2019. The pipeline maps these dates to their actual historical release dates, preventing retroactive lookahead.
3. **2013 Government Shutdown (Oct 1–16, 2013):**
   Reports delayed to late October 2013 are similarly aligned to their actual catch-up release dates.

---

## 17. Research-Ready Derived Fields

The normalized dataset includes pre-computed candidate features strictly as mathematical transformations:
- `net_speculative_pos`
- `net_commercial_pos`
- `net_spec_zscore_3y`
- `net_comm_zscore_3y`
- `net_spec_4w_change`
- `net_leveraged_money_pos`
- `net_asset_mgr_pos`

> [!NOTE]
> **Non-Predictive Disclaimer:**
> These derived fields are provided to establish data readiness only. They have **NOT** been evaluated for predictive accuracy, directional edge, or trading profitability in Phase 28.

---

## 18. Limitations

1. **Weekly Frequency:** COT data is published weekly, capturing multi-week institutional rebalancing rather than short-horizon intraday dynamics.
2. **Publication Delay:** The mandatory 3-to-6-day lag between Tuesday position collection and Monday trading availability means high-frequency sentiment shifts are invisible.
3. **Futures-Only Aggregation:** The report measures CME exchange-traded futures contracts, which represent a significant fraction but not 100% of global OTC Forex turnover.

---

## 19. Scientific Decision

# **`READY FOR FUTURE EXPERIMENT`**

### Evidence-Based Justification:
1. Complete 16.72-year continuous weekly archive (873 reports) successfully acquired from the official CFTC regulatory portal with verified SHA-256 provenance.
2. Zero data defects, zero negative counts, zero duplicate dates, and 100% adherence to Open Interest accounting identities.
3. Fully implemented point-in-time publication schedule with IANA DST timezone conversion, holiday handling, and government shutdown adjustments.
4. Empirical alignment against 25,800 EURUSD H4 bars confirmed **zero leakage violations** across all 16 years.
5. All 11 dedicated causal alignment unit tests passed with 100% success rate.

---

## 20. Authorization Status for Future Modeling

**DO NOT TRAIN A MODEL IN PHASE 28.**

Phase 28 establishes the verified CFTC COT data foundation required for a future pre-registered Phase 29 experiment. **No predictive claims have been made.**

Any future experiment must:
1. Pre-register a single hypothesis (e.g. multi-week D1 swing mean reversion on positioning extremes).
2. Pre-define frozen model architecture and hyperparameters without parameter sweeps.
3. Quarantine all locked test partitions.
4. Establish unambiguous success and failure gates prior to execution.
