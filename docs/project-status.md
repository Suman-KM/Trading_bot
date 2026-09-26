# AI Autonomous Trading System — Project Status

Canonical Environment: Ubuntu Linux (Single Canonical Environment)  
Last Updated: 2026-09-27

---

## Architecture & Codebase Ownership
- **Repository:** Unified single repository (`https://github.com/Suman-KM/Trading_bot.git`)
- **Canonical Development Environment:** Ubuntu Linux x86_64
- **Trading Core (`trading/`):** RiskEngine, KillSwitch, PaperBroker, FastAPI signal API, MT5 bridge.
- **AI Core (`ai/`):** Data ingestion & validation, feature engineering, baseline models, ML engine, backtesting.
- **Shared Source of Truth:** GitHub (`develop` branch).

---

## Locked Research Configuration
- **Primary Instrument:** EURUSD
- **Primary Timeframe:** M15 (900 seconds)
- **Broker / Server:** MetaQuotes Ltd. / MetaQuotes-Demo
- **Account Type:** Hedging Demo (Leverage 1:100, ~$100,000 balance)
- **Research Timezone:** UTC

---

## Phase Status Matrix

| Phase | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Phase 1** | MT5 infrastructure | COMPLETE | Wine 10.0, MT5 Build 6205, Python IPC verified operational |
| **Phase 2** | Instrument discovery | COMPLETE | EURUSD selected and locked as primary instrument |
| **Phase 2** | EURUSD M15 locked | COMPLETE | Primary timeframe locked to M15 |
| **Phase 2** | EURUSD M15 read-only validation | COMPLETE | Price feeds, tick sizes, spread dynamics validated |
| **Phase 3** | Data pipeline architecture | COMPLETE | Schema validation, gap analysis, quality reporting in `ai/data/` |
| **Phase 3** | GitHub synchronization | COMPLETE | Unrelated histories unified on `develop`; all commits preserved |
| **Phase 3** | Raw MT5 data export | COMPLETE | 100,000 bars exported to `data/raw/eurusd_m15/eurusd_m15_raw.parquet` |
| **Phase 3** | Real-data pipeline validation | IN PROGRESS | Running `validate_eurusd_m15.py` against exported MT5 Parquet |
| **Phase 4** | Exploratory Data Analysis (EDA) | NEXT | Volatility profiling, return distributions, regime detection |
| **Phase 5** | Feature engineering | NOT STARTED | Point-in-time technical and price action indicators |
| **Phase 6** | Label engineering | NOT STARTED | Forward return labeling, triple barrier method |
| **Phase 7** | Baseline strategies | NOT STARTED | Naive benchmarks (Buy & Hold, SMA cross, RSI rules) |
| **Phase 8** | Backtesting engine | NOT STARTED | Vectorized and event simulation with cost models |
| **Phase 9** | ML model training | NOT STARTED | Walk-forward training (Linear, Random Forest, XGBoost) |
| **Phase 10**| Out-of-sample evaluation | NOT STARTED | Walk-forward and holdout stress tests |
| **Phase 11**| Signal contract integration | NOT STARTED | Binding ML signal schema to FastAPI endpoint |
| **Phase 12**| RiskEngine integration | NOT STARTED | Enforcing position sizing, drawdown, kill switch gating |
| **Phase 13**| Paper / MT5 demo execution | NOT STARTED | Live forward testing with RiskEngine gating |
| **Phase 14**| Monitoring & Dashboard | NOT STARTED | Telemetry and performance analytics |

---

## Current Milestone Summary
The project has successfully unified all development onto the canonical Ubuntu Linux environment. Both Member 1 (`trading/`, MT5 export, RiskEngine) and Member 2 (`ai/`, data pipeline, validation suite) commit histories are fully preserved. The raw EURUSD M15 historical dataset (100,000 bars) has been exported and is ready for full pipeline validation.
