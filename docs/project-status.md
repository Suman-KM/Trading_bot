# AI Autonomous Trading System — Project Status

Last Updated: 2026-09-26 (Member 1)

## Team Architecture & Ownership
- **Member 1 (Ubuntu/Linux):** MT5 infrastructure, broker connectivity, historical data export, execution safety core, RiskEngine, portfolio management, order routing.
- **Member 2 (macOS):** Historical data processing, real-data validation, EDA, feature engineering, ML models, backtesting, out-of-sample evaluation, signal generation, research dashboard.
- **Repository / GitHub:** Authoritative shared source of truth for code, tests, documentation, manifests, and architecture contracts.

## Locked Project Configuration
- **Primary Instrument:** EURUSD
- **Primary Timeframe:** M15
- **Broker / Server:** MetaQuotes Ltd. / MetaQuotes-Demo
- **Account Type:** Hedging Demo (Leverage 1:100, ~$100,000 balance)
- **Research Timezone:** UTC

## Phase Status Matrix

| Phase | Status | Owner | Notes |
| :--- | :--- | :--- | :--- |
| MT5 infrastructure | COMPLETE | Member 1 | Wine 10.0, MT5 Build 6205, Python IPC verified operational |
| Instrument discovery | COMPLETE | Member 1 | EURUSD selected and locked as primary instrument |
| EURUSD M15 locked | COMPLETE | Team | Primary timeframe locked to M15 |
| EURUSD M15 validation | COMPLETE | Member 1 | Price feeds, tick sizes, spread dynamics validated |
| Member 2 data pipeline | COMPLETE | Member 2 | Pipeline architecture established |
| Raw data export | COMPLETE | Member 1 | 100,000 bars exported to `eurusd_m15_raw.parquet` (2.10 MB) |
| Raw data transfer | PENDING | Team | Ready for transfer; awaiting Member 2 receipt confirmation |
| Real-data validation | PENDING — MEMBER 2 | Member 2 | Member 2 to execute validation on transferred Parquet |
| EDA | NOT STARTED | Member 2 | Post-validation exploratory data analysis |
| Features | NOT STARTED | Member 2 | Feature engineering on M15 bars |
| Labels | NOT STARTED | Member 2 | Label generation for ML targets |
| Baseline strategies | NOT STARTED | Member 2 | Rule-based and statistical benchmarks |
| Backtesting | NOT STARTED | Member 2 | Simulation engine execution |
| ML | NOT STARTED | Member 2 | Machine learning training and hyperparameter tuning |
| Out-of-sample evaluation | NOT STARTED | Member 2 | Walk-forward and holdout test set verification |
| Signal contract integration | NOT STARTED | Team | Binding ML signal outputs to FastAPI signal API schemas |
| Demo execution | NOT STARTED | Member 1 | Controlled paper/demo trading with RiskEngine gating |
| Monitoring | NOT STARTED | Member 1 | Real-time health, risk limits, and telemetry |

## Current Milestone Summary
Member 1 has completed the raw historical data export (100,000 M15 bars covering 2022-09-16 through 2026-09-25) into `data/raw/eurusd_m15/eurusd_m15_raw.parquet`. The data passed all read-only integrity checks (0 duplicates, strictly chronological, 0 OHLC violations, 0 NaNs). Member 1 is now awaiting Member 2's confirmation of data transfer and real-data validation.
