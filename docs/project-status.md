# AI Autonomous Trading System — Project Status

## Current Architecture

Member 1:
- MT5
- broker connectivity
- execution
- RiskEngine
- portfolio/order management
- infrastructure

Member 2:
- historical data
- data validation
- EDA
- feature engineering
- ML
- backtesting
- evaluation
- signal generation

## Locked Research Configuration

Instrument: EURUSD
Timeframe: M15
Broker: MetaQuotes Ltd.
Server: MetaQuotes-Demo
Research timezone: UTC

## Completed

- MT5 infrastructure
- Instrument discovery
- EURUSD M15 validation
- Member 2 Phase 3 data pipeline
- 48 tests passing
- Ruff passing

## Current Phase

Phase 3 — Historical Data Acquisition + Validation

## Current Blocker

The actual raw EURUSD M15 MT5 dataset has not yet been transferred to Member 2's Mac.

## Next Action

Member 1:
Export the validated EURUSD M15 historical dataset from the MetaQuotes-Demo MT5 environment.

Member 2:
Import the raw dataset into:

data/raw/eurusd_m15/

Then run:

uv run python scripts/validate_eurusd_m15.py

## Safety

No live trading.

No orders.

No execution bridge.

No RiskEngine changes.

No ML before data validation.

## Next Phase

Phase 4 — Exploratory Data Analysis + Data Characteristics
