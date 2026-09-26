# Member 1 → Member 2

Current task:
MT5 infrastructure & EURUSD M15 raw data export

Status:
INFRASTRUCTURE VERIFIED — READY TO EXPORT EURUSD M15 DATASET

Verified Environment & Connectivity:
- OS: Ubuntu 26.04.1 LTS
- Wine: Wine 10.0
- MT5: Build 6205
- Python: 3.12.9
- MT5 Package: MetaTrader5 5.0.6180
- Python ↔ MT5 IPC: Operational
- Broker: MetaQuotes Ltd.
- Server: MetaQuotes-Demo
- Account: Demo configured, Hedging mode, 1:100 leverage, ~$100,000 balance/equity

Implemented & Tested Components:
- RiskEngine
- PaperBroker
- Local FastAPI signal API

Read-Only Data Validation Completed:
- Instrument: EURUSD
- Timeframe: M15
- Visible / tradable: Yes
- Total bars: ~75,000 M15 candles
- Date range: 2023-09-18 → 2026-09-25
- Quality checks: No duplicate timestamps, strict chronological ordering, valid OHLC geometry, positive/finite prices, tick_volume present, spread present, real_volume is 0 (standard OTC FX), weekend and holiday gaps verified normal.

Execution Bridge Status:
- MT5 demo order execution has NOT yet been tested.
- MT5DemoBroker/RPC execution bridge is NOT the current task.
- Zero orders should be placed at this stage.

Next Member 1 Action:
- Export the validated EURUSD M15 historical dataset from MT5.
- Provide dataset file: `eurusd_m15_raw.parquet` (or `eurusd_m15_raw.csv`).
- Ensure columns: `time, open, high, low, close, tick_volume, spread, real_volume`.
- Transfer to Member 2's `data/raw/eurusd_m15/` directory.
