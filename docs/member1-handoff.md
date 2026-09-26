# Member 1 → Member 2 / Unified Project Handoff

Date/time:
2026-09-27 00:50:00 UTC

Task:
EURUSD M15 MT5 historical data export & Unified Environment Synchronization

Status:
SYNCHRONIZED

Canonical Environment:
Ubuntu Linux x86_64

GitHub Repository:
https://github.com/Suman-KM/Trading_bot.git

Branch:
develop

Preserved History:
- Member 1 commits: `c355384`, `c753a07`
- Member 2 commits: `0e77f32`, `36a6c63`, `c1922a5`

Instrument:
EURUSD

Timeframe:
M15

Broker:
MetaQuotes Ltd.

Server:
MetaQuotes-Demo

Rows:
100,000

Earliest:
1663320600 (2022-09-16 09:30:00 UTC)

Latest:
1790379900 (2026-09-25 23:45:00 UTC)

Format:
Parquet (Snappy compression)

File path:
/home/cino/projects/ai-trading-system/data/raw/eurusd_m15/eurusd_m15_raw.parquet

File size:
2,201,015 bytes (2.10 MB)

SHA256 Checksum:
0870f2733a0b7ed08c937a239cb9e45543c1a599e57c20efb04585ac7d35c286

Validation:

Duplicate timestamps:
0

Chronological:
True (strictly increasing, min step = 900s, max step = 260,100s across weekends/holidays)

OHLC violations:
0 (all 6 consistency inequalities passed across all 100,000 bars)

NaN/Inf:
0 NaN, 0 Inf across all numeric price and volume columns

Tick volume:
Range [1, 21483], 0 zeros, 0 negative values

Spread:
Range [0, 129] points, 0 negative values

Real volume:
100% zero (`uint64`), standard for retail Forex OTC pricing

Historical completeness:
MT5-LIMITED (terminal chart buffer limit `maxbars = 100,000` reached; covers 4 full years: Sep 2022 – Sep 2026)

Known limitations:
- MT5 terminal chart buffer cap: Exactly 100,000 bars retrievable from the terminal buffer. Older bars beyond offset 100,000 return `(-1, 'Terminal: Call failed')`.
- Real volume: Unpopulated for OTC FX feeds; `tick_volume` is the activity proxy.
- Market closure gaps: Normal weekend and holiday trading halts are preserved as-is without synthetic interpolation.

Dataset Location & Status:
The raw dataset is located at `data/raw/eurusd_m15/eurusd_m15_raw.parquet`. Since development is now unified on Ubuntu, no network transfer is required.

Real-Data Pipeline Validation:
Executed: `uv run python scripts/validate_eurusd_m15.py`
Status: PASS (100,000 candles verified with 0 duplicates, 0 OHLC violations, strictly monotonic chronological order).
Processed dataset generated at: `data/processed/eurusd_m15/eurusd_m15_processed.parquet`.

Next Action:
Proceed to Phase 4 — Exploratory Data Analysis (EDA) on the validated EURUSD M15 dataset.

