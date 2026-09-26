# EURUSD M15 Raw MT5 Export

Instrument:
EURUSD

Timeframe:
M15

Broker:
MetaQuotes Ltd.

Server:
MetaQuotes-Demo

Source:
MetaTrader 5 historical bars

Research timezone:
UTC

Raw fields:
- `time` (int64, Unix epoch seconds)
- `open` (float64)
- `high` (float64)
- `low` (float64)
- `close` (float64)
- `tick_volume` (uint64)
- `spread` (int32, points)
- `real_volume` (uint64)

Rows:
100,000

Earliest raw timestamp:
1663320600

Latest raw timestamp:
1790379900

Earliest interpreted UTC:
2022-09-16 09:30:00 UTC

Latest interpreted UTC:
2026-09-25 23:45:00 UTC

Timestamp representation:
Unix epoch seconds

Raw timestamp policy:
Preserved exactly as returned by MT5 without time-zone shifting, synthetic interpolation, or conversion to strings.

Export format:
Parquet

Compression:
Snappy

Dataset location:
data/raw/eurusd_m15/eurusd_m15_raw.parquet

File size:
2,201,015 bytes (2.10 MB)

SHA256 Checksum:
0870f2733a0b7ed08c937a239cb9e45543c1a599e57c20efb04585ac7d35c286

Historical completeness:
MT5-LIMITED

Retrieval method:
Deterministic chunked retrieval using `copy_rates_from_pos` in sequential 25,000-bar blocks from position 0 up to the terminal limit (100,000 bars ceiling). Chunks were sorted in ascending order and verified across block boundaries to ensure zero duplicate timestamps and strict monotonic chronology.

Validation:

Duplicate timestamps:
0

Chronological:
True (strictly monotonic increasing, min step = 900s, max step = 260,100s across weekends/holidays)

OHLC violations:
0 (verified: high >= open, high >= close, high >= low, low <= open, low <= close, low <= high)

Zero/negative prices:
0 (all open, high, low, close > 0)

NaN:
0

Inf:
0

Tick volume:
Range [1, 21483], 0 zeros, 0 negative values

Spread:
Range [0, 129] points, 0 negative values

Real volume:
100% zero (`uint64`), expected for OTC retail Forex broker feeds

Known limitations:
- Terminal chart buffer ceiling: MT5 terminal configuration caps chart history at `maxbars = 100,000`. Retrieval queries beyond 100,000 bars return `(-1, 'Terminal: Call failed')`. 100,000 bars (spanning Sep 16, 2022 to Sep 25, 2026) is the maximal retrievable history from the active broker server buffer.
- OTC retail Forex tick volume: Real volume is not reported by MetaQuotes-Demo for FX pairs; tick volume serves as activity proxy.
- Market closure gaps: Weekends and market holidays represent normal trading halts and are preserved without synthetic candle fabrication.

IMPORTANT:
The raw Parquet dataset is excluded from Git via `.gitignore` and is NOT committed to GitHub.
