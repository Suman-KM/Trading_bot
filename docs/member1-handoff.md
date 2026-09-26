# Member 1 → Member 2

Date/time:
2026-09-26 23:40:00 UTC

Task:
EURUSD M15 raw MT5 historical data export

Status:
READY

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
- Real volume: Unpopulated for OTC FX feeds; `tick_volume` should be used as activity proxy.
- Market closure gaps: Normal weekend and holiday trading halts are preserved as-is without synthetic interpolation.

Transfer method:
Secure copy (SCP) or Rsync from Member 1 host to Member 2 macOS machine:
```bash
mkdir -p ~/projects/ai-trading-system/data/raw/eurusd_m15
scp cino@192.168.1.21:/home/cino/projects/ai-trading-system/data/raw/eurusd_m15/eurusd_m15_raw.parquet ~/projects/ai-trading-system/data/raw/eurusd_m15/
```
*(Secondary Member 1 IP: `192.168.1.14` if Wi-Fi interface is preferred).*

Member 2 next action:
1. Pull the latest repository changes.
2. Transfer the raw dataset into `data/raw/eurusd_m15/eurusd_m15_raw.parquet`.
3. Verify file integrity using SHA256 checksum: `0870f2733a0b7ed08c937a239cb9e45543c1a599e57c20efb04585ac7d35c286`.
4. Run real-data validation pipeline (`uv run python scripts/validate_eurusd_m15.py`).
5. Record real-data validation results in shared documentation.
6. Commit and push validation results to GitHub.

Member 1 next action:
WAIT FOR MEMBER 2 DATA VALIDATION
