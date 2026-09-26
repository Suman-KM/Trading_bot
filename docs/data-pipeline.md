# Historical Market Data Pipeline & Validation Specification

## 1. Executive Summary & Objective

This document defines the quantitative market-data ingestion, validation, and storage pipeline for **Phase 3** of the AI Autonomous Trading System (Member 2 — Research & Modeling).

The primary objective is to establish an immutable, verifiable, and statistically honest data foundation for the locked research instrument: **EURUSD M15** (MetaTrader 5 feed from MetaQuotes Ltd. / MetaQuotes-Demo server).

---

## 2. Locked Research Configuration

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Instrument** | `EURUSD` | Spot Foreign Exchange |
| **Timeframe** | `M15` (15 minutes) | Nominal spacing: 900 seconds |
| **Broker** | `MetaQuotes Ltd.` | Demo execution environment |
| **Server** | `MetaQuotes-Demo` | Primary research bridge via Member 1 |
| **Data Provider** | MetaTrader 5 (MT5) | Sourced directly from Member 1 Linux node |
| **Historical Range** | 2023-09-18 to 2026-09-25 | ~75,000 candles |
| **Timestamp Format** | Unix Epoch Seconds | Integer seconds since 1970-01-01 00:00:00 UTC |
| **Research Timezone**| `UTC` | Timezone-aware `datetime64[ns, UTC]` |

---

## 3. Data Dictionary & Schema

The raw MT5 market feed delivers 8 standard columns:

| Column | Type | Unit / Representation | Description |
| :--- | :--- | :--- | :--- |
| `time` | `int64` | Seconds | MT5 Unix epoch timestamp |
| `open` | `float64` | Quote currency (USD) | Opening price of the 15-minute candle |
| `high` | `float64` | Quote currency (USD) | Highest price during the 15-minute interval |
| `low` | `float64` | Quote currency (USD) | Lowest price during the 15-minute interval |
| `close` | `float64` | Quote currency (USD) | Closing price of the 15-minute candle |
| `tick_volume` | `int64` | Count | Number of price updates (ticks) during the candle |
| `spread` | `int64` | MT5 Points | Bid-Ask spread (1 point = 0.00001 for EURUSD) |
| `real_volume` | `int64` | Lots / Units | Actual traded contracts (0 for retail OTC FX) |

### Processed Schema Enrichment
The pipeline preserves all 8 raw columns unchanged and prepends an explicit, timezone-aware UTC datetime column:
- `timestamp`: `datetime64[ns, UTC]` (e.g., `2026-09-25 22:45:00+00:00`).

---

## 4. Architectural Data Flow

```text
MetaTrader 5 Export (Member 1 Linux Server)
                     ↓
        data/raw/eurusd_m15/ [IMMUTABLE]
                     ↓
         Raw Dataset Ingestion
                     ↓
          Strict Schema Validation
                     ↓
       Timestamp Monotonicity & Uniqueness
                     ↓
         OHLC Physical Integrity Checks
                     ↓
       Spread & Volume Distribution Audit
                     ↓
         Market Session Gap Analysis
                     ↓
  ┌──────────────────┴──────────────────┐
  ↓                                     ↓
data_quality_report.json     dataset_metadata.json
  ↓                                     ↓
data/processed/eurusd_m15/eurusd_m15_processed.parquet
                     ↓
    READY FOR PHASE 4 (Exploratory Data Analysis)
```

---

## 5. Validation Rules & Acceptance Criteria

To guarantee data integrity before modeling, the pipeline enforces strict, deterministic verification rules. A failure of any critical check marks the dataset as `FAIL` and halts progression:

### 5.1 Schema Integrity
- All 8 required columns must exist.
- No unexpected columns (in strict mode).
- Price columns must be numeric floating-point types (`float64`).
- Epoch, volume, and spread must be integer types (`int64`).
- No string or object dtypes allowed in numerical columns.

### 5.2 Timestamp Monotonicity & Uniqueness
- Epoch seconds must be strictly positive (`time > 0`).
- Zero duplicate timestamps allowed (`duplicated().sum() == 0`).
- Strict chronological monotonicity enforced: for every row $i$, $t_i > t_{i-1}$.

### 5.3 Physical OHLC Consistency
For every individual candle, the following geometric price relationships must hold:
1. $\text{High} \ge \max(\text{Open}, \text{Close})$
2. $\text{Low} \le \min(\text{Open}, \text{Close})$
3. $\text{High} \ge \text{Low}$
4. $\text{Open}, \text{High}, \text{Low}, \text{Close} > 0$
5. No $\text{NaN}$, $+\infty$, or $-\infty$.

### 5.4 Spread Validation
- Spread is quoted in **MT5 points**. For EURUSD, 1 point = $0.00001$ ($0.1$ pip).
- Negative spreads are physically impossible in FX markets and are rejected as critical corruption.
- Zero spreads are tracked and flagged as warnings (characteristic of specific institutional zero-spread accounts or spread-filter glitches).
- Percentile distributions (25th, 50th, 75th, 95th, 99th) are logged in the quality report.

### 5.5 Volume Validation
- `tick_volume` measures market activity (tick frequency). Must be non-negative and is checked for anomalous zero-tick rows during active trading sessions.
- `real_volume` is expected to be $0$ ($100\%$ zero rate) on retail OTC FX demo servers. It is verified and recorded without fabrication.

---

## 6. Gap Analysis & Market Session Rules

EURUSD trades approximately 24 hours a day, 5 days a week. Nominal candle interval for M15 is 900 seconds.

### Gap Classification
Any interval between consecutive candles $> 900$ seconds is analyzed and classified:
1. **`EXPECTED_WEEKEND`**: Friday close (approx. 20:00–22:00 UTC) to Sunday open (approx. 20:00–22:00 UTC). Duration typically spans 45 to 52 hours.
2. **`EXPECTED_HOLIDAY`**: Recognized market closures including Christmas (Dec 24–26), New Year's Day (Dec 31–Jan 2), and Good Friday.
3. **`UNEXPECTED_GAP`**: Gaps occurring within regular Monday–Friday trading sessions during normal market hours.
4. **`UNCLASSIFIED_GAP`**: Unexplained interval irregularities that do not match regular session schedules.

> [!IMPORTANT]
> **No Gap Filling Policy:**
> Missing bars and weekend closures are **never** filled with synthetic, forward-filled, or interpolated data. Filling market gaps distorts volatility, corrupts rolling indicator lookback windows, and invents phantom liquidity.

---

## 7. Data Leakage & Machine Learning Safety

To prevent look-ahead bias and cross-temporal leakage:
- **No Data Shuffling:** Time series are stored and processed strictly in chronological order.
- **Zero Future Information in Cleaning:** Cleaning logic operates row-by-row or backward-looking only.
- **No Global Normalization:** Standardization (e.g., z-scores, min-max scaling) must **not** be performed in the data ingestion pipeline. Preprocessing scalers must only be fitted on historical training slices during model cross-validation in later phases.
- **Raw Immutability:** Raw data files in `data/raw/eurusd_m15/` are treated as read-only historical truth.

---

## 8. Known Market Feed Limitations

1. **`real_volume` is 0:** The MetaQuotes-Demo FX feed provides tick volume rather than exchange volume. Modeling pipelines must not assume exchange-traded contract quantities.
2. **Weekend Rollover Spread Widening:** At Friday market close and Sunday market open, spreads temporarily widen due to reduced interbank liquidity. The pipeline preserves these true market conditions.
3. **Bank Holiday Liquidity Drops:** Reduced tick volume during regional bank holidays is normal market behavior, not data corruption.

---

## 9. Next Steps: Phase 4 (Exploratory Data Analysis)

Following validation and acceptance of the raw dataset:
1. **Return Distribution Analysis:** Calculate log returns across 15m, 1h, and 4h horizons; test for normality, skewness, and fat-tailed kurtosis.
2. **Volatility & Regime Profiling:** Analyze ATR distributions, realized volatility by session (Asian, European, New York), and volatility clustering.
3. **Autocorrelation & Stationarity:** Augmented Dickey-Fuller (ADF) tests and autocorrelation analysis of returns and volumes.
4. **Baseline Strategy Formulation:** Preparation for Phase 5 baseline benchmarks (Buy & Hold, SMA cross, RSI rules).
