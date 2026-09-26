# EURUSD M15 Exploratory Data Analysis (EDA) Report

**Phase:** Phase 4 — Quantitative Research & Exploratory Data Analysis  
**Instrument:** EURUSD  
**Timeframe:** M15 (15-Minute Candles)  
**Dataset Path:** `data/processed/eurusd_m15/eurusd_m15_processed.parquet` (and `data/raw/eurusd_m15/eurusd_m15_raw.parquet`)  
**Environment:** Canonical Ubuntu Linux x86_64, Python 3.13.15  
**Evaluation Scope:** Strictly descriptive empirical analysis. No machine learning training, feature engineering, label creation, or backtesting.

---

## 1. Dataset Overview

The dataset analyzed comprises 100,000 validated M15 historical candles for the primary instrument **EURUSD**, exported via official Python-to-MetaTrader 5 IPC from the MetaQuotes-Demo server.

| Property | Value |
| :--- | :--- |
| **Asset Class** | Spot Foreign Exchange (Forex) |
| **Broker / Server** | MetaQuotes Ltd. / `MetaQuotes-Demo` |
| **Total Rows** | 100,000 candles |
| **Columns** | 9 (`time`, `timestamp`, `open`, `high`, `low`, `close`, `tick_volume`, `spread`, `real_volume`) |
| **Primary Storage Format** | Apache Parquet (Snappy compression, 2.10 MB) |
| **Base Currency / Quote** | EUR / USD |
| **Digits / Precision** | 5 decimals (0.00001 USD = 1 point = 0.1 pip) |
| **Contract Size** | 100,000 EUR |

---

## 2. Time Coverage

The dataset spans over four full calendar years of continuous market activity:

- **Earliest Timestamp:** `2022-09-16 09:30:00 UTC` (epoch `1663320600`)
- **Latest Timestamp:** `2026-09-25 23:45:00 UTC` (epoch `1790466300`)
- **Total Duration:** 1,470 calendar days (~4.02 calendar years)
- **Timezone Standardization:** Strictly UTC (`datetime64[ms, UTC]`), verified with zero timezone offsets or ambiguous local time transitions.

---

## 3. Data Quality Findings

The dataset was subjected to automated verification through `ai.data.validation`:

- **Missing / Null Values:** Exactly 0 missing values across all columns.
- **Infinite Values:** Exactly 0 infinite values detected.
- **Timestamp Monotonicity:** Strictly monotonically increasing epoch timestamps (`dt > 0` for all consecutive rows; zero retrograde timestamps; zero duplicate timestamps).
- **OHLC Geometry Validity:** 100,000 / 100,000 candles (100.0%) satisfy `high >= max(open, close)` and `low <= min(open, close)`.
- **Price Positivity:** All price values are strictly positive (>0.95 USD). Zero negative or zero-priced bars observed.
- **Negative Spreads:** Exactly 0 negative spreads observed.

---

## 4. Price Behavior

### 4.1 OHLC Descriptive Summary

| Metric | Open | High | Low | Close |
| :--- | :--- | :--- | :--- | :--- |
| **Min** | 0.95392 | 0.95517 | 0.95357 | 0.95391 |
| **Max** | 1.20494 | 1.20819 | 1.20368 | 1.20494 |
| **Mean** | 1.10361 | 1.10395 | 1.10328 | 1.10361 |
| **Median** | 1.09203 | 1.09234 | 1.09175 | 1.09204 |
| **Std Dev** | 0.04943 | 0.04940 | 0.04946 | 0.04943 |
| **P01** | 0.97624 | 0.97686 | 0.97561 | 0.97624 |
| **P25** | 1.07237 | 1.07266 | 1.07207 | 1.07238 |
| **P75** | 1.15389 | 1.15418 | 1.15361 | 1.15389 |
| **P99** | 1.18576 | 1.18612 | 1.18536 | 1.18574 |

### 4.2 Bar Range and Candle Geometry

- **High-Low Range (Absolute):**
  - Mean: `0.000669` (6.69 pips / 66.9 points)
  - Median: `0.000530` (5.30 pips / 53.0 points)
  - Min: `0.000000` | Max: `0.016190` (161.9 pips)
  - 95th Percentile: `0.001570` (15.7 pips) | 99th Percentile: `0.002680` (26.8 pips)
- **High-Low Range (Percentage of Close):**
  - Mean: `0.0610%` | Median: `0.0482%` | Std: `0.0504%` | Max: `1.600%`
- **Candle Body (`|Close - Open|`):**
  - Mean: `0.000334` (3.34 pips) | Median: `0.000220` (2.20 pips) | Max: `0.016140` (161.4 pips)
- **Wicks:**
  - Upper Wick Mean: `0.000165` (1.65 pips) | Max: `0.007150` (71.5 pips)
  - Lower Wick Mean: `0.000170` (1.70 pips) | Max: `0.005610` (56.1 pips)
- **Candle Classification:**
  - Bullish (`Close > Open`): 49,686 bars (49.69%)
  - Bearish (`Close < Open`): 48,800 bars (48.80%)
  - Doji (`Close == Open`): 1,514 bars (1.51%)

---

## 5. Return Behavior

### 5.1 Simple & Log Return Distributions

M15 percentage returns and continuous log returns exhibit near-identical central moments due to small bar delta scales:

| Statistic | Simple Returns (`pct_change`) | Log Returns (`ln(C_t / C_{t-1})`) |
| :--- | :--- | :--- |
| **Mean** | `+0.0000014` (+0.00014% per bar) | `+0.0000013` |
| **Std Dev** | `0.0004900` (4.90 bps) | `0.0004899` (4.90 bps) |
| **Min** | `-0.0108356` (-1.08%) | `-0.0108947` (-1.09%) |
| **Max** | `+0.0161465` (+1.61%) | `+0.0160176` (+1.60%) |
| **Skewness** | `+0.2373` | `+0.2050` |
| **Excess Kurtosis** | `+42.1985` | `+41.8936` |

### 5.2 Return Quantiles

- **P01:** `-0.001324` (-13.2 bps)
- **P05:** `-0.000687` (-6.87 bps)
- **P25:** `-0.000197` (-1.97 bps)
- **P50 (Median):** `0.000000` (0.00 bps)
- **P75:** `+0.000198` (+1.98 bps)
- **P95:** `+0.000682` (+6.82 bps)
- **P99:** `+0.001343` (+13.4 bps)

### 5.3 Distribution Shape & Heavy Tails

The empirical distribution of EURUSD M15 returns is **strongly leptokurtic** with extreme fat tails (excess kurtosis ~41.9). While 50% of all 15-minute bars move less than 2.0 basis points (IQR = 3.95 bps), the distribution produces tail excursions exceeding 100 basis points during scheduled macroeconomic releases (e.g. US Non-Farm Payrolls, CPI, ECB/Fed rate statements). Standard normal distribution assumptions are strongly invalidated by the empirical Q-Q plot.

---

## 6. Volatility Profile

- **Bar-Level Volatility ($\sigma_{\text{bar}}$):** `0.0004899` per 15-minute bar.
- **Annualized Volatility:** `7.62%` (computed using standard convention: $252 \times 24 \times 4 = 24,192$ M15 bars/year).
- **14-Period Average True Range (ATR-14):** Average `0.000672` (6.72 pips).
- **Volatility Clustering:** Inspection of rolling 20-bar (5-hour) and 96-bar (24-hour) volatility demonstrates sustained autoregressive clustering: periods of calm (annualized volatility < 4%) frequently persist for several sessions, punctuated by sharp spikes (> 25% annualized volatility).

---

## 7. Time-of-Day Behavior (UTC)

Market dynamics vary systematically across the 24 UTC hours, reflecting global liquidity handoffs:

| Session / Window | UTC Hours | Avg Tick Volume | Avg Spread (pts) | Return Std (bps) | Market Character |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Asian Session** | 00:00 – 08:00 | 467.3 | 3.72 | 3.38 | Subdued volatility, low volume, elevated rollover spread at opening |
| **London Morning** | 07:00 – 12:00 | 1,003.6 | 1.65 | 5.39 | European institutional order flow, volume acceleration, tight spreads |
| **NY / London Overlap**| 12:00 – 16:00 | 1,283.9 | 1.60 | 6.09 | Highest daily liquidity and volatility; US macro announcements |
| **NY Afternoon** | 16:00 – 21:00 | 1,403.2 | 1.62 | 5.75 | Volume peak at 17:00 UTC (1,922 ticks); orderly price discovery |
| **Daily Rollover** | 21:00 – 00:00 | 659.1 | 2.56 | 3.88 | Bank settlement window; spread widens to 14.5 pts at 00:00 UTC |

---

## 8. Day-of-Week Behavior

Trading occurs from Sunday evening (~21:00 UTC) through Friday evening (~23:59 UTC):

| Weekday | Day Index | Candle Count | Avg Tick Volume | Avg Spread (pts) | Return Std (bps) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Monday** | 0 | 19,906 | 824.4 | 2.59 | 4.88 |
| **Tuesday** | 1 | 20,097 | 870.2 | 2.41 | 4.52 |
| **Wednesday** | 2 | 19,968 | 919.8 | 2.39 | 4.98 |
| **Thursday** | 3 | 19,916 | 928.8 | 2.39 | 5.08 |
| **Friday** | 4 | 20,113 | 899.8 | 2.41 | 5.02 |

*Midweek sessions (Wednesday and Thursday) demonstrate the highest average tick volumes and return volatility, while Mondays experience slightly subdued volume and wider average spreads.*

---

## 9. Spread Analysis

| Metric | Value |
| :--- | :--- |
| **Unit** | Points ($10^{-5}$ USD, 1 point = 0.1 pip) |
| **Min** | 0.0 points |
| **Max** | 129.0 points (12.9 pips) |
| **Mean** | 2.44 points (0.24 pips) |
| **Median** | 0.0 points |
| **Standard Deviation** | 4.52 points |
| **Zero Rate** | 50,171 bars (50.17%) |
| **P75 / P95 / P99** | 4.0 pts / 8.0 pts / 21.0 pts |

*Spread Observation:* The recorded spread in this MetaQuotes-Demo feed is 0 points for over 50% of all candles during liquid daytime trading. Spread expansion is heavily localized to the rollover period (21:00–00:00 UTC) and holiday/illiquid boundary intervals.

---

## 10. Tick Volume Analysis

| Metric | Value |
| :--- | :--- |
| **Min** | 1 tick |
| **Max** | 21,483 ticks |
| **Mean** | 888.6 ticks |
| **Median** | 594.0 ticks |
| **Standard Deviation** | 930.4 ticks |
| **Zero Rate** | 0 bars (0.00%) |
| **P01 / P05 / P25** | 62 / 128 / 332 ticks |
| **P75 / P95 / P99** | 1,097 / 2,638 / 4,442 ticks |

*Tick Volume Observation:* Every candle in the 100,000 dataset contains at least 1 tick. Tick volume tracks the number of bid/ask quote changes transmitted by the broker and serves as an empirical proxy for market activity.

---

## 11. Real Volume Analysis

- **Total Rows:** 100,000
- **Zero Count:** 100,000 (100.0%)
- **Non-Zero Count:** 0
- **Mean / Min / Max / Std:** 0.0

*Real Volume Observation:* Real traded volume is **completely unavailable** in retail OTC Forex feeds from MetaQuotes-Demo. Decentralized OTC foreign exchange transactions are not cleared through a centralized exchange, meaning no authoritative traded volume metric exists. `tick_volume` must NOT be replaced with `real_volume` without explicit documentation.

---

## 12. Gap Analysis

Evaluating timestamp deltas across the 99,999 consecutive candle intervals:

| Category | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **Normal M15 Intervals** | 99,768 | 99.77% | Exact 900-second spacing |
| **Total Gaps (> 900s)** | 231 | 0.23% | Total inter-bar closures |
| **Expected Weekend Closures**| 209 | 0.21% | Friday close (~21:00-23:59 UTC) to Sunday open (~21:00-23:00 UTC); ~48-49h |
| **Expected Holiday Closures** | 6 | <0.01% | Christmas / New Year closures (~24h - 72h) |
| **Unexpected Intraday Gaps** | 12 | <0.01% | Brief feed interruptions (<24h) during weekday trading |
| **Unclassified Boundary Gaps**| 4 | <0.01% | Non-standard boundary transitions |
| **Maximum Observed Gap** | 72.25 hours | — | Extended holiday weekend closure (260,100 seconds) |

*Gap Handling Policy:* Market closures are legitimate structural events. In accordance with quantitative best practices, gaps are preserved without synthetic forward-filling or candle interpolation.

---

## 13. Important Quantitative Observations

1. **Leptokurtic Return Structure:** Returns display extreme excess kurtosis (~42). Risk modeling must account for tail events and cannot rely on Gaussian distribution assumptions.
2. **Session Seasonality:** Clear intraday regime shifts occur between low-volatility Asian hours (spread ~3.7 pts, std ~3.4 bps) and high-volatility London/NY overlap (spread ~1.6 pts, std ~6.1 bps, volume peaking >1,900 ticks).
3. **Rollover Fragility:** Spreads spike by an order of magnitude (averaging 14.5 points at 00:00 UTC) concurrent with liquidity exhaustion. Execution algorithms must respect rollover avoidance windows.
4. **Zero-Spread Modeling:** The demo feed reports 0-point spreads for over 50% of daytime bars. In real live trading accounts, EURUSD spreads rarely drop below 0.1–0.4 pips plus commissions.

---

## 14. Limitations of the Dataset

- **Demo Broker Feed:** Data originates from MetaQuotes-Demo. Real institutional liquidity, execution latency, and retail broker markups are not reflected.
- **Absence of Real Volume:** `real_volume` is 100% zero; quantitative analysis is restricted to broker quote update frequency (`tick_volume`).
- **Snapshot Spreads:** Spreads represent candle-close snapshots rather than the complete bid-ask trajectory across the 15-minute bar.
- **Regime Specificity:** The period September 2022 to September 2026 encompasses a specific macro regime (rapid central bank rate hikes followed by gradual easing), which may differ from other historical eras.

---

## 15. What Should NOT Be Concluded from the EDA

1. **No Directional Alpha:** Historical mean returns (1.31e-06) and bullish/bearish candle counts (49.69% vs 48.80%) reflect empirical balance and do NOT constitute a directional edge or predictive signal.
2. **No Execution Guarantee:** High tick volume does not guarantee fills without slippage, nor does zero spread imply zero cost in production environments.
3. **No Unconditional Stationarity:** Descriptive statistics aggregated over 100,000 bars obscure time-varying regime shifts; models cannot assume strict parameter stationarity across years.
4. **No Causal Inferences:** Time-of-day or day-of-week volume patterns are empirical descriptions of historical market schedules, not causal laws.

---

## 16. Research Figures Generated

The non-interactive visualization pipeline generated 8 high-resolution figures saved under the ignored `reports/figures/` directory:

1. `eurusd_close_price.png` — Historical EURUSD close price series (2022–2026).
2. `eurusd_return_distribution.png` — M15 log return density vs. normal fit and Q-Q diagnostic.
3. `eurusd_return_timeseries.png` — Continuous return time series highlighting volatility clustering.
4. `eurusd_rolling_volatility.png` — 20-bar and 96-bar rolling annualized volatility.
5. `eurusd_spread_distribution.png` — Spread distribution histogram and empirical CDF.
6. `eurusd_volume_distribution.png` — Tick volume distribution histogram and empirical CDF.
7. `eurusd_hourly_activity.png` — Diurnal volume and spread mechanics across 24 UTC hours.
8. `eurusd_weekday_activity.png` — Weekday distribution of tick volume and return volatility.
