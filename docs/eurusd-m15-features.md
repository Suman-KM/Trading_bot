# EURUSD M15 Feature Engineering Specification

**Phase:** Phase 5 — Quantitative Feature Engineering  
**Instrument:** EURUSD  
**Timeframe:** M15 (15-Minute Candles)  
**Input Source:** `data/processed/eurusd_m15/eurusd_m15_processed.parquet` (100,000 candles)  
**Output Dataset:** `data/features/eurusd_m15/eurusd_m15_features.parquet` (100,000 rows $\times$ 89 columns)  
**Metadata Spec:** `reports/feature_metadata.json` (Machine-readable metadata contract)  
**Environment:** Canonical Ubuntu Linux x86_64, Python 3.13.15  

---

## 1. Feature Engineering Philosophy

The objective of Phase 5 is to construct a compact, mathematically rigorous, and strictly causal set of quantitative market features. 

Key principles governing the design:
1. **Strict Point-in-Time Causality:** Any feature evaluated at candle index $t$ uses exclusively information that became observable at or before the completion of candle $t$. Look-ahead operations, forward-filling across market closures, and future target leakage are strictly prevented.
2. **Dimensionally Grounded & Scale-Free:** Where appropriate, features are normalized relative to price or rolling variance (e.g. relative wicks, normalized ranges, percentage distances from moving averages, z-scores) to ensure scale-invariance across varying macro price regimes (e.g., EURUSD moving from 0.95 to 1.20).
3. **Multi-Scale Temporal Horizons:** M15 candles exhibit microstructure noise on short scales and structural regimes on longer scales. Features leverage structured lookback windows of 5, 10, 20, 40, and 80 bars (spanning 1.25 hours to 20.0 hours).
4. **No Premature Model Assumptions:** Feature engineering is descriptive transformation. No predictive claims or alpha assertions are made during Phase 5.

---

## 2. Feature Groups Summary

The pipeline extracts **80 distinct features** organized across 7 functional groups:

| Feature Group | Feature Count | Primary Input Columns | Max Lookback | Description |
| :--- | :---: | :--- | :---: | :--- |
| **Price Action** | 14 | `open`, `high`, `low`, `close` | 20 bars | Returns, candle geometry, wick ratios, directional signs |
| **Momentum** | 10 | `close` | 80 bars | Multi-scale Rate of Change (ROC), distance from SMAs, 14-period RSI |
| **Volatility** | 13 | `high`, `low`, `close` | 80 bars | Log return std dev, annualized vol, ATR, normalized ATR, vol ratios |
| **Trend** | 16 | `close` | 80 bars | SMAs, EMAs, normalized price-to-EMA distances, EMA spreads, slopes |
| **Market Activity**| 9 | `tick_volume`, `spread`, `close` | 80 bars | Volume z-scores, volume ratios, normalized spread, spread z-score |
| **Time / Calendar**| 13 | `timestamp` | 0 bars | Harmonic sine/cosine cyclical encodings, FX session indicators |
| **Gap / Continuity**| 5 | `time`, `timestamp` | 1 bar | Delta seconds, normal interval flag, post-gap flag, weekend reopen |
| **Total** | **80** | — | **80 bars** | Unified feature matrix |

---

## 3. Mathematical Definitions

### 3.1 Price Action & Candle Geometry (14 Features)

Let $O_t, H_t, L_t, C_t$ denote the open, high, low, and close prices at bar $t$.

- **Simple 1-Bar Return (`return_1`):**  
  $$R_t = \frac{C_t - C_{t-1}}{C_{t-1}}$$
- **Logarithmic 1-Bar Return (`log_return_1`):**  
  $$r_t = \ln\left(\frac{C_t}{C_{t-1}}\right)$$
- **Intra-Bar Return (`open_to_close_return`):**  
  $$R^{\text{intra}}_t = \frac{C_t - O_t}{O_t}$$
- **High-Low Range (`hl_range`):**  
  $$\text{Range}_t = H_t - L_t$$
- **Normalized High-Low Range (`hl_range_norm`):**  
  $$\text{Range}^{\text{norm}}_t = \frac{H_t - L_t}{C_t}$$
- **Candle Body (`candle_body`):**  
  $$\text{Body}_t = |C_t - O_t|$$
- **Normalized Candle Body (`candle_body_norm`):**  
  $$\text{Body}^{\text{norm}}_t = \frac{|C_t - O_t|}{C_t}$$
- **Upper Wick (`upper_wick`):**  
  $$\text{Wick}^{\text{upper}}_t = H_t - \max(O_t, C_t)$$
- **Upper Wick Ratio (`upper_wick_ratio`):**  
  $$\text{Ratio}^{\text{upper}}_t = \frac{\text{Wick}^{\text{upper}}_t}{\text{Range}_t + 10^{-8}}$$
- **Lower Wick (`lower_wick`):**  
  $$\text{Wick}^{\text{lower}}_t = \min(O_t, C_t) - L_t$$
- **Lower Wick Ratio (`lower_wick_ratio`):**  
  $$\text{Ratio}^{\text{lower}}_t = \frac{\text{Wick}^{\text{lower}}_t}{\text{Range}_t + 10^{-8}}$$
- **Candle Direction (`candle_direction`):**  
  $$\text{Dir}_t = \text{sign}(C_t - O_t) \in \{-1.0, 0.0, +1.0\}$$
- **Rolling Mean Returns (`return_mean_5`, `return_mean_20`):**  
  $$\bar{r}_{t, k} = \frac{1}{k} \sum_{i=0}^{k-1} r_{t-i}, \quad k \in \{5, 20\}$$

---

### 3.2 Momentum Features (10 Features)

- **Rate of Change (`roc_5`, `roc_10`, `roc_20`, `roc_40`, `roc_80`):**  
  $$\text{ROC}_{t, w} = \frac{C_t - C_{t-w}}{C_{t-w}}, \quad w \in \{5, 10, 20, 40, 80\}$$
- **Price Distance from SMA (`dist_sma_10`, `dist_sma_20`, `dist_sma_40`, `dist_sma_80`):**  
  $$\text{DistSMA}_{t, w} = \frac{C_t - \text{SMA}_{t, w}}{\text{SMA}_{t, w}}, \quad \text{where } \text{SMA}_{t, w} = \frac{1}{w}\sum_{i=0}^{w-1} C_{t-i}$$
- **14-Period Relative Strength Index (`rsi_14`):**  
  Using Wilder's exponential smoothing ($\alpha = 1/14$):  
  $$\Delta C_t = C_t - C_{t-1}$$  
  $$U_t = \max(\Delta C_t, 0), \quad D_t = \max(-\Delta C_t, 0)$$  
  $$\text{RS}_t = \frac{\text{EMA}_{14}(U)_t}{\text{EMA}_{14}(D)_t + 10^{-12}}, \quad \text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}$$

---

### 3.3 Volatility & Dispersion Features (13 Features)

- **Rolling Return Standard Deviation (`vol_std_10`, `vol_std_20`, `vol_std_40`, `vol_std_80`):**  
  $$\sigma_{t, w} = \sqrt{\frac{1}{w-1}\sum_{i=0}^{w-1} (r_{t-i} - \bar{r}_{t, w})^2}, \quad w \in \{10, 20, 40, 80\}$$
- **Rolling Annualized Volatility (`vol_ann_20`, `vol_ann_80`):**  
  Assuming standard M15 annual trading density ($252 \times 24 \times 4 = 24,192$ bars/year):  
  $$\sigma^{\text{ann}}_{t, w} = \sigma_{t, w} \times \sqrt{24,192}$$
- **True Range ($\text{TR}_t$):**  
  $$\text{TR}_t = \max\left(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|\right)$$
- **Average True Range (`atr_14`, `atr_40`):**  
  $$\text{ATR}_{t, w} = \frac{1}{w}\sum_{i=0}^{w-1} \text{TR}_{t-i}, \quad w \in \{14, 40\}$$
- **Normalized ATR (`atr_norm_14`, `atr_norm_40`):**  
  $$\text{ATR}^{\text{norm}}_{t, w} = \frac{\text{ATR}_{t, w}}{C_t}$$
- **Volatility Term Structure Ratios (`vol_ratio_10_40`, `vol_ratio_20_80`):**  
  $$\text{VolRatio}_{t, 10/40} = \frac{\sigma_{t, 10}}{\sigma_{t, 40} + 10^{-8}}, \quad \text{VolRatio}_{t, 20/80} = \frac{\sigma_{t, 20}}{\sigma_{t, 80} + 10^{-8}}$$
- **Rolling High-Low Envelope Ratio (`rolling_hl_ratio_20`):**  
  $$\text{EnvelopeRatio}_{t, 20} = \frac{\max_{i \in [0, 19]} H_{t-i} - \min_{i \in [0, 19]} L_{t-i}}{C_t}$$

---

### 3.4 Trend & Moving Average Features (16 Features)

- **Simple Moving Averages (`sma_10`, `sma_20`, `sma_40`, `sma_80`):**  
  $$\text{SMA}_{t, w} = \frac{1}{w}\sum_{i=0}^{w-1} C_{t-i}$$
- **Exponential Moving Averages (`ema_10`, `ema_20`, `ema_40`, `ema_80`):**  
  $$\text{EMA}_{t, w} = \alpha C_t + (1 - \alpha) \text{EMA}_{t-1, w}, \quad \alpha = \frac{2}{w + 1}$$
- **Normalized Price Distance from EMA (`dist_ema_10`, `dist_ema_20`, `dist_ema_40`, `dist_ema_80`):**  
  $$\text{DistEMA}_{t, w} = \frac{C_t - \text{EMA}_{t, w}}{\text{EMA}_{t, w}}$$
- **Moving Average Spread (`ema_spread_10_40`, `ema_spread_20_80`):**  
  $$\text{Spread}_{t, 10/40} = \frac{\text{EMA}_{t, 10} - \text{EMA}_{t, 40}}{\text{EMA}_{t, 40}}, \quad \text{Spread}_{t, 20/80} = \frac{\text{EMA}_{t, 20} - \text{EMA}_{t, 80}}{\text{EMA}_{t, 80}}$$
- **Causal Trend Slope (`ema_slope_10`, `ema_slope_20`):**  
  5-bar lookback velocity normalized by lagged value:  
  $$\text{Slope}_{t, w} = \frac{\text{EMA}_{t, w} - \text{EMA}_{t-5, w}}{5 \times \text{EMA}_{t-5, w}}$$

---

### 3.5 Market Activity & Volume Features (9 Features)

- **Log Tick Volume (`log_tick_volume`):**  
  $$v^{\text{log}}_t = \ln(1 + V^{\text{tick}}_t)$$
- **Rolling Mean Tick Volume (`tick_vol_sma_20`):**  
  $$\bar{V}_{t, 20} = \frac{1}{20}\sum_{i=0}^{19} V^{\text{tick}}_{t-i}$$
- **Tick Volume Ratio (`tick_vol_ratio_20`):**  
  $$\text{VRatio}_{t, 20} = \frac{V^{\text{tick}}_t}{\bar{V}_{t, 20} + 10^{-8}}$$
- **Tick Volume Z-Scores (`tick_vol_zscore_20`, `tick_vol_zscore_80`):**  
  $$z^V_{t, w} = \frac{V^{\text{tick}}_t - \bar{V}_{t, w}}{\sigma^V_{t, w} + 10^{-8}}, \quad w \in \{20, 80\}$$
- **Normalized Spread (`spread_norm`):**  
  Expressing raw broker spread points ($10^{-5}$ USD) relative to spot price:  
  $$\text{Spread}^{\text{norm}}_t = \frac{S_t \times 10^{-5}}{C_t}$$
- **Rolling Mean Spread (`spread_sma_20`):**  
  $$\bar{S}_{t, 20} = \frac{1}{20}\sum_{i=0}^{19} S_{t-i}$$
- **Spread Ratio (`spread_ratio_20`):**  
  $$\text{SRatio}_{t, 20} = \frac{S_t}{\bar{S}_{t, 20} + 10^{-8}}$$
- **Spread Z-Score (`spread_zscore_20`):**  
  $$z^S_{t, 20} = \frac{S_t - \bar{S}_{t, 20}}{\sigma^S_{t, 20} + 10^{-8}}$$

---

### 3.6 Time & Diurnal Cyclical Features (13 Features)

- **Calendar Components (`hour`, `minute`, `day_of_week`):**  
  Extracted strictly from the UTC timestamp: $h_t \in [0, 23]$, $m_t \in \{0, 15, 30, 45\}$, $d_t \in [0, 6]$ (0 = Monday).
- **Harmonic Cyclical Encodings:**  
  Non-linear continuous projections preserving cyclical continuity across midnight and week boundaries:  
  $$\text{sin\_hour}_t = \sin\left(\frac{2\pi h_t}{24}\right), \quad \text{cos\_hour}_t = \cos\left(\frac{2\pi h_t}{24}\right)$$  
  $$\text{sin\_minute}_t = \sin\left(\frac{2\pi m_t}{60}\right), \quad \text{cos\_minute}_t = \cos\left(\frac{2\pi m_t}{60}\right)$$  
  $$\text{sin\_day\_of\_week}_t = \sin\left(\frac{2\pi d_t}{7}\right), \quad \text{cos\_day\_of\_week}_t = \cos\left(\frac{2\pi d_t}{7}\right)$$
- **Session Indicators (Binary Flags $\in \{0.0, 1.0\}$):**  
  - `is_asian_session`: $h_t \in [0, 7]$ (Tokyo / Sydney liquidity window)
  - `is_london_session`: $h_t \in [7, 15]$ (European institutional open)
  - `is_ny_session`: $h_t \in [12, 20]$ (London / New York overlap and US session)
  - `is_rollover_window`: $h_t \in [21, 23]$ (Global interbank rollover settlement)

---

### 3.7 Gap & Market Continuity Features (5 Features)

- **Timestamp Delta (`delta_seconds`):**  
  $$\Delta t_t = \text{Epoch}_t - \text{Epoch}_{t-1}$$
- **Normal Interval Indicator (`is_normal_m15`):**  
  $$\mathbf{1}(\Delta t_t == 900)$$
- **Post-Gap Indicator (`is_post_gap`):**  
  $$\mathbf{1}(\Delta t_t > 900)$$
- **Weekend Reopening Indicator (`is_weekend_reopen`):**  
  $$\mathbf{1}(\Delta t_t \ge 129,600 \land d_t \in \{0, 6\})$$
- **Bars Since Previous Gap (`bars_since_last_gap`):**  
  Causal integer counter of consecutive normal bars since the last structural gap (clipped at 96 bars = 24 hours of trading).

---

## 4. Lookback Windows & Justification

The selected lookback horizons directly reflect the structural dynamics of M15 foreign exchange trading:

| Window ($w$) | Equivalent Duration | Economic & Quantitative Rationale |
| :---: | :---: | :--- |
| **5 bars** | 1 hour 15 minutes | Immediate short-term momentum and noise filtering |
| **10 bars** | 2 hours 30 minutes | Intra-session impulse horizon |
| **14 bars** | 3 hours 30 minutes | Classical Wilder horizon for RSI and ATR |
| **20 bars** | 5 hours 00 minutes | Core session scale (roughly one active trading session) |
| **40 bars** | 10 hours 00 minutes | Trans-session scale (covering London to New York transitions) |
| **80 bars** | 20 hours 00 minutes | Full-day baseline regime (maximum lookback in pipeline) |

---

## 5. Feature NaN & Warm-Up Policy

### 5.1 Initial Warm-Up NaNs
Because rolling features require historical data to initialize, features with lookback window $w$ produce exactly $w$ initial `NaN` values. The maximum lookback window across the pipeline is **80 bars** (20 hours).

- **Warm-Up Range:** Rows 0 to 79 (80 bars out of 100,000, or 0.08% of the dataset).
- **Post-Warm-Up Data (Rows 80 to 99,999):** Exactly **99,920 bars with zero NaNs** across all 80 features.

### 5.2 Downstream Handling Policy
1. **Never Arbitrarily Zero-Fill:** Rolling features (such as SMA, ATR, and volatilities) must **not** be filled with zero, as zero has economic meaning (e.g. zero volatility) and injects severe distortion.
2. **Explicit Truncation:** For supervised ML training, the warm-up period is explicitly dropped (`df.iloc[80:]`), leaving 99,920 clean, complete rows.
3. **Preservation in Master File:** The primary feature parquet retains the raw index with NaNs intact to maintain exact 1-to-1 alignment with the input dataset.

---

## 6. Mandatory Leakage Safeguards & Empirical Verification

Point-in-time causality is enforced and verified via automated test suite `tests/test_features.py`:

1. **Future-Shift Test (`test_leakage_future_shift`):**  
   Perturbing candle $t+1$ (e.g. multiplying prices by 3.0 and volume by 10) produces mathematically **identical** feature values at candle $t$ and all prior candles ($t-1, \dots$).
2. **Truncation Test (`test_leakage_truncation`):**  
   Computing features on a truncated dataset ending at bar $t$ produces the exact same row vector at $t$ as computing on the complete 100,000-candle dataset.
3. **Timestamp Ordering Test (`test_leakage_timestamp_ordering`):**  
   Output features maintain strict 1-to-1 monotonic alignment with input timestamps without any re-indexing distortion.
4. **No-Target Test (`test_leakage_no_future_targets`):**  
   The feature matrix is strictly free of future return targets, forward labels, or lead operators.
5. **Gap Preservation Test (`test_leakage_gap_preservation`):**  
   Market closures (such as 48-hour weekends) are preserved as authentic structural events without artificial candle fabrication or forward interpolation.

---

## 7. Online Computability Contract

Every feature in the 80-feature library is **100% online computable**:
- To compute all features for a newly arriving live M15 bar at timestamp $t$, an execution engine requires only the trailing 80 bars of historical OHLCV data.
- The state required is bounded by a fixed-size ring buffer of size 80 ($O(1)$ memory, sub-millisecond computation time).

---

## 8. Limitations & Excluded Features

### 8.1 Explicit Exclusion of `real_volume`
- **Finding:** Phase 4 EDA verified that `real_volume` is **100.0% zero** in the MetaQuotes-Demo OTC forex feed.
- **Action:** Any features based on `real_volume` (e.g., VWAP, Volume-Weighted ATR, real volume ratios) are **strictly excluded**. No synthetic volume is fabricated.

### 8.2 Broker Quote Specificity
- `tick_volume` measures the frequency of price updates from the broker's liquidity feed rather than matched institutional contract lots.
- `spread` represents the snapshot spread at bar close, not the full intraday bid-ask trajectory.
- Models must not assume zero spread or infinite liquidity in production execution.
