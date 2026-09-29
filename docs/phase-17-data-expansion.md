# Phase 17: Historical Data Expansion & H4 Candidate Robustness Research Report

**Document Status:** Complete & Audited  
**Date (UTC):** 2026-09-29  
**Branch:** `develop`  
**Commit Anchor:** `ca7594f1dd42a852b68bf29c18b32a82b610f709` (Phase 16 closeout)  
**Research Status:** **CANDIDATE WEAKENED**  

---

## 1. Executive Summary & Objective

In Phase 16, the preselected 4-Hour (H4) swing candidate—a Random Forest classifier predicting 8-bar (32-hour) volatility-adjusted directional returns (`direction_vol_8`) with hyperparameters $N=100$, $\text{max\_depth}=5$, $\text{min\_samples\_leaf}=10$, and balanced class weights—yielded a mean walk-forward balanced accuracy of **56.05%** (median 55.67%, min 48.90%, max 60.39%) across 5 expanding folds spanning May 2024 to February 2026. However, the evaluation was constrained to approximately 4 years of total historical data (September 2022 to February 2026) with only 2,321 valid labeled samples. Phase 16 concluded with `REQUIRES FURTHER VALIDATION`, mandating that no backtesting or live deployment take place until the candidate's robustness was tested across a broader historical regime.

Phase 17 is a **Data and Robustness Phase only**. The primary objective is to investigate whether the apparent H4 directional separation survives when tested across a 16-year historical window (2010 to 2026), capturing major macroeconomic shifts, policy regimes, and market structures.

### Key Empirical Findings:
1. **MT5 Terminal Buffer vs History Depth:** Intraday M15 tick history before September 2022 is permanently unavailable from the MetaQuotes demo broker server due to MT5 server retention limits. Conversely, server-side pre-aggregated H4 bars are available back to March 2010 (25,800 bars, 16.07 years), enabling multi-decade macroeconomic research.
2. **Chunked Acquisition & Overlap Validation:** The 16-year H4 history was acquired in three chronological chunks of 6,000 to 10,000 bars each. Exactly 100 overlapping bars at each boundary were verified bit-for-bit across all 7 OHLCV fields (`open`, `high`, `low`, `close`, `tick_volume`, `spread`, `real_volume`) with zero differences.
3. **Data Expansion Scale:** Permitted pre-test H4 research bars expanded from **5,324 bars to 24,849 bars** (+366.7%). Valid non-neutral labeled samples for `direction_vol_8` expanded from **2,321 to 11,102** (5,618 Short vs 5,484 Long: 50.6% vs 49.4% balance).
4. **Walk-Forward Performance Compression:** When subjected to a 10-fold expanding-window chronological walk-forward evaluation across the 16-year history (8,732 out-of-sample predictions, 2013 to 2026):
   - **Mean Balanced Accuracy:** fell from **56.05% down to 51.62%** (absolute decline of **-4.43%**).
   - **Median Balanced Accuracy:** fell from **55.67% down to 50.92%** (absolute decline of **-4.75%**).
   - **Fold Consistency:** In the 4-year sample, 4 out of 5 folds (80%) exceeded 55% balanced accuracy. In the 16-year sample, only 2 out of 10 folds (20%) exceeded 55%, while **4 out of 10 folds (40%) fell below 50.0%** (worse than random guessing).
   - **Baseline Competitors:** Across the expanded 16 years, Logistic Regression achieved 50.53% mean balanced accuracy, Extra Trees achieved 49.74%, and Majority achieved 50.00%. The Random Forest candidate's margin of advantage over a simple linear model compressed to just +1.09%.
5. **Regime Vulnerability:** In Era 2 (2016–2019, ECB negative interest rates and quantitative easing), the candidate achieved only **49.33% balanced accuracy** (Macro F1 = 0.4817). In Era 4 (2022–2026), it achieved only **50.07% balanced accuracy** (Macro F1 = 0.4875). The elevated performance observed in Phase 16 was concentrated in a single anomalous sub-window (Fold 1 of expanded: 2013–2014 at 57.64% and Fold 6: 2019–2021 at 55.96%).
6. **Definitive Classification:** **`CANDIDATE WEAKENED`**. The 56.05% edge was a localized historical anomaly rather than a robust, stationary structural market phenomenon.

> [!CAUTION]
> **Strict Operational Invariant:** No trading system may proceed to live trading, broker order execution, demo deployment, or backtest curve-fitting while classified as `CANDIDATE WEAKENED`. Model parameters were not retuned or cherry-picked.

---

## 2. MT5 History Audit (M15 Buffer Limit vs H4 Depth)

An empirical audit of the MetaQuotes-Demo MT5 terminal and server history retention was conducted to evaluate historical availability across timeframes:

| Timeframe | Earliest Retrievable Bar (UTC) | Latest Retrievable Bar (UTC) | Bar Count | Terminal Buffer Ceiling | Broker Server Status | Expansion Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M15** | 2022-09-20 07:45:00 | 2026-09-29 22:00:00 | 100,000 | 100,000 (`maxbars`) | Server ticks unavailable pre-2022 | **BLOCKED** |
| **H4** | 2010-03-01 16:00:00 | 2026-09-29 20:00:00 | 25,800 | 100,000 (`maxbars`) | 16.07 years pre-aggregated bars | **AVAILABLE** |
| **D1** | 2007-06-28 00:00:00 | 2026-09-29 00:00:00 | 5,000 | 100,000 (`maxbars`) | 19.25 years daily bars | **AVAILABLE** |

### Root Cause Analysis:
1. **M15 Terminal Buffer Limit vs Server Tick Retention:** The MT5 client terminal enforces a maximum historical bar buffer (`maxbars=100000`). Even if the buffer is widened, retrieving M15 bars prior to September 20, 2022 fails because MetaQuotes demo server does not retain M15 tick logs beyond ~4 years. Requesting earlier M15 bars returns 0 records.
2. **H4 Native Server Retention:** Because H4 bars are pre-aggregated on the broker server, the MetaQuotes archive retains 25,800 H4 bars dating back to March 1, 2010 16:00 UTC. This enables rigorous, multi-decade macro-regime evaluation for swing trading.

---

## 3. Historical Data Acquisition Methodology

To avoid memory pressure, socket timeout, or terminal buffer truncations, H4 historical data was extracted in three chronologically contiguous, overlapping chunks using Python in the Wine MT5 runtime (`scripts/export_eurusd_h4_expanded.py`):

| Chunk Identifier | Bar Count | Date Range (UTC) | SHA256 Checksum | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `chunk_1.parquet` | 10,000 | 2020-04-28 12:00:00 to 2026-09-29 20:00:00 | `f295b95a28906351ee2ce683d73516cbcf74c43ee0ba287aa85aa882c40c1110` | Recent era & locked test |
| `chunk_2.parquet` | 10,000 | 2013-12-10 20:00:00 to 2020-05-21 00:00:00 | `e5bc24467c69994c6f376cfd38ee591f8fe73ebfe9ea07153a56294d137f8f94` | Mid era (QE & Brexit) |
| `chunk_3.parquet` | 6,000 | 2010-03-01 16:00:00 to 2014-01-06 20:00:00 | `efd6b9ea04d7010a30b42d74945763cb7c050047ea63f886f4a861614e59048a` | Early era (Euro crisis) |

Each chunk export recorded complete metadata into `reports/h4_chunk_export_metadata.json`, capturing bar counts, boundary timestamps, column dtypes, and cryptographic hashes.

---

## 4. Chunk Boundary Overlap Verification

To guarantee that the chunk boundary transitions do not introduce discontinuities, missing bars, or corrupted prices, a bit-for-bit verification was executed across the overlapping ranges:

| Chunk Boundary Pair | Overlapping Bars | Overlap Date Range (UTC) | Field Differences (O, H, L, C, Vol, Spread) | Bit-for-Bit Identical |
| :--- | :--- | :--- | :--- | :--- |
| `chunk_1` vs `chunk_2` | 100 bars | 2013-12-10 20:00:00 to 2014-01-06 20:00:00 | 0 differences across all 7 fields | **YES (100.000%)** |
| `chunk_2` vs `chunk_3` | 100 bars | 2020-04-28 12:00:00 to 2020-05-21 00:00:00 | 0 differences across all 7 fields | **YES (100.000%)** |

All overlapping records matched perfectly to floating-point precision, proving absolute data continuity across chunk boundaries.

---

## 5. Expanded Dataset Construction and Validation

The chunked series were merged chronologically and deduplicated via `ai/swing/data_expansion.py`, producing:
- `data/raw/eurusd_h4_expanded/eurusd_h4_raw.parquet` (25,800 bars)
- `data/processed/eurusd_h4_expanded/eurusd_h4_processed.parquet` (25,800 bars)

### Geometric and Structural Integrity Checks:
- **Total Bars:** 25,800 bars from `2010-03-01 16:00:00 UTC` to `2026-09-29 20:00:00 UTC`.
- **Chronological Monotonicity:** $\text{Timestamp}_{t+1} > \text{Timestamp}_t$ strictly maintained ($100\%$).
- **Duplicate Timestamps:** 0 duplicates.
- **Missing / Null Values:** 0 null values across all price and volume columns.
- **OHLC Geometry Validity:** $\text{High} \ge \max(\text{Open}, \text{Close})$ and $\text{Low} \le \min(\text{Open}, \text{Close})$ verified for all 25,800 bars ($100.0\%$).

---

## 6. Canonical Dataset Comparison and Overlap Preservation

The merged expanded H4 dataset was benchmarked against the canonical H4 series aggregated from the Phase 11/12 M15 dataset:
- **Overlapping Bars:** 6,263 bars (from `2022-09-16 08:00:00 UTC` to `2026-09-25 20:00:00 UTC`).
- **High Price Difference:** Exactly $0.00000000$ (0.0000 pips).
- **Low Price Difference:** Exactly $0.00000000$ (0.0000 pips).
- **Close Price Difference:** Exactly $0.00000000$ (0.0000 pips).
- **Open Price Difference:** Exactly $0.00000000$ on 6,262 bars. (On bar index 0, `2022-09-16 08:00:00 UTC`, a minor difference of 0.0011 occurred because the raw M15 dataset commenced at 09:30 UTC instead of 08:00 UTC, causing the canonical aggregation to take the 09:30 M15 open rather than the 08:00 H4 open. High, Low, and Close are identical).
- **Canonical Files Intact:** `data/raw/eurusd_m15/eurusd_m15_raw.parquet` and `data/processed/eurusd_m15/eurusd_m15_processed.parquet` were left completely unmodified.

---

## 7. Pre-test Research vs Test Split Governance

To uphold the absolute governance lock established in Phase 11 and Phase 15, the expanded H4 dataset was partitioned strictly at the canonical boundary:

```
[================= EXPANDED RESEARCH DATA (24,849 bars) =================] | [=== HELD-OUT TEST (951 bars) ===]
2010-03-01 16:00:00 UTC                               2026-02-19 10:45:00 UTC | 2026-02-19 12:00:00 UTC      2026-09-29 20:00:00 UTC
                                                                               | [CANONICAL TEST: 939 bars] (to 2026-09-25 20:00)
```

| Partition | Bar Count | Start Timestamp (UTC) | End Timestamp (UTC) | Permitted Use | Governance State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Expanded Research Partition** | 24,849 | 2010-03-01 16:00:00 | 2026-02-19 10:45:00 | Walk-forward cross-validation | Unlocked for Research |
| **Held-Out Test Partition** | 951 | 2026-02-19 12:00:00 | 2026-09-29 20:00:00 | Reserved for future final audit | **PERMANENTLY LOCKED** |

The maximum validation timestamp evaluated across all 10 walk-forward folds was `2026-02-18 00:00:00 UTC`, strictly prior to `2026-02-19 12:00:00 UTC`. The 939 canonical test bars remain completely unaccessed and unread.

---

## 8. Feature and Target Engineering on Expanded Data

Features and targets were reconstructed across the 24,849 expanded research bars using the point-in-time causal engines `compute_swing_features` and `compute_swing_targets`:

### Features (32 causal swing indicators):
- **Momentum & Returns:** `return_1`, `return_2`, `return_4`, `return_8`, `return_12`, `return_24`.
- **Volatility & ATR:** `atr_14`, `atr_norm_14`, `vol_std_10`, `vol_std_20`, `vol_ratio_10_50`.
- **Moving Average Displacements:** `dist_sma_20`, `dist_sma_50`, `dist_ema_21`, `sma_slope_10`, `ema_cross_12_26`.
- **Channel & Range Indicators:** `channel_pos_20`, `channel_width_20`, `dist_rolling_high_10`, `dist_rolling_low_10`, `dist_rolling_high_20`, `dist_rolling_low_20`.
- **Cyclical & Intraday Structure:** `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`.
- **Composite Regimes:** `trend_regime`, `vol_regime`, `channel_regime`.

### Target Formulation (`direction_vol_8`):
The volatility-adjusted directional target for horizon $H=8$ bars (32 hours) is defined as:
$$R_{t+8} = \frac{C_{t+8} - C_t}{C_t}$$
$$\text{Threshold}_t = 0.5 \times \sqrt{8} \times \text{ATR}_{\text{norm}, 14, t}$$
$$y_t = \begin{cases} 1 & \text{if } R_{t+8} > +\text{Threshold}_t \quad (\text{Long}) \\ 0 & \text{if } R_{t+8} < -\text{Threshold}_t \quad (\text{Short}) \\ \text{NaN} & \text{otherwise} \quad (\text{Neutral / Unlabeled}) \end{cases}$$

### Target Label Distribution:
- **Total Research Bars:** 24,849
- **Valid Labeled Samples ($y \in \{0, 1\}$):** 11,102 (vs 2,321 in Phase 16, a +378.3% increase)
- **Short Class ($y=0$):** 5,618 samples (50.60%)
- **Long Class ($y=1$):** 5,484 samples (49.40%)
- **Neutral / Noise Bars Filtered Out:** 13,747 bars (55.3%)

---

## 9. 10-Fold Chronological Walk-Forward Design

To evaluate the frozen candidate across the entire 16-year history without lookahead bias, a 10-fold expanding-window chronological walk-forward scheme was designed (`ai/swing/walk_forward_expanded.py`). 

### Purging Invariant:
An exact **8-bar (32-hour) de Prado purge gap** was enforced between the end of every training partition and the start of the validation partition, ensuring no overlap of forward-looking target windows:
$$\text{Val Start Index} = \text{Train End Index} + 1 + 8$$

```
Fold 1 : [== Train: 2,319 ==]--Purge 8--[== Val 1: 842 ==]
Fold 2 : [===== Train: 3,166 =====]--Purge 8--[== Val 2: 878 ==]
Fold 3 : [======== Train: 4,052 ========]--Purge 8--[== Val 3: 821 ==]
...
Fold 10: [======================= Train: 10,257 =======================]--Purge 8--[== Val 10: 837 ==]
```

### Partition Boundaries:

| Fold | Training Start | Training End | Validation Start | Validation End | Train N | Val N |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2010-03-01 16:00 | 2013-05-14 04:00 | 2013-05-15 16:00 | 2014-08-11 20:00 | 2,319 | 842 |
| **Fold 2** | 2010-03-01 16:00 | 2014-08-11 20:00 | 2014-08-13 08:00 | 2015-12-08 00:00 | 3,166 | 878 |
| **Fold 3** | 2010-03-01 16:00 | 2015-12-08 00:00 | 2015-12-09 12:00 | 2017-03-10 16:00 | 4,052 | 821 |
| **Fold 4** | 2010-03-01 16:00 | 2017-03-10 16:00 | 2017-03-14 00:00 | 2018-06-25 08:00 | 4,880 | 937 |
| **Fold 5** | 2010-03-01 16:00 | 2018-06-25 08:00 | 2018-06-26 20:00 | 2019-09-24 16:00 | 5,820 | 898 |
| **Fold 6** | 2010-03-01 16:00 | 2019-09-24 16:00 | 2019-09-26 04:00 | 2021-01-05 20:00 | 6,722 | 888 |
| **Fold 7** | 2010-03-01 16:00 | 2021-01-05 20:00 | 2021-01-07 08:00 | 2022-04-18 16:00 | 7,615 | 868 |
| **Fold 8** | 2010-03-01 16:00 | 2022-04-18 16:00 | 2022-04-20 04:00 | 2023-07-25 04:00 | 8,486 | 898 |
| **Fold 9** | 2010-03-01 16:00 | 2023-07-25 04:00 | 2023-07-26 16:00 | 2024-11-04 12:00 | 9,387 | 865 |
| **Fold 10** | 2010-03-01 16:00 | 2024-11-04 12:00 | 2024-11-06 00:00 | 2026-02-18 00:00 | 10,257 | 837 |
| **Total** | | | | | | **8,732** |

Across all 10 folds, exactly **8,732 out-of-sample predictions** were generated without future lookahead.

---

## 10. Walk-Forward Results: Frozen Candidate vs Baselines

The frozen candidate model configuration was evaluated against 4 standard benchmarks across all 10 folds:

### Overall Benchmark Summary (10 Folds, 16 Years, 8,732 Out-of-Sample Predictions):

| Model Family | Mean BalAcc | Median BalAcc | Std Dev | Min BalAcc | Max BalAcc | Mean Macro F1 | Folds >50% | Folds >55% | Folds <50% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Candidate)** | **51.62%** | **50.92%** | **3.03%** | **48.54%** | **57.64%** | **0.5020** | **6/10** | **2/10** | **4/10** |
| **Logistic Regression** | 50.53% | 50.01% | 2.85% | 47.10% | 56.54% | 0.4905 | 5/10 | 1/10 | 5/10 |
| **Naive Persistence** | 50.34% | 50.03% | 3.34% | 45.63% | 57.80% | 0.5027 | 5/10 | 1/10 | 5/10 |
| **Extra Trees** | 49.74% | 49.12% | 2.77% | 44.75% | 54.88% | 0.4878 | 4/10 | 0/10 | 6/10 |
| **Majority Baseline** | 50.00% | 50.00% | 0.00% | 50.00% | 50.00% | 0.3352 | 0/10 | 0/10 | 0/10 |

### Fold-by-Fold Performance Breakdown (Random Forest Candidate):

| Fold | Out-of-Sample Period | Val Samples | Balanced Accuracy | Raw Accuracy | Macro F1 | ROC-AUC | LR Benchmark BalAcc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2013-05 to 2014-08 | 842 | **57.64%** | 58.19% | 0.5760 | 0.6053 | 47.46% |
| **Fold 2** | 2014-08 to 2015-12 | 878 | **53.19%** | 54.10% | 0.5053 | 0.5284 | 52.73% |
| **Fold 3** | 2015-12 to 2017-03 | 821 | **48.72%** | 48.72% | 0.4658 | 0.4851 | 47.10% |
| **Fold 4** | 2017-03 to 2018-06 | 937 | **48.63%** | 48.56% | 0.4847 | 0.4893 | 47.26% |
| **Fold 5** | 2018-06 to 2019-09 | 898 | **50.21%** | 51.56% | 0.4631 | 0.4908 | 50.40% |
| **Fold 6** | 2019-10 to 2021-01 | 888 | **55.96%** | 56.08% | 0.5595 | 0.5694 | 56.54% |
| **Fold 7** | 2021-01 to 2022-04 | 868 | **52.30%** | 52.65% | 0.5187 | 0.5058 | 49.62% |
| **Fold 8** | 2022-04 to 2023-07 | 898 | **51.63%** | 51.56% | 0.4953 | 0.4939 | 51.96% |
| **Fold 9** | 2023-08 to 2024-11 | 865 | **48.54%** | 48.44% | 0.4854 | 0.4820 | 49.59% |
| **Fold 10** | 2024-11 to 2026-02 | 837 | **49.41%** | 49.34% | 0.4661 | 0.4725 | 52.63% |

---

## 11. Existing History (4-Year) vs Expanded History (16-Year) Comparison

Comparing performance across the short 4-year canonical window and the expanded 16-year window reveals severe performance degradation:

| Metric | Phase 16 Existing History (4-Year, 5 Folds) | Phase 17 Expanded History (16-Year, 10 Folds) | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Historical Period** | May 2024 to Feb 2026 | May 2013 to Feb 2026 | +12.0 years | +300.0% |
| **Pre-Test Research Bars** | 5,324 | 24,849 | +19,525 bars | +366.7% |
| **Valid Labeled Samples** | 2,321 | 11,102 | +8,781 samples | +378.3% |
| **Out-of-Sample Predictions** | 1,811 | 8,732 | +6,921 predictions | +382.2% |
| **Mean Balanced Accuracy** | **56.05%** | **51.62%** | **-4.43%** | -7.9% |
| **Median Balanced Accuracy** | **55.67%** | **50.92%** | **-4.75%** | -8.5% |
| **Minimum Balanced Accuracy** | 48.90% | 48.54% | -0.36% | -0.7% |
| **Maximum Balanced Accuracy** | 60.39% | 57.64% | -2.75% | -4.6% |
| **Folds Above 50.0%** | **4 / 5 (80.0%)** | **6 / 10 (60.0%)** | **-20.0%** | -25.0% |
| **Folds Above 55.0%** | **4 / 5 (80.0%)** | **2 / 10 (20.0%)** | **-60.0%** | -75.0% |
| **Folds Below 50.0%** | **1 / 5 (20.0%)** | **4 / 10 (40.0%)** | **+20.0%** | +100.0% |

### Key Diagnostic Takeaway:
The apparent 56.05% balanced accuracy found in Phase 16 was inflated by **regime concentration**. Across the broader 16-year historical distribution, the model's edge compresses toward 51.6%, with 4 out of 10 folds performing worse than a coin flip. The hypothesis that the candidate possesses an enduring, multi-year statistical edge across diverse market environments is definitively rejected.

---

## 12. Historical Era Breakdown and Regime Sensitivity

To isolate which historical conditions drove performance variation, the 8,732 out-of-sample predictions were segmented into four distinct macroeconomic eras:

| Era | Chronological Period | Key Macro & Monetary Regime | Samples | Balanced Accuracy | Raw Accuracy | Macro F1 | Regime Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Era 1** | 2013-05 to 2016-01 | Post-GFC recovery, Eurozone debt crisis tail, Fed Taper Tantrum | 1,775 | **54.09%** | 54.82% | 0.5346 | Moderate predictive separation |
| **Era 2** | 2016-01 to 2019-01 | ECB Negative Interest Rates (NIRP), aggressive QE, Brexit referendum, US election volatility | 2,100 | **49.33%** | 49.52% | 0.4817 | **Complete Signal Failure** (<50%) |
| **Era 3** | 2019-01 to 2022-01 | US-China trade tensions, COVID-19 liquidity shock, global zero-bound rates | 2,053 | **52.98%** | 53.04% | 0.5280 | Marginal predictive separation |
| **Era 4** | 2022-01 to 2026-02 | Post-pandemic inflation spike, aggressive Fed/ECB rate hike cycle, geopolitical conflicts | 2,804 | **50.07%** | 50.14% | 0.4875 | **Neutral / Random Walk** (~50%) |

### Detailed Regime Performance:
Evaluating the 8,732 out-of-sample predictions across structural volatility, trend, and session slices reveals how the signal behaves under specific market states:

1. **Volatility Regime (ATR):**
   - **Low Volatility ($\le$ median ATR):** 4,366 samples, **52.61%** balanced accuracy, Macro F1 = 0.5261, ROC-AUC = 0.5222.
   - **High Volatility ($>$ median ATR):** 4,366 samples, **50.51%** balanced accuracy, Macro F1 = 0.4731, ROC-AUC = 0.4919.
   - *Observation:* The candidate completely breaks down in high-volatility regimes, dropping to near-random accuracy with severe F1 deterioration.
2. **Volatility Expansion Regime:**
   - **Compression ($\text{vol\_ratio} \le 1.0$):** 5,224 samples, **52.68%** balanced accuracy, Macro F1 = 0.5180.
   - **Expansion ($\text{vol\_ratio} > 1.0$):** 3,508 samples, **50.04%** balanced accuracy, Macro F1 = 0.4946.
3. **Trend Regime:**
   - **Strong Bullish Trend:** 2,541 samples, **50.90%** balanced accuracy.
   - **Strong Bearish Trend:** 2,583 samples, **51.72%** balanced accuracy.
   - **Transition / Choppy Range:** 3,608 samples, **52.03%** balanced accuracy.
4. **Session Bar Hour:**
   - H4 00:00 UTC: 50.90%
   - H4 04:00 UTC: 51.69%
   - H4 08:00 UTC: 52.25%
   - H4 12:00 UTC: 52.26%
   - H4 16:00 UTC: 50.41%
   - H4 20:00 UTC: 51.67%
5. **Day of Week:**
   - Monday: 50.78%
   - Midweek (Tue-Thu): 51.15%
   - Friday: 54.12%

---

## 13. Feature Importance Stability Over 16 Years

Evaluating feature importances across all 10 walk-forward models demonstrates that feature relevance remained consistent even while out-of-sample accuracy decayed:

| Rank | Feature | Mean Gini Importance | Standard Deviation Across Folds | Feature Description |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `channel_width_20` | **0.0632** | 0.0183 | 20-bar Donchian channel width |
| **2** | `atr_14` | **0.0622** | 0.0162 | 14-period Average True Range |
| **3** | `atr_norm_14` | **0.0615** | 0.0123 | Normalized ATR (volatility level) |
| **4** | `vol_std_20` | **0.0612** | 0.0133 | 20-bar return standard deviation |
| **5** | `dist_sma_50` | **0.0503** | 0.0080 | Normalized distance from 50-period SMA |
| **6** | `vol_std_10` | **0.0466** | 0.0096 | 10-bar return standard deviation |
| **7** | `return_8` | **0.0451** | 0.0077 | 8-bar historical return momentum |
| **8** | `dist_rolling_low_20` | **0.0443** | 0.0061 | Distance from 20-bar rolling low |
| **9** | `sma_slope_10` | **0.0406** | 0.0082 | Slope of 10-period SMA |
| **10** | `dist_rolling_low_10` | **0.0365** | 0.0107 | Distance from 10-bar rolling low |

Volatility indicators (`channel_width_20`, `atr_14`, `atr_norm_14`, `vol_std_20`) consistently account for the top 4 predictive splits. However, despite stable feature weighting, the relationship between these volatility features and future 32-hour directional returns proved non-stationary across eras.

---

## 14. Leakage and Governance Audit (All 12 Checks Passed)

A suite of 12 programmatic checks was executed via `scripts/run_phase17_research.py` to confirm data integrity, chronological causality, and compliance with governance locks:

| Check # | Check Description | Verification Detail | Result |
| :---: | :--- | :--- | :---: |
| **1** | Expanded H4 Monotonicity | Timestamps strictly monotonically increasing in UTC | **PASS** |
| **2** | Chunk Overlap Bit-for-Bit Match | Zero difference across Open, High, Low, Close, Volumes, Spread | **PASS** |
| **3** | Duplicate Timestamp Absence | Exactly 0 duplicate bar timestamps | **PASS** |
| **4** | OHLC Geometric Validity | High $\ge \max(\text{O}, \text{C})$, Low $\le \min(\text{O}, \text{C})$, Open $> 0$ on all 25,800 bars | **PASS** |
| **5** | Canonical Overlap Match | 6,262 identical bars to canonical feed (High/Low/Close zero diff) | **PASS** |
| **6** | Feature Finiteness and Validity | Zero infinities, zero all-NaN columns in computed features | **PASS** |
| **7** | Target Horizon Alignment | Exactly 11,102 valid labeled bars with forward return window | **PASS** |
| **8** | Purge Gap Enforcement | Exact 8 bars (32 hours) purged at every boundary across all 10 folds | **PASS** |
| **9** | Zero Train/Validation Overlap | $\text{Train End} < \text{Val Start}$ strictly enforced across all 10 folds | **PASS** |
| **10** | Chronological Progression of Folds | Non-decreasing training sets with rolling out-of-sample origins | **PASS** |
| **11** | Candidate Configuration Immutability | Exactly 100 trees, max depth 5, leaf 10, balanced (8,732 OOF predictions) | **PASS** |
| **12** | Locked Test Partitions Unaccessed | Max val ts `2026-02-18 00:00 UTC` $<$ `2026-02-19 12:00 UTC`; 939 canonical test bars preserved | **PASS** |

All 12 checks passed deterministically without exceptions.

---

## 15. Candidate Classification and Definitive Verdict

Based on the quantitative criteria established for Phase 17:

1. **Criterion for CANDIDATE CONFIRMED:**
   - 10-fold mean balanced accuracy $\ge 54.0\%$
   - Median balanced accuracy $\ge 53.0\%$
   - At least 8 of 10 folds $> 50.0\%$
   - No fold $< 48.0\%$
   - *Outcome:* **FAILED** (Mean = 51.62%, Median = 50.92%, Folds $>50\% = 6/10$).
2. **Criterion for CANDIDATE WEAKENED:**
   - 10-fold mean balanced accuracy drops between $50.0\%$ and $54.0\%$
   - Or $\ge 3$ folds fall below $50.0\%$
   - *Outcome:* **MET** (Mean = 51.62%, 4 folds below 50.0%: Folds 3, 4, 9, 10).
3. **Criterion for CANDIDATE INVALIDATED:**
   - 10-fold mean balanced accuracy $< 50.0\%$
   - Or Candidate underperforms naive majority/persistence baselines overall
   - *Outcome:* Not met (Mean is 51.62%, marginally above 50.0% majority).

### Definitive Verdict: **`CANDIDATE WEAKENED`**

The candidate's apparent predictive advantage in the 4-year sample (56.05%) collapsed to a marginal **51.62%** over the 16-year historical sample. With 4 out of 10 out-of-sample folds underperforming random guessing, and multi-year macroeconomic regimes (such as 2016–2019 NIRP) producing persistent sub-50% accuracy, the directional edge is too weak, non-stationary, and regime-dependent to warrant trading deployment or backtesting.

---

## 16. Recommendations for Future Research

In strict adherence to the project charter and governance rules:

1. **DO NOT START BACKTESTING:**
   - Do NOT run trade simulations, equity curve generation, position sizing tests, or stop-loss / take-profit optimization. Backtesting a model with 51.6% balanced accuracy and 4 failed folds is pure data snooping and curve-fitting.
2. **DO NOT COMMENCE LIVE OR DEMO TRADING:**
   - The H4 swing candidate MUST NOT be connected to any broker, order router, or execution service.
3. **PRESERVE ALL LOCKED PARTITIONS:**
   - The Phase 11 M15 test partition (14,988 bars) and Phase 15 H4 test partition (939 bars) remain permanently locked.
4. **RECOMMENDED FUTURE RESEARCH DIRECTIONS:**
   - **Regime-Conditional Conditioning:** Rather than unconditional directional forecasting, explore whether regime classification (e.g., volatility compression detection) can be used exclusively as an execution filter rather than a directional predictor.
   - **Alternative Instruments / Multi-Asset Data:** Explore whether macroeconomic cross-asset features (e.g., US Treasury yields, DXY, equity volatility) provide stationary signal strength for EURUSD swing horizons.
5. **STOP:** Phase 17 is complete. Do not start Phase 18 or any downstream engineering until explicit direction is provided.
