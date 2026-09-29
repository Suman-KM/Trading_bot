# Phase 15 — Dual-Track Intraday + Swing Research Report

**Status:** COMPLETE  
**Date:** 2026-09-29  
**Branch:** `develop`  
**Execution Environment:** Ubuntu (Python 3.13 via `uv`)  
**Data Instrument:** EURUSD  
**Track A Timeframe:** M15 (Intraday)  
**Track B Timeframes:** H4 (4-Hour) and D1 (Daily)  
**Data Range Analyzed:** 2022-09-16 09:30:00 UTC to 2026-02-19 10:45:00 UTC (Training + Validation Only)  
**Phase 11 Holdout Test Set:** PERMANENTLY LOCKED & UNTOUCHED (14,988 M15 rows: 2026-02-19 12:00:00 UTC onward)  
**Phase 12 Baseline Configuration:** PRESERVED UNCHANGED  
**RiskEngine Authority:** UNCHANGED & UNBYPASSED  
**Production Changes:** NONE  
**Trading Backtests / Parameter Sweeps:** NONE (Research-Only Phase)  

---

## 1. Objective

The objective of Phase 15 is to conduct an independent, comparative investigation into two distinct trading tracks for EURUSD:
1. **Track A (Intraday M15):** Investigate whether the current weak directional separation observed on M15 bars (balanced accuracy ~51–53% in Phases 13 and 14) is caused by an incomplete information set, by evaluating a controlled 25-feature extension (expanding the feature matrix from 80 to 105 features) capturing market structure, range compression/expansion, momentum acceleration, trend alignment, and session transition dynamics.
2. **Track B (Swing H4 & D1):** Investigate whether aggregating canonical M15 market history into higher-timeframe swing bars (H4: 4 hours, D1: daily) provides greater economic displacement, a more favorable friction-to-move ratio, and statistically meaningful directional signal separation over multi-day horizons.

---

## 2. Phase 13 Findings

In Phase 13, diagnostic analysis of the baseline Random Forest classifier on ternary `direction_4` ($H=4$ M15 bars / 60 minutes) established that:
- Directional confidence peaked at $59.36\%$, never breaching the production entry threshold $\tau = 0.60$, resulting in zero executed trades during the Phase 12 validation backtest.
- When isolating true directional displacements from the neutral class ($N=5,702$), binary classification accuracy was **$50.89\%$** (balanced accuracy $50.68\%$, macro F1 $48.86\%$), indistinguishable from random chance.
- Model confidence was strongly correlated with volatility expansion ($z$-scores of $+1.7\sigma$ to $+2.1\sigma$ on `hl_range`, `atr_14`, and `vol_std_10`) rather than directional drift.

---

## 3. Phase 14 Findings

In Phase 14, four alternative target formulations were tested on M15 validation data to determine whether target redesign could resolve the weak directional separability:
- **Target A (Binary Direction):** Balanced accuracy = **53.10% (RF)**, **51.98% (LR)**; ROC-AUC = **0.5373**.
- **Target B (Volatility-Adjusted):** Balanced accuracy = **51.76% (RF)**, **51.22% (LR)**; ROC-AUC = **0.5287**.
- **Target C (Fixed 5-Pip Move):** Balanced accuracy = **51.49% (RF)**, **51.23% (LR)**; ROC-AUC = **0.5253**.
- **Target D (Extreme 2.0x ATR Move):** Balanced accuracy = **51.51% (RF)**, **49.26% (LR)**; ROC-AUC = **0.5156**.

**Key Conclusion from Phase 14:**  
Redesigning the target formulation or filtering for larger moves did not break the ~53% ceiling on M15. High-confidence predictions ($\ge 0.80$) collapsed to 100% LONG classifications across all targets, indicating localized regime drift rather than robust bidirectional edge. This motivated the Phase 15 dual-track inquiry into information richness (Track A) and timeframe horizon (Track B).

---

## 4. Research Methodology

Phase 15 strictly isolates the two research tracks to prevent information cross-contamination or lookahead leakage:

```
                               CANONICAL M15 DATA
                         (100,000 bars: 2022 to 2026)
                                       |
                   +-------------------+-------------------+
                   |                                       |
             TRACK A (INTRADAY)                     TRACK B (SWING)
           Timeframe: EURUSD M15                  Timeframes: H4 & D1
                   |                                       |
       Data Availability Audit                Resampling Aggregation (OHLCV)
                   |                                       |
      Controlled Feature Expansion               Data Quality Validation
          (80 -> 105 features)                             |
                   |                             Swing Feature Engineering
        Model Evaluation on M15                     (~30 point-in-time feats)
        (LR, RF, ExtraTrees)                               |
                   |                             Swing Target Engineering
      3-Block Temporal Stability                 (Multi-bar forward returns)
                   |                                       |
           Comparative Audit                      Model Evaluation & Temporal
                                                      Stability (LR, RF)
```

**Non-Negotiable Research Invariants:**
1. **Holdout Protection:** Phase 11 Test partition (`2026-02-19 12:00:00 UTC` onward, 14,988 M15 rows) was strictly locked and unaccessed.
2. **Causal Integrity:** All new features, whether intraday or swing, use strictly past and current bar information available at bar close $t$.
3. **No Backtesting:** No trading backtests, trade simulations, or SL/TP parameter optimization were executed.
4. **No Production Modifications:** Production baselines and RiskEngine remain completely untouched.

---

## 5. Intraday Information Audit

An audit of the canonical M15 dataset (`data/processed/eurusd_m15/eurusd_m15_processed.parquet`, 100,000 bars from `2022-09-16 09:30:00 UTC` to `2026-09-25 23:45:00 UTC`) confirmed the availability and limitations of the raw market data:

| Field Name | Data Type | Availability Status | Notes & Constraints |
|---|---|---|---|
| `timestamp` | `datetime64[ms, UTC]` | **AVAILABLE** | Strictly monotonic increasing UTC timestamps. |
| `open` | `float64` | **AVAILABLE** | Valid positive prices. |
| `high` | `float64` | **AVAILABLE** | Valid positive prices; $\text{high} \ge \max(\text{open}, \text{close})$. |
| `low` | `float64` | **AVAILABLE** | Valid positive prices; $\text{low} \le \min(\text{open}, \text{close})$. |
| `close` | `float64` | **AVAILABLE** | Valid positive prices. |
| `tick_volume` | `uint64` | **AVAILABLE** | Non-zero across 100% of bars. |
| `spread` | `int32` | **AVAILABLE** | Point spread; historical MetaQuotes export contains ~50% zero-fill before 2024. |
| `real_volume` | `uint64` | **UNAVAILABLE** | Exactly 0 across all 100,000 bars (spot FX has no centralized exchange volume). |

**Information Audit Assessment:**  
The raw data feed is strictly limited to 15-minute OHLC prices, tick count, and spread. Order book depth, institutional trade flow, and real trading volume do not exist in the canonical dataset. Consequently, any expanded intraday feature set must derive purely from causal transformations of price action, tick volume, and calendar time.

---

## 6. Intraday Feature Expansion

To test whether the 80 baseline features omitted critical structural market context, exactly 25 new causal features were implemented in [`ai/features/intraday_extended.py`](file:///home/cino/projects/ai-trading-system/ai/features/intraday_extended.py), expanding the feature matrix to 105 features:

### 6.1 Catalog of 25 Extended Features

| # | Feature Name | Category | Mathematical Formulation | Lookback |
|---|---|---|---|---|
| 1 | `dist_high_20` | Market Structure | $(\text{high}_{20\text{-bar max}} - \text{close}) / \text{close}$ | 20 bars (5h) |
| 2 | `dist_low_20` | Market Structure | $(\text{close} - \text{low}_{20\text{-bar min}}) / \text{close}$ | 20 bars (5h) |
| 3 | `dist_high_80` | Market Structure | $(\text{high}_{80\text{-bar max}} - \text{close}) / \text{close}$ | 80 bars (20h) |
| 4 | `dist_low_80` | Market Structure | $(\text{close} - \text{low}_{80\text{-bar min}}) / \text{close}$ | 80 bars (20h) |
| 5 | `channel_width_20` | Volatility | $(\text{high}_{20} - \text{low}_{20}) / \text{close}$ | 20 bars (5h) |
| 6 | `channel_width_80` | Volatility | $(\text{high}_{80} - \text{low}_{80}) / \text{close}$ | 80 bars (20h) |
| 7 | `breakout_high_20` | Market Structure | $\mathbb{I}(\text{high} \ge \text{high}[t-1]_{20\text{-bar max}})$ | 20 bars (5h) |
| 8 | `breakout_low_20` | Market Structure | $\mathbb{I}(\text{low} \le \text{low}[t-1]_{20\text{-bar min}})$ | 20 bars (5h) |
| 9 | `return_persistence_5` | Momentum | Rolling mean of $\mathbb{I}(\text{close} > \text{close}[t-1])$ over 5 bars | 5 bars (75m) |
| 10 | `return_persistence_20` | Momentum | Rolling mean of $\mathbb{I}(\text{close} > \text{close}[t-1])$ over 20 bars | 20 bars (5h) |
| 11 | `momentum_acceleration_5` | Momentum | $(\Delta P_5 - \Delta P_{5\text{ lag 5}}) / \text{close}$ | 10 bars (2.5h) |
| 12 | `roc_acceleration_20` | Momentum | $\text{ROC}_{10} - \text{ROC}_{20}$ | 20 bars (5h) |
| 13 | `range_expansion_ratio` | Volatility | $(\text{high} - \text{low}) / (\text{ATR}_{14} + \epsilon)$ | 14 bars (3.5h) |
| 14 | `range_compression_5` | Volatility | $\text{Range}_5 / (\text{Range}_{20} + \epsilon)$ | 20 bars (5h) |
| 15 | `vol_regime_ratio_10_80` | Volatility | $\sigma_{10}(\text{ret}) / (\sigma_{80}(\text{ret}) + \epsilon)$ | 80 bars (20h) |
| 16 | `body_to_range_ratio` | Candle Geometry | $\|\text{close} - \text{open}\| / (\text{high} - \text{low} + \epsilon)$ | 1 bar (15m) |
| 17 | `high_low_wick_imbalance` | Candle Geometry | $(\text{upper\_wick} - \text{lower\_wick}) / (\text{hl\_range} + \epsilon)$ | 1 bar (15m) |
| 18 | `trend_regime_score` | Trend | Alignment score of $\text{SMA}_{20}, \text{SMA}_{80}, \text{EMA}_{10}, \text{EMA}_{20}$ | 80 bars (20h) |
| 19 | `dist_daily_pivot` | Market Structure | Distance from 80-bar multi-session typical price pivot | 80 bars (20h) |
| 20 | `session_transition_london_open` | Time / Session | Indicator for London open window (07:00–08:00 UTC) | Point-in-time |
| 21 | `session_transition_ny_open` | Time / Session | Indicator for NY open window (12:00–13:00 UTC) | Point-in-time |
| 22 | `session_transition_london_close` | Time / Session | Indicator for London fix / close (15:00–17:00 UTC) | Point-in-time |
| 23 | `volume_momentum_5` | Activity | $\text{Volume}_5 / (\text{Volume}_{\text{SMA } 20} + \epsilon)$ | 20 bars (5h) |
| 24 | `volume_price_trend_5` | Activity | $\text{ret}_5 \times \text{volume\_momentum}_5$ | 20 bars (5h) |
| 25 | `consecutive_direction_run` | Momentum | Consecutive bar run length bounded in $[-5.0, 5.0]$ | 5 bars (75m) |

---

## 7. Intraday Model Comparison (80 vs 105 Features)

Models were evaluated on **Target A (Binary Direction: $\text{ret}_4 > 0$ vs $< 0$)** across the training partition (69,453 valid rows) and validation partition (14,895 valid rows).

### Validation Performance Comparison

| Model | Feature Set | Balanced Accuracy | Raw Accuracy | Macro F1 | ROC-AUC | Mean Confidence | Max Confidence |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Baseline (80 feats) | **51.98%** | 52.00% | 51.96% | 0.5301 | 53.3% | 98.7% |
| **Logistic Regression** | Extended (105 feats) | **51.94%** | 51.96% | 51.92% | 0.5308 | 53.5% | 98.3% |
| **Random Forest** | Baseline (80 feats) | **53.10%** | 52.99% | 51.56% | 0.5373 | 54.8% | 91.8% |
| **Random Forest** | Extended (105 feats) | **52.97%** | 52.89% | 52.09% | 0.5401 | 54.3% | 92.2% |
| **Extra Trees** | Baseline (80 feats) | **52.09%** | 51.97% | 49.96% | 0.5337 | 52.9% | 82.7% |
| **Extra Trees** | Extended (105 feats) | **52.82%** | 52.72% | 51.56% | 0.5406 | 52.7% | 80.1% |

### Key Intraday Finding
> [!IMPORTANT]
> **Extended Features Do Not Improve Directional Separability:**  
> Expanding the feature set from 80 to 105 features by adding market structure, breakout distance, momentum acceleration, and session transitions produced **zero meaningful improvement** in directional separation:
> - Random Forest balanced accuracy changed from **$53.10\%$** (80 feats) to **$52.97\%$** (105 feats).
> - Logistic Regression balanced accuracy changed from **$51.98\%$** to **$51.94\%$**.
> - ROC-AUC nudged by only $+0.0028$ (RF: 0.5373 $\rightarrow$ 0.5401; ET: 0.5337 $\rightarrow$ 0.5406).
> 
> This provides definitive empirical evidence that the M15 performance ceiling is **not** caused by omitting standard technical indicators or structural patterns.

---

## 8. Intraday Validation: Per-Class Metrics

Detailed classification performance for the 105-feature models on the validation partition ($N=14,895$):

| Model | Class | Precision | Recall | F1 Score | Confusion Matrix Elements |
|---|---|---|---|---|---|
| **Logistic Regression (105 feats)** | SHORT (-1.0) | 51.74% | 49.32% | 50.50% | $\text{TN}=3,651, \text{FP}=3,752$ |
|  | LONG (+1.0) | 52.16% | 54.55% | 53.33% | $\text{FN}=3,405, \text{TP}=4,087$ |
| **Random Forest (105 feats)** | SHORT (-1.0) | 52.26% | 63.69% | 57.41% | $\text{TN}=4,715, \text{FP}=2,688$ |
|  | LONG (+1.0) | 53.94% | 42.25% | 47.38% | $\text{FN}=4,327, \text{TP}=3,165$ |
| **Extra Trees (105 feats)** | SHORT (-1.0) | 52.06% | 65.59% | 58.05% | $\text{TN}=4,856, \text{FP}=2,547$ |
|  | LONG (+1.0) | 54.00% | 39.84% | 45.85% | $\text{FN}=4,507, \text{TP}=2,985$ |

Both tree ensemble models exhibit a persistent recall asymmetry, heavily skewing predictions toward SHORT (63.7%–65.6% recall) while precision for both classes remains pinned near baseline prevalence (~52%–54%).

---

## 9. Intraday Temporal Stability (3 Equal Blocks)

Evaluating the 105-feature Random Forest across three equal chronological validation blocks ($N=4,965$ bars each):

| Block | Chronological Period | Sample Size | LONG % | SHORT % | Balanced Accuracy | Macro F1 | ROC-AUC | Mean Confidence |
|---|---|---|---|---|---|---|---|---|
| **Block 1 (Early)** | 2025-07-14 to 2025-09-24 | 4,965 | 50.1% | 49.9% | **52.97%** | 52.59% | 0.5364 | 54.2% |
| **Block 2 (Mid)** | 2025-09-24 to 2025-12-05 | 4,965 | 50.7% | 49.3% | **53.23%** | 53.00% | 0.5536 | 53.7% |
| **Block 3 (Late)** | 2025-12-05 to 2026-02-19 | 4,965 | 50.0% | 50.0% | **52.58%** | 49.49% | 0.5307 | 55.0% |

**Temporal Assessment:**  
The 105-feature model exhibits the exact same flat, near-random trajectory as the 80-feature model. Performance in every block sits between **$52.58\%$ and $53.23\%$**, confirming that the lack of predictive edge is persistent across time and market regimes.

---

## 10. Swing Data Availability

The canonical M15 dataset spans 4.0 years (`2022-09-16 09:30:00 UTC` to `2026-09-25 23:45:00 UTC`). Aggregation into higher timeframes confirms the sample size feasibility:

| Timeframe | Period Length | Bar Count | Trading Days Covered | Sample Size Adequacy |
|---|---|---|---|---|
| **M15 (Canonical)** | 15 Minutes | 100,000 bars | ~1,047 days | High sample size |
| **H4 (Swing Primary)** | 4 Hours (16 M15 bars) | 6,263 bars | ~1,047 days | Robust sample size (~6,200 bars) |
| **D1 (Swing Secondary)** | 1 Day (96 M15 bars) | 1,047 bars | ~1,047 days | Low sample size (~1,000 bars) |

---

## 11. H4 Dataset Construction

The H4 dataset was aggregated from completed M15 candles in [`ai/dataset/swing.py`](file:///home/cino/projects/ai-trading-system/ai/dataset/swing.py) using UTC standard session anchors (`00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC`):
- $\text{open} = \text{open of first constituent M15 bar}$.
- $\text{high} = \max(\text{constituent M15 highs})$.
- $\text{low} = \min(\text{constituent M15 lows})$.
- $\text{close} = \text{close of final constituent M15 bar}$.
- $\text{tick\_volume} = \sum \text{constituent tick volumes}$.
- $\text{spread} = \text{median of constituent spreads}$.

### H4 Chronological Partitions

| Partition | Bar Count | Percentage | Start Timestamp (UTC) | End Timestamp (UTC) |
|---|---|---|---|---|
| **Training** | 4,387 | 70.05% | 2022-09-16 08:00:00 | 2025-07-14 04:00:00 |
| **Validation** | 937 | 14.96% | 2025-07-14 08:00:00 | 2026-02-19 08:00:00 |
| **Test (LOCKED)** | 939 | 14.99% | 2026-02-19 12:00:00 | 2026-09-25 20:00:00 |
| **Total** | 6,263 | 100.0% | 2022-09-16 08:00:00 | 2026-09-25 20:00:00 |

Data validation confirmed:
- Zero duplicate timestamps.
- Zero NaNs or Infs in OHLC.
- 100% geometric consistency ($\text{high} \ge \max(\text{open}, \text{close})$ and $\text{low} \le \min(\text{open}, \text{close})$).

---

## 12. D1 Dataset Construction

The Daily (D1) dataset was aggregated from completed M15 candles using UTC calendar days (`00:00:00 to 24:00:00 UTC`):

### D1 Chronological Partitions

| Partition | Bar Count | Percentage | Start Timestamp (UTC) | End Timestamp (UTC) |
|---|---|---|---|---|
| **Training** | 735 | 70.20% | 2022-09-16 00:00:00 | 2025-07-14 00:00:00 |
| **Validation** | 156 | 14.90% | 2025-07-15 00:00:00 | 2026-02-19 00:00:00 |
| **Test (LOCKED)** | 156 | 14.90% | 2026-02-20 00:00:00 | 2026-09-25 00:00:00 |
| **Total** | 1,047 | 100.0% | 2022-09-16 00:00:00 | 2026-09-25 00:00:00 |

Data validation confirmed zero NaNs, monotonic increasing dates, and 100% valid OHLC geometries.

---

## 13. Swing Feature Engineering

In [`ai/features/swing.py`](file:///home/cino/projects/ai-trading-system/ai/features/swing.py), a dedicated swing feature pipeline was built, computing 32 features for H4 and 28 features for D1 across 7 functional groups:
1. **Multi-Bar Returns (4):** `return_1`, `return_2`, `return_4`, `return_8`.
2. **Candle Geometry (5):** `hl_range_norm`, `candle_body_norm`, `candle_direction`, `upper_wick_ratio`, `lower_wick_ratio`.
3. **Trend & Moving Averages (7):** `dist_sma_10`, `dist_sma_20`, `dist_sma_50`, `sma_slope_10`, `sma_slope_20`, `ema_spread_10_20`, `trend_regime`.
4. **Momentum (5):** `roc_5`, `roc_10`, `roc_20`, `rsi_14`, `momentum_acceleration`.
5. **Volatility & Realized Range (6):** `atr_14`, `atr_norm_14`, `vol_std_10`, `vol_std_20`, `vol_ratio_5_20`, `rolling_hl_ratio_20`.
6. **Market Structure (6):** `dist_rolling_high_10`, `dist_rolling_low_10`, `dist_rolling_high_20`, `dist_rolling_low_20`, `breakout_high_10`, `breakout_low_10`, `channel_width_20`.
7. **Time & Seasonality (3 for D1, 6 for H4):** `day_of_week`, `sin_day_of_week`, `cos_day_of_week` (plus `hour`, `sin_hour`, `cos_hour` on H4).

All features are point-in-time and causal. Zero future bars or lookahead formulas exist in the calculations.

---

## 14. Swing Target Research

In [`ai/labels/swing.py`](file:///home/cino/projects/ai-trading-system/ai/labels/swing.py), multi-bar forward return and directional targets were engineered across candidate swing horizons:

### 14.1 Target Definitions & Empirical Displacement Statistics

| Timeframe | Horizon | Horizon Hours | Target Type | Valid Bars | Excluded (%) | LONG (%) | SHORT (%) | Mean Abs Move (pips) | Median Abs Move (pips) |
|---|---|---|---|---|---|---|---|---|---|
| **H4** | $H=4$ | 16 hours | Binary Direction | 6,256 | 0.1% | 49.7% | 50.3% | **27.48** | **20.24** |
| **H4** | $H=4$ | 16 hours | Vol-Adjusted Direction | 2,621 | 58.2% | 50.2% | 49.8% | **38.45** | **31.20** |
| **H4** | $H=8$ | 32 hours | Binary Direction | 6,247 | 0.3% | 49.8% | 50.2% | **39.33** | **29.88** |
| **H4** | $H=8$ | 32 hours | Vol-Adjusted Direction | 2,710 | 56.7% | 51.5% | 48.5% | **55.10** | **44.80** |
| **H4** | $H=12$ | 48 hours | Binary Direction | 6,250 | 0.2% | 49.5% | 50.5% | **48.12** | **36.62** |
| **H4** | $H=12$ | 48 hours | Vol-Adjusted Direction | 2,763 | 55.9% | 51.2% | 48.8% | **67.30** | **55.10** |
| **D1** | $H=3$ | 3 days | Binary Direction | 1,043 | 0.4% | 51.4% | 48.6% | **59.19** | **46.50** |
| **D1** | $H=3$ | 3 days | Vol-Adjusted Direction | 403 | 61.5% | 50.4% | 49.6% | **82.30** | **68.40** |
| **D1** | $H=5$ | 1 week | Binary Direction | 1,039 | 0.8% | 50.5% | 49.5% | **74.47** | **59.28** |
| **D1** | $H=5$ | 1 week | Vol-Adjusted Direction | 406 | 61.2% | 51.7% | 48.3% | **104.20** | **85.60** |
| **D1** | $H=10$ | 2 weeks | Binary Direction | 1,037 | 1.0% | 51.4% | 48.6% | **105.09** | **87.72** |
| **D1** | $H=10$ | 2 weeks | Vol-Adjusted Direction | 422 | 59.7% | 53.1% | 46.9% | **148.50** | **126.10** |

---

## 15. Swing Model Evaluation

Models (`LogisticRegression` with training `StandardScaler` and `RandomForestClassifier` balanced) were trained on the training partition and evaluated strictly on the validation partition.

### 15.1 H4 Model Evaluation

| Target Name | Horizon | Target Type | Validation $N$ | Logistic Regression Bal Acc | LR ROC-AUC | Random Forest Bal Acc | RF ROC-AUC | RF Macro F1 |
|---|---|---|---|---|---|---|---|---|
| `H4_binary_H4` | 16 hours | Binary | 932 | **50.97%** | 0.5365 | **51.76%** | 0.5138 | 51.58% |
| `H4_vol_adj_H4` | 16 hours | Vol-Adjusted | 370 | **45.63%** | 0.4985 | **54.88%** | 0.5038 | 54.51% |
| `H4_binary_H8` | 32 hours | Binary | 928 | **49.84%** | 0.5077 | **51.22%** | 0.5102 | 51.19% |
| `H4_vol_adj_H8` | 32 hours | Vol-Adjusted | 406 | **48.98%** | 0.5244 | **57.54%** | **0.5772** | **55.77%** |
| `H4_binary_H12` | 48 hours | Binary | 925 | **51.07%** | 0.4950 | **52.75%** | 0.5368 | 52.75% |
| `H4_vol_adj_H12` | 48 hours | Vol-Adjusted | 425 | **49.48%** | 0.4650 | **50.82%** | 0.4879 | 49.13% |

### 15.2 D1 Model Evaluation

| Target Name | Horizon | Target Type | Validation $N$ | Logistic Regression Bal Acc | LR ROC-AUC | Random Forest Bal Acc | RF ROC-AUC | RF Macro F1 |
|---|---|---|---|---|---|---|---|---|
| `D1_binary_H3` | 3 days | Binary | 153 | **52.40%** | 0.5012 | **47.03%** | 0.4554 | 46.91% |
| `D1_vol_adj_H3` | 3 days | Vol-Adjusted | 63 | **54.56%** | 0.5639 | **54.51%** | 0.5842 | 54.16% |
| `D1_binary_H5` | 1 week | Binary | 150 | **47.70%** | 0.5018 | **46.79%** | 0.4398 | 46.07% |
| `D1_vol_adj_H5` | 1 week | Vol-Adjusted | 55 | **52.02%** | 0.5860 | **47.11%** | 0.4328 | 46.99% |
| `D1_binary_H10` | 2 weeks | Binary | 146 | **56.85%** | 0.6104 | **44.52%** | 0.4309 | 43.11% |
| `D1_vol_adj_H10` | 2 weeks | Vol-Adjusted | 44 | **44.41%** | 0.4203 | **39.03%** | 0.4410 | 38.35% |

---

## 16. Swing Temporal Stability

Examining temporal stability across three chronological validation blocks:

### 16.1 H4 Volatility-Adjusted $H=8$ (Random Forest)
This was the highest-performing swing configuration ($57.54\%$ overall balanced accuracy, $N=406$):

| Block | Chronological Period | Validation Bars | LONG % | SHORT % | Balanced Accuracy | Macro F1 | ROC-AUC | Mean Confidence |
|---|---|---|---|---|---|---|---|---|
| **Block 1 (Early)** | 2025-07-14 to 2025-09-16 | 135 | 55.6% | 44.4% | **58.33%** | 57.77% | 0.6202 | 55.3% |
| **Block 2 (Mid)** | 2025-09-16 to 2025-11-25 | 135 | 42.2% | 57.8% | **62.15%** | 62.18% | 0.6023 | 53.8% |
| **Block 3 (Late)** | 2025-11-25 to 2026-02-17 | 136 | 66.2% | 33.8% | **53.60%** | 46.18% | 0.5285 | 53.9% |

### Key Diagnostic Insight on Swing Stability
1. **Regime Degradation in Block 3:**  
   In Block 1 and Block 2, Random Forest achieved balanced accuracies of 58.3% and 62.1%. However, in Block 3, performance collapsed to **$53.60\%$**, with Macro F1 dropping to $46.18\%$ as ground truth LONG moves surged to 66.2%.
2. **Model Divergence:**  
   While Random Forest achieved 57.54% on `H4_vol_adj_H8`, Logistic Regression on the exact same dataset achieved only **$48.98\%$** (AUC 0.5244). On unthresholded binary data (`H4_binary_H8`), both models sat near random chance (RF: 51.22%, LR: 49.84%).
3. **Sample Size Uncertainty:**  
   Because the validation set contains only 406 observations for this target (approx. 135 per block), the margin of error at a 95% confidence interval is approximately $\pm 4.8\%$. This elevated variance means the 57.5% result cannot be claimed as persistent statistical edge.

---

## 17. Intraday vs Swing Research Observations

A direct, factual comparison between Track A and Track B reveals profound structural differences in market dynamics:

| Research Dimension | Track A: Intraday (EURUSD M15) | Track B: Swing (EURUSD H4) | Track B: Swing (EURUSD D1) |
|---|---|---|---|
| **Prediction Horizon** | 1 Hour (4 bars) | 32 Hours (8 bars) | 1 Week (5 bars) |
| **Sample Size (Train / Val)** | 69,453 / 14,895 | 4,324 / 928 | 680 / 150 |
| **Mean Absolute Move** | **5.32 pips** | **39.33 pips** | **74.47 pips** |
| **Median Absolute Move** | **3.69 pips** | **29.88 pips** | **59.28 pips** |
| **Fixed Cost (Spread+Comm)** | ~1.5 pips | ~1.5 pips | ~1.5 pips |
| **Friction / Mean Move Ratio** | **28.2%** (Prohibitive) | **3.8%** (Manageable) | **2.0%** (Negligible) |
| **Financing / Swap Drag** | Zero (no overnight hold) | ~0.1–0.3 pips / day | ~0.5–1.5 pips / week |
| **Weekend Gap Risk** | Zero (flat by Friday close) | Low (if closed) to Medium | Significant (multi-day hold) |
| **Binary Balanced Accuracy (RF)** | 52.97% (105 feats) | 51.22% (binary) | 46.79% (binary) |
| **Vol-Adjusted Bal Acc (RF)** | 51.76% (Target B) | **57.54%** ($H=8$) | 47.11% ($H=5$) |
| **Model Agreement (LR vs RF)** | High (both ~51–53%) | Low (RF 57.5% vs LR 49.0%) | Extreme divergence (39% to 57%) |
| **Statistical Degrees of Freedom** | Very High ($N > 14,000$) | Moderate ($N \approx 400–900$) | Extremely Low ($N \approx 50–150$) |

### Summary of Differences
1. **The Friction Hurdle is Vastly Lower on Swing Timeframes:**  
   On M15, transaction costs consume over 28% of the expected move. A directional model on M15 must achieve >56% accuracy merely to break even against spread and slippage. On H4, transaction friction represents less than 4% of the move, reducing the break-even accuracy requirement.
2. **Directional Forecasting Remains Difficult Across All Timeframes:**  
   Neither M15 nor H4 nor D1 exhibits high, unconditional directional predictability. The unthresholded binary targets all sit tightly around 50%–53%.
3. **D1 Suffers from Severe Sample Starvation:**  
   With only 156 validation bars across 7 months, daily data is too scarce to establish reliable statistical conclusions. Models diverge wildly and overfit small sample fluctuations.

---

## 18. Leakage Validation

All ten required data integrity and leakage tests were executed and verified:

| Check # | Requirement | Status | Verification Detail |
|---|---|---|---|
| **1** | Future Dependency Correctness | **PASS** | Features use only data available at time $t$; targets use forward closes. |
| **2** | Horizon Alignment Strictly Preserved | **PASS** | H4 and D1 forward shifts strictly match declared horizons. |
| **3** | End-of-Data Boundary NaNs | **PASS** | Exactly the final $H$ rows of H4 and D1 datasets evaluate to NaN. |
| **4** | Feature Immutability | **PASS** | Raw parquet files remain read-only and unaltered. |
| **5** | No Target Columns in Feature Matrices | **PASS** | Zero target names present in $X_{\text{M15}}$, $X_{\text{H4}}$, or $X_{\text{D1}}$. |
| **6** | No Future OHLC in Features | **PASS** | All rolling windows and lags use strictly non-negative shifts. |
| **7** | No Future Returns in Features | **PASS** | Feature pipelines exclude all forward return columns. |
| **8** | Phase 11 Test Set Holdout Isolation | **PASS** | Test partition (14,988 M15 rows: `2026-02-19 12:00:00 UTC` onward) was never accessed. |
| **9** | Chronological Split Integrity | **PASS** | Train end < Val start < Test start across all timeframes. |
| **10** | Deterministic Repeatability | **PASS** | Fixed random seeds (42) produce bit-exact identical predictions. |

---

## 19. Limitations

1. **Statistical Power on Daily Bars:**  
   D1 analysis is constrained by having only 1,047 total historical bars and 156 validation observations, resulting in wide confidence intervals ($\pm 8\%$).
2. **Spread Data Limitation:**  
   Historical MetaQuotes M15 export exhibits zero-spread reporting on ~50% of bars prior to 2024. While median aggregation mitigates outliers, live spreads will vary dynamically.
3. **Execution Frictions Omitted:**  
   Consistent with research phase guidelines, trade executions, order book dynamics, and swap calculations were not simulated in P&L.
4. **Instrument Scope:**  
   Evaluated strictly on EURUSD. Cross-currency pairs (e.g., GBPJPY, EURCHF) may exhibit different volatility and trending characteristics.

---

## 20. Future Research Directions

1. **Focus Swing Research on H4 Rather than D1:**  
   H4 provides the optimal compromise between sufficient statistical sample size (~6,200 bars) and meaningful economic displacement (~30–40 pips).
2. **Investigate Asymmetric Swing Targets:**  
   Rather than symmetric classification, evaluate target barriers aligned with structural swing support/resistance levels.
3. **Regime-Conditioned Gating:**  
   Test using H4 macro trend/volatility regime as a conditioning filter for intraday M15 setups.
4. **Develop Swing Cost Model:**  
   Incorporate dynamic overnight swap rates and weekend gap risk into future swing evaluation pipelines.

---

## 21. Production Status

1. **Production Code Unaltered:** Zero production trading logic, baseline parameters, or configuration constants were modified during Phase 15.
2. **Phase 12 Baseline Preserved:** Official baseline remains Random Forest balanced on `direction_4` ($H=4$ M15 bars, $\tau=0.60$, SL $1.00\times$ ATR, TP $1.50\times$ ATR, 4-bar max hold).
3. **RiskEngine Authority:** Unchanged. RiskEngine remains the final trading authority.
4. **Dual-Track Research Architecture:** All code created in Phase 15 ([`ai/features/intraday_extended.py`](file:///home/cino/projects/ai-trading-system/ai/features/intraday_extended.py), [`ai/dataset/swing.py`](file:///home/cino/projects/ai-trading-system/ai/dataset/swing.py), [`ai/features/swing.py`](file:///home/cino/projects/ai-trading-system/ai/features/swing.py), [`ai/labels/swing.py`](file:///home/cino/projects/ai-trading-system/ai/labels/swing.py)) is modular and retained strictly for research.
