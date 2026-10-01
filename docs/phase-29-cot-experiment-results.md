# Phase 29 — Pre-Registered CFTC COT Positioning Experiment Results

**Date:** 2026-10-02  
**Project:** AI Autonomous Trading System  
**Repository:** `https://github.com/Suman-KM/Trading_bot.git`  
**Branch:** `develop`  
**Base Commit:** [`2a3138a`](https://github.com/Suman-KM/Trading_bot/commit/2a3138af62b205c3febb6e8e7719e1594abe7fd4) (`research: establish CFTC COT data foundation`)  
**Scientific Verdict:** **`NOT SUPPORTED`**  

---

## 1. Executive Summary

In accordance with Phase 26 governance and the pre-registered protocol in [`docs/phase-29-cot-experiment-preregistration.md`](file:///home/cino/projects/ai-trading-system/docs/phase-29-cot-experiment-preregistration.md), Phase 29 evaluated whether adding three causally aligned CFTC Commitments of Traders (COT) positioning features to a frozen EURUSD D1 price-derived benchmark provides statistically meaningful incremental directional predictive power.

The experiment was conducted across **10 expanding chronological walk-forward folds** on EURUSD D1 bars covering the pre-holdout historical research period (`2010-03-01` to `2024-11-04`, 3,816 daily bars) across two pre-registered holding horizons ($H = 10$ and $H = 20$ business days):

1. **Horizon $H = 10$ Business Days (~2 Weeks):**
   - Baseline Mean Balanced Accuracy: **51.78%**
   - COT Candidate Mean Balanced Accuracy: **53.00%**
   - Mean Difference: **+1.23%** (95% CI: $[-3.50\%, +5.95\%]$)
   - Paired Student's t-test: $t = 0.588$, $p = 0.5710$ (Statistically Insignificant, $p \ge 0.05$)
   - Success Gate: **FAIL** | Failure Gate: **TRIGGERED**

2. **Horizon $H = 20$ Business Days (~4 Weeks):**
   - Baseline Mean Balanced Accuracy: **49.32%**
   - COT Candidate Mean Balanced Accuracy: **49.59%**
   - Mean Difference: **+0.27%** (95% CI: $[-6.48\%, +7.01\%]$)
   - Paired Student's t-test: $t = 0.089$, $p = 0.9308$ (Statistically Insignificant, $p \ge 0.05$)
   - Success Gate: **FAIL** | Failure Gate: **TRIGGERED**

**Overall Scientific Decision:** **`NOT SUPPORTED`**. Weekly CFTC speculative and commercial positioning features do not provide statistically meaningful incremental directional predictive information beyond EURUSD price-derived features.

---

## 2. Hypothesis & Null Hypothesis

- **Hypothesis ($H_1$):** Extreme weekly speculative positioning in CME Euro FX futures contains incremental information about the medium-term directional movement of EURUSD beyond information already contained in EURUSD price-derived features.
- **Null Hypothesis ($H_0$):** CFTC positioning provides no statistically meaningful incremental directional information beyond the frozen EURUSD price-only benchmark.

**Verdict:** The null hypothesis **cannot be rejected** ($p = 0.5710$ for $H=10$, $p = 0.9308$ for $H=20$).

---

## 3. Dataset & Causal Alignment

- **Exogenous Source:** Official weekly CFTC COT Legacy and TFF reports (CME Euro FX contract code `099741`).
- **Cryptographic Provenance:** 34 SHA-256 verified annual archives (2010–2026), 873 weekly reports.
- **Price Target:** EURUSD D1 OHLCV bars aggregated causally from validated EURUSD H4 data.
- **Point-in-Time Lag:** Tuesday position snapshots are officially released on Friday at 15:30 US Eastern Time. Converted to UTC via `America/New_York` (19:30 UTC during EDT, 20:30 UTC during EST). Actionable timestamp is strictly Monday 00:00:00 UTC post-release.
- **Causal Alignment Audit:** Zero publication leakage violations, zero effective availability violations across all 4,309 bars.

---

## 4. Evaluated Features

### Baseline (30 Features)
The canonical 30 D1 technical swing features (`EURUSD_BASELINE_32_COLS[:30]`):
- Return momentum (`return_1`, `return_2`, `return_4`, `return_8`)
- Price structure (`hl_range_norm`, `candle_body_norm`)
- Moving average relationships (`dist_sma_10`, `dist_sma_20`, `dist_sma_50`, `sma_slope_10`, `sma_slope_20`, `ema_spread_10_20`, `trend_regime`)
- Oscillators & ROC (`roc_5`, `roc_10`, `roc_20`, `rsi_14`, `momentum_acceleration`)
- Volatility (`atr_14`, `atr_norm_14`, `vol_std_10`, `vol_std_20`, `vol_ratio_5_20`)
- Channel dynamics (`dist_rolling_high_10`, `dist_rolling_low_10`, `dist_rolling_high_20`, `dist_rolling_low_20`, `channel_width_20`)
- Calendar seasonality (`sin_day_of_week`, `cos_day_of_week`)

### Candidate (33 Features)
Baseline 30 features + strictly 3 pre-registered COT features:
1. `speculative_net_zscore_3y`: Rolling 3-year (156-week) standardized Z-score of NonCommercial net positioning.
2. `commercial_position_zscore_3y`: Rolling 3-year (156-week) standardized Z-score of Commercial net positioning.
3. `speculative_net_4w_change`: 4-week delta in NonCommercial net positioning.

---

## 5. Walk-Forward Validation Results

### Horizon $H = 10$ Business Days (Target: `direction_vol_10`)

| Fold | Training Window | Validation Window | Train N | Val N | Baseline BalAcc | COT BalAcc | Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2011-01-05 -> 2013-12-03 | 2014-01-31 -> 2015-01-16 | 366 | 130 | 56.19% | 61.62% | +5.43% |
| 2 | 2011-01-05 -> 2014-12-16 | 2015-02-17 -> 2016-02-17 | 498 | 116 | 57.76% | 48.28% | -9.48% |
| 3 | 2011-01-05 -> 2016-01-19 | 2016-03-15 -> 2017-03-14 | 616 | 111 | 49.14% | 54.03% | +4.89% |
| 4 | 2011-01-05 -> 2017-02-16 | 2017-04-07 -> 2018-04-17 | 736 | 116 | 47.79% | 47.87% | +0.08% |
| 5 | 2011-01-05 -> 2018-03-21 | 2018-05-09 -> 2019-04-26 | 858 | 73 | 48.21% | 61.37% | +13.16% |
| 6 | 2011-01-05 -> 2019-04-24 | 2019-06-13 -> 2020-06-01 | 945 | 101 | 59.41% | 61.96% | +2.55% |
| 7 | 2011-01-05 -> 2020-05-27 | 2020-07-09 -> 2021-07-09 | 1053 | 101 | 47.02% | 39.41% | -7.61% |
| 8 | 2011-01-05 -> 2021-06-25 | 2021-08-17 -> 2022-08-16 | 1161 | 123 | 53.50% | 54.59% | +1.08% |
| 9 | 2011-01-05 -> 2022-07-29 | 2022-09-07 -> 2023-09-13 | 1291 | 106 | 44.90% | 49.39% | +4.49% |
| 10 | 2011-01-05 -> 2023-09-08 | 2023-10-20 -> 2024-10-11 | 1397 | 130 | 53.85% | 51.54% | -2.31% |

**Summary Statistics ($H = 10$):**
- Baseline Balanced Accuracy: Mean = 51.78%, Median = 51.32%, Std = 5.25%
- COT Candidate Balanced Accuracy: Mean = 53.00%, Median = 52.78%, Std = 7.15%
- Paired Difference: Mean = +1.23%, Std = 6.60%
- Paired t-test: $t = 0.588$, $p = 0.5710$
- Wilcoxon Signed-Rank Test: $W = 20.0$, $p = 0.4922$
- 95% Confidence Interval: $[-3.50\%, +5.95\%]$

---

### Horizon $H = 20$ Business Days (Target: `direction_vol_20`)

| Fold | Training Window | Validation Window | Train N | Val N | Baseline BalAcc | COT BalAcc | Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2011-01-05 -> 2013-11-19 | 2014-01-21 -> 2015-01-02 | 318 | 130 | 57.83% | 70.00% | +12.17% |
| 2 | 2011-01-05 -> 2014-12-02 | 2015-02-09 -> 2016-02-03 | 451 | 97 | 50.34% | 55.53% | +5.19% |
| 3 | 2011-01-05 -> 2016-01-05 | 2016-03-14 -> 2017-03-02 | 560 | 84 | 71.83% | 55.26% | -16.57% |
| 4 | 2011-01-05 -> 2017-02-02 | 2017-04-07 -> 2018-04-03 | 655 | 110 | 32.32% | 40.91% | +8.59% |
| 5 | 2011-01-05 -> 2018-03-07 | 2018-05-17 -> 2019-03-21 | 771 | 38 | 38.12% | 39.08% | +0.96% |
| 6 | 2011-01-05 -> 2019-04-10 | 2019-06-24 -> 2020-05-27 | 833 | 100 | 61.94% | 62.90% | +0.96% |
| 7 | 2011-01-05 -> 2020-05-13 | 2020-07-09 -> 2021-06-25 | 941 | 113 | 56.04% | 55.95% | -0.10% |
| 8 | 2011-01-05 -> 2021-06-11 | 2021-08-10 -> 2022-08-01 | 1066 | 109 | 27.54% | 28.22% | +0.68% |
| 9 | 2011-01-05 -> 2022-07-15 | 2022-09-09 -> 2023-08-30 | 1178 | 133 | 37.05% | 43.22% | +6.17% |
| 10 | 2011-01-05 -> 2023-08-25 | 2023-10-13 -> 2024-10-02 | 1320 | 113 | 60.22% | 44.82% | -15.40% |

**Summary Statistics ($H = 20$):**
- Baseline Balanced Accuracy: Mean = 49.32%, Median = 53.19%, Std = 14.54%
- COT Candidate Balanced Accuracy: Mean = 49.59%, Median = 50.39%, Std = 12.30%
- Paired Difference: Mean = +0.27%, Std = 9.42%
- Paired t-test: $t = 0.089$, $p = 0.9308$
- Wilcoxon Signed-Rank Test: $W = 24.0$, $p = 0.7695$
- 95% Confidence Interval: $[-6.48\%, +7.01\%]$

---

## 6. Pre-Registered Gate Evaluation

| Gate | Requirement | Actual $H=10$ | Actual $H=20$ | Result |
| :--- | :--- | :---: | :---: | :---: |
| **Success Gate 1** | Mean COT BalAcc $\ge 55.0\%$ | 53.00% | 49.59% | **FAIL** |
| **Success Gate 2** | Statistically significant improvement ($p < 0.05$) | $p = 0.5710$ | $p = 0.9308$ | **FAIL** |
| **Failure Gate 1** | Mean COT BalAcc $< 53.0\%$ | 53.00% | 49.59% | **TRIGGERED ($H=20$)** |
| **Failure Gate 2** | Mean difference $\le 0.0\%$ or $p \ge 0.05$ | $p = 0.5710$ | $p = 0.9308$ | **TRIGGERED (Both)** |

---

## 7. Governance & Safety Adherence

- **Models Trained:** 40 models total (10 folds $\times$ 2 configurations $\times$ 2 horizons).
- **Backtests Executed:** 0 (predictive criteria failed; backtesting was strictly prohibited).
- **Live / Demo Trades:** 0.
- **Locked Test Partitions:** Untouched and quarantined (`2026-02-19 12:00:00 UTC` onward).
- **No Indicator Mining / Parameter Sweeps:** Executed exactly one frozen model configuration with zero hyperparameter searches.

---

## 8. Conclusion & Next Step

The empirical evidence definitively shows that weekly CFTC Commitments of Traders institutional positioning data does not produce a stationary, out-of-sample directional edge for EURUSD swing trading on daily bars. Across both multi-week horizons ($H=10$ and $H=20$), out-of-sample performance hovered at 49.6%–53.0%, with differences against the price-only baseline statistically indistinguishable from zero noise ($p > 0.50$).

In accordance with strict scientific protocol, no further models will be trained on this configuration.
