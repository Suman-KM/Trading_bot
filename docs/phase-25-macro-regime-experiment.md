# Phase 25 — Controlled H4 Macro Regime Experiment: US-Germany 2Y Yield Spread Conditioning for EURUSD

**Date:** 2026-10-02  
**Project:** AI Autonomous Trading System  
**Repository:** `https://github.com/Suman-KM/Trading_bot.git`  
**Branch:** `develop`  
**Base Commit:** [`c2069da`](https://github.com/Suman-KM/Trading_bot/commit/c2069da) (`research: verify German 2Y yield series`)  
**Status:** COMPLETE — SCIENTIFIC DECISION: `NOT SUPPORTED`

---

## 1. Research Question

Does adding a causally aligned US-Germany 2-Year sovereign yield spread macro regime feature set improve the out-of-sample directional prediction of the frozen EURUSD H4 swing trading model relative to a technical-only baseline?

---

## 2. Hypothesis

Conditioning the frozen EURUSD H4 swing prediction model on:
1. The point-in-time US-Germany 2-Year sovereign yield spread (`US_Germany_2Y_Spread`), and
2. Its 5-business-day change (`US_Germany_2Y_Spread_5D_Change`)

provides stationary, statistically significant out-of-sample predictive alpha over the frozen 30-feature technical baseline across 10 chronological walk-forward folds and on the sealed Fresh Research Holdout.

---

## 3. Data Sources

| Series | Source Authority | Identifier / Code | Frequency | Definition |
| :--- | :--- | :--- | :--- | :--- |
| **US 2Y Treasury Yield** | Federal Reserve Bank of St. Louis (FRED) / US Department of the Treasury | `DGS2` | Daily | 2-Year Treasury Constant Maturity nominal yield at 15:30 US Eastern close. |
| **German 2Y Bund Yield** | Deutsche Bundesbank Open Data Portal | `BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A` | Daily (`P1D`) | Coupon-bearing par yield on German Federal debt securities, residual maturity 2.0 years (formally approved in Phase 24.1). |
| **EURUSD H4 Bars** | MetaQuotes-Demo Server (UTC) | `EURUSD` H4 | 4-Hour | 25,800 H4 bars spanning 2010-03-01 16:00 UTC through 2026-09-29 20:00 UTC. |

> [!IMPORTANT]
> The rejected series `FRED: IRLTLT01DEM156N` (which was empirically proved in Phase 24.1 to be an OECD monthly 10-year yield) was strictly excluded. Only the formally approved Deutsche Bundesbank daily 2-year benchmark par yield was utilized.

---

## 4. Data Provenance

Every exogenous observation was ingested with full cryptographic provenance and verified in Phase 24 and Phase 24.1:
- **US 2Y Treasury Raw Artifact:** `data/external/macro_yields/raw/fred_dgs2_raw.csv` (SHA-256: `6eef40d437016cb2c2b3dc80eb7c5553e168f1879301da287e0fa0c1bfaf8fa4`).
- **German 2Y Bund Raw Artifact:** `data/external/macro_yields/raw/bundesbank_german_2y_raw.csv` (SHA-256: `3b80eb176e336b13e54b6b6e4fa8ddaa67104b2816fe0ea203b5ba572d4aeebf`).
- **Harmonized Daily Parquet:** `data/external/macro_yields/processed/macro_yields_daily_harmonized.parquet` (4,369 business days, 2010 to 2026).
- **Causally Aligned Parquet:** `data/external/macro_yields/processed/eurusd_h4_yield_aligned.parquet` (25,800 bars, zero NaN values).

---

## 5. Point-in-Time Alignment

To eliminate lookahead bias and timing bleed:
1. **Publication Clocks:**
   - German Federal debt closing yields are computed at Frankfurt close (~17:00 CET / ~16:00 UTC).
   - US Treasury constant maturity yields are posted by the US Treasury / FRED between 16:15 and 17:00 ET (~21:15–22:00 UTC).
2. **Point-in-Time Rule:**
   - Any yield observed on Day $D$ is **strictly forbidden** from being made available to any EURUSD H4 bar opening or closing on Day $D$.
   - **First Permitted Usage:** `00:00:00 UTC` on Day $D+1$ (the bar closing at `04:00:00 UTC`).
   - Latency buffer between publication (~22:00 UTC Day $D$) and bar opening (`00:00 UTC` Day $D+1$): **Minimum 2 hours**.
3. **Causal Alignment Verification:**
   - Across all 25,800 H4 bars, exactly **zero** same-day or future-day yield observations were made available to intraday bars (`pit_violation = 0`).

---

## 6. Feature Definition

### A. Baseline Features (30 Technical Features)
Derived strictly from EURUSD OHLCV price action up to candle close $t$:
1. **Multi-Bar Returns:** `return_1`, `return_2`, `return_4`, `return_8`
2. **Price Action Geometry:** `hl_range_norm`, `candle_body_norm`
3. **Moving Average Displacements:** `dist_sma_10`, `dist_sma_20`, `dist_sma_50`
4. **Trend Dynamics:** `sma_slope_10`, `sma_slope_20`, `ema_spread_10_20`, `trend_regime`
5. **Momentum & Oscillators:** `roc_5`, `roc_10`, `roc_20`, `rsi_14`, `momentum_acceleration`
6. **Volatility Channels:** `atr_14`, `atr_norm_14`, `vol_std_10`, `vol_std_20`, `vol_ratio_5_20`
7. **Support / Resistance Extremes:** `dist_rolling_high_10`, `dist_rolling_low_10`, `dist_rolling_high_20`, `dist_rolling_low_20`, `channel_width_20`
8. **Seasonality:** `sin_day_of_week`, `cos_day_of_week`

### B. Approved Macro Features (2 Features)
1. **`US_Germany_2Y_Spread`:**
   $$\Delta Y_{2Y, t} = Y_{US, 2Y, t} - Y_{DE, 2Y, t}$$
2. **`US_Germany_2Y_Spread_5D_Change`:**
   $$\Delta(\Delta Y_{2Y})_{5D, t} = \Delta Y_{2Y, t} - \Delta Y_{2Y, t-5\text{ business days}}$$

Total features:
- **Baseline:** Exactly 30 features.
- **Macro Candidate:** Exactly 32 features (30 technical + 2 approved macro).

---

## 7. Target Definition

The prediction target is identical to the established Phase 16/17 H4 research candidate:
- **Timeframe:** EURUSD H4
- **Holding Horizon:** $H = 8$ bars (32 hours)
- **Target Variable:** Volatility-adjusted directional return (`direction_vol_8`):
  $$Y_t = \begin{cases} +1.0 & \text{if } R_{t \to t+8} > 0.5 \times \sqrt{8} \times \text{ATR}_{\text{norm}, t} \\ -1.0 & \text{if } R_{t \to t+8} < -0.5 \times \sqrt{8} \times \text{ATR}_{\text{norm}, t} \\ \text{NaN} & \text{otherwise (neutral/noise zone)} \end{cases}$$
- Zero threshold alterations, zero ATR parameter alterations, zero redesign.

---

## 8. Baseline Model

- **Model Family:** `RandomForestClassifier` (Scikit-Learn)
- **Configuration (Frozen Phase 15/16/17):**
  - `n_estimators`: 100
  - `max_depth`: 5
  - `min_samples_leaf`: 10
  - `class_weight`: `"balanced"`
  - `random_state`: 42
  - `n_jobs`: -1
- **Input Matrix:** 30 technical features.

---

## 9. Macro Candidate Model

- **Model Family:** `RandomForestClassifier`
- **Configuration:** Identical frozen hyperparameters (`n_estimators=100`, `max_depth=5`, `min_samples_leaf=10`, `class_weight="balanced"`, `random_state=42`).
- **Input Matrix:** 30 technical features + 2 approved macro features.
- **Hyperparameter Tuning:** Absolutely zero (no GridSearchCV, no Optuna, no parameter tuning).

---

## 10. Walk-Forward Design

- **Validation Structure:** 10 chronological expanding walk-forward folds.
- **Research Partition:** `2010-03-01 16:00:00 UTC` through `2024-11-04 12:00:00 UTC` (22,847 H4 bars).
- **Initial Training Warmup:** 5,000 H4 bars (~3.2 years).
- **Purge Gap:** 8 H4 bars (32 hours) at training fold boundary (eliminates forward label lookahead).
- **Embargo Buffer:** 4 H4 bars (16 hours) at validation fold start (eliminates autoregressive feature memory).
- **Boundary Separation:** Minimum 12 H4 bars (48 hours) separating training end from validation start.

---

## 11. Fold Boundaries

| Fold | Training Start | Training End | Validation Start | Validation End | Train $N$ | Val $N$ | Separation Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2010-03-11 | 2013-05-10 | 2013-05-16 | 2014-07-07 | 2,319 | 749 | 26 bars (104h) |
| **2** | 2010-03-11 | 2014-07-07 | 2014-07-09 | 2015-08-31 | 3,074 | 807 | 15 bars (60h) |
| **3** | 2010-03-11 | 2015-08-31 | 2015-09-02 | 2016-10-20 | 3,886 | 746 | 13 bars (52h) |
| **4** | 2010-03-11 | 2016-10-20 | 2016-10-26 | 2017-12-13 | 4,636 | 794 | 20 bars (80h) |
| **5** | 2010-03-11 | 2017-12-13 | 2017-12-18 | 2019-02-08 | 5,434 | 855 | 16 bars (64h) |
| **6** | 2010-03-11 | 2019-02-08 | 2019-02-12 | 2020-04-02 | 6,293 | 783 | 13 bars (52h) |
| **7** | 2010-03-11 | 2020-04-02 | 2020-04-08 | 2021-05-27 | 7,084 | 815 | 24 bars (96h) |
| **8** | 2010-03-11 | 2021-05-27 | 2021-06-01 | 2022-07-18 | 7,905 | 745 | 17 bars (68h) |
| **9** | 2010-03-11 | 2022-07-18 | 2022-07-25 | 2023-09-06 | 8,651 | 808 | 28 bars (112h) |
| **10** | 2010-03-11 | 2023-09-06 | 2023-09-13 | 2024-11-01 | 9,462 | 787 | 28 bars (112h) |

---

## 12. Fold-Level Results

Chronological comparison of Baseline (30 features) versus Macro Candidate (32 features):

| Fold | Validation Period | Val $N$ | Baseline BalAcc | Macro BalAcc | BalAcc Delta | Baseline F1 | Macro F1 | Baseline Confusion Matrix | Macro Confusion Matrix |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2013-05 to 2014-07 | 749 | **58.25%** | 54.56% | **-3.69%** | 0.5828 | 0.5460 | `[[247, 137], [176, 189]]` | `[[224, 160], [180, 185]]` |
| **2** | 2014-07 to 2015-08 | 807 | **55.14%** | 53.81% | **-1.33%** | 0.5401 | 0.5369 | `[[369, 114], [236, 88]]` | `[[329, 154], [206, 118]]` |
| **3** | 2015-09 to 2016-10 | 746 | 51.12% | **51.44%** | **+0.32%** | 0.5103 | 0.5126 | `[[183, 178], [187, 198]]` | `[[170, 191], [171, 214]]` |
| **4** | 2016-10 to 2017-12 | 794 | 44.17% | **49.44%** | **+5.27%** | 0.4435 | 0.4939 | `[[134, 184], [258, 218]]` | `[[155, 163], [238, 238]]` |
| **5** | 2017-12 to 2019-02 | 855 | 46.96% | **47.73%** | **+0.77%** | 0.4673 | 0.4770 | `[[235, 230], [224, 166]]` | `[[221, 244], [202, 188]]` |
| **6** | 2019-02 to 2020-04 | 783 | 49.05% | **49.37%** | **+0.32%** | 0.4908 | 0.4934 | `[[202, 192], [207, 182]]` | `[[187, 207], [189, 200]]` |
| **7** | 2020-04 to 2021-05 | 815 | 52.42% | **55.00%** | **+2.58%** | 0.5230 | 0.5501 | `[[187, 179], [209, 240]]` | `[[207, 159], [207, 242]]` |
| **8** | 2021-06 to 2022-07 | 745 | **53.05%** | 52.02% | **-1.03%** | 0.5310 | 0.5186 | `[[268, 160], [190, 127]]` | `[[286, 142], [214, 103]]` |
| **9** | 2022-07 to 2023-09 | 808 | **49.05%** | 45.70% | **-3.36%** | 0.4916 | 0.4578 | `[[191, 208], [204, 205]]` | `[[193, 206], [233, 176]]` |
| **10** | 2023-09 to 2024-11 | 787 | 47.36% | **47.54%** | **+0.18%** | 0.4727 | 0.4754 | `[[219, 219], [195, 154]]` | `[[221, 217], [196, 153]]` |

---

## 13. Aggregate Results

| Metric | Baseline (30 Features) | Macro Candidate (32 Features) | Difference (Macro - Base) |
| :--- | :---: | :---: | :---: |
| **Mean Balanced Accuracy** | **50.66%** | **50.66%** | **+0.00%** (+0.0035 pp) |
| **Median Balanced Accuracy** | **50.09%** | **50.44%** | **+0.35%** |
| **Standard Deviation** | 3.98% | 3.04% | -0.94% |
| **Minimum Balanced Accuracy** | 44.17% (Fold 4) | 45.70% (Fold 9) | +1.53% |
| **Maximum Balanced Accuracy** | 58.25% (Fold 1) | 55.00% (Fold 7) | -3.25% |
| **Folds Above 50.0%** | **5 / 10** | **5 / 10** | 0 |
| **Folds Above 52.0%** | **4 / 10** | **4 / 10** | 0 |
| **Folds Below 50.0%** | **5 / 10** | **5 / 10** | 0 |

---

## 14. Paired Statistical Test

A formal paired statistical test was conducted across the 10 chronological fold pairs:

- **Test Name:** Paired Student's t-test (two-tailed)
- **Degrees of Freedom:** $df = 9$
- **Test Statistic:** $t = 0.0029$
- **p-value:** $p = 0.9978$
- **Mean Paired Difference:** $+0.0035$ percentage points ($+0.000035$)
- **95% Confidence Interval:** $[-1.89\%, +1.89\%]$
- **Non-Parametric Wilcoxon Signed-Rank Test:** $W = 27.0000, p = 1.0000$

> [!CAUTION]
> With $p = 0.9978$, the difference between the Macro Candidate model and the Baseline is indistinguishable from random noise. The 95% confidence interval spans zero symmetric bounds from $-1.89\%$ to $+1.89\%$.

---

## 15. Fresh Holdout Result

The sealed Fresh Research Holdout was unsealed for a single, one-shot out-of-sample evaluation:
- **Date Range:** `2024-11-06 00:00:00 UTC` to `2026-02-19 10:45:00 UTC`
- **Training Set:** Full Pre-Holdout Partition (`2010-03-01 16:00:00 UTC` to `2024-11-04 12:00:00 UTC`, 10,256 valid labeled bars)
- **Holdout Evaluation Samples:** Exactly 838 valid labeled bars (zero lookahead into locked test)

| Model | Balanced Accuracy | Accuracy | Macro F1 | ROC AUC | Confusion Matrix `[[-1, 1], [-1, 1]]` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (30 Features)** | 48.96% | 50.60% | 0.4708 | 0.4935 | `[[104, 284], [130, 320]]` |
| **Macro Candidate (32 Features)** | 50.17% | 52.15% | 0.4726 | 0.5008 | `[[91, 297], [104, 346]]` |
| **Incremental Difference** | **+1.21%** | +1.55% | +0.0018 | +0.0073 | — |

While the Macro Candidate produced a slight $+1.21\%$ delta on the holdout partition, its absolute Balanced Accuracy reached only **50.17%**, which falls substantially below both the 52.0% failure threshold and the 55.0% success threshold.

---

## 16. Failure/Success Gate Evaluation

### A. Pre-Specified Success Gate Evaluation (ALL must pass)

| # | Success Condition | Required | Actual Observed | Result |
| :---: | :--- | :---: | :---: | :---: |
| **1** | Mean Walk-Forward Balanced Accuracy | $\ge 55.0\%$ | 50.66% | **FAILED** |
| **2** | Folds with Balanced Accuracy $\ge 52.0\%$ | $\ge 8 / 10$ | 4 / 10 | **FAILED** |
| **3** | Paired Fold Comparison Statistical Significance | $p < 0.05$ | $p = 0.9978$ | **FAILED** |
| **4** | Fresh Holdout Balanced Accuracy | $\ge 55.0\%$ | 50.17% | **FAILED** |
| **OVERALL** | **Success Gate Verdict** | **All 4 Pass** | **0 / 4 Met** | **`FAILED`** |

### B. Pre-Specified Failure Gate Evaluation (ANY triggers failure)

| # | Failure Criterion | Falsification Threshold | Actual Observed | Trigger Status |
| :---: | :--- | :---: | :---: | :---: |
| **1** | Mean Walk-Forward Balanced Accuracy | $< 53.5\%$ | 50.66% | **`TRIGGERED`** |
| **2** | Folds with Balanced Accuracy $< 50.0\%$ | $> 3 / 10$ | 5 / 10 | **`TRIGGERED`** |
| **3** | Mean Improvement Over Baseline | $\le +1.0\%$ | +0.00% | **`TRIGGERED`** |
| **4** | Fresh Holdout Balanced Accuracy | $< 52.0\%$ | 50.17% | **`TRIGGERED`** |
| **OVERALL** | **Failure Gate Verdict** | **Any 1 Triggers** | **All 4 Triggered** | **`TRIGGERED`** |

Every single failure criterion was triggered independently.

---

## 17. Leakage Audit

Ten deterministic audit checks were executed programmatically:

```
[PASS] 1. Point-in-Time Lag Verification     : Day D yield never appears before Day D+1 00:00 UTC
[PASS] 2. Causal 5-Day Lookback              : Spread momentum strictly uses Spread[t] - Spread[t-5 business days]
[PASS] 3. Forward-Fill Integrity             : Zero future yields filled backward into historical bars
[PASS] 4. Purge Gap Enforcement              : Exact 8 H4 bars (32h) purged between train and validation
[PASS] 5. Embargo Buffer Enforcement         : At least 12 bars (purge 8 + embargo 4) separate train and val
[PASS] 6. Zero Train/Val Overlap             : Train end ts strictly precedes val start ts across all folds
[PASS] 7. Expanding Train Monotonicity       : Training sample count strictly increases monotonically across folds
[PASS] 8. Feature Precedes Target Outcome    : Features at candle close t; target resolves at candle close t+8
[PASS] 9. Locked Test Partition Protection   : Zero bars or labels accessed from 2026-02-19 12:00:00 UTC onward
[PASS] 10. Feature Count Immutability        : Exactly 30 baseline features + exactly 2 approved macro features
```

---

## 18. Limitations

1. **Unconditional Model Family Invariance:** The Random Forest tree splitting mechanism did not identify split points in the 2-Year yield spread that generalize out of sample. At tree depth 5, price volatility and momentum features absorb almost all tree split criteria.
2. **Horizon Asymmetry:** Daily macro yield differentials evolve over weeks and months, whereas the H4 swing target ($H=8$ bars = 32 hours) operates on a 1.3-day horizon. The rate of change in sovereign spreads over 32 hours is typically fractional basis points, insufficient to produce directional displacement over transaction frictions.
3. **Regime Non-Stationarity:** In earlier eras (e.g. 2013–2016 during the European sovereign debt crisis), wider yield differentials had strong currency correlation; in later eras (2020–2024 quantitative easing and zero lower bound), central bank asset purchases compressed yield differentials and severed the classic relationship.

---

## 19. Scientific Decision

### **`NOT SUPPORTED`**

### Evidence-Based Justification:
1. **Predictive Failure Gate Fully Triggered:** The macro candidate triggers all four pre-specified falsification criteria:
   - Mean walk-forward Balanced Accuracy of **50.66%** is below the 53.5% threshold.
   - **5 out of 10 folds** fall below 50.0% (failure threshold: $> 3$ folds).
   - Net improvement over the frozen baseline is **+0.00%** (failure threshold: $\le +1.0\%$).
   - Fresh holdout Balanced Accuracy of **50.17%** is below the 52.0% threshold.
2. **Statistical Insignificance:** The paired t-test between fold balanced accuracies yields $t = 0.0029, p = 0.9978$, proving that the macro features provide zero statistically detectable edge.
3. **No Trading Backtest Justified:** Because the candidate failed the pre-specified predictive gate, no trading simulation, transaction-cost backtest, or parameter optimization is authorized.

---

## Mandatory Governance Confirmation

- **Phase 11 Test Partition:** 951 bars starting `2026-02-19 12:00:00 UTC` remain **PERMANENTLY LOCKED & UNTOUCHED**. Zero labels inspected.
- **Phase 18 Holdout:** Evaluated strictly once without tuning.
- **Live Trading:** NOT STARTED.
- **Demo Trading:** NOT STARTED.
- **Hyperparameter Search:** ZERO performed.
- **Feature Mining:** ZERO performed.
- **Phase 26:** **NOT STARTED.**
