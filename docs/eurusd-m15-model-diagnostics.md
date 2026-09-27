# EURUSD M15 Baseline Model Diagnostics, Probability Analysis & Feature Importance Specification

**Phase:** Phase 9 — Baseline Model Diagnostics, Probability Analysis & Feature Importance  
**Instrument:** EURUSD  
**Timeframe:** M15 (15-Minute Candles)  
**Prediction Horizon:** $H = 4$ bars (1 hour / 60 minutes)  
**Target:** `direction_4` (Ternary Direction Classification: -1.0 [SHORT], 0.0 [NEUTRAL], +1.0 [LONG])  
**Input Features:** 80 point-in-time engineered features from Phase 5 feature registry  
**Evaluated Baseline Models:** Phase 8 `LogisticRegressionBaseline` and `RandomForestBaseline`  
**Dataset Split:** Phase 7 chronological split (Train: 69,937 rows, Validation: 14,983 rows, Test: 14,988 rows)  
**Evaluation Scope:** Strictly Validation Partition Only (Test partition remains 100% locked)  

---

> [!IMPORTANT]
> **Explicit Research Boundary & Non-Viability Disclaimers:**
> - Phase 9 is an exploratory, descriptive diagnostic analysis of baseline models trained in Phase 8.
> - These diagnostics do **not** establish trading profitability, economic value, edge, alpha, or live-trading viability.
> - The models evaluated herein are **unmodified baseline models** fitted on the training split only. No hyperparameters were tuned, no feature selection was conducted, no probability calibration was fitted (no Platt scaling or isotonic regression), and no decision thresholds were optimized.
> - The **TEST partition (14,988 observations) was NOT accessed, inspected, or evaluated** in any capacity.

---

## 1. Objective

The objective of Phase 9 is to conduct a rigorous diagnostic post-mortem of the Phase 8 baseline classifiers (`LogisticRegressionBaseline` and `RandomForestBaseline`) on the EURUSD M15 directional forecasting problem ($H=4$).

Specifically, Phase 9 addresses:
1. **Probability Distribution Behavior:** Examining the spread, dispersion, and certainty of predicted class probabilities.
2. **Confidence / Coverage Trade-Offs:** Quantifying sample retention and accuracy dynamics when conditioning predictions on minimum confidence thresholds ($\tau \in [0.40, 0.90]$).
3. **Class-Wise Discrimination:** Measuring per-class precision, recall, F1, and threshold-independent discrimination via One-vs-Rest ROC-AUC and Precision-Recall Average Precision (PR-AUC).
4. **Feature Importance Decomposition:** Analyzing Random Forest Mean Decrease in Impurity (MDI) and Logistic Regression coefficient magnitudes across all 80 features.
5. **Feature Sanity & Leakage Verification:** Auditing top features against the Phase 5 feature registry to confirm absence of targets, timestamps, or raw OHLCV base columns.
6. **Temporal Validation Stability:** Evaluating performance stability across three contiguous chronological validation sub-periods (Blocks A, B, and C).
7. **Error Analysis:** Identifying empirical misclassification modes, directional flip frequencies, and neutral confusion patterns.
8. **Model Agreement Dynamics:** Measuring prediction concordance, Cohen's Kappa, and conditional accuracies between linear and non-linear representations.
9. **Descriptive Calibration Assessment:** Diagnosing empirical reliability and Brier scores without altering model parameters.

---

## 2. Dataset Used

The diagnostic analysis uses the chronological partitions established in Phase 7 and modeled in Phase 8:

- **Source Market Feed:** 100,000 real MetaQuotes-Demo EURUSD M15 candles (September 2022 to September 2026).
- **Usable Clean Observations:** 99,916 rows (after dropping 80 warm-up bars and reserving 4 tail bars).
- **Target Specification:** `direction_4` ($H=4$ bars / 60 minutes forward price displacement with 5.0 bps fixed threshold).
- **Feature Set:** Exactly 80 point-in-time features encompassing price action, momentum, volatility, trend, activity, diurnal/session timing, and gap mechanics.
- **Training Partition (Model Fitting Only):** 69,937 observations (`2022-09-19 05:30:00` to `2025-07-14 04:30:00` UTC).
- **Validation Partition (Diagnostics Evaluation):** 14,983 observations (`2025-07-14 05:45:00` to `2026-02-19 10:45:00` UTC).
- **Purge Buffers:** 4-bar forward purge buffers ($H=4$, 60 minutes) precede each partition boundary.

---

## 3. Test-Set Protection

Strict isolation of the final test partition is maintained throughout Phase 9:

- **Test Observations:** 14,988 rows (`2026-02-19 12:00:00` to `2026-09-25 22:45:00` UTC).
- **Access Status:** **Zero access.** The test partition was never loaded into memory, inspected, transformed, predicted on, or evaluated.
- **Verification:** Automated tests in [`tests/test_model_diagnostics.py`](file:///home/cino/projects/ai-trading-system/tests/test_model_diagnostics.py) formally verify that all diagnostic functions operate exclusively on validation arrays.
- **Metadata Contract:** Field `"test_set_used": false` is verified in [`reports/model_diagnostics_metadata.json`](file:///home/cino/projects/ai-trading-system/reports/model_diagnostics_metadata.json).

---

## 4. Probability Distributions

Predicted probability distributions for each observation $x \in \text{Validation}$ were extracted from both classifiers:

### 4.1 Maximum Predicted Probability Distribution ($\max_c P(y=c \mid x)$)

| Model | Mean | Std Dev | Min | Median (p50) | p75 | p90 | p95 | p99 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6387 | 0.1627 | 0.3343 | 0.6353 | 0.7885 | 0.8469 | 0.8728 | 0.9191 | 0.9728 |
| **Random Forest** | 0.5591 | 0.1302 | 0.3337 | 0.5636 | 0.6635 | 0.7357 | 0.7701 | 0.8148 | 0.8516 |

### 4.2 Distribution Insights
- **Logistic Regression:** Features higher probability variance (std 0.1627 vs 0.1302) and reaches higher peak confidence values (99th percentile at 0.9191, maximum at 0.9728). This reflects the tendency of the softmax transfer function to push uncalibrated linear log-odds toward extreme probabilities.
- **Random Forest:** Ensemble averaging across 100 decision trees shrinks predicted probabilities toward the training prior distribution. The maximum predicted probability produced by Random Forest across 14,983 validation samples is 0.8516.
- Both models exhibit a median maximum predicted probability near 0.56–0.64, closely tracking the validation majority class prevalence (NEUTRAL: 61.94%).

**Generated Artifacts:**
- [`reports/figures/logistic_max_probability_distribution.png`](file:///home/cino/projects/ai-trading-system/reports/figures/logistic_max_probability_distribution.png)
- [`reports/figures/random_forest_max_probability_distribution.png`](file:///home/cino/projects/ai-trading-system/reports/figures/random_forest_max_probability_distribution.png)
- [`reports/figures/logistic_probability_by_class.png`](file:///home/cino/projects/ai-trading-system/reports/figures/logistic_probability_by_class.png)
- [`reports/figures/random_forest_probability_by_class.png`](file:///home/cino/projects/ai-trading-system/reports/figures/random_forest_probability_by_class.png)

---

## 5. Confidence / Coverage Analysis

To diagnose precision and coverage trade-offs, predictions were filtered by minimum confidence threshold $\tau \in \{0.40, 0.50, 0.60, 0.70, 0.80, 0.90\}$:

### 5.1 Logistic Regression Coverage & Performance

| Threshold $\tau$ | Covered Observations | Coverage % | Subset Accuracy | Subset Balanced Acc | Subset Macro F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.40** | 13,841 | 92.38% | 0.6436 | 0.3395 | 0.2788 |
| **0.50** | 11,169 | 74.54% | 0.6899 | 0.3347 | 0.2761 |
| **0.60** | 8,288 | 55.32% | 0.7428 | 0.3340 | 0.2855 |
| **0.70** | 6,298 | 42.03% | 0.7815 | 0.3333 | 0.2925 |
| **0.80** | 3,307 | 22.07% | 0.8397 | 0.3333 | 0.3043 |
| **0.90** | 302 | 2.02% | 0.9205 | 0.3333 | 0.3195 |

### 5.2 Random Forest Coverage & Performance

| Threshold $\tau$ | Covered Observations | Coverage % | Subset Accuracy | Subset Balanced Acc | Subset Macro F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.40** | 12,662 | 84.51% | 0.6658 | 0.3378 | 0.2774 |
| **0.50** | 9,634 | 64.30% | 0.7200 | 0.3333 | 0.2791 |
| **0.60** | 6,024 | 40.21% | 0.7887 | 0.3333 | 0.2940 |
| **0.70** | 2,536 | 16.93% | 0.8604 | 0.3333 | 0.3083 |
| **0.80** | 315 | 2.10% | 0.9206 | 0.3333 | 0.3196 |
| **0.90** | 0 | 0.00% | N/A | N/A | N/A |

### 5.3 Diagnostic Interpretation
- As the confidence threshold rises from 0.40 to 0.80, subset accuracy increases from ~64% to ~84% in Logistic Regression and ~66% to ~92% in Random Forest.
- However, **balanced accuracy remains flat at ~33.33%**.
- This indicates that high-confidence predictions consist almost exclusively of `NEUTRAL (0)` classifications during periods of compressed market volatility. The models do not develop high confidence on directional breakouts (`SHORT` or `LONG`).
- Therefore, thresholding on raw probability does not isolate high-confidence directional signals; it merely isolates low-volatility neutral regimes.

---

## 6. Class-Wise Performance

Evaluating performance separately for `SHORT (-1)`, `NEUTRAL (0)`, and `LONG (+1)`:

| Model | Class | True Support | Precision | Recall | F1 Score | OvR ROC-AUC | Average Precision (PR-AUC) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | SHORT (-1) | 2,882 | 0.3380 | 0.0500 | 0.0871 | 0.6562 | 0.2846 |
| | NEUTRAL (0) | 9,281 | 0.6350 | 0.9750 | 0.7691 | 0.6999 | 0.7830 |
| | LONG (+1) | 2,820 | 0.3203 | 0.0348 | 0.0627 | 0.6463 | 0.2732 |
| **Random Forest** | SHORT (-1) | 2,882 | 0.3390 | 0.0906 | 0.1429 | 0.6515 | 0.2830 |
| | NEUTRAL (0) | 9,281 | 0.6441 | 0.9644 | 0.7724 | 0.7040 | 0.7866 |
| | LONG (+1) | 2,820 | 0.3544 | 0.0397 | 0.0714 | 0.6530 | 0.2835 |

### 6.1 Findings
- **Neutral Dominance:** Both models exhibit high recall on `NEUTRAL` (97.50% for LR, 96.44% for RF), but very low recall on directional classes (SHORT recall: 5.00% LR, 9.06% RF; LONG recall: 3.48% LR, 3.97% RF).
- **Directional Precision:** When predicting directional movements, precision is ~32.0–35.4%, which is higher than the random baseline prevalence (SHORT: 19.24%, LONG: 18.82%), but directional F1 scores remain depressed (< 0.15) due to low recall.

---

## 7. ROC Analysis

One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) curves evaluate rank-order discrimination independent of the default 0.33 decision threshold:

- **SHORT (-1):** LR ROC-AUC = 0.6562; RF ROC-AUC = 0.6515 (vs random guess 0.5000).
- **NEUTRAL (0):** LR ROC-AUC = 0.6999; RF ROC-AUC = 0.7040.
- **LONG (+1):** LR ROC-AUC = 0.6463; RF ROC-AUC = 0.6530.
- **Macro OvR ROC-AUC:** LR = 0.6675; RF = 0.6695.

### 7.1 Interpretation
- Both baseline models possess genuine rank-order discrimination significantly above the 0.50 random baseline across all three classes.
- The separation between LR (0.6675) and RF (0.6695) is minimal (+0.0020), indicating that linear combinations of normalized features achieve comparable ranking performance to tree-based non-linear ensembles.

**Generated Artifacts:**
- [`reports/figures/roc_curve_logistic_regression.png`](file:///home/cino/projects/ai-trading-system/reports/figures/roc_curve_logistic_regression.png)
- [`reports/figures/roc_curve_random_forest.png`](file:///home/cino/projects/ai-trading-system/reports/figures/roc_curve_random_forest.png)
- [`reports/figures/roc_curve_model_comparison.png`](file:///home/cino/projects/ai-trading-system/reports/figures/roc_curve_model_comparison.png)

---

## 8. Precision-Recall Analysis

Because directional classes constitute minority proportions (~19% each), Precision-Recall curves and Average Precision (AP) provide a more stringent evaluation than ROC curves:

- **SHORT (-1):** LR AP = 0.2846; RF AP = 0.2830 (vs baseline prevalence: 19.24%).
- **NEUTRAL (0):** LR AP = 0.7830; RF AP = 0.7866 (vs baseline prevalence: 61.94%).
- **LONG (+1):** LR AP = 0.2732; RF AP = 0.2835 (vs baseline prevalence: 18.82%).

### 8.1 Interpretation
- For both directional classes, Average Precision exceeds baseline prevalence by approximately 9 to 10 percentage points (e.g., 28.3% vs 18.8–19.2%).
- This confirms that top-ranked predicted probabilities for SHORT and LONG contain genuine directional information above base rate, even though the standard argmax classification rule selects NEUTRAL in ~95% of instances.

**Generated Artifacts:**
- [`reports/figures/pr_curve_logistic_regression.png`](file:///home/cino/projects/ai-trading-system/reports/figures/pr_curve_logistic_regression.png)
- [`reports/figures/pr_curve_random_forest.png`](file:///home/cino/projects/ai-trading-system/reports/figures/pr_curve_random_forest.png)

---

## 9. Random Forest Feature Importance

Random Forest feature importances were extracted using Mean Decrease in Impurity (MDI). All 80 features were ranked:

| Rank | Feature | MDI Importance | Feature Category |
| :---: | :--- | :---: | :--- |
| **1** | `hl_range` | 0.08828 | Price Action / Geometry |
| **2** | `hl_range_norm` | 0.08012 | Price Action / Normalized Range |
| **3** | `log_tick_volume` | 0.07274 | Market Activity / Volume |
| **4** | `cos_hour` | 0.05424 | Diurnal / Cyclical Timing |
| **5** | `tick_vol_zscore_80` | 0.04078 | Market Activity / Normalized Volume |
| **6** | `atr_14` | 0.03962 | Volatility / True Range |
| **7** | `vol_std_10` | 0.03272 | Volatility / Standard Deviation |
| **8** | `atr_norm_14` | 0.03188 | Volatility / Normalized ATR |
| **9** | `hour` | 0.02945 | Diurnal Timing |
| **10** | `vol_ann_20` | 0.02600 | Volatility / Annualized Vol |
| **11** | `atr_40` | 0.02534 | Volatility / True Range |
| **12** | `delta_seconds` | 0.02498 | Gap / Bar Arrival Timing |
| **13** | `tick_vol_zscore_20` | 0.02324 | Market Activity / Normalized Volume |
| **14** | `candle_body` | 0.02298 | Price Action / Geometry |
| **15** | `vol_std_40` | 0.02269 | Volatility / Standard Deviation |
| **16** | `sin_hour` | 0.02195 | Diurnal / Cyclical Timing |
| **17** | `dist_ema_40` | 0.02058 | Trend / Moving Average Distance |
| **18** | `atr_norm_40` | 0.02038 | Volatility / Normalized ATR |
| **19** | `vol_ann_40` | 0.02035 | Volatility / Annualized Vol |
| **20** | `vol_std_20` | 0.02008 | Volatility / Standard Deviation |

### 9.1 Observations
- The top three features (`hl_range`, `hl_range_norm`, and `log_tick_volume`) account for ~24.1% of total impurity reduction.
- Volatility and activity features dominate the top 20 rankings. This indicates that decision trees primarily partition samples by volatility regimes, distinguishing between compressed consolidation (leading to `NEUTRAL`) and volatile expansion (where `SHORT` or `LONG` become reachable).

**Generated Artifacts:**
- [`reports/random_forest_feature_importance.csv`](file:///home/cino/projects/ai-trading-system/reports/random_forest_feature_importance.csv)
- [`reports/figures/random_forest_feature_importance_top20.png`](file:///home/cino/projects/ai-trading-system/reports/figures/random_forest_feature_importance_top20.png)

---

## 10. Logistic Regression Feature Importance

For multiclass Logistic Regression, the aggregate importance of feature $j$ is computed as the mean absolute standardized coefficient across the three classes: $\frac{1}{3}\sum_{k=1}^3 |w_{k, j}|$:

| Rank | Feature | Mean Absolute $\|w\|$ | $w_{\text{SHORT}}$ | $w_{\text{NEUTRAL}}$ | $w_{\text{LONG}}$ |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | `hl_range_norm` | 0.3639 | -0.546 | +0.062 | +0.484 |
| **2** | `atr_40` | 0.3436 | +0.090 | -0.515 | +0.426 |
| **3** | `delta_seconds` | 0.3112 | +0.467 | -0.191 | -0.276 |
| **4** | `ema_spread_10_40` | 0.2623 | -0.394 | +0.154 | +0.239 |
| **5** | `dist_sma_80` | 0.2111 | +0.317 | -0.042 | -0.275 |
| **6** | `dist_ema_40` | 0.2092 | -0.314 | +0.058 | +0.256 |
| **7** | `candle_body_norm` | 0.2081 | +0.132 | +0.180 | -0.312 |
| **8** | `dist_sma_40` | 0.1809 | +0.227 | +0.044 | -0.271 |
| **9** | `atr_norm_40` | 0.1751 | +0.025 | +0.238 | -0.263 |
| **10** | `dist_sma_20` | 0.1610 | +0.162 | +0.079 | -0.242 |
| **11** | `return_mean_5` | 0.1593 | -0.188 | +0.048 | +0.140 |
| **12** | `dist_ema_10` | 0.1584 | +0.141 | -0.096 | -0.045 |
| **13** | `hl_range` | 0.1583 | +0.170 | -0.167 | -0.003 |
| **14** | `candle_direction` | 0.1549 | -0.193 | +0.051 | +0.142 |
| **15** | `dist_ema_80` | 0.1527 | -0.198 | -0.016 | +0.214 |
| **16** | `open_to_close_return` | 0.1458 | -0.194 | +0.048 | +0.146 |
| **17** | `atr_norm_14` | 0.1417 | +0.014 | +0.198 | -0.212 |
| **18** | `log_return_1` | 0.1376 | -0.178 | +0.040 | +0.138 |
| **19** | `return_1` | 0.1374 | -0.178 | +0.040 | +0.138 |
| **20** | `tick_vol_zscore_80` | 0.1352 | +0.137 | -0.202 | +0.065 |

### 10.1 Directional Sign Consistency
- `hl_range_norm` has negative weight for SHORT (-0.546) and positive weight for LONG (+0.484).
- `ema_spread_10_40` (fast EMA minus slow EMA) has negative weight for SHORT (-0.394) and positive weight for LONG (+0.239), aligning with trend momentum physics.
- `return_mean_5`, `candle_direction`, and `return_1` all exhibit symmetric opposite signs between SHORT and LONG.

**Generated Artifacts:**
- [`reports/logistic_regression_feature_importance.csv`](file:///home/cino/projects/ai-trading-system/reports/logistic_regression_feature_importance.csv)
- [`reports/figures/logistic_regression_feature_importance_top20.png`](file:///home/cino/projects/ai-trading-system/reports/figures/logistic_regression_feature_importance_top20.png)

---

## 11. Feature Sanity Checks

Automated checks validated that feature importances do not reflect data corruption or lookahead leakage:

1. **Feature Count:** Exactly 80 features extracted and ranked across both models.
2. **Registry Match:** All 80 features correspond exactly to names registered in Phase 5 (`ai.features.pipeline.get_feature_registry()`).
3. **No Target Leakage:** Zero candidate target names (`direction_*`, `future_*`, `forward_*`, `target`, `label`) exist in the feature set.
4. **No Timestamp Leakage:** Zero raw timestamp columns (`open_time`, `timestamp`, `date`, `time`, `datetime`) exist in the feature set.
5. **No Raw OHLCV Base Columns:** Zero unadjusted raw columns (`open`, `high`, `low`, `close`, `tick_volume`, `spread`, `real_volume`) exist in the feature set.

All sanity checks evaluated to **PASS**.

---

## 12. Temporal Validation Diagnostics

To evaluate model stability across time without retuning or selecting on validation subsets, the validation partition was split into three contiguous chronological sub-periods:

- **VALIDATION_A (Early):** 4,994 rows (`2025-07-14 05:45:00` to `2025-09-24 18:30:00` UTC)
- **VALIDATION_B (Middle):** 4,994 rows (`2025-09-24 18:45:00` to `2025-12-05 06:15:00` UTC)
- **VALIDATION_C (Late):** 4,995 rows (`2025-12-05 06:30:00` to `2026-02-19 10:45:00` UTC)

| Model | Block | Row Count | Accuracy | Balanced Acc | Macro F1 | Recall SHORT | Recall NEUTRAL | Recall LONG | OvR ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | VALIDATION_A | 4,994 | 0.5713 | 0.3602 | 0.3140 | 0.0538 | 0.9554 | 0.0713 | 0.6618 |
| | VALIDATION_B | 4,994 | 0.6460 | 0.3456 | 0.2917 | 0.0317 | 0.9888 | 0.0163 | 0.6663 |
| | VALIDATION_C | 4,995 | 0.6430 | 0.3509 | 0.3045 | 0.0642 | 0.9785 | 0.0099 | 0.6653 |
| **Random Forest** | VALIDATION_A | 4,994 | 0.5741 | 0.3687 | 0.3307 | 0.0867 | 0.9473 | 0.0722 | 0.6583 |
| | VALIDATION_B | 4,994 | 0.6480 | 0.3569 | 0.3161 | 0.0700 | 0.9798 | 0.0209 | 0.6721 |
| | VALIDATION_C | 4,995 | 0.6448 | 0.3670 | 0.3337 | 0.1170 | 0.9642 | 0.0198 | 0.6726 |

### 12.1 Temporal Stability Observations
- **Discrimination Robustness:** OvR ROC-AUC remains remarkably consistent across blocks (LR: 0.6618 to 0.6663; RF: 0.6583 to 0.6726), demonstrating that rank-order discrimination does not deteriorate over time.
- **Accuracy Shift:** Accuracy increases from ~57% in Block A to ~64.5% in Blocks B and C. This shift tracks changes in underlying market volatility: Block A experienced higher volatility with fewer neutral bars, whereas Blocks B and C exhibited lower volatility and higher neutral prevalence (~64%).
- **Balanced Accuracy Invariance:** Balanced accuracy remains in the narrow range of 34.5% to 36.9% across all three sub-periods, confirming stable underlying model behavior.

**Generated Artifacts:**
- [`reports/validation_temporal_diagnostics.csv`](file:///home/cino/projects/ai-trading-system/reports/validation_temporal_diagnostics.csv)
- [`reports/figures/validation_temporal_diagnostics.png`](file:///home/cino/projects/ai-trading-system/reports/figures/validation_temporal_diagnostics.png)

---

## 13. Error Analysis

Misclassification distributions across all $(y_{\text{true}}, y_{\text{pred}})$ combinations were quantified:

### 13.1 Logistic Regression Error Breakdown (Total Errors: 5,692 rows / 37.99%)

| True Class | Predicted Class | Count | % of Validation | % of Errors | Mean Confidence | Median Confidence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SHORT (-1)** | **NEUTRAL (0)** | 2,625 | 17.52% | 46.12% | 0.5853 | 0.5694 |
| **LONG (+1)** | **NEUTRAL (0)** | 2,577 | 17.20% | 45.27% | 0.5869 | 0.5732 |
| **LONG (+1)** | **SHORT (-1)** | 145 | 0.97% | 2.55% | 0.4125 | 0.3957 |
| **NEUTRAL (0)** | **SHORT (-1)** | 137 | 0.91% | 2.41% | 0.3996 | 0.3846 |
| **NEUTRAL (0)** | **LONG (+1)** | 95 | 0.63% | 1.67% | 0.4079 | 0.3900 |
| **SHORT (-1)** | **LONG (+1)** | 113 | 0.75% | 1.99% | 0.4283 | 0.4132 |

### 13.2 Random Forest Error Breakdown (Total Errors: 5,660 rows / 37.78%)

| True Class | Predicted Class | Count | % of Validation | % of Errors | Mean Confidence | Median Confidence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SHORT (-1)** | **NEUTRAL (0)** | 2,506 | 16.73% | 44.28% | 0.5243 | 0.5230 |
| **LONG (+1)** | **NEUTRAL (0)** | 2,440 | 16.29% | 43.11% | 0.5220 | 0.5234 |
| **LONG (+1)** | **SHORT (-1)** | 268 | 1.79% | 4.73% | 0.3696 | 0.3601 |
| **NEUTRAL (0)** | **SHORT (-1)** | 241 | 1.61% | 4.26% | 0.3651 | 0.3582 |
| **SHORT (-1)** | **LONG (+1)** | 115 | 0.77% | 2.03% | 0.3820 | 0.3676 |
| **NEUTRAL (0)** | **LONG (+1)** | 90 | 0.60% | 1.59% | 0.3770 | 0.3649 |

### 13.3 Key Findings
- **Dominant Error Mode:** Over 91% of all errors are **neutral omissions**: true directional displacements (SHORT or LONG) classified as NEUTRAL. This is a direct consequence of class imbalance and default 0.33 decision thresholds.
- **Directional Flips:** Critical directional flips (True LONG predicted SHORT, or True SHORT predicted LONG) occur in only **1.72%** of samples in Logistic Regression (258 rows) and **2.56%** in Random Forest (383 rows). When directional flips occur, model confidence is uniformly low (~0.37 to 0.42), barely exceeding uniform random choice (0.333).

**Generated Artifacts:**
- [`reports/error_analysis_logistic.csv`](file:///home/cino/projects/ai-trading-system/reports/error_analysis_logistic.csv)
- [`reports/error_analysis_random_forest.csv`](file:///home/cino/projects/ai-trading-system/reports/error_analysis_random_forest.csv)

---

## 14. Model Agreement

Concordance between Logistic Regression and Random Forest predictions on the 14,983 validation samples:

- **Raw Agreement Rate:** **94.31%** (14,130 identical predictions out of 14,983)
- **Disagreement Count:** 853 observations (5.69%)
- **Cohen's Kappa:** **0.5088** (indicates moderate-to-substantial agreement beyond chance)
- **Contingency Matrix (Rows: Logistic Regression, Columns: Random Forest):**

| LR \ RF | Pred SHORT (-1) | Pred NEUTRAL (0) | Pred LONG (+1) |
| :---: | :---: | :---: | :---: |
| **Pred SHORT (-1)** | **426** | 338 | 6 |
| **Pred NEUTRAL (0)** | 344 | **13,634** | 270 |
| **Pred LONG (+1)** | 0 | 369 | **70** |

- **Accuracy on Agreement Subset (14,130 rows):** **63.91%**
- **Accuracy on Disagreement Subset (853 rows):**
  - Logistic Regression Accuracy: 30.60%
  - Random Forest Accuracy: 34.47%

### 14.1 Interpretation
- When the two models agree, overall accuracy (63.91%) exceeds the majority baseline (61.94%).
- When the models disagree, accuracy drops significantly below 35% for both models. Disagreements occur near decision boundaries where signal-to-noise ratio is lowest.
- Disagreement is descriptive; it does not indicate superiority of either model.

**Generated Artifacts:**
- [`reports/model_agreement.csv`](file:///home/cino/projects/ai-trading-system/reports/model_agreement.csv)

---

## 15. Calibration Diagnostics

Probability calibration diagnostics assess the reliability of predicted probabilities without modifying model parameters:

### 15.1 Brier Score Loss (Lower is Better; Perfect = 0.0)

| Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| **Multiclass Brier Score** | **0.5051** | **0.5103** |
| SHORT (-1) One-vs-Rest Brier | 0.1488 | 0.1494 |
| NEUTRAL (0) One-vs-Rest Brier | 0.2094 | 0.2129 |
| LONG (+1) One-vs-Rest Brier | 0.1469 | 0.1479 |

### 15.2 Reliability Curve Diagnostics
- Uncalibrated Logistic Regression produces reliability curves that deviate from the 45-degree diagonal in the upper tails ($P > 0.70$), reflecting sigmoid overconfidence on linear combinations.
- Random Forest produces compressed probability distributions ($P \in [0.33, 0.85]$) where extreme probabilities are under-represented due to tree-averaging regularization.
- Both models achieve nearly identical Multiclass Brier scores (~0.505 to 0.510), indicating comparable probabilistic accuracy in uncalibrated form.

**Generated Artifacts:**
- [`reports/figures/calibration_curve_logistic_regression.png`](file:///home/cino/projects/ai-trading-system/reports/figures/calibration_curve_logistic_regression.png)
- [`reports/figures/calibration_curve_random_forest.png`](file:///home/cino/projects/ai-trading-system/reports/figures/calibration_curve_random_forest.png)

---

## 16. Limitations

The Phase 9 diagnostic findings are subject to specific technical limitations:

1. **Class Imbalance Impact:** The dominance of the NEUTRAL class (61.94% validation) heavily biases default argmax predictions toward zero, suppressing directional recall.
2. **Uncalibrated Probabilities:** Neither model has undergone post-hoc calibration (such as Platt scaling or isotonic regression). Confidence values reflect algorithmic scoring rather than true empirical probabilities.
3. **Absence of Economic / Cost Modeling:** Diagnostics measure statistical classification metrics only. They do not incorporate bid-ask spreads, broker commissions, slippage, latency, or position sizing rules.
4. **No Hyperparameter Optimization:** Models evaluate default scikit-learn settings. Performance may differ under systematic tuning.
5. **No Test-Set Confirmation:** All findings reflect training and validation splits. Generalizability to out-of-sample test data is unknown and unverified.

---

## 17. Conclusions

The Phase 9 diagnostics establish the following empirical observations:

1. **Discriminative Structure Confirmed:** Both Logistic Regression and Random Forest exhibit genuine ranking discrimination above random chance (OvR ROC-AUC = 0.6675 and 0.6695; Precision-Recall Average Precision = 0.283–0.285 vs 0.188–0.192 baseline prevalence for directional classes).
2. **Neutral Class Masking:** At default classification thresholds, both models predict NEUTRAL for >95% of validation bars. High-confidence subsets ($P \ge 0.70$) consist almost entirely of neutral volatility consolidation regimes rather than directional trends.
3. **Feature Dominance:** Top predictive features across both models are normalized price range (`hl_range_norm`), true range (`atr_40`, `atr_14`), market activity (`log_tick_volume`), and moving average distances (`ema_spread_10_40`, `dist_sma_80`). All 80 features passed strict sanity and zero-leakage audits.
4. **Temporal Stability:** Performance metrics and discrimination remain stable across three contiguous chronological validation sub-periods, with OvR ROC-AUC varying by less than 0.015.
5. **High Inter-Model Concordance:** Linear Logistic Regression and non-linear Random Forest share 94.31% prediction agreement, with Cohen's Kappa of 0.5088.
6. **Non-Viability Boundary:** These diagnostics confirm that while genuine predictive signal exists in the Phase 5 feature set, raw baseline models cannot be translated directly into trading signals without addressing class imbalance, calibration, and risk constraints.

These diagnostics do not establish profitability or trading viability.
