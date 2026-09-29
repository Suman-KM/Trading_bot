# Phase 13 — Intraday Signal Improvement Research

**Instrument:** EURUSD  
**Timeframe:** M15  
**Data Partition:** Training (69,937 bars) + Validation (14,983 bars)  
**Holdout Test Partition:** 14,988 bars (2026-02-19 12:00:00 UTC onward) — **PERMANENTLY LOCKED & UNTOUCHED**  
**Safety Core:** Sovereign, non-bypassable deterministic `RiskEngine` with hard capital limits  
**Research Scope:** Descriptive analysis of intraday signal quality, directional separability, probability calibration, and feature information. Zero live trading, zero MT5 execution, zero hyperparameter optimization.

---

## 1. Objective

Phase 13 investigates the fundamental empirical behavior of the EURUSD M15 directional forecasting pipeline to answer a core research question:

> *"Why does the current EURUSD M15 intraday model produce zero actionable signals at the baseline confidence threshold ($\tau = 0.60$), and can the existing research pipeline produce higher-quality, more actionable directional information without compromising chronological data integrity or sovereign risk controls?"*

This phase focuses exclusively on **signal generation mechanics**. It does **not** perform trading parameter sweeps (SL/TP optimization), does **not** search for profitable configurations, and does **not** deploy a new production threshold.

---

## 2. Data Used

The research strictly utilizes the established chronological partitions:
- **Training Partition:** 69,937 bars (`2022-09-19 05:30:00 UTC` to `2025-07-14 04:30:00 UTC`). Used exclusively to fit the ML models.
- **Validation Partition:** 14,983 bars (`2025-07-14 05:45:00 UTC` to `2026-02-19 10:45:00 UTC`). Used exclusively for descriptive signal audits, probability distribution analysis, calibration curves, and temporal stability checks.

All transformations and feature inputs remain strictly point-in-time and causal.

---

## 3. Phase 11 Holdout Protection

The Phase 11 Test partition comprises **14,988 bars** (`2026-02-19 12:00:00 UTC` to `2026-09-25 22:45:00 UTC`).
- **Isolation Status:** **PERMANENTLY LOCKED & UNTOUCHED.**
- Zero rows of the test set were loaded, inspected, evaluated, or referenced in Phase 13.
- No model was fitted, tuned, or selected using test performance.
- The integrity of the final holdout remains 100% intact.

---

## 4. Current Baseline

The official Phase 12 reference baseline remains unmodified:
- **Model:** Random Forest with `class_weight="balanced"`, `max_depth=10`, `min_samples_leaf=20`, `n_estimators=100`, `random_state=42`.
- **Target:** `direction_4` ($H=4$ M15 bars / 60 minutes, fixed return threshold $\theta = 0.00050 = 5.0\text{ pips}$).
- **Confidence Threshold:** $\tau = 0.60$.
- **Stop Loss / Take Profit:** $\text{SL} = 1.0\times\text{ATR}_{14}$, $\text{TP} = 1.5\times\text{ATR}_{14}$.
- **Holding Horizon:** $H=4$ bars (60 minutes).
- **Execution:** Next-bar executable Open price with historical bid-ask spread.
- **Validation Backtest Outcome:** Exactly **0 trades** executed across 14,983 validation bars because directional confidence never reached the $\tau = 0.60$ threshold.

The baseline is preserved as the sovereign reference point and is not altered.

---

## 5. Signal Confidence Distribution

On the 14,983 validation bars, model output probabilities across the three classes ($\text{SHORT} = -1.0, \text{NEUTRAL} = 0.0, \text{LONG} = 1.0$) were evaluated via vectorized batch inference:

### Probability Distributions by Class

| Statistic | $P(\text{LONG})$ | $P(\text{SHORT})$ | $P(\text{NEUTRAL})$ | Directional Confidence $\max(P_{\text{LONG}}, P_{\text{SHORT}})$ |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** | 0.3054 | 0.3252 | 0.3694 | **0.3484 (34.84%)** |
| **Std Dev** | 0.0702 | 0.0994 | 0.1506 | 0.0863 |
| **Min** | 0.1048 | 0.0744 | 0.0724 | 0.1048 |
| **25th Percentile** | 0.2599 | 0.2506 | 0.2422 | 0.2878 |
| **Median (50th)** | 0.3112 | 0.3315 | 0.3443 | **0.3602 (36.02%)** |
| **75th Percentile** | 0.3559 | 0.4118 | 0.4810 | 0.4182 |
| **90th Percentile** | 0.3865 | 0.4486 | 0.5898 | **0.4518 (45.18%)** |
| **95th Percentile** | 0.4084 | 0.4637 | 0.6488 | **0.4668 (46.68%)** |
| **99th Percentile** | 0.4589 | 0.4932 | 0.7420 | **0.4932 (49.32%)** |
| **Maximum** | **0.5936** | **0.5628** | **0.8191** | **0.5936 (59.36%)** |

### Directional Confidence Coverage

| Threshold ($\tau$) | Validation Rows Meeting Condition | Percentage of Validation Dataset |
| :---: | :---: | :---: |
| $\ge 0.40$ | 5,069 | 33.83% |
| $\ge 0.45$ | 1,617 | 10.79% |
| $\ge 0.50$ | 95 | 0.63% |
| $\ge 0.55$ | 7 | 0.05% |
| $\ge 0.60$ | **0** | **0.00%** |

**Empirical Finding:** The maximum directional confidence generated across the entire 7-month validation period was **$59.36\%$** ($P_{\text{LONG}} = 0.5936$ on 2025-07-31 00:00 UTC). Consequently, at the baseline threshold $\tau = 0.60$, directional entry signals are mathematically impossible. In contrast, the $\text{NEUTRAL}$ class frequently reaches high confidence (maximum $81.91\%$, with 90th percentile at $58.98\%$).

---

## 6. Directional-vs-Neutral Diagnostic

To test whether the model possesses directional discriminatory power that is being masked or absorbed by the dominant $\text{NEUTRAL}$ class, two complementary diagnostics were performed:

### A. Full 3-Class Validation Performance

| Metric | Value |
| :--- | :---: |
| **Validation Samples** | 14,983 |
| **Accuracy** | 50.31% |
| **Balanced Accuracy** | 43.61% |
| **Macro Precision** | 42.99% |
| **Macro Recall** | 43.61% |
| **Macro F1 Score** | 40.87% |

**3-Class Confusion Matrix:**
```
                Predicted SHORT (-1)   Predicted NEUTRAL (0)   Predicted LONG (+1)
True SHORT (-1)        1,592                    821                     469
True NEUTRAL (0)       2,917                  5,479                     885
True LONG (+1)         1,536                    817                     467
```

### B. Binary Directional Diagnostic (True Directional Bars Only)

Evaluated strictly on the **5,702 bars** (38.06% of the validation set) where the ground-truth label was directional ($y \in \{-1.0, 1.0\}$), comparing $P(\text{LONG})$ vs. $P(\text{SHORT})$:

| Metric | Binary Diagnostic Value | Random Benchmark |
| :--- | :---: | :---: |
| **Directional Sample Count** | 5,702 bars (38.06% of validation) | — |
| **True LONG Count** | 2,820 (49.46%) | 50.0% |
| **True SHORT Count** | 2,882 (50.54%) | 50.0% |
| **Directional Accuracy** | **50.89%** | 50.0% |
| **Balanced Accuracy** | **50.68%** | 50.0% |
| **Macro Precision** | 50.80% | 50.0% |
| **Macro Recall** | 50.68% | 50.0% |
| **Macro F1 Score** | 48.86% | 50.0% |

**Binary Directional Confusion Matrix:**
```
                Predicted SHORT (-1)   Predicted LONG (+1)
True SHORT (-1)        2,020                    862
True LONG (+1)         1,938                    882
```

### Neutral Absorption Analysis
- Of the 5,702 true directional bars, **1,638 bars (28.73%)** were assigned $\text{NEUTRAL}$ ($0.0$) by 3-class argmax.
- More critically: when predicting between LONG and SHORT on actual directional bars, the model's accuracy is **$50.89\%$**—virtually identical to an uninformative coin flip.
- The model exhibits a pronounced structural bias toward SHORT predictions ($3,958$ predicted SHORT vs $1,744$ predicted LONG on directional bars).

**Core Diagnostic Finding:** The lack of signal actionability is **not** merely caused by the $\text{NEUTRAL}$ class suppressing predictions. When the neutral bars are completely stripped away, the model's conditional directional accuracy ($50.89\%$) displays near-zero edge over random chance.

---

## 7. Confidence Bucket Analysis

To determine how prediction accuracy behaves as model confidence increases, validation observations were segmented into descriptive confidence buckets based on directional confidence $C_{\text{dir}} = \max(P_{\text{LONG}}, P_{\text{SHORT}})$:

| Confidence Bucket | Count | % of Val | Pred LONG | Pred SHORT | Pred NEUTRAL | Dir Preds Count | Directional Accuracy | True LONG % | True SHORT % | True NEUTRAL % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.40–0.45** | 3,452 | 23.04% | 549 | 2,876 | 27 | 3,425 | **26.86%** | 25.35% | 27.11% | 47.54% |
| **0.45–0.50** | 1,522 | 10.16% | 152 | 1,369 | 1 | 1,521 | **31.03%** | 31.01% | 29.76% | 39.22% |
| **0.50–0.55** | 88 | 0.59% | 41 | 47 | 0 | 88 | **34.09%** | 34.09% | 23.86% | 42.05% |
| **0.55–0.60** | 7 | 0.05% | 5 | 2 | 0 | 7 | **57.14%** | 71.43% | 0.00% | 28.57% |
| **0.60–0.65** | 0 | 0.00% | 0 | 0 | 0 | 0 | N/A | 0.00% | 0.00% | 0.00% |
| **0.65–0.70** | 0 | 0.00% | 0 | 0 | 0 | 0 | N/A | 0.00% | 0.00% | 0.00% |
| **0.70–0.75** | 0 | 0.00% | 0 | 0 | 0 | 0 | N/A | 0.00% | 0.00% | 0.00% |
| **0.75–0.80** | 0 | 0.00% | 0 | 0 | 0 | 0 | N/A | 0.00% | 0.00% | 0.00% |
| **0.80+** | 0 | 0.00% | 0 | 0 | 0 | 0 | N/A | 0.00% | 0.00% | 0.00% |

### Observations:
1. **Low Signal Accuracy in Primary Operating Range:** In the $[0.40, 0.55)$ confidence range (accounting for 4,974 bars), directional prediction accuracy ranges between $26.86\%$ and $34.09\%$. Because the true label distribution contains $40\%-47\%$ neutral bars, a directional signal is more likely to encounter market chop than a completed 5-pip target.
2. **Tiny Sample at Higher Confidence:** The $[0.55, 0.60)$ bucket contains only **7 observations** out of 14,983 bars over 7 months. While directional accuracy appears higher ($57.14\%$), the sample size ($N=7$) is far too small to establish statistical significance.
3. **Empty Tail:** All buckets $\ge 0.60$ have zero observations.

---

## 8. Probability Calibration

Calibration evaluates whether predicted probabilities correspond to true empirical frequencies:

### Multi-Class Brier Scores
*(Lower is better; naive uninformative reference $\approx 0.222$)*

| Class | Brier Score Loss | Benchmark / Interpretation |
| :--- | :---: | :--- |
| **SHORT ($-1.0$)** | **0.16761** | Marginally better than naive prior |
| **NEUTRAL ($0.0$)** | **0.27114** | Substantial penalty due to misclassifying chop |
| **LONG ($+1.0$)** | **0.16089** | Marginally better than naive prior |
| **Overall Multi-Class Mean** | **0.19988** | Modest probability accuracy across 3 classes |

### Calibration Curves (Predicted Probability vs. Observed Empirical Hit Rate)

| Probability Range | Class | Observations | Mean Predicted Probability | Observed Empirical Hit Rate | Calibration Bias |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.33–0.40** | LONG | 4,956 | 36.08% | 25.20% | Overconfident (+10.88%) |
|  | SHORT | 3,138 | 36.68% | 23.58% | Overconfident (+13.10%) |
| **0.40–0.45** | LONG | 755 | 41.92% | 27.81% | Overconfident (+14.11%) |
|  | SHORT | 2,992 | 42.53% | 27.07% | Overconfident (+15.46%) |
| **0.45–0.50** | LONG | 155 | 46.66% | 41.29% | Overconfident (+5.37%) |
|  | SHORT | 1,370 | 46.67% | 29.85% | Overconfident (+16.82%) |
| **0.50–0.55** | LONG | 41 | 51.57% | 39.02% | Overconfident (+12.55%) |
|  | SHORT | 47 | 51.45% | 29.79% | Overconfident (+21.66%) |
| **0.55–0.60** | LONG | 5 | 57.48% | 80.00% | Small sample ($N=5$) |
|  | SHORT | 2 | 55.85% | 0.00% | Small sample ($N=2$) |

**Calibration Finding:** The Random Forest model is **systematically overconfident** across both directional classes. When the model assigns a $45\%-55\%$ probability to SHORT, the actual market achieves the 5-pip downward target only $29.8\%$ of the time. For LONG, predicted probabilities of $51.6\%$ materialize as positive returns only $39.0\%$ of the time. The raw probabilities cannot be treated as true Bayesian likelihoods without post-processing or recalibration.

---

## 9. Signal Frequency Analysis

Why does the baseline confidence threshold of $\tau = 0.60$ produce zero trades?

| Threshold | Number of Directional Predictions Meeting Threshold |
| :---: | :---: |
| **$\ge 0.50$** | 95 |
| **$\ge 0.55$** | 7 |
| **$\ge 0.60$** | **0** |
| **$\ge 0.65$** | **0** |
| **$\ge 0.70$** | **0** |

### Evaluation of Potential Explanations:

- **A. Model Confidence is Intrinsically Low:** **SUPPORTED.** Random Forest ensembles average class probability predictions across 100 decision trees. With `min_samples_leaf=20` and noisy intraday financial data, tree leaf distributions are rarely unanimous. Ensemble bagging naturally shrinks predicted probabilities toward the training prior ($\approx 33\%$), establishing an empirical ceiling for directional classes around $55\%-59\%$.
- **B. Model Mostly Predicts NEUTRAL:** **SUPPORTED.** Out of 14,983 bars, the argmax prediction is $\text{NEUTRAL}$ ($0.0$) on **7,117 bars (47.50%)**. Furthermore, NEUTRAL confidence reaches up to **$81.91\%$**, while directional confidence never reaches $60\%$.
- **C. Directional Classes are Difficult to Separate:** **STRONGLY SUPPORTED.** As demonstrated in Section 6, when restricting evaluation strictly to bars that move $>5$ pips in either direction, binary classification accuracy is only **$50.89\%$**. The feature set has difficulty distinguishing between upcoming upward and downward moves.
- **D. Probability Calibration is Poor:** **SUPPORTED.** Observed hit rates ($27\%-39\%$) lag predicted confidence ($40\%-55\%$) by 10 to 20 percentage points.
- **E. Combination of Above Factors:** **PRIMARY CONCLUSION.** The zero-trade outcome is not caused by a single isolated flaw. It is the logical consequence of **ensemble probability shrinkage (A)** operating on **weak underlying directional separability (C)** within a **dominant neutral market regime (B)**, evaluated against a threshold ($\tau = 0.60$) that was chosen without accounting for the multi-class prior ($1/3$).

---

## 10. Existing Feature Diagnostic

The feature pipeline produces 80 strictly point-in-time technical and temporal indicators. Analyzing Gini impurity reduction from the fitted Random Forest reveals how features contribute to predictions:

### Top 20 Features by Gini Importance

| Rank | Feature Name | Importance | Functional Category | Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `hl_range` | **7.89%** | Volatility | Absolute high-low range of bar $t$ |
| **2** | `log_tick_volume` | **6.85%** | Activity | Log-transformed tick volume |
| **3** | `hl_range_norm` | **6.57%** | Volatility | High-low range normalized by close price |
| **4** | `cos_hour` | **5.07%** | Time-of-Day | Cyclical diurnal time feature |
| **5** | `atr_14` | **3.65%** | Volatility | 14-period Average True Range |
| **6** | `tick_vol_zscore_80` | **3.32%** | Activity | 80-bar standardized volume z-score |
| **7** | `hour` | **3.00%** | Time-of-Day | Raw integer hour of day (0–23) |
| **8** | `vol_std_10` | **2.94%** | Volatility | 10-bar return standard deviation |
| **9** | `atr_norm_14` | **2.94%** | Volatility | ATR normalized by price |
| **10** | `vol_ann_20` | **2.46%** | Volatility | 20-bar annualized return volatility |
| **11** | `tick_vol_sma_20` | **2.14%** | Activity | 20-bar simple moving average of volume |
| **12** | `rolling_hl_ratio_20` | **2.11%** | Volatility | Ratio of current bar range to 20-bar range |
| **13** | `is_london_session` | **2.05%** | Time-of-Day | Binary flag for London trading session |
| **14** | `vol_std_20` | **2.02%** | Volatility | 20-bar return standard deviation |
| **15** | `vol_ann_80` | **1.78%** | Volatility | 80-bar annualized volatility |
| **16** | `atr_norm_40` | **1.62%** | Volatility | 40-bar normalized ATR |
| **17** | `vol_std_80` | **1.62%** | Volatility | 80-bar return standard deviation |
| **18** | `vol_std_40` | **1.40%** | Volatility | 40-bar return standard deviation |
| **19** | `tick_vol_zscore_20` | **1.37%** | Activity | 20-bar volume z-score |
| **20** | `tick_vol_ratio_20` | **1.33%** | Activity | Short-to-long volume ratio |

### Category Breakdown (Top 30 Features)
- **Volatility:** 10 features (**33.3%**)
- **Activity & Volume:** 8 features (**26.7%**)
- **Trend & Momentum:** 7 features (**23.3%**)
- **Time-of-Day:** 5 features (**16.7%**)

### Feature Shift During Directional Predictions
Comparing feature values on the 95 bars where directional confidence reached $P_{\text{dir}} \ge 0.50$ against the full validation population:

| Feature Name | Category | Gini Imp | All Validation Mean | High-Confidence Mean | Normalized Shift ($z$-score) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `hl_range` | Volatility | 0.0789 | 0.00063 | 0.00155 | **$+2.07\sigma$** |
| `hl_range_norm` | Volatility | 0.0657 | 0.00054 | 0.00133 | **$+2.06\sigma$** |
| `vol_std_10` | Volatility | 0.0294 | 0.00034 | 0.00079 | **$+2.07\sigma$** |
| `rolling_hl_ratio_20` | Volatility | 0.0211 | 0.00253 | 0.00548 | **$+1.91\sigma$** |
| `vol_ann_20` | Volatility | 0.0246 | 0.05442 | 0.10921 | **$+1.77\sigma$** |
| `atr_14` | Volatility | 0.0365 | 0.00063 | 0.00112 | **$+1.74\sigma$** |
| `tick_vol_zscore_80` | Activity | 0.0332 | 0.03908 | 1.09271 | **$+0.91\sigma$** |
| `hour` | Time-of-Day | 0.0300 | 11.49 | 10.14 | $-0.20\sigma$ |

**Feature Finding:** Model confidence is predominantly driven by **volatility expansion**. When directional confidence spikes above $0.50$, volatility features shift by $+1.7\sigma$ to $+2.1\sigma$ above normal. The model is effectively detecting that **the market is active and volatile**, rather than identifying clean directional drift.

---

## 11. Temporal Stability

The 14,983 validation observations were divided chronologically into three equal contiguous temporal blocks (~4,994 bars / ~2.4 months each):

| Metric | Block 1 (Early) | Block 2 (Mid) | Block 3 (Late) |
| :--- | :---: | :---: | :---: |
| **Start Time (UTC)** | 2025-07-14 05:45 | 2025-09-24 06:30 | 2025-12-05 09:45 |
| **End Time (UTC)** | 2025-09-24 06:15 | 2025-12-05 09:30 | 2026-02-19 10:45 |
| **Row Count** | 4,994 | 4,994 | 4,995 |
| **True LONG %** | 21.07% | 17.24% | 18.16% |
| **True SHORT %** | 21.95% | 18.30% | 17.46% |
| **True NEUTRAL %** | 56.99% | 64.46% | 64.38% |
| **Mean Directional Confidence** | **0.3653** | **0.3375** | **0.3423** |
| **Max Directional Confidence** | **0.5936** | **0.5659** | **0.5838** |
| **Directional Predictions (Argmax)** | 2,852 (57.11%) | 2,417 (48.40%) | 2,597 (51.99%) |
| **Directional Signals with Conf $\ge 0.50$** | **68** | **10** | **17** |
| **Balanced Accuracy** | 42.54% | 45.25% | 43.05% |
| **Macro F1 Score** | 41.05% | 42.16% | 38.79% |

### Temporal Findings:
1. **Regime Consistency:** General classification metrics remain remarkably stable across all three blocks (Balanced Accuracy: $42.5\%-45.3\%$, Macro F1: $38.8\%-42.2\%$).
2. **Signal Frequency Clustering:** Although general accuracy was stable, signals meeting the $\tau \ge 0.50$ threshold clustered heavily in **Block 1** (68 signals) compared to Block 2 (10) and Block 3 (17). This directly mirrors the seasonal volatility surge in late summer 2025, reinforcing the finding that model confidence tracks volatility spikes rather than steady alpha.

---

## 12. Controlled Research Model Comparison

To evaluate whether the confidence ceiling and directional separability are specific to Random Forest, a controlled comparison was executed using `LogisticRegressionBaseline` with `class_weight="balanced"` and `StandardScaler` fitted strictly on the Training partition:

| Metric | Random Forest (Balanced) | Logistic Regression (Balanced) | Comparison / Interpretation |
| :--- | :---: | :---: | :--- |
| **Training Fit Time** | 7.16s | 17.31s | Both execute rapidly |
| **3-Class Balanced Accuracy** | **43.61%** | **42.74%** | RF slightly better on multi-class |
| **3-Class Macro F1** | **40.87%** | **42.08%** | LR slightly better balanced across classes |
| **Binary Directional Accuracy** | **50.89%** | **51.35%** | **Both models perform near random (50%)** |
| **Mean Directional Confidence** | 0.3484 | 0.3089 | RF mean is slightly higher |
| **Max Directional Confidence** | **0.5936** | **0.8227** | **LR reaches high confidence ($>80\%$)** |
| **Signals with Conf $\ge 0.50$** | 95 | 219 | LR generates $2.3\times$ more signals |
| **Signals with Conf $\ge 0.60$** | **0** | **34** | LR generates actionable signals at 0.60 |
| **Signals with Conf $\ge 0.70$** | **0** | **3** | LR has observations in high tails |

### Comparative Insight:
- **Probability Compression is Model-Specific:** Logistic Regression readily generates directional confidence exceeding $0.60$ (34 bars) and $0.70$ (3 bars), proving that the $0.60$ ceiling in Random Forest is an artifact of ensemble averaging (bagging probability shrinkage).
- **Directional Edge Remains Weak Across Architectures:** Despite generating higher confidence, Logistic Regression's binary directional accuracy on actual directional moves is **$51.35\%$** (compared to Random Forest's $50.89\%$). Switching model families expands confidence spread, but does **not** inherently create directional predictive edge.

---

## 13. Findings

1. **Root Cause of Baseline Zero-Trade Result:** At $\tau = 0.60$, zero trades are generated because Random Forest bagging averages 100 leaf nodes across a 3-class distribution with a $33\%$ prior, capping maximum directional confidence at $59.36\%$.
2. **Directional Separation is the Fundamental Bottleneck:** On bars where the market undergoes a decisive $\ge 5$-pip move, the model predicts the correct direction only $50.89\%$ of the time. The primary performance hurdle is not the presence of the neutral class, but the low signal-to-noise ratio in directional forecasting.
3. **Confidence Measures Volatility, Not Edge:** Features with highest importance are short-term volatility indicators (`hl_range`, `atr_14`, `vol_std_10`). High-confidence predictions occur almost exclusively when historical range expands by $+2\sigma$.
4. **Probability Overconfidence:** Predicted directional probabilities between $0.40$ and $0.55$ systematically overestimate empirical success rates by 10 to 20 percentage points.
5. **Linear Model Contrast:** Logistic regression confirms that probability calibration varies by algorithm (LR reaches $82\%$ confidence), but the underlying directional classification edge remains near $51\%$ for both.

---

## 14. Limitations

1. **Fixed Threshold Target:** The target label (`direction_4`) uses a fixed 5.0-pip return barrier ($\theta = 0.00050$). In low-volatility regimes (e.g. Asian session), 5 pips represents a large multi-standard-deviation move; in high-volatility regimes (e.g. US CPI releases), 5 pips represents normal intra-candle noise.
2. **Fixed 4-Bar Horizon:** The 60-minute prediction window ($H=4$) forces all position liquidations at bar $t+4$, which may prematurely truncate trending moves or prematurely realize adverse noise.
3. **Feature Information Limits:** Standard technical oscillators derived solely from single-pair M15 OHLCV data provide limited forward-looking information regarding institutional liquidity flows or macroeconomic drivers.

---

## 15. What Should NOT Be Concluded

- **DO NOT conclude the strategy is profitable.** None of the investigated configurations demonstrated a statistically robust positive edge.
- **DO NOT conclude the strategy is definitively unprofitable.** This was a diagnostic study of existing signals, not an exhaustive bound on ML capabilities.
- **DO NOT conclude that intraday trading is impossible.** Weakness in a 4-bar M15 fixed-threshold model on a single currency pair does not invalidate intraday systematic trading broadly.
- **DO NOT conclude that longer timeframes or daily bars are required.** Timeframe selection is an empirical design choice, not an all-or-nothing dichotomy.
- **DO NOT adopt a new production threshold or modify Phase 12 baseline.** Lowering $\tau$ to $0.50$ in production without addressing the $50.89\%$ directional accuracy and transaction frictions would merely trade random noise.

---

## 16. Candidate Directions for a FUTURE Phase

If a subsequent research phase is authorized, the following hypotheses warrant investigation:

1. **Volatility-Normalized Target Thresholds:** Instead of a static 5-pip threshold ($\theta = 0.00050$), define directional classes relative to prevailing volatility:
   $$\theta_t = k \times \text{ATR}_{14}(t)$$
   This ensures that target difficulty scales dynamically with market conditions.
2. **Decoupled Two-Stage Architecture:**
   - *Stage 1 (Regime / Expansion Filter):* Binary model predicting whether the market will expand beyond noise levels ($\text{Chop}$ vs. $\text{Breakout}$).
   - *Stage 2 (Directional Classifier):* Binary model ($\text{LONG}$ vs. $\text{SHORT}$) evaluated only when Stage 1 predicts an expansion regime.
3. **Continuous Return Regression or Ranking:** Replace 3-class categorization with continuous expected return forecasting, allowing dynamic risk-adjusted trade selection.
4. **Order Flow & Multi-Asset Feature Enrichment:** Incorporate cross-currency proxies (e.g. Dollar Index / DXY, EURGBP, US 10-year Treasury yield momentum) to provide external directional context beyond single-pair OHLC.
5. **Probability Recalibration:** Apply post-hoc isotonic regression or Platt scaling on out-of-fold validation splits prior to confidence thresholding.

---

**PHASE 13 RESEARCH COMPLETE — PHASE 14 NOT STARTED**
