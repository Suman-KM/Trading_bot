# Phase 10 — Controlled Model Improvement Study (EURUSD M15)

## 1. Executive Summary & Context

This document details the design, methodology, empirical results, and diagnostic conclusions of **Phase 10: Controlled Model Improvement Study** for the automated AI trading system on **EURUSD M15** data.

### System Configuration
- **Instrument**: EURUSD
- **Timeframe**: M15 (15-minute bars)
- **Data Source**: MetaQuotes Ltd. / MetaQuotes-Demo (100,000 real bars, 2022-09-19 to 2026-02-19 UTC)
- **Prediction Horizon**: $H = 4$ bars (60 minutes forward)
- **Target Variable**: `direction_4` (Fixed-threshold ternary return classification at $\tau = 12.0$ pips / 0.00120)
  - `SHORT (-1.0)`: Forward return $\le -0.00120$
  - `NEUTRAL (0.0)`: Forward return $\in (-0.00120, +0.00120)$
  - `LONG (+1.0)`: Forward return $\ge +0.00120$
- **Feature Set**: 80 derived point-in-time features from Phase 5 (volatility, momentum, price action, volume, time encoding)
- **Partitioning**: Chronological split with 4-bar purge buffer (Train: 69,937 rows / 70%, Validation: 14,983 rows / 15%, Test: 14,988 rows / 15%)
- **Test Set Protection**: **STRICTLY ENFORCED**. The 14,988-row Test partition was **NEVER** loaded, inspected, predicted on, or evaluated. Validation is the sole out-of-sample evaluation partition.

### Motivation
In Phase 8 and Phase 9, default unweighted machine learning baselines (Logistic Regression and Random Forest) exhibited **majority-class collapse**. Because the underlying market is in a low-displacement state ~62% of the time, unweighted models minimized standard loss functions (cross-entropy and Gini impurity) by predicting the majority class (`NEUTRAL`) >96% of the time. While achieving ~62% raw accuracy, their directional recall was near zero (SHORT recall 5–9%, LONG recall 3–4%), with balanced accuracy barely exceeding the random baseline of 33.33%.

Phase 10 systematically investigated whether:
1. **Class Weighting** (`class_weight="balanced"`) forces models to attend equally to directional price displacements.
2. **Alternative Tree Ensembles** (`ExtraTreesClassifier`) improve generalization by introducing extreme threshold randomization.
3. **Decision / Confidence Filtering** isolates high-precision directional regimes.

---

## 2. Methodology & Experimental Matrix

All candidate models were trained strictly on the 69,937 training observations and evaluated out-of-sample on the 14,983 validation observations.

```
+---------------------------------------------------------------------------------------------+
|                                  PHASE 10 MODEL MATRIX                                      |
+----+----------------------------------+---------------------+-------------------------------+
| ID | Model Architecture               | Class Weighting     | Key Hyperparameters           |
+----+----------------------------------+---------------------+-------------------------------+
| 1  | Majority Class Reference         | None                | Predicts constant 0.0         |
| 2  | Logistic Regression (Baseline)   | None (unweighted)   | C=1.0, L2, lbfgs, Scaler      |
| 3  | Logistic Regression (Balanced)   | Inverse Frequency   | C=1.0, L2, lbfgs, Scaler      |
| 4  | Random Forest (Baseline)         | None (unweighted)   | 100 trees, depth 10, leaf 20  |
| 5  | Random Forest (Balanced)         | Inverse Frequency   | 100 trees, depth 10, leaf 20  |
| 6  | Extra Trees (Candidate)          | None (unweighted)   | 100 trees, depth 10, leaf 20  |
| 7  | Extra Trees (Balanced)           | Inverse Frequency   | 100 trees, depth 10, leaf 20  |
+----+----------------------------------+---------------------+-------------------------------+
```

### Data Preprocessing & Leakage Controls
- **Scaling**: For Logistic Regression models, `StandardScaler` was fitted strictly on `splits.train.X`. Validation features were transformed using the frozen training parameters (`mean_`, `scale_`).
- **Class Weights**: Inverse class frequencies were computed exclusively from training target labels:
  $$w_c = \frac{N_{\text{train}}}{K \cdot N_{c, \text{train}}}$$
  where $K=3$ classes. Training weights: `SHORT: 1.761`, `NEUTRAL: 0.538`, `LONG: 1.762`.
- **Ensemble Hyperparameters**: Bounded tree depth (`max_depth=10`, `min_samples_leaf=20`) to prevent overfitting on financial noise.

---

## 3. Overall Performance Comparison (Validation Set)

The table below summarizes the out-of-sample validation metrics across all 7 evaluated models (from `reports/model_improvement_comparison.csv`):

| Model Architecture | Class Weight | Raw Accuracy | Balanced Accuracy | Macro F1 | SHORT Recall | LONG Recall | SHORT F1 | LONG F1 | Macro ROC-AUC | Macro PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | None | 61.94% | 33.33% | 0.2550 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.5000 | 0.3333 |
| **Logistic Regression (unweighted)** | None | 62.01% | 35.32% | 0.3063 | 5.00% | 3.48% | 0.0871 | 0.0627 | 0.6675 | 0.4469 |
| **Logistic Regression (balanced)** | Balanced | **53.95%** | **42.74%** | **0.4208** | **39.04%** | **20.43%** | **0.3218** | **0.2303** | 0.6642 | 0.4432 |
| **Random Forest (unweighted)** | None | 62.23% | 36.49% | 0.3289 | 9.06% | 3.97% | 0.1429 | 0.0714 | 0.6695 | 0.4510 |
| **Random Forest (balanced)** | Balanced | **50.31%** | **43.61%** | **0.4087** | **55.24%** | **16.56%** | **0.3567** | **0.2012** | 0.6626 | 0.4436 |
| **Extra Trees (unweighted)** | None | 62.20% | 34.40% | 0.2814 | 2.74% | 1.28% | 0.0510 | 0.0247 | 0.6714 | 0.4529 |
| **Extra Trees (balanced)** | Balanced | **49.48%** | **43.14%** | **0.4039** | **54.89%** | **16.81%** | **0.3545** | **0.1968** | 0.6636 | 0.4460 |

---

## 4. In-Depth Analysis Answering the 6 Core Diagnostic Questions

### Question 1: Does class weighting improve minority directional recall?
**Answer: YES, decisively.**
Across all three model families, applying inverse frequency class weighting produced massive, statistically robust improvements in directional recall:
- **SHORT Recall**:
  - Logistic Regression: jumps from **5.00%** to **39.04%** (a **7.8x increase**).
  - Random Forest: jumps from **9.06%** to **55.24%** (a **6.1x increase**).
  - Extra Trees: jumps from **2.74%** to **54.89%** (a **20.0x increase**).
- **LONG Recall**:
  - Logistic Regression: jumps from **3.48%** to **20.43%** (a **5.9x increase**).
  - Random Forest: jumps from **3.97%** to **16.56%** (a **4.2x increase**).
  - Extra Trees: jumps from **1.28%** to **16.81%** (a **13.1x increase**).

**Mechanism**: In unweighted models, predicting the majority class (`NEUTRAL`) minimises empirical loss because false negatives on the minority classes incur equal penalty to false positives on the majority class. Class weighting multiplies minority loss gradients by ~1.76 while downweighting neutral gradients by 0.54, forcing the decision boundaries outward into directional space.

---

### Question 2: Does class weighting improve Macro F1 and Balanced Accuracy, or only shift confusion?
**Answer: Class weighting improves true multi-class discrimination (Macro F1 and Balanced Accuracy), but does not overcome intrinsic SNR limits.**
- **Balanced Accuracy**:
  - Logistic Regression: **+7.42%** (35.32% $\rightarrow$ 42.74%)
  - Random Forest: **+7.12%** (36.49% $\rightarrow$ 43.61%)
  - Extra Trees: **+8.74%** (34.40% $\rightarrow$ 43.14%)
- **Macro F1**:
  - Logistic Regression: **+0.1145** (0.3063 $\rightarrow$ 0.4208)
  - Random Forest: **+0.0798** (0.3289 $\rightarrow$ 0.4087)
  - Extra Trees: **+0.1225** (0.2814 $\rightarrow$ 0.4039)
- **Trade-off Analysis**:
  Because the models now actively predict directional moves, raw accuracy decreases from ~62.2% to ~50.3% (Random Forest). Directional precision is ~26.3% for SHORT and ~25.6% for LONG (compared to base class prevalence of ~19.0%). However, because recall improved by 4x to 20x while precision experienced only modest compression, the harmonic mean (F1 score) for both directional classes improved substantially (e.g. SHORT F1 for RF grew from 0.1429 to 0.3567). 
  
  Importantly, **Macro ROC-AUC remains essentially invariant** (~0.663 vs ~0.669). This demonstrates that class weighting does not extract new underlying ranking information from the feature space; rather, it shifts the operational operating point along the ROC curve from a degenerate majority corner to a balanced multi-class regime.

---

### Question 3: Does Extra Trees outperform Random Forest or Logistic Regression?
**Answer: NO.**
- **Extra Trees vs Random Forest**:
  - Extra Trees (balanced) achieved **43.14%** balanced accuracy and **0.4039** Macro F1.
  - Random Forest (balanced) achieved **43.61%** balanced accuracy and **0.4087** Macro F1.
  - Random Forest retains a slight performance advantage over Extra Trees in both balanced accuracy (+0.47%) and Macro F1 (+0.0048).
- **Extra Trees vs Logistic Regression**:
  - Logistic Regression (balanced) achieved the highest Macro F1 overall (**0.4208**) and a higher LONG recall (**20.43%** vs **16.81%**), despite having a slightly lower balanced accuracy (**42.74%** vs **43.14%**).
- **Conclusion**: Extreme randomization in candidate split thresholds does not provide measurable benefit over standard Random Forest for EURUSD M15 features. Random Forest and Logistic Regression remain the preferred baseline architectures.

---

### Question 4: Does high-confidence prediction isolation solve the low precision issue?
**Answer: NO. High-confidence thresholding causes directional predictions to vanish and collapses back into majority-class prediction.**

The empirical breakdown of `reports/model_improvement_confidence.csv` reveals a critical insight into probability behavior:

```
+---------------------------------------------------------------------------------------------------------+
|                  CONFIDENCE FILTERING BREAKDOWN: RANDOM FOREST (BALANCED)                               |
+--------+---------------+-----------+----------+----------+----------+----------+----------+-------------+
| Thresh | Covered Count | Coverage% | Accuracy | Bal. Acc | Macro F1 | Pred S   | Pred N   | Pred L      |
+--------+---------------+-----------+----------+----------+----------+----------+----------+-------------+
|  0.35  |    14,648     |   97.76%  |  50.74%  |  43.88%  |  0.4108  |  5,926   |  7,012   |    1,710    |
|  0.40  |    10,847     |   72.40%  |  55.82%  |  46.26%  |  0.4270  |  4,294   |  5,806   |      747    |
|  0.45  |     6,069     |   40.51%  |  68.99%  |  47.61%  |  0.4546  |  1,418   |  4,453   |      198    |
|  0.50  |     3,375     |   22.53%  |  83.08%  |  36.90%  |  0.3723  |     49   |  3,280   |       46    |
|  0.55  |     2,192     |   14.63%  |  86.86%  |  34.37%  |  0.3306  |      2   |  2,185   |        5    |
|  0.60  |     1,340     |    8.94%  |  88.43%  |  33.33%  |  0.3129  |      0   |  1,340   |        0    |
|  0.70  |       300     |    2.00%  |  95.33%  |  33.33%  |  0.3254  |      0   |    300   |        0    |
|  0.80  |         4     |    0.03%  | 100.00%  |  33.33%  |  1.0000  |      0   |      4   |        0    |
+--------+---------------+-----------+----------+----------+----------+----------+----------+-------------+
```

#### Diagnostic Takeaway:
- When the threshold is increased to $\tau \ge 0.50$, the subset raw accuracy climbs from 69% to >88%, but **balanced accuracy collapses to 33.33%**.
- At $\tau = 0.50$, only **49 SHORT** and **46 LONG** predictions remain out of 3,375 covered bars; **3,280 (97.2%)** are NEUTRAL.
- At $\tau \ge 0.60$, **100% of covered predictions are NEUTRAL**.
- **Explanation**: High model confidence in financial time series reflects **market consolidation and low volatility**, not high-probability directional trends. The model is confident only when volatility is crushed and the market is quiescent. Therefore, filtering by high confidence does NOT isolate directional alpha; it isolates trivial non-trading regimes.
- The highest balanced accuracy for directional models occurs in the **moderate confidence range** ($\tau \in [0.40, 0.45]$), retaining 40–72% coverage with balanced accuracy ~46–48%.

---

### Question 5: Are the improvements consistent across validation time blocks A, B, C?
**Answer: YES.**
Evaluating models across three contiguous chronological thirds of the validation partition (each ~5,000 bars) confirms strong temporal stationarity:

```
+-------------------------------------------------------------------------------------------------+
|                       TEMPORAL STABILITY ACROSS VALIDATION BLOCKS A, B, C                       |
+----------------------------------+---------------+-------------+----------+----------+----------+
| Model                            | Block         | Date Range  | Bal. Acc | Macro F1 | ROC-AUC  |
+----------------------------------+---------------+-------------+----------+----------+----------+
| Logistic Regression (balanced)   | VAL_A (Early) | Jul-Sep '25 |  41.16%  |  0.4101  |  0.6555  |
| Logistic Regression (balanced)   | VAL_B (Mid)   | Sep-Dec '25 |  43.77%  |  0.4216  |  0.6644  |
| Logistic Regression (balanced)   | VAL_C (Late)  | Dec-Feb '26 |  43.03%  |  0.4195  |  0.6638  |
+----------------------------------+---------------+-------------+----------+----------+----------+
| Random Forest (balanced)         | VAL_A (Early) | Jul-Sep '25 |  42.54%  |  0.4105  |  0.6481  |
| Random Forest (balanced)         | VAL_B (Mid)   | Sep-Dec '25 |  45.25%  |  0.4216  |  0.6696  |
| Random Forest (balanced)         | VAL_C (Late)  | Dec-Feb '26 |  43.05%  |  0.3879  |  0.6667  |
+----------------------------------+---------------+-------------+----------+----------+----------+
| Extra Trees (balanced)           | VAL_A (Early) | Jul-Sep '25 |  42.25%  |  0.4062  |  0.6560  |
| Extra Trees (balanced)           | VAL_B (Mid)   | Sep-Dec '25 |  44.91%  |  0.4160  |  0.6681  |
| Extra Trees (balanced)           | VAL_C (Late)  | Dec-Feb '26 |  42.27%  |  0.3863  |  0.6610  |
+----------------------------------+---------------+-------------+----------+----------+----------+
```

Across all three balanced architectures:
- Balanced accuracy remains steadily within the **41.2% to 45.3%** range across all 7 months of validation.
- Macro F1 remains steadily within the **0.386 to 0.422** range.
- ROC-AUC remains steadily within **0.648 to 0.670**.
- There is no catastrophic regime breakdown or severe degradation between 2025 and 2026. The improvement gained via class weighting is temporally robust.

---

### Question 6: Overall conclusion: Do any of these improvements justify live trading?
**Answer: ABSOLUTELY NOT.**

#### Quantitative Justification:
1. **Low Directional Precision**:
   Even with class weighting and moderate thresholding, directional precision remains between **25.6% and 27.4%**. Approximately 3 out of every 4 directional predictions are false positives (the market remained neutral or moved in the opposite direction).
2. **Transaction Friction Deficit**:
   On EURUSD M15, entering and exiting positions incurs bid-ask spread (typically 0.2–0.5 pips), broker commission ($3–$7 per round lot), and latency slippage. With a 12-pip take-profit target, a precision of 26% yields severe negative mathematical expectancy:
   $$\mathbb{E}[\text{Return}] < 0.26 \times (+12.0) - 0.74 \times (\text{friction} + \text{loss}) < 0$$
3. **Execution Safety Boundaries**:
   These machine learning baselines are diagnostic statistical tools designed to measure feature signal content and class dynamics. They are **not** an end-to-end trading strategy.
4. **Safety System Unchanged**:
   - `RiskEngine` remains authoritative and unchanged.
   - `PaperBroker` remains the sole execution path.
   - Live trading remains strictly disabled.
   - MT5 execution connects exclusively to demo/paper accounts if enabled in separate phases.

---

## 5. Artifacts Generated in Phase 10

The following artifacts have been generated and validated:

1. **`reports/model_improvement_comparison.csv`**:
   Full tabular performance comparison across all 7 candidate architectures.
2. **`reports/model_improvement_temporal.csv`**:
   Temporal block stability metrics across Validation Blocks A, B, and C.
3. **`reports/model_improvement_confidence.csv`**:
   Confidence threshold grid ($\tau \in [0.35, 0.80]$) for balanced and unweighted models.
4. **`reports/model_experiments_metadata.json`**:
   Reproducible study metadata including dataset shapes, class frequencies, timestamp bounds, random seed (42), and explicit verification that `test_set_used = false`.
5. **`scripts/run_model_experiments.py`**:
   Automated, reproducible CLI entry point to run the Phase 10 study.
6. **`tests/test_model_experiments.py`**:
   Suite of 8 automated tests ensuring zero lookahead, test set protection, scaler training isolation, and reproducibility.

---

## 6. Phase 10 Quality Gates & Verification

- **Automated Tests**: 8 dedicated Phase 10 tests passed; 200 total repository tests passing.
- **Linter & Formatter**: Ruff check and format clean across all files.
- **Test Set Isolation**: Verified `test_set_used == False`; test partition length (14,988 rows) remains intact and completely unread.
- **Git State**: All changes committed under `develop`.

---

**PHASE 10 COMPLETE — PHASE 11 NOT STARTED**
