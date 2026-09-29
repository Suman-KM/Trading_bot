# Phase 14 — Intraday Target & Signal Redesign Research Report

**Status:** COMPLETE  
**Date:** 2026-09-29  
**Branch:** `develop`  
**Execution Environment:** Ubuntu (Python 3.13 via `uv`)  
**Data Instrument:** EURUSD M15  
**Data Range Analyzed:** 2022-09-19 05:30:00 UTC to 2026-02-19 10:45:00 UTC (Training + Validation Only)  
**Phase 11 Holdout Test Set:** PERMANENTLY LOCKED & UNTOUCHED (14,988 rows: 2026-02-19 12:00:00 UTC onward)  
**Phase 12 Baseline Configuration:** PRESERVED UNCHANGED  
**Production Changes:** NONE  
**Trading Backtests / Parameter Sweeps:** NONE (Research-Only Phase)  

---

## 1. Objective

The objective of Phase 14 is to investigate whether the weak directional separation observed in Phase 13 (balanced accuracy of 50.68% and binary accuracy of 50.89% on EURUSD M15) can be meaningfully improved by redesigning the prediction target and signal formulation, while preserving strict chronological validation, causal feature constraints, and holdout isolation.

Specifically, this research examines whether:
1. Eliminating the large ternary `NEUTRAL` class (which represented ~61.9% of all validation observations) and training directly on binary directional displacement improves class separability.
2. Volatility-scaled dynamic thresholds ($\pm 0.5 \times \sqrt{H} \times \text{ATR14\_norm}$) produce higher quality directional labels than fixed pip thresholds ($\pm 5.0\text{ pips}$).
3. A simple, unthresholded binary directional target (price up vs price down over $H=4$ bars) provides cleaner signal learning for baseline models.
4. Filtering exclusively for extreme tail moves ($\pm 1.0 \times \sqrt{H} \times \text{ATR14\_norm} = \pm 2.0 \times \text{ATR14\_norm}$, representing $\approx 12.5\%$ of observations) extracts meaningful directional signal by removing intra-horizon noise.

---

## 2. Why Phase 14 Was Required

In Phase 12, backtesting the baseline configuration (Random Forest balanced on ternary `direction_4`, confidence threshold $\tau = 0.60$, SL $1.0\times$ ATR, TP $1.5\times$ ATR, 4-bar max hold) produced **zero executed trades** over the 7-month validation window (14,983 bars). 

In Phase 13, diagnostic investigation revealed that:
1. **Confidence Compression:** The maximum directional confidence produced by the baseline Random Forest on validation data was $59.36\%$, strictly below the $\tau = 0.60$ entry threshold. Tree bagging probability shrinkage naturally bounds maximum class probabilities.
2. **Neutral Class Masking:** At default classification thresholds, the ternary model predicted `NEUTRAL (0.0)` for 94.62% of validation samples, suppressing 28.73% of true directional moves.
3. **Coin-Flip Directional Separation:** When isolated strictly to non-neutral moves ($N=5,702$), the model's binary LONG-vs-SHORT accuracy was **$50.89\%$** (balanced accuracy $50.68\%$, macro F1 $48.86\%$).
4. **Volatility Confounding:** High directional confidence was strongly correlated with volatility expansion ($z$-scores of $+1.7\sigma$ to $+2.1\sigma$ on `hl_range`, `atr_14`, and `vol_std_10`) rather than true directional drift.

Phase 14 was commissioned to address the root modeling question: **"Is the current ternary target itself poorly aligned with the type of intraday prediction required, and can alternative, economically defensible target formulations produce statistically meaningful directional separation?"**

---

## 3. Phase 13 Findings Summary

The baseline findings established in Phase 13 serve as the empirical starting point for Phase 14:

| Metric | Phase 13 Baseline Diagnostic (RF on `direction_4`) | Interpretation |
|---|---|---|
| **Validation Rows** | 14,983 M15 bars | Full chronological validation window |
| **Max Directional Confidence** | 59.36% | Never triggers $\tau = 0.60$ baseline entry threshold |
| **Directional Subset Rows** | 5,702 bars (38.06%) | Ground-truth moves with $\| \text{ret}_4 \| > 0.00050$ |
| **Neutral Subset Rows** | 9,281 bars (61.94%) | Moves with $\| \text{ret}_4 \| \le 0.00050$ |
| **Binary Directional Accuracy** | 50.89% | Indistinguishable from random chance (50.0%) |
| **Binary Balanced Accuracy** | 50.68% | Symmetric coin-flip between LONG and SHORT |
| **Binary Macro F1** | 48.86% | Suppressed by asymmetric directional precision |
| **Primary Confidence Driver** | Volatility expansion ($z > +1.7$) | Model senses volatility, not directional sign |

Phase 14 tests four candidate target representations to see whether this near-random baseline can be improved.

---

## 4. Data Used

The research strictly utilized the established, reproducible chronological partitions for EURUSD M15 assembled from MetaQuotes demo tick history:

- **Instrument:** EURUSD
- **Timeframe:** M15 (15-minute bars)
- **Timezone:** UTC
- **Feature Set:** Exactly 80 point-in-time, causal technical and market structure features (price action, candle geometry, momentum, volatility, trend, activity, time-of-day, gap/session).
- **Partitions:**
  - **Training Partition:** 69,937 M15 bars  
    `2022-09-19 05:30:00 UTC` to `2025-07-14 04:30:00 UTC` (approx. 2.8 years)
  - **Validation Partition:** 14,983 M15 bars  
    `2025-07-14 05:45:00 UTC` to `2026-02-19 10:45:00 UTC` (approx. 7.2 months)
  - **Purge Window:** 4 bars (60 minutes) at boundaries to eliminate label overlap leakage.

---

## 5. Phase 11 Holdout Protection

Strict governance protocols were enforced throughout Phase 14 to preserve the integrity of the final holdout test set:

- **Test Partition Size:** Exactly 14,988 M15 rows (`2026-02-19 12:00:00 UTC` onward).
- **Holdout Status:** **PERMANENTLY LOCKED & UNTOUCHED**.
- **Access Restrictions:**
  - Zero test partition rows were loaded into memory for feature transformation.
  - Zero test labels were evaluated, filtered, or inspected.
  - Zero model training, hyperparameter tuning, or threshold selection accessed test data.
  - Model comparison was conducted strictly on the validation partition.

Programmatic verification confirmed that the test partition start timestamp remained bit-exact at `2026-02-19 12:00:00+00:00` and row count remained exactly 14,988.

---

## 6. Candidate Target Definitions

All candidate targets share a forward prediction horizon of $H = 4$ M15 bars (60 minutes forward price displacement relative to bar close $t$). The four research targets are formulated as follows:

### Candidate Target A — Binary Direction (Unthresholded)
- **Concept:** Simplest possible directional target; classifies whether price moves up or down over 60 minutes, removing neutral consolidation.
- **Formula:**
  $$\text{Target A}(t) = \begin{cases} +1.0 & \text{if } \text{ret}_4(t) > 0.0 \\ -1.0 & \text{if } \text{ret}_4(t) < 0.0 \\ \text{NaN (Excluded)} & \text{if } \text{ret}_4(t) = 0.0 \end{cases}$$
  where $\text{ret}_4(t) = \frac{\text{close}[t+4] - \text{close}[t]}{\text{close}[t]}$.

### Candidate Target B — Volatility-Adjusted Binary Direction
- **Concept:** Directional movement conditioned on exceeding a point-in-time volatility barrier, excluding low-volatility chop.
- **Volatility Threshold:**
  $$\theta_t = 0.5 \times \sqrt{H} \times \text{ATR14\_norm}(t) = 1.0 \times \frac{\text{ATR14}(t)}{\text{close}[t]}$$
- **Formula:**
  $$\text{Target B}(t) = \begin{cases} +1.0 & \text{if } \text{ret}_4(t) > +\theta_t \\ -1.0 & \text{if } \text{ret}_4(t) < -\theta_t \\ \text{NaN (Excluded)} & \text{if } |\text{ret}_4(t)| \le \theta_t \end{cases}$$
  *(Matches non-neutral rows of `direction_vol_4` from Phase 6).*

### Candidate Target C — Fixed Economic-Move Binary
- **Concept:** Directional movement conditioned on exceeding a fixed minimum spread/slippage friction barrier of 5.0 pips (50 points), excluding sub-barrier noise.
- **Fixed Threshold:** $\theta = 0.00050 = 5.0\text{ pips}$.
- **Formula:**
  $$\text{Target C}(t) = \begin{cases} +1.0 & \text{if } \text{ret}_4(t) > +0.00050 \\ -1.0 & \text{if } \text{ret}_4(t) < -0.00050 \\ \text{NaN (Excluded)} & \text{if } |\text{ret}_4(t)| \le 0.00050 \end{cases}$$
  *(Matches non-neutral rows of `direction_4` from Phase 6).*

### Candidate Target D — Extreme-Move Volatility Filter
- **Concept:** Investigates whether directional predictability is concentrated in tail events by filtering out 87.5% of bars and retaining only large volatility expansions.
- **Higher Volatility Barrier:**
  $$\theta_t = 1.0 \times \sqrt{H} \times \text{ATR14\_norm}(t) = 2.0 \times \frac{\text{ATR14}(t)}{\text{close}[t]}$$
  *(Equivalent to $|\text{future\_vol\_adj\_return\_4}(t)| > 2.0$).*
- **Formula:**
  $$\text{Target D}(t) = \begin{cases} +1.0 & \text{if } \text{ret}_4(t) > +2.0 \times \text{ATR14\_norm}(t) \\ -1.0 & \text{if } \text{ret}_4(t) < -2.0 \times \text{ATR14\_norm}(t) \\ \text{NaN (Excluded)} & \text{if } |\text{ret}_4(t)| \le 2.0 \times \text{ATR14\_norm}(t) \end{cases}$$

---

## 7. Target Distributions

The empirical sample sizes, exclusion percentages, and class balance across both partitions are detailed below:

| Target | Partition | Total Partition Rows | Valid Sample Size | Excluded Rows (%) | LONG Count (%) | SHORT Count (%) |
|---|---|---|---|---|---|---|
| **Target A** | Training | 69,937 | 69,453 | 484 (0.69%) | 35,230 (50.72%) | 34,223 (49.28%) |
| **Target A** | Validation | 14,983 | 14,895 | 88 (0.59%) | 7,492 (50.30%) | 7,403 (49.70%) |
| **Target B** | Training | 69,937 | 27,514 | 42,423 (60.66%) | 14,057 (51.09%) | 13,457 (48.91%) |
| **Target B** | Validation | 14,983 | 5,622 | 9,361 (62.48%) | 2,799 (49.79%) | 2,823 (50.21%) |
| **Target C** | Training | 69,937 | 30,578 | 39,359 (56.28%) | 15,560 (50.89%) | 15,018 (49.11%) |
| **Target C** | Validation | 14,983 | 5,702 | 9,281 (61.94%) | 2,820 (49.46%) | 2,882 (50.54%) |
| **Target D** | Training | 69,937 | 9,739 | 60,198 (86.07%) | 4,960 (50.93%) | 4,779 (49.07%) |
| **Target D** | Validation | 14,983 | 1,876 | 13,107 (87.48%) | 923 (49.20%) | 953 (50.80%) |

**Observed Quality:**
1. Across all four candidate targets, the LONG vs SHORT ratio is remarkably balanced, sitting between 49.2% and 51.1% in both training and validation.
2. The percentage of excluded rows increases monotonically with threshold stringency: 0.6% (Target A) $\rightarrow$ 61.9% (Target C) $\rightarrow$ 62.5% (Target B) $\rightarrow$ 87.5% (Target D).

---

## 8. Target Quality Analysis

### 8.1 Future Return Displacement Statistics

| Target | Partition | Mean Return | Median Return | Return Std Dev | Mean Abs Return | Median Abs Return |
|---|---|---|---|---|---|---|
| **Target A** | Training | +0.09 bps | +0.18 bps | 10.32 bps | **6.63 pips** | **4.25 pips** |
| **Target A** | Validation | +0.03 bps | +0.09 bps | 8.01 bps | **5.32 pips** | **3.69 pips** |
| **Target B** | Training | +0.17 bps | +2.85 bps | 15.56 bps | **12.02 pips** | **9.31 pips** |
| **Target B** | Validation | +0.06 bps | -2.49 bps | 12.35 bps | **9.87 pips** | **8.06 pips** |
| **Target C** | Training | +0.16 bps | +5.10 bps | 15.27 bps | **12.20 pips** | **9.34 pips** |
| **Target C** | Validation | +0.02 bps | -5.05 bps | 12.50 bps | **10.29 pips** | **8.29 pips** |
| **Target D** | Training | +0.29 bps | +4.71 bps | 22.06 bps | **17.78 pips** | **14.26 pips** |
| **Target D** | Validation | +0.36 bps | -5.17 bps | 17.99 bps | **14.74 pips** | **12.50 pips** |

*Note: 1 pip = $0.00010 = 1.0\text{ bps} \times 10$.*

### 8.2 Contingency Analysis: Overlap with Baseline `direction_4`

The validation contingency matrices demonstrate how each candidate target relates to the original ternary `direction_4` labels:

#### Target A Overlap
```
                 Target A: -1.0  Target A: +1.0  Target A: NaN (0.0)    Total
direction_4: -1.0         2,882               0                    0    2,882
direction_4:  0.0         4,521           4,672                   88    9,281
direction_4: +1.0             0           2,820                    0    2,820
Total                     7,403           7,492                   88   14,983
```
- **Finding:** Target A retains 100% of the true directional displacement bars, but forces 9,193 small consolidation bars ($|\text{ret}_4| \le 5.0\text{ pips}$) into binary LONG/SHORT classifications based on sub-pip noise.

#### Target B Overlap
```
                 Target B: -1.0  Target B: +1.0  Target B: Excluded     Total
direction_4: -1.0         2,358               0                  524    2,882
direction_4:  0.0           465             487                8,329    9,281
direction_4: +1.0             0           2,312                  508    2,820
Total                     2,823           2,799                9,361   14,983
```
- **Finding:** Target B adapts to market regime: during low-volatility regimes where $1.0 \times \text{ATR} < 5.0\text{ pips}$, it promotes 952 bars previously labeled NEUTRAL into directional moves. During high-volatility regimes, it filters out 1,032 moves that exceeded 5 pips but failed to clear 1 ATR.

#### Target C Overlap
```
                 Target C: -1.0  Target C: +1.0  Target C: Excluded     Total
direction_4: -1.0         2,882               0                    0    2,882
direction_4:  0.0             0               0                9,281    9,281
direction_4: +1.0             0           2,820                    0    2,820
Total                     2,882           2,820                9,281   14,983
```
- **Finding:** Target C is mathematically identical to isolating non-neutral rows from `direction_4`. It excludes exactly the 9,281 neutral bars, creating a clean binary prediction task on meaningful 5-pip displacements.

#### Target D Overlap
```
                 Target D: -1.0  Target D: +1.0  Target D: Excluded     Total
direction_4: -1.0           940               0                1,942    2,882
direction_4:  0.0            13              12                9,256    9,281
direction_4: +1.0             0             911                1,909    2,820
Total                       953             923               13,107   14,983
```
- **Finding:** Target D selects exclusively tail excursions: 940 of 2,882 Short displacements and 911 of 2,820 Long displacements, discarding 87.5% of all market bars.

---

## 9. Model Configurations

To ensure that performance differences reflect target properties rather than modeling or hyperparameter artifacts, standard baseline models established in Phases 8–13 were utilized without alteration:

1. **Logistic Regression Baseline (`LogisticRegressionBaseline`):**
   - Preprocessing: `StandardScaler` fitted **strictly on training observations** $X_{\text{train}}$. Validation features $X_{\text{val}}$ transformed using fitted training parameters.
   - Solver: `lbfgs`
   - Class Weight: `balanced`
   - Max Iterations: 1,000
   - Regularization $C$: 1.0
   - Random State: 42

2. **Random Forest Baseline (`RandomForestBaseline`):**
   - Estimators: 100 decision trees
   - Max Depth: 10
   - Min Samples Leaf: 20
   - Class Weight: `balanced`
   - Features: Operating directly on raw causal features
   - Parallel Jobs: -1
   - Random State: 42

---

## 10. Validation Results

All candidate target and model combinations were fitted on the training partition and evaluated strictly on the validation partition using vectorized batch inference.

### 10.1 Primary & Secondary Classification Metrics

| Target | Model | Balanced Accuracy (Primary) | Raw Accuracy | Macro F1 | ROC-AUC | PR-AUC (Long) | PR-AUC (Short) | Mean Conf | Max Conf | 90th % Conf | Fit Time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Ph 12)** | Random Forest (3-class) | **43.61%** | 50.31% | 40.87% | N/A | N/A | N/A | 45.68% | 59.36% | 51.2% | ~25s |
| **Target A** | Logistic Regression | **51.98%** | 52.00% | 51.96% | 0.5301 | 0.5415 | 0.5154 | 53.27% | 98.66% | 56.21% | 3.94s |
| **Target A** | Random Forest | **53.10%** | 52.99% | 51.56% | 0.5373 | 0.5520 | 0.5282 | 54.82% | 91.75% | 60.00% | 7.93s |
| **Target B** | Logistic Regression | **51.22%** | 51.28% | 50.07% | 0.5227 | 0.5293 | 0.5244 | 54.72% | 99.58% | 59.12% | 2.56s |
| **Target B** | Random Forest | **51.76%** | 51.83% | 50.37% | 0.5287 | 0.5391 | 0.5315 | 56.42% | 94.10% | 63.39% | 2.80s |
| **Target C** | Logistic Regression | **51.23%** | 51.35% | 50.73% | 0.5171 | 0.5184 | 0.5171 | 54.28% | 100.0% | 58.20% | 2.92s |
| **Target C** | Random Forest | **51.49%** | 51.77% | 48.18% | 0.5253 | 0.5286 | 0.5261 | 55.98% | 91.10% | 63.02% | 3.16s |
| **Target D** | Logistic Regression | **49.26%** | 49.73% | 44.72% | 0.5044 | 0.4945 | 0.5231 | 57.18% | 98.79% | 63.41% | 1.81s |
| **Target D** | Random Forest | **51.51%** | 52.03% | 46.14% | 0.5156 | 0.5151 | 0.5141 | 56.23% | 84.82% | 61.60% | 1.03s |

### 10.2 Detailed Confusion Matrices and Per-Class Performance

- **Target A — Logistic Regression ($N=14,895$):**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=3,660 & \text{FP}=3,743 \\ \text{FN}=3,407 & \text{TP}=4,085 \end{bmatrix}$
  - SHORT (-1.0): Precision = 51.79%, Recall = 49.44%, F1 = 50.59%
  - LONG (+1.0): Precision = 52.19%, Recall = 54.52%, F1 = 53.33%

- **Target A — Random Forest ($N=14,895$):**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=5,227 & \text{FP}=2,176 \\ \text{FN}=4,826 & \text{TP}=2,666 \end{bmatrix}$
  - SHORT (-1.0): Precision = 51.99%, Recall = 70.61%, F1 = 59.90%
  - LONG (+1.0): Precision = 55.06%, Recall = 35.59%, F1 = 43.23%

- **Target B — Logistic Regression ($N=5,622$):**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=1,879 & \text{FP}=944 \\ \text{FN}=1,795 & \text{TP}=1,004 \end{bmatrix}$
  - SHORT (-1.0): Precision = 51.14%, Recall = 66.56%, F1 = 57.84%
  - LONG (+1.0): Precision = 51.54%, Recall = 35.87%, F1 = 42.26%

- **Target B — Random Forest ($N=5,622$):**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=1,939 & \text{FP}=884 \\ \text{FN}=1,824 & \text{TP}=975 \end{bmatrix}$
  - SHORT (-1.0): Precision = 51.53%, Recall = 68.69%, F1 = 58.88%
  - LONG (+1.0): Precision = 52.45%, Recall = 34.83%, F1 = 41.87%

- **Target C — Logistic Regression ($N=5,702$):**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=1,784 & \text{FP}=1,098 \\ \text{FN}=1,676 & \text{TP}=1,144 \end{bmatrix}$
  - SHORT (-1.0): Precision = 51.56%, Recall = 61.90%, F1 = 56.26%
  - LONG (+1.0): Precision = 51.03%, Recall = 40.57%, F1 = 45.20%

- **Target C — Random Forest ($N=5,702$):**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=2,227 & \text{FP}=655 \\ \text{FN}=2,095 & \text{TP}=725 \end{bmatrix}$
  - SHORT (-1.0): Precision = 51.53%, Recall = 77.27%, F1 = 61.83%
  - LONG (+1.0): Precision = 52.54%, Recall = 25.71%, F1 = 34.52%

- **Target D — Logistic Regression ($N=1,876$):**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=749 & \text{FP}=204 \\ \text{FN}=739 & \text{TP}=184 \end{bmatrix}$
  - SHORT (-1.0): Precision = 50.34%, Recall = 78.59%, F1 = 61.37%
  - LONG (+1.0): Precision = 47.42%, Recall = 19.93%, F1 = 28.07%

- **Target D — Random Forest ($N=1,876$):**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=798 & \text{FP}=155 \\ \text{FN}=745 & \text{TP}=178 \end{bmatrix}$
  - SHORT (-1.0): Precision = 51.72%, Recall = 83.74%, F1 = 63.92%
  - LONG (+1.0): Precision = 53.45%, Recall = 19.28%, F1 = 28.34%

---

## 11. Confidence Analysis

To examine whether higher model confidence isolates genuine directional edge or merely concentrates model bias, predictions were segmented into descriptive confidence intervals ($p_{\text{dir}} = \max(p_{\text{long}}, p_{\text{short}})$).

### Random Forest Confidence Bucket Breakdown

#### Target A (Binary Direction)
| Confidence Bucket | Count | Pct of Subset | Observed Accuracy | Mean Confidence | Precision LONG | Precision SHORT |
|---|---|---|---|---|---|---|
| **0.50–0.55** | 9,792 | 65.74% | 52.33% | 52.4% | 53.3% | 51.7% |
| **0.55–0.60** | 3,611 | 24.24% | 51.73% | 56.8% | 55.2% | 51.2% |
| **0.60–0.65** | 948 | 6.36% | 55.27% | 61.8% | 55.6% | 55.2% |
| **0.65–0.70** | 211 | 1.42% | 62.09% | 67.2% | 57.3% | 72.1% |
| **0.70–0.75** | 121 | 0.81% | 67.77% | 72.4% | 67.8% | 66.7% |
| **0.75–0.80** | 105 | 0.70% | 69.52% | 77.2% | 69.5% | 0.0% (0 preds) |
| **0.80+** | 107 | 0.72% | **85.05%** | 84.2% | **85.0%** | **0.0% (0 preds)** |

#### Target B (Volatility-Adjusted Direction)
| Confidence Bucket | Count | Pct of Subset | Observed Accuracy | Mean Confidence | Precision LONG | Precision SHORT |
|---|---|---|---|---|---|---|
| **0.50–0.55** | 2,873 | 51.10% | 50.40% | 52.5% | 49.5% | 51.1% |
| **0.55–0.60** | 1,568 | 27.89% | 51.98% | 57.1% | 56.0% | 50.9% |
| **0.60–0.65** | 819 | 14.57% | 50.79% | 62.2% | 53.1% | 50.5% |
| **0.65–0.70** | 233 | 4.14% | 60.09% | 66.4% | 53.6% | 61.0% |
| **0.70–0.75** | 32 | 0.57% | 65.62% | 72.7% | 59.3% | 100.0% (5 preds) |
| **0.75–0.80** | 25 | 0.44% | 64.00% | 77.9% | 64.0% | 0.0% (0 preds) |
| **0.80+** | 72 | 1.28% | **80.56%** | 86.0% | **80.6%** | **0.0% (0 preds)** |

#### Target C (Fixed 5-Pip Direction)
| Confidence Bucket | Count | Pct of Subset | Observed Accuracy | Mean Confidence | Precision LONG | Precision SHORT |
|---|---|---|---|---|---|---|
| **0.50–0.55** | 3,097 | 54.31% | 50.27% | 52.7% | 48.3% | 51.0% |
| **0.55–0.60** | 1,646 | 28.87% | 53.40% | 56.8% | 61.5% | 51.5% |
| **0.60–0.65** | 641 | 11.24% | 51.79% | 62.5% | 47.0% | 52.9% |
| **0.65–0.70** | 214 | 3.75% | 52.80% | 66.3% | 45.7% | 54.8% |
| **0.70–0.75** | 23 | 0.40% | 52.17% | 72.5% | 47.6% | 100.0% (2 preds) |
| **0.75–0.80** | 24 | 0.42% | 58.33% | 77.2% | 58.3% | 0.0% (0 preds) |
| **0.80+** | 57 | 1.00% | **78.95%** | 84.8% | **78.9%** | **0.0% (0 preds)** |

#### Target D (Extreme Move Filter)
| Confidence Bucket | Count | Pct of Subset | Observed Accuracy | Mean Confidence | Precision LONG | Precision SHORT |
|---|---|---|---|---|---|---|
| **0.50–0.55** | 828 | 44.14% | 52.66% | 52.5% | 53.6% | 52.3% |
| **0.55–0.60** | 724 | 38.59% | 51.38% | 57.3% | 49.2% | 51.6% |
| **0.60–0.65** | 288 | 15.35% | 51.39% | 61.9% | 58.3% | 51.1% |
| **0.65–0.70** | 13 | 0.69% | 30.77% | 66.7% | 16.7% | 42.9% |
| **0.70–0.75** | 8 | 0.43% | 75.00% | 71.9% | 75.0% | 0.0% (0 preds) |
| **0.75–0.80** | 8 | 0.43% | 62.50% | 78.4% | 62.5% | 0.0% (0 preds) |
| **0.80+** | 7 | 0.37% | 71.43% | 82.9% | 71.4% | 0.0% (0 preds) |

### Key Diagnostic Discovery on Confidence Buckets
> [!IMPORTANT]
> **One-Sided Collapse at High Confidence:**  
> In all four candidate targets, while raw accuracy appears high in the top confidence bucket ($\ge 0.80$, reaching 78.9%–85.1%), **100% of all predictions in the $\ge 0.80$ bucket are LONG**. Zero SHORT trades were emitted at high confidence.  
> 
> This demonstrates that high model confidence does **not** reflect bidirectional forecasting edge. Instead, the model exhibits localized probability distortion: during specific high-volatility session regimes (primarily London/NY overlap or weekend re-openings), tree splits strongly favor LONG classifications due to training set drift. When subjected to a symmetric trading rule, this asymmetry produces severe unhedged directional exposure.

---

## 12. Temporal Stability

To ensure that candidate model performance is not an artifact of an isolated market regime, the 7.2-month validation partition was segmented chronologically into three equal blocks of approximately 4,994 M15 bars:

- **Block 1 (Early):** `2025-07-14 05:45:00 UTC` to `2025-09-24 06:15:00 UTC`
- **Block 2 (Mid):** `2025-09-24 06:30:00 UTC` to `2025-12-05 09:30:00 UTC`
- **Block 3 (Late):** `2025-12-05 09:45:00 UTC` to `2026-02-19 10:45:00 UTC`

### Random Forest Sub-Period Results Across Targets

| Target | Chronological Block | Included Rows | LONG Count (%) | SHORT Count (%) | Balanced Accuracy | Macro F1 | ROC-AUC | Mean Conf | Max Conf |
|---|---|---|---|---|---|---|---|---|---|
| **Target A** | Block 1 (Early) | 4,968 | 2,491 (50.1%) | 2,477 (49.9%) | **53.10%** | 52.29% | 0.5390 | 54.5% | 88.3% |
| **Target A** | Block 2 (Mid) | 4,965 | 2,519 (50.7%) | 2,446 (49.3%) | **53.66%** | 53.00% | 0.5539 | 53.9% | 91.8% |
| **Target A** | Block 3 (Late) | 4,962 | 2,482 (50.0%) | 2,480 (50.0%) | **52.43%** | 48.15% | 0.5213 | 56.0% | 89.2% |
| **Target B** | Block 1 (Early) | 1,904 | 943 (49.5%) | 961 (50.5%) | **49.82%** | 49.57% | 0.5128 | 56.1% | 93.6% |
| **Target B** | Block 2 (Mid) | 1,851 | 919 (49.6%) | 932 (50.4%) | **52.97%** | 52.37% | 0.5574 | 54.9% | 94.1% |
| **Target B** | Block 3 (Late) | 1,867 | 937 (50.2%) | 930 (49.8%) | **52.72%** | 47.34% | 0.5291 | 58.3% | 88.4% |
| **Target C** | Block 1 (Early) | 2,148 | 1,052 (49.0%) | 1,096 (51.0%) | **51.04%** | 49.70% | 0.5096 | 55.4% | 90.5% |
| **Target C** | Block 2 (Mid) | 1,775 | 861 (48.5%) | 914 (51.5%) | **51.80%** | 49.73% | 0.5417 | 54.6% | 91.1% |
| **Target C** | Block 3 (Late) | 1,779 | 907 (51.0%) | 872 (49.0%) | **52.33%** | 42.82% | 0.5495 | 58.1% | 88.0% |
| **Target D** | Block 1 (Early) | 686 | 330 (48.1%) | 356 (51.9%) | **51.07%** | 49.74% | 0.5096 | 55.0% | 82.5% |
| **Target D** | Block 2 (Mid) | 561 | 281 (50.1%) | 280 (49.9%) | **51.76%** | 43.29% | 0.5041 | 55.6% | 84.8% |
| **Target D** | Block 3 (Late) | 629 | 312 (49.6%) | 317 (50.4%) | **52.29%** | 42.21% | 0.5366 | 58.2% | 83.2% |

### Temporal Stability Analysis
1. **Consistency of Near-Random Behavior:** Balanced accuracy remains compressed within the narrow range of **49.82% to 53.66%** across all blocks and all candidate targets. No single block demonstrates robust predictive edge.
2. **Degradation of Macro F1 in Block 3:** In Block 3 (Late 2025 – Early 2026), Macro F1 collapses across Targets B, C, and D (dropping to 42.2%–47.3%) due to an increasing imbalance in model recall toward SHORT classifications, while mean predicted confidence rises ($58.1\%–58.3\%$).

---

## 13. Feature Diagnostics

Feature importances extracted from the fitted Random Forest models reveal how tree decision boundaries adapt to each target formulation across the seven quantitative feature categories:

### 13.1 Top 10 Features by Target Formulation

| Rank | Target A (Binary) | Target B (Vol-Adjusted) | Target C (Fixed 5-Pip) | Target D (Extreme Move) |
|---|---|---|---|---|
| **1** | `hour` (0.0486) | `hour` (0.0448) | `cos_hour` (0.0398) | `vol_ann_80` (0.0253) |
| **2** | `cos_hour` (0.0465) | `cos_hour` (0.0391) | `hour` (0.0357) | `vol_std_80` (0.0234) |
| **3** | `spread_norm` (0.0354) | `spread_norm` (0.0301) | `spread_norm` (0.0306) | `spread_norm` (0.0233) |
| **4** | `spread_zscore_20` (0.0230) | `ema_spread_20_80` (0.0221) | `roc_80` (0.0224) | `sma_10` (0.0218) |
| **5** | `log_tick_volume` (0.0199) | `spread_zscore_20` (0.0219) | `vol_ann_80` (0.0208) | `tick_vol_sma_20` (0.0216) |
| **6** | `roc_80` (0.0198) | `vol_ann_80` (0.0218) | `vol_std_80` (0.0199) | `sma_20` (0.0212) |
| **7** | `ema_spread_20_80` (0.0197) | `ema_10` (0.0196) | `vol_std_40` (0.0191) | `ema_spread_20_80` (0.0211) |
| **8** | `dist_ema_80` (0.0196) | `vol_std_80` (0.0188) | `ema_spread_20_80` (0.0191) | `ema_10` (0.0207) |
| **9** | `vol_ann_80` (0.0187) | `vol_std_40` (0.0184) | `ema_10` (0.0185) | `rolling_hl_ratio_20` (0.0205) |
| **10** | `roc_5` (0.0179) | `atr_norm_40` (0.0183) | `dist_sma_80` (0.0185) | `roc_40` (0.0204) |

### 13.2 Category Distribution in Top 20 Features

| Feature Category | Available (Total 80) | Target A Top 20 | Target B Top 20 | Target C Top 20 | Target D Top 20 |
|---|---|---|---|---|---|
| **Trend** | 18 | 4 (20%) | **7 (35%)** | **8 (40%)** | 6 (30%) |
| **Volatility** | 13 | 5 (25%) | 5 (25%) | 4 (20%) | **7 (35%)** |
| **Activity** | 11 | **6 (30%)** | 5 (25%) | 5 (25%) | 4 (20%) |
| **Momentum** | 8 | 3 (15%) | 1 (5%) | 1 (5%) | 2 (10%) |
| **Time** | 13 | 2 (10%) | 2 (10%) | 2 (10%) | 1 (5%) |
| **Price / Candle** | 12 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| **Gap / Session** | 5 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |

### 13.3 Diagnostic Insights from Feature Shifts
1. **Dominance of Session Timing:** In Targets A, B, and C, `hour` and `cos_hour` constitute the top two individual features. Trees split primarily on time of day, aligning with daily EURUSD liquidity cycles (Asian quiet vs European open vs NY overlap).
2. **Shift Toward Structural Volatility in Target D:** When small moves are excluded (Target D), time-of-day features drop out of the top ranks. The ensemble shifts its primary split criteria to multi-session volatility metrics (`vol_ann_80`, `vol_std_80`) and trend moving averages (`sma_10`, `sma_20`), demonstrating that extreme moves are identified by volatility regime rather than clock time.
3. **Absence of Micro-Price Action:** Across all four targets, zero features from the `price/candle geometry` category (`candle_body`, `upper_wick`, `lower_wick_ratio`, `open_to_close_return`) appeared in the top 20, confirming that individual candle morphology carries negligible explanatory power for 1-hour forward displacement.

---

## 14. Leakage and Governance Checks

All ten required leakage and data integrity checks were executed and verified programmatically:

| Check # | Verification Requirement | Status | Verification Detail |
|---|---|---|---|
| **1** | Future Dependency Correctness | **PASS** | Features use only data available at time $t$; targets use forward close at $t+4$. |
| **2** | Horizon Alignment Strictly $H=4$ | **PASS** | Target forward shift exactly equals 4 bars (60 minutes). |
| **3** | End-of-Data Boundary NaNs | **PASS** | Exactly the final 4 rows of the dataset evaluate to NaN for all targets. |
| **4** | Feature Immutability | **PASS** | Feature parquet file remains read-only and unmodified; sha256 checksums verified. |
| **5** | No Target Columns in Feature Matrix $X$ | **PASS** | Evaluated `ds.feature_names`: zero `direction*`, `future*`, or `target*` columns. |
| **6** | No Future OHLCV in Features | **PASS** | All rolling windows and lag operations use strictly non-negative shifts. |
| **7** | No Future Returns in Features | **PASS** | Feature pipeline excludes `future_return_*` and `future_log_return_*`. |
| **8** | Phase 11 Test Set Holdout Isolation | **PASS** | Test partition (14,988 rows: `2026-02-19 12:00:00 UTC` onward) was never accessed. |
| **9** | Chronological Split Integrity | **PASS** | Train end (`2025-07-14 04:30`) < Val start (`2025-07-14 05:45`); 4-bar purge verified. |
| **10** | Deterministic Repeatability | **PASS** | Fixed random seeds (42) produce bit-exact identical predictions across repeated runs. |

---

## 15. Limitations

The empirical findings of Phase 14 are subject to the following known research boundaries:
1. **Instrument Limitation:** Evaluated strictly on EURUSD M15 data from MetaQuotes demo feeds. Findings cannot be generalized to higher timeframes (H1, H4, D1) or other currency pairs without independent testing.
2. **Horizon Constraint:** Fixed strictly to $H=4$ bars (60 minutes). Longer horizons ($H=8, 16$) may exhibit different signal-to-noise ratios.
3. **Model Space:** Evaluated on standard reference classifiers (`LogisticRegressionBaseline` and `RandomForestBaseline`). Advanced architectures (e.g., gradient boosted trees, temporal convolutional networks) were intentionally omitted to maintain strict target comparison controls.
4. **Execution Frictions Omitted:** Because Phase 14 is a target and signal research phase, trading backtests with spread, commission, and slippage were not simulated.

---

## 16. Interpretation

> [!NOTE]
> ### Distinction of Research Evidence
> - **OBSERVED RESULT:** Direct, indisputable empirical measurement from the validation partition.
> - **INTERPRETATION:** Statistical deduction explaining the observed results.
> - **FUTURE HYPOTHESIS:** Proposed question or direction for subsequent research.

### 16.1 Target Redesign Does Not Solve Directional Separability
- **OBSERVED RESULT:**  
  - Target A (Binary Direction) produced validation balanced accuracy of **51.98% (LR)** and **53.10% (RF)** with ROC-AUC of **0.5373**.
  - Target B (Volatility-Adjusted) produced validation balanced accuracy of **51.22% (LR)** and **51.76% (RF)** with ROC-AUC of **0.5287**.
  - Target C (Fixed 5-Pip Move) produced validation balanced accuracy of **51.23% (LR)** and **51.49% (RF)** with ROC-AUC of **0.5253**.
  - Target D (Extreme Move Filter) produced validation balanced accuracy of **49.26% (LR)** and **51.51% (RF)** with ROC-AUC of **0.5156**.
- **INTERPRETATION:**  
  Converting the multi-class target into a binary target and filtering out neutral observations does **not** create meaningful directional separation. Across all formulations and model architectures, validation balanced accuracy remains tightly confined between **49.3% and 53.1%**, and ROC-AUC never exceeds **0.538**. The EURUSD M15 directional process at $H=4$ behaves overwhelmingly as an efficient martingale where past technical features offer negligible linear or non-linear directional forecasting advantage.
- **FUTURE HYPOTHESIS:**  
  Any observed trading edge in live execution must originate from asymmetric risk/reward structure, execution alpha, or regime conditioning, rather than unconditioned directional forecasting on M15 bars.

### 16.2 Filtering for Large Moves Does Not Increase Predictability
- **OBSERVED RESULT:**  
  Excluding 87.5% of samples to focus exclusively on extreme moves ($|\text{ret}_4| > 2.0 \times \text{ATR}$, Target D) resulted in a balanced accuracy of **51.51% (RF)** and **49.26% (LR)**, lower than the unfiltered binary target (53.10%).
- **INTERPRETATION:**  
  Extreme intraday excursions on EURUSD M15 are primarily driven by exogenous macroeconomic events, scheduled news releases (NFP, CPI, interest rate decisions), and liquidity vacuums. While these events create large absolute displacements, their directional sign (+ vs -) is not predictable from past 15-minute price action or momentum indicators.

### 16.3 High Model Confidence Reflects Regime Drift, Not Edge
- **OBSERVED RESULT:**  
  When model confidence exceeds 0.80, 100% of emitted predictions across all four targets are LONG, with 0 SHORT predictions.
- **INTERPRETATION:**  
  High model probability does not reflect symmetric certainty. Rather, it reflects localized overfitting to training-period drift where specific volatility/session combinations coincided with net upward moves. Applying high-confidence thresholding to this model creates unhedged directional exposure rather than a selective trading filter.

---

## 17. Production Status

1. **Production Code Unchanged:** No production files, baseline parameters, or configuration constants were altered during Phase 14.
2. **Phase 12 Baseline Preserved:** The official production baseline remains:
   - Model: Random Forest balanced
   - Target: `direction_4` (ternary: -1.0, 0.0, +1.0)
   - Horizon: $H=4$ M15 bars
   - Confidence Threshold: $\tau = 0.60$
   - Stop Loss: $1.00 \times \text{ATR14}$
   - Take Profit: $1.50 \times \text{ATR14}$
   - Maximum Hold: 4 bars
   - Position Sizing: RiskEngine fixed fractional authority
3. **Candidate Targets Retained as Research Artifacts:** Targets A, B, C, and D are documented strictly as research diagnostics and are **not** promoted to production.

---

## 18. Future Research Directions

Based on the empirical findings of Phase 14, the following directions are recommended for future research phases:

1. **Horizon Expansion ($H \ge 16$ or Multi-Hour):** Investigate whether directional signal improves at multi-hour or daily horizons where macroeconomic trends and monetary policy differentials exert greater influence relative to intraday spread and microstructure noise.
2. **Regime Conditioning:** Rather than predicting unconditional direction, evaluate models trained strictly as volatility regime or trend-state classifiers (e.g., predicting whether volatility will expand or contract, regardless of direction).
3. **Alternative Feature Representations:** Explore order book depth, real volume tick imbalances, or macroeconomic calendar indicators, as price-derived technical indicators show insufficient signal.
4. **Execution-Centric Edge:** Focus future backtesting on structural market inefficiencies (e.g., Asian session mean reversion, rollover spread dynamics) with strictly deterministic entry rules rather than statistical direction classifiers.

---

## 19. Governance Sign-off & Audit Log

- **Phase Status:** COMPLETE
- **Holdout Test Set:** UNTOUCHED (14,988 rows verified)
- **Leakage Checks:** 10 / 10 PASSED
- **Test Suite Status:** 244 / 244 tests passing (8 new Phase 14 tests)
- **Linter & Formatter:** 0 errors, 100% compliant (`ruff check`, `ruff format`)
- **Git Target Branch:** `develop`
- **Execution Timestamp:** 2026-09-29T16:55:00 UTC
