# EURUSD M15 Baseline Machine Learning Models Specification

**Phase:** Phase 8 — Baseline Machine Learning Models  
**Instrument:** EURUSD  
**Timeframe:** M15 (15-Minute Candles)  
**Prediction Horizon:** $H = 4$ bars (1 hour / 60 minutes)  
**Target:** `direction_4` (Ternary Direction Classification: -1.0 [SHORT], 0.0 [NEUTRAL], +1.0 [LONG])  
**Input Features:** 80 derived point-in-time features from Phase 5 feature registry  
**Dataset Partitions:** Phase 7 chronological 70% / 15% / 15% split ($H=4$ forward purge)  
**Environment:** Canonical Ubuntu Linux x86_64, Python 3.13.15  

---

> [!IMPORTANT]
> **Explicit Research Boundary & Non-Viability Disclaimers:**
> - Phase 8 establishes predictive baselines only. It does **not** establish trading profitability, economic value, or live-trading viability.
> - The test set was **not** used for model selection or baseline evaluation. The test partition remains strictly untouched and reserved for future final verification.

---

## 1. Phase Objective

The objective of Phase 8 is to establish rigorous, reproducible, and leakage-safe statistical baseline models for the EURUSD M15 directional prediction problem. Prior to experimenting with complex non-linear architectures, deep learning, or hyperparameter optimization, the research pipeline requires an empirical benchmark against which all future algorithmic claims can be evaluated.

---

## 2. Prediction Task Definition

At candle completion timestamp $t$, an algorithm observes input feature vector $X_t \in \mathbb{R}^{80}$ (constructed point-in-time from historical and concurrent candle data up to $t$) and predicts the direction of market price displacement over the subsequent 1-hour interval ($H=4$ candles):

$$\hat{y}_t = f_\theta(X_t) \in \{-1.0 \text{ (SHORT)}, \; 0.0 \text{ (NEUTRAL)}, \; +1.0 \text{ (LONG)}\}$$

The ground truth target $y_t$ was engineered in Phase 6 using fixed thresholds ($\tau = 5.0$ bps / 5.0 pips):

$$y_{t, 4} = \begin{cases} 
+1.0 \text{ (LONG)}, & R_{t, 4} > +0.00050 \\ 
-1.0 \text{ (SHORT)}, & R_{t, 4} < -0.00050 \\ 
0.0 \text{ (NEUTRAL)}, & -0.00050 \le R_{t, 4} \le +0.00050 
\end{cases}$$

---

## 3. Dataset Partitions and Split Dimensions

The dataset was assembled and partitioned strictly using the Phase 7 temporal holdout pipeline:

- **Source Dataset:** 100,000 real MetaQuotes-Demo EURUSD M15 candles (September 2022 to September 2026).
- **Feature Warm-Up:** First 80 rows dropped to eliminate rolling window initialization NaNs.
- **Assembled Usable Candles:** 99,916 complete rows (tail 4 rows reserved due to missing future realizations).
- **Split Configuration:** Chronological 70% Train / 15% Validation / 15% Test with forward purge of $H=4$ bars.

| Partition | Observation Count | Percentage | Start Timestamp (UTC) | End Timestamp (UTC) | Role in Phase 8 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | **69,937** | 70.00% | `2022-09-19 05:30:00` | `2025-07-14 04:30:00` | Model Fitting Only |
| *Purged Gap* | *4 bars* | *60 min* | `2025-07-14 04:45:00` | `2025-07-14 05:30:00` | Lookahead Elimination |
| **VALIDATION** | **14,983** | 15.00% | `2025-07-14 05:45:00` | `2026-02-19 10:45:00` | Model Evaluation & Comparison |
| *Purged Gap* | *4 bars* | *60 min* | `2026-02-19 11:00:00` | `2026-02-19 11:45:00` | Lookahead Elimination |
| **TEST** | **14,988** | 15.00% | `2026-02-19 12:00:00` | `2026-09-25 22:45:00` | **UNTOUCHED / RESERVED** |

---

## 4. Class Frequencies and Imbalance

The directional classes exhibit natural market drift balance with a neutral majority:

| Class | Label | Training Count (70%) | Training % | Validation Count (15%) | Validation % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SHORT** | `-1.0` | 15,017 | 21.47% | 2,882 | 19.24% |
| **NEUTRAL** | `0.0` | 39,360 | **56.28%** | 9,281 | **61.94%** |
| **LONG** | `+1.0` | 15,560 | 22.25% | 2,820 | 18.82% |
| **Total** | | **69,937** | 100.00% | **14,983** | 100.00% |

The majority class in the training partition is `0.0 (NEUTRAL)` (56.28%). In the validation partition, the proportion of neutral drift rises to 61.94%.

---

## 5. Baseline Models Architecture & Specifications

Three distinct model families were implemented in `ai/models/baselines.py`:

### Model A: Majority Class Baseline (`MajorityClassClassifier`)
- **Type:** Heuristic / Zero-Rule Reference
- **Mechanism:** Finds the modal class of the training target (`0.0 / NEUTRAL`) and predicts it unconditionally for all validation samples.
- **Probabilities:** Constant empirical class frequencies observed in training ($P(\text{SHORT})=0.2147, P(\text{NEUTRAL})=0.5628, P(\text{LONG})=0.2225$).
- **Purpose:** Establishes the trivial statistical lower bound that any viable machine learning model must beat.

### Model B: Regularized Logistic Regression (`LogisticRegressionBaseline`)
- **Type:** Linear Multiclass Classifier (`sklearn.linear_model.LogisticRegression`)
- **Hyperparameters:** `solver='lbfgs'`, `C=1.0`, `max_iter=1000`, `random_state=42`.
- **Preprocessing:** `StandardScaler` fitted **strictly on the training feature matrix**. The validation matrix is transformed using only the training-derived mean and scale parameters.
- **Purpose:** Measures the extent of linear separability in the 80 engineered features.

### Model C: Random Forest (`RandomForestBaseline`)
- **Type:** Non-linear Decision Tree Ensemble (`sklearn.ensemble.RandomForestClassifier`)
- **Hyperparameters:** `n_estimators=100`, `max_depth=10`, `min_samples_leaf=20`, `random_state=42`, `n_jobs=-1`.
- **Preprocessing:** Operates directly on raw features without scaling.
- **Purpose:** Captures non-linear interactions, regime thresholds, and feature splits while controlling tree depth to minimize overfitting.

---

## 6. Preprocessing and Causal Standardization

In time-series machine learning, fitting scalers across the full dataset causes lookahead leakage by exposing future means and standard deviations to earlier training observations.

Phase 8 implements strict causal standardization:
1. **Fit on Train:** $\mu_{\text{train}} = \frac{1}{N_{\text{tr}}} \sum X_{\text{tr}}, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{tr}}} \sum (X_{\text{tr}} - \mu_{\text{train}})^2}$.
2. **Transform Train:** $X_{\text{tr}}^{\text{scaled}} = (X_{\text{tr}} - \mu_{\text{train}}) / \sigma_{\text{train}}$.
3. **Transform Validation:** $X_{\text{val}}^{\text{scaled}} = (X_{\text{val}} - \mu_{\text{train}}) / \sigma_{\text{train}}$.
4. **Validation Isolation:** The validation set mean $\mu_{\text{val}}$ and scale $\sigma_{\text{val}}$ are never computed or accessed during model fitting.

---

## 7. Deterministic Randomness & Reproducibility

To guarantee exact reproducibility across runs:
- Global random seed: `random_state = 42`.
- Algorithm solver convergence tolerance: default `tol = 1e-4`.
- Re-running `scripts/run_baseline_models.py` produces byte-for-byte identical predictions, confusion matrices, and metrics.
- Automated test `test_deterministic_model_reproducibility` verifies identical predictions across multiple invocations.

---

## 8. Validation Evaluation Metrics

All metrics were computed strictly on the validation partition (14,983 observations):

| Evaluation Metric | Majority Baseline | Logistic Regression | Random Forest |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.6194 (61.94%) | 0.6201 (62.01%) | **0.6223 (62.23%)** |
| **Balanced Accuracy** | 0.3333 (33.33%) | 0.3532 (35.32%) | **0.3649 (36.49%)** |
| **Macro Precision** | 0.2065 | 0.4311 | **0.4458** |
| **Macro Recall** | 0.3333 | 0.3532 | **0.3649** |
| **Macro F1-Score** | 0.2550 | 0.3063 | **0.3289** |
| **ROC-AUC (One-vs-Rest)** | 0.5000 | 0.6675 | **0.6695** |

### Per-Class Validation F1-Scores

| Model | SHORT (-1.0) F1 | NEUTRAL (0.0) F1 | LONG (+1.0) F1 |
| :--- | :---: | :---: | :---: |
| **Majority Class Baseline** | 0.0000 | 0.7650 | 0.0000 |
| **Logistic Regression** | 0.0871 | 0.7691 | 0.0627 |
| **Random Forest** | **0.1429** | **0.7724** | **0.0714** |

---

## 9. Baseline Comparison and Analysis

1. **Accuracy Relative to Majority Baseline:**
   - The naive majority baseline predicts `NEUTRAL (0.0)` for all samples, achieving an accuracy of **61.94%** purely because the validation partition is 61.94% neutral.
   - Logistic Regression achieves **62.01%**, exceeding the naive baseline by **+0.07 percentage points**.
   - Random Forest achieves **62.23%**, exceeding the naive baseline by **+0.29 percentage points**.

2. **Balanced Accuracy & Macro F1:**
   - Balanced accuracy removes the bias of the neutral majority:
     - Majority baseline: 33.33% (equivalent to random guessing on a balanced distribution).
     - Logistic Regression: 35.32% (+1.99 percentage points).
     - Random Forest: 36.49% (+3.16 percentage points).
   - Macro F1:
     - Majority baseline: 0.2550 (zero precision/recall for directional moves).
     - Logistic Regression: 0.3063.
     - Random Forest: 0.3289.

3. **Multiclass ROC-AUC:**
   - Majority baseline has an ROC-AUC of **0.5000**, confirming zero discriminative power.
   - Logistic Regression achieves an ROC-AUC of **0.6675**.
   - Random Forest achieves an ROC-AUC of **0.6695**.
   - This indicates that while hard thresholding at standard probabilities (argmax) is dominated by the neutral class, the ranked predicted class probabilities possess modest directional signal ($AUC \approx 0.67$).

---

## 10. Confusion Matrix Analysis

Confusion matrices were generated and saved to `reports/figures/`:

1. **Majority Baseline (`reports/figures/confusion_matrix_majority.png`):**
   - Predicted SHORT: 0 (0.0%)
   - Predicted NEUTRAL: 14,983 (100.0%)
   - Predicted LONG: 0 (0.0%)
   - Completely ignores directional moves; all errors are false neutrals.

2. **Logistic Regression (`reports/figures/confusion_matrix_logistic_regression.png`):**
   - True SHORT (2,882): 154 correctly identified (5.3% recall), 2,654 classified neutral, 74 classified long.
   - True NEUTRAL (9,281): 9,039 correctly identified (97.4% recall), 127 classified short, 115 classified long.
   - True LONG (2,820): 98 correctly identified (3.5% recall), 2,624 classified neutral, 98 classified long.

3. **Random Forest (`reports/figures/confusion_matrix_random_forest.png`):**
   - True SHORT (2,882): 278 correctly identified (9.6% recall), 2,525 classified neutral, 79 classified long.
   - True NEUTRAL (9,281): 8,938 correctly identified (96.3% recall), 209 classified short, 134 classified long.
   - True LONG (2,820): 108 correctly identified (3.8% recall), 2,618 classified neutral, 94 classified long.

---

## 11. Leakage Controls and Test-Set Protection

Phase 8 adheres strictly to 12 automated leakage verification tests in `tests/test_models.py`:

1. **Test Data Never Fitted:** Model fitting calls strictly receive `train_X` and `train_y`.
2. **Scaler Training-Only Fitting:** `StandardScaler.mean_` and `scale_` match training statistics and differ from validation statistics.
3. **Validation Transformation Invariance:** Scaler parameters remain immutable during validation inference.
4. **Target Absence:** The target column `direction_4` is completely absent from input feature matrices.
5. **Timestamp Absence:** Timestamps and integer times are excluded from feature matrices.
6. **Feature Dimensionality:** Feature matrix width is invariant at exactly 80 derived features.
7. **Chronological Sequence:** Train timestamps strictly precede validation timestamps.
8. **Deterministic Reproducibility:** Repeated runs produce identical predictions.
9. **Majority Class Training Mode:** Majority baseline computes class frequencies from training data only.
10. **Validation Metric Isolation:** Validation metrics evaluate validation targets strictly.
11. **Test Split Protection:** `run_baseline_training_pipeline()` executes without accessing `test_X` or `test_y`. Guarded proxy raises an exception if test data is queried.
12. **Point-in-Time Causality:** Past feature values are unaffected by future candle changes.

---

## 12. Limitations of Baseline Models

1. **Argmax Probability Threshold Bias:** Because the neutral class represents ~56%–62% of observations, argmax probability assignment naturally favors the neutral class, leading to low directional recall (~4%–10%).
2. **Absence of Hyperparameter Optimization:** Models used fixed default parameters without tuning depth, regularization, or tree count.
3. **Absence of Probability Calibration:** Raw class probabilities have not been calibrated using isotonic regression or Platt scaling.
4. **Fixed Horizon Evaluation:** Only $H=4$ was evaluated in this baseline run.
5. **No Economic Feasibility:** Statistical classification metrics (accuracy, F1, AUC) do not account for bid-ask spread, slippage, rollover costs, or execution latency.
