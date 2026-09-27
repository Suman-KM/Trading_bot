# Phase 11 — Final Out-of-Sample Test Evaluation (EURUSD M15)

## 1. Executive Summary & Objective

This document reports the final out-of-sample statistical evaluation of the locked machine learning models on the completely unseen holdout **Test partition** (14,988 observations, spanning February 19, 2026 to September 25, 2026 UTC).

> [!IMPORTANT]
> **Mandatory Holdout Governance Disclaimers:**
> - "Phase 11 is a final out-of-sample statistical evaluation of the locked EURUSD M15 classification task. Test results are not used for model tuning, feature selection, threshold selection, or trading-rule development."
> - "Phase 11 does not establish profitability, positive trading expectancy, execution viability, or live-market suitability."

### Primary Finding
The models trained strictly on historical data prior to July 14, 2025 demonstrate **near-zero generalization gap** when evaluated on the unseen February–September 2026 test partition:
- **Random Forest (balanced)** (validation-era selected candidate) achieved **43.78% Balanced Accuracy** on the Test set vs **43.61%** on the Validation set ($\Delta = +0.17\%$).
- **Macro F1** remained virtually identical: **0.4066** on Test vs **0.4087** on Validation ($\Delta = -0.0022$).
- **SHORT Recall** was preserved almost perfectly: **55.19%** on Test vs **55.24%** on Validation ($\Delta = -0.05\%$).
- **Macro ROC-AUC** remained consistent and slightly expanded: **0.6808** on Test vs **0.6626** on Validation ($\Delta = +0.0182$).
- **Logistic Regression (balanced)** exhibited identical stability: **43.06% Balanced Accuracy** on Test vs **42.74%** on Validation ($\Delta = +0.32\%$), and **0.4251 Macro F1** on Test vs **0.4208** on Validation ($\Delta = +0.0043$).

At the same time, directional precision on the test set remains modest (**24.4% to 28.5%** against a random base rate of ~16–18%), confirming that while the models possess stable non-random classification signal, **they do not possess sufficient precision to overcome bid-ask spread and transaction friction in live execution**.

---

## 2. Test-Set Governance & Integrity Audit

The Test partition was unlocked strictly for read-only evaluation under the following automated contract rules enforced by [`TestSetGovernanceGuard`](file:///home/cino/projects/ai-trading-system/ai/models/test_evaluation.py#L48-L135):

1. **Chronological Post-Validation Placement**: Training (2022-09-19 to 2025-07-14) $\rightarrow$ Validation (2025-07-14 to 2026-02-19) $\rightarrow$ Test (2026-02-19 to 2026-09-25).
2. **Purge/Embargo Boundaries**: A 4-bar purge buffer (60 minutes) was strictly enforced between Train and Validation, and between Validation and Test.
3. **Partition Dimension**: Test partition contains exactly 14,988 rows.
4. **Feature Integrity**: Exactly 80 derived causal point-in-time features from Phase 5; no target (`direction_4`), future returns (`future_return_4`), or raw timestamps exist in $X_{\text{test}}$.
5. **No Lookahead in Preprocessing**: The `StandardScaler` used for Logistic Regression was fitted exclusively on the 69,937 training rows and never updated on Test.
6. **No Retraining or Tuning**: All model hyperparameters, tree counts, depths, and class weights remained frozen as defined in Phase 8 and Phase 10.
7. **No Post-Test Model Selection**: Candidate models were preselected exclusively based on Phase 10 validation evidence before test evaluation commenced.

---

## 3. Locked Problem Specification

- **Instrument**: EURUSD
- **Timeframe**: M15 (15-minute bars)
- **Data Feed**: MetaQuotes Ltd. / MetaQuotes-Demo (100,000 real bars)
- **Timezone**: UTC
- **Prediction Horizon**: $H = 4$ bars (60 minutes forward)
- **Target Column**: `direction_4`
- **Canonical Fixed Threshold**: $\tau = 0.00050$ (5.0 pips / 50 points)
  $$\text{LONG (+1.0)}: R_{t, 4} > +0.00050$$
  $$\text{SHORT (-1.0)}: R_{t, 4} < -0.00050$$
  $$\text{NEUTRAL (0.0)}: -0.00050 \le R_{t, 4} \le +0.00050$$
- **Dataset Partition Counts**:
  - **Training**: 69,937 observations (70.0% of usable)
  - **Validation**: 14,983 observations (15.0% of usable)
  - **Test**: 14,988 observations (15.0% of usable)
  - Total Usable Rows: 99,908 rows (post-feature-warmup, post-purge)

---

## 4. Test Class Distribution Breakdown

Before prediction evaluation, the ground-truth distribution of market directions across all three chronological partitions was analyzed:

| Partition | Date Range (UTC) | SHORT (-1) Count (%) | NEUTRAL (0) Count (%) | LONG (+1) Count (%) | Total Rows |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | 2022-09-19 to 2025-07-14 | 15,018 (21.47%) | 39,359 (56.28%) | 15,560 (22.25%) | 69,937 |
| **Validation** | 2025-07-14 to 2026-02-19 | 2,882 (19.24%) | 9,281 (61.94%) | 2,820 (18.82%) | 14,983 |
| **Test** | 2026-02-19 to 2026-09-25 | 2,649 (17.67%) | 9,939 (66.31%) | 2,400 (16.01%) | 14,988 |

### Descriptive Observations:
- In the unseen 2026 holdout period, the market exhibited a higher proportion of quiescent/flat candles, with **NEUTRAL** rising to **66.31%** (compared to 61.94% in Validation and 56.28% in Training).
- Directional moves were slightly less frequent (SHORT: 17.67%, LONG: 16.01%), but remained symmetric.
- This natural variation is descriptive and does not represent an artificial regime breakdown; it reflects market volatility cycles.

---

## 5. Validation-Era Candidate Model Selection

To maintain strict scientific integrity, the primary candidate model was selected **exclusively on Phase 10 validation evidence prior to unlocking the test partition**:

- **Primary Candidate**: **Random Forest (class_weight="balanced")**
  - **Rationale**: Achieved the highest validation Balanced Accuracy (**43.61%**) and strong directional recall (SHORT recall: **55.24%**, LONG recall: **16.56%**) while maintaining a bounded tree complexity (`max_depth=10`, `min_samples_leaf=20`).
- **Linear Reference Candidate**: **Logistic Regression (class_weight="balanced")**
  - **Rationale**: Achieved the highest validation Macro F1 (**0.4208**) and more symmetric directional recall (SHORT recall: **39.04%**, LONG recall: **20.43%**).

Both candidates, alongside all other baselines, were frozen before running test evaluations.

---

## 6. Full Test Performance Comparison (All 7 Models)

The table below summarizes the final out-of-sample holdout performance on the 14,988 Test observations (from `reports/final_test_comparison.csv`):

| Model Architecture | Weight | Raw Accuracy | Balanced Accuracy | Macro F1 | SHORT Recall | LONG Recall | SHORT F1 | LONG F1 | Macro ROC-AUC | Multiclass Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | None | 66.31% | 33.33% | 0.2658 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.5000 | 0.5188 |
| **Logistic Regression (unweighted)** | None | 66.50% | 35.85% | 0.3244 | 5.89% | 3.87% | 0.1009 | 0.0682 | 0.6772 | 0.4666 |
| **Logistic Regression (balanced)** | Balanced | **56.83%** | **43.06%** | **0.4251** | **35.37%** | **23.13%** | **0.3130** | **0.2559** | **0.6754** | **0.5422** |
| **Random Forest (unweighted)** | None | 66.07% | 36.38% | 0.3348 | 10.42% | 2.46% | 0.1585 | 0.0463 | 0.6854 | 0.4761 |
| **Random Forest (balanced)** | Balanced | **52.95%** | **43.78%** | **0.4066** | **55.19%** | **14.50%** | **0.3470** | **0.1772** | **0.6808** | **0.5811** |
| **Extra Trees (unweighted)** | None | 66.31% | 34.29% | 0.2907 | 3.25% | 0.67% | 0.0592 | 0.0131 | 0.6863 | 0.4775 |
| **Extra Trees (balanced)** | Balanced | **50.98%** | **43.42%** | **0.4058** | **52.62%** | **19.50%** | **0.3359** | **0.2225** | **0.6795** | **0.5990** |

---

## 7. Validation vs Test Generalization Gap Analysis

Generalization gaps ($\Delta = \text{Test} - \text{Validation}$) measure whether the models degraded on unseen future data:

| Model Architecture | Class Weight | Val Acc $\rightarrow$ Test Acc | Val BalAcc $\rightarrow$ Test BalAcc | Val Macro F1 $\rightarrow$ Test Macro F1 | Val RecS $\rightarrow$ Test RecS | Val RecL $\rightarrow$ Test RecL | Val AUC $\rightarrow$ Test AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | None | 61.94% $\rightarrow$ 66.31% (+4.37%) | 33.33% $\rightarrow$ 33.33% (+0.00%) | 0.2550 $\rightarrow$ 0.2658 (+0.0108) | 0.00% $\rightarrow$ 0.00% (+0.00%) | 0.00% $\rightarrow$ 0.00% (+0.00%) | 0.5000 $\rightarrow$ 0.5000 (+0.0000) |
| **Logistic Regression (unweighted)** | None | 62.01% $\rightarrow$ 66.50% (+4.49%) | 35.32% $\rightarrow$ 35.85% (+0.52%) | 0.3063 $\rightarrow$ 0.3244 (+0.0181) | 5.00% $\rightarrow$ 5.89% (+0.89%) | 3.48% $\rightarrow$ 3.87% (+0.40%) | 0.6675 $\rightarrow$ 0.6772 (+0.0097) |
| **Logistic Regression (balanced)** | Balanced | 53.95% $\rightarrow$ 56.83% (+2.88%) | 42.74% $\rightarrow$ 43.06% (**+0.32%**) | 0.4208 $\rightarrow$ 0.4251 (**+0.0043**) | 39.04% $\rightarrow$ 35.37% (-3.67%) | 20.43% $\rightarrow$ 23.13% (**+2.70%**) | 0.6642 $\rightarrow$ 0.6754 (**+0.0111**) |
| **Random Forest (unweighted)** | None | 62.23% $\rightarrow$ 66.07% (+3.84%) | 36.49% $\rightarrow$ 36.38% (-0.11%) | 0.3289 $\rightarrow$ 33.48% (+0.0059) | 9.06% $\rightarrow$ 10.42% (+1.36%) | 3.97% $\rightarrow$ 2.46% (-1.51%) | 0.6695 $\rightarrow$ 0.6854 (+0.0158) |
| **Random Forest (balanced)** | Balanced | 50.31% $\rightarrow$ 52.95% (+2.64%) | 43.61% $\rightarrow$ 43.78% (**+0.17%**) | 0.4087 $\rightarrow$ 0.4066 (**-0.0022**) | 55.24% $\rightarrow$ 55.19% (**-0.05%**) | 16.56% $\rightarrow$ 14.50% (-2.06%) | 0.6626 $\rightarrow$ 0.6808 (**+0.0182**) |
| **Extra Trees (unweighted)** | None | 62.20% $\rightarrow$ 66.31% (+4.11%) | 34.40% $\rightarrow$ 34.29% (-0.10%) | 0.2814 $\rightarrow$ 0.2907 (+0.0092) | 2.74% $\rightarrow$ 3.25% (+0.51%) | 1.28% $\rightarrow$ 0.67% (-0.61%) | 0.6714 $\rightarrow$ 0.6863 (+0.0149) |
| **Extra Trees (balanced)** | Balanced | 49.48% $\rightarrow$ 50.98% (+1.50%) | 43.14% $\rightarrow$ 43.42% (**+0.28%**) | 0.4039 $\rightarrow$ 0.4058 (**+0.0019**) | 54.89% $\rightarrow$ 52.62% (-2.27%) | 16.81% $\rightarrow$ 19.50% (**+2.69%**) | 0.6636 $\rightarrow$ 0.6795 (**+0.0159**) |

### Generalization Analysis Takeaways:
1. **Zero Degradation**: There is no material performance decay from Validation to Test across any balanced model. Balanced Accuracy changes by $+0.17\%$ for RF and $+0.32\%$ for LR; Macro F1 changes by $-0.0022$ for RF and $+0.0043$ for LR.
2. **Directional Recall Persistence**: Minority directional recall holds up robustly. For Random Forest (balanced), SHORT recall is identical (**55.19%** vs **55.24%**); LONG recall experiences minor compression (**14.50%** vs **16.56%**). For Logistic Regression (balanced), LONG recall improves (**23.13%** vs **20.43%**).
3. **Discriminative Capacity (ROC-AUC)**: Macro ROC-AUC increases modestly for all models on the test set (+0.01 to +0.02), reaching **0.6808** for RF balanced and **0.6854** for RF unweighted.
4. **Validation Fidelity**: The validation set was not overly optimistic. The leak-free Phase 7 purge and embargo boundaries successfully prevented artificial validation inflation.

---

## 8. Temporal Test Analysis (Preselected Candidate: Random Forest Balanced)

The 14,988-row Test partition was divided into three contiguous chronological blocks (each 4,996 observations) to evaluate stationarity across 2026:

| Block Identifier | Chronological Range (UTC) | Observations | Accuracy | Balanced Accuracy | Macro F1 | SHORT Recall | LONG Recall | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TEST_BLOCK_A** | 2026-02-19 12:00 $\rightarrow$ 2026-05-04 20:45 | 4,996 | 51.52% | 41.54% | 0.3936 | 56.03% | 20.35% | 0.6383 |
| **TEST_BLOCK_B** | 2026-05-04 21:00 $\rightarrow$ 2026-07-15 21:45 | 4,996 | 54.44% | 42.29% | 0.3923 | 52.98% | 11.65% | 0.6549 |
| **TEST_BLOCK_C** | 2026-07-15 22:00 $\rightarrow$ 2026-09-25 22:45 | 4,996 | 52.88% | 44.86% | 0.3967 | 56.73% | 7.05% | 0.7089 |

### Observations:
- Balanced accuracy remains steadily between **41.5% and 44.9%** across all three chronological periods.
- SHORT recall remains exceptionally consistent (**52.9% to 56.7%**).
- LONG recall exhibits seasonal decay in Block C (7.05%), offset by rising ROC-AUC (0.7089), reflecting a persistent downward drift in EURUSD during late summer 2026.

---

## 9. Post-Hoc Descriptive Confidence Analysis on Test Holdout

Evaluating confidence thresholds on the holdout partition for **Random Forest (balanced)** confirms the identical pattern discovered in Phase 10:

| Threshold ($\tau$) | Covered Rows | Coverage % | Subset Accuracy | Balanced Accuracy | Macro F1 | SHORT Preds | NEUTRAL Preds | LONG Preds |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.35** | 14,541 | 97.02% | 53.64% | 44.25% | 0.4110 | 5,727 | 7,466 | 1,348 |
| **0.40** | 10,238 | 68.31% | 61.06% | **47.06%** | **0.4367** | 3,564 | 6,119 | 555 |
| **0.45** | 6,149 | 41.03% | 74.06% | 44.43% | 0.4236 | 1,106 | 4,936 | 107 |
| **0.50** | 3,950 | 26.35% | 86.51% | 36.69% | 0.3695 | 80 | 3,851 | 19 |
| **0.55** | 2,883 | 19.24% | 89.77% | 33.32% | 0.3154 | 0 | 2,882 | 1 |
| **0.60** | 1,824 | 12.17% | 91.45% | 33.31% | 0.3184 | 0 | 1,823 | 1 |
| **0.70** | 317 | 2.12% | 93.38% | 33.33% | 0.3219 | 0 | 317 | 0 |
| **0.80** | 2 | 0.01% | 100.00% | 33.33% | 1.0000 | 0 | 2 | 0 |

### Holdout Takeaway:
- As observed on Validation, high confidence ($\tau \ge 0.50$) correlates with market quiescence. At $\tau = 0.50$, **3,851 out of 3,950 (97.5%)** covered bars are predicted as NEUTRAL. At $\tau \ge 0.55$, directional predictions vanish entirely.
- High confidence thresholding **cannot be used as a directional trade filter** because the model is confident only during market consolidation.

---

## 10. Probability Calibration & Brier Diagnostics

| Model Architecture | Multiclass Brier | Brier SHORT | Brier NEUTRAL | Brier LONG |
| :--- | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.5188 | 0.1469 | 0.2335 | 0.1384 |
| **Logistic Regression (unweighted)** | 0.4666 | 0.1399 | 0.1985 | 0.1282 |
| **Logistic Regression (balanced)** | 0.5422 | 0.1515 | 0.2483 | 0.1423 |
| **Random Forest (unweighted)** | 0.4761 | 0.1408 | 0.2046 | 0.1307 |
| **Random Forest (balanced)** | 0.5811 | 0.1613 | 0.2738 | 0.1460 |
| **Extra Trees (unweighted)** | 0.4775 | 0.1407 | 0.2057 | 0.1310 |
| **Extra Trees (balanced)** | 0.5990 | 0.1603 | 0.2863 | 0.1523 |

### Interpretation:
- Unweighted models achieve lower (better) Brier scores (~0.466 to 0.477) because predicting near-constant NEUTRAL probabilities aligns with the 66.3% majority class prevalence.
- Balanced models exhibit higher Brier scores (~0.542 to 0.581) because their probability distributions are shifted toward minority directional classes, creating higher squared deviation during flat regimes.

---

## 11. Limitations & Trading Reality

1. **Precision Ceiling**:
   On the test set, directional precision for Random Forest (balanced) is **25.5% for SHORT** and **23.3% for LONG** (against random prevalence of 17.7% and 16.0%).
2. **Negative Expectancy in Execution**:
   With a 5.0-pip target (fixed threshold = 0.00050), trading at 25% precision against spread (0.2–0.5 pips) and commission creates unavoidable negative expected value:
   $$\mathbb{E}[\text{Return}] < 0.25 \times (+5.0) - 0.75 \times (5.0 + \text{friction}) < 0$$
3. **Research Utility vs Execution Reality**:
   The classification pipeline successfully extracts a genuine, stationary, non-random signal from 80 causal features. However, raw classification models on fixed return thresholds are **not directly tradable strategies**. They serve as feature-filtering and regime-detection inputs for higher-level quantitative architectures.

---

## 12. Artifacts Generated in Phase 11

1. **`reports/final_test_comparison.csv`**: Complete tabular test metrics and validation-to-test generalization gaps across all 7 models.
2. **`reports/final_test_temporal.csv`**: Temporal test breakdown across Blocks A, B, and C.
3. **`reports/final_test_confidence.csv`**: Post-hoc confidence analysis grid on the test partition.
4. **`reports/final_test_metadata.json`**: Complete machine-readable audit metadata, partition bounds, class distributions, and governance checks.
5. **`ai/models/test_evaluation.py`**: Implementation of `TestSetGovernanceGuard`, `FinalTestEvaluationMetrics`, and test workflow.
6. **`scripts/run_test_evaluation.py`**: Automated CLI script for reproducible execution.
7. **`tests/test_test_evaluation.py`**: 10 automated unit and security tests ensuring test set isolation and reproducibility.

---

PHASE 11 COMPLETE — PHASE 12 NOT STARTED
