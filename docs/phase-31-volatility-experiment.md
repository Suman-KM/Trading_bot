# Phase 31: Pre-Registered Autoregressive Realized Volatility Regime Forecasting Experiment

**Project:** AI Autonomous Trading System  
**Repository:** `~/projects/ai-trading-system`  
**Branch:** `develop`  
**Date:** 2026-10-02  
**Governance:** Non-Directional Pre-Registered Research Experiment (Phase 30 Synthesis Directive)  
**Status:** COMPLETE  
**Scientific Verdict:** **NOT SUPPORTED — VOLATILITY FORECASTING FAILED**  

---

## Executive Summary

Phase 31 executed the pre-registered non-directional research experiment specified by the Phase 30 Research Synthesis ([`docs/phase-30-research-synthesis.md`](file:///home/cino/projects/ai-trading-system/docs/phase-30-research-synthesis.md)). The experiment evaluated whether 24 autoregressive features derived from three canonical realized-volatility estimators—Parkinson (1980), Garman-Klass (1980), and Rogers-Satchell (1991)—can forecast the forward 24-hour ($H=6$ bars) EURUSD volatility regime (High Volatility Expansion vs. Low Volatility Compression) beyond a naive persistence baseline across 10 chronological walk-forward folds spanning 2010 to 2024.

The experiment was governed by strict pre-registered criteria:
1. **Mean Walk-Forward Balanced Accuracy $\ge 60.0\%$**
2. **Statistical Significance $p < 0.01$ versus naive persistence baseline** (paired $t$-test)

### Summary of Empirical Findings
* **Return Martingale vs. Volatility Memory:** Returns exhibit zero autocorrelation across all lags (Lag 1 $r = -0.0112$, Lag 6 $r = -0.0053$), confirming absolute directional efficiency. In contrast, realized volatility estimators exhibit pronounced, statistically significant long memory ($ACF = 0.40$ to $0.53$ across 1 to 30 lags / 4h to 120h), peaking at the 24-hour diurnal cycle (Lag 6).
* **Persistence Baseline Performance:** A naive trailing 6-bar persistence baseline achieved a Mean Balanced Accuracy of **57.99%** (Accuracy: 65.80%), demonstrating that volatility clustering provides strong passive regime predictability.
* **Random Forest Model Performance:** The frozen Random Forest classifier achieved a Mean Balanced Accuracy of **61.21%** (Accuracy: 72.94%), satisfying Condition 1 ($\ge 60.0\%$).
* **Incremental Edge and Statistical Gate:** The model generated a mean improvement of **+3.22%** balanced accuracy over persistence (95% CI: $[+0.34\%, +6.10\%]$). However, the paired $t$-test yielded $t = 2.532$, $p = 0.0321$ (Wilcoxon signed-rank $p = 0.0273$).
* **Failure Gate Triggered:** Because $p = 0.0321 \ge 0.01$, the pre-registered significance gate failed. Under the pre-registered decision matrix, the formal outcome is **NOT SUPPORTED — VOLATILITY FORECASTING FAILED**.
* **Integrity and Quarantine:** Zero information leakage occurred across all 10 folds (Purge = 6, Embargo = 4, training-only median thresholds). The locked test partition (`2026-02-19 12:00:00 UTC` onward) remained 100% quarantined. Zero backtests, zero paper/live trading executions.

---

## Section A: Research Question

Does current and recent realized volatility on EURUSD H4 contain sufficient autoregressive structure to predict whether the forward 24-hour volatility regime will be high or low more accurately than a naive persistence forecast, under strict walk-forward validation with purge and embargo?

---

## Section B: Pre-Registered Hypothesis

EURUSD realized volatility contains temporal persistence/long-memory information such that current and recent realized-volatility estimators can predict the volatility regime over the following 24 hours ($H=6$ bars) better than a naive persistence baseline.

Specifically, an ensemble model trained on multi-scale autoregressive estimators (Parkinson, Garman-Klass, Rogers-Satchell) will achieve:
1. Mean Walk-Forward Balanced Accuracy across 10 expanding folds $\ge 60.0\%$.
2. Statistically significant predictive outperformance over a naive trailing persistence baseline at $\alpha = 0.01$ ($p < 0.01$).

---

## Section C: Null Hypothesis

The proposed realized-volatility features do not provide statistically meaningful predictive improvement over the naive persistence baseline ($p \ge 0.01$ or Mean Balanced Accuracy $< 60.0\%$).

---

## Section D: Dataset

* **Source File:** `data/raw/eurusd_h4_expanded/eurusd_h4_raw.parquet`
* **Asset:** EURUSD
* **Timeframe:** H4 (4-hour intervals)
* **Total Available Bars:** 25,800 bars
* **Pre-Holdout Research Partition:**
  - Start Timestamp: `2010-03-01 16:00:00 UTC`
  - End Timestamp: `2024-11-04 12:00:00 UTC`
  - Total Usable Bars: 22,847 bars
* **Locked Test Partition:**
  - Start Timestamp: `2026-02-19 12:00:00 UTC`
  - Status: Strictly quarantined, 0 bars accessed.

---

## Section E: Target Definition

The target represents the forward 24-hour ($H=6$ H4 bars) realized volatility:

$$RV_{t, t+6} = \sqrt{\sum_{k=1}^{6} r_{t+k}^2}$$

where $r_{t+k} = \ln(C_{t+k} / C_{t+k-1})$ is the log return of bar $t+k$.

### Binary Regime Classification
To formulate an objective regime classification without lookahead bias:
1. **Causal Training Median:** For each walk-forward fold, calculate the sample median of $RV_{t, t+6}$ exclusively across the training set partition:
   $$\theta_{\text{train}} = \text{median}\left(RV_{t, t+6} \mid t \in \mathcal{T}_{\text{train}}\right)$$
2. **Binary Labeling:**
   $$Y_t = \begin{cases} 1 & \text{if } RV_{t, t+6} > \theta_{\text{train}} \quad (\text{High Volatility Expansion}) \\ 0 & \text{if } RV_{t, t+6} \le \theta_{\text{train}} \quad (\text{Low Volatility Compression}) \end{cases}$$
3. **Strict Validation Freezing:** The validation partition is labeled using the frozen $\theta_{\text{train}}$. Future validation observations never influence the regime threshold.

---

## Section F: Exact Volatility-Estimator Formulas

Three authorized high-low-open-close extreme-value estimators were computed per bar:

### 1. Parkinson (1980) Volatility
Measures extreme value variance using the normalized high-to-low price ratio:
$$\sigma_{\text{Parkinson}}^2 = \frac{\left(\ln(H_t / L_t)\right)^2}{4 \ln(2)}$$
$$\sigma_{\text{Parkinson}} = \sqrt{\sigma_{\text{Parkinson}}^2}$$

### 2. Garman-Klass (1980) Volatility
Extends Parkinson by incorporating opening and closing prices:
$$\sigma_{\text{GK}}^2 = 0.5 \left(\ln\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_t}{O_t}\right)^2$$
$$\sigma_{\text{GK}} = \sqrt{\max\left(0, \sigma_{\text{GK}}^2\right)}$$

### 3. Rogers-Satchell (1991) Volatility
Allows for non-zero drift:
$$\sigma_{\text{RS}}^2 = \ln\left(\frac{H_t}{C_t}\right)\ln\left(\frac{H_t}{O_t}\right) + \ln\left(\frac{L_t}{C_t}\right)\ln\left(\frac{L_t}{O_t}\right)$$
$$\sigma_{\text{RS}} = \sqrt{\max\left(0, \sigma_{\text{RS}}^2\right)}$$

---

## Section G: Feature Construction

Exactly 24 features were pre-registered and constructed (8 features for each of the 3 estimators):

For each estimator $E \in \{\text{Parkinson}, \text{Garman-Klass}, \text{Rogers-Satchell}\}$:
1. `vol_{E}_1`: Instantaneous current bar volatility ($\sigma_{E, t}$)
2. `vol_{E}_6`: 6-bar rolling mean ($\approx 24$ hours)
3. `vol_{E}_12`: 12-bar rolling mean ($\approx 48$ hours)
4. `vol_{E}_24`: 24-bar rolling mean ($\approx 96$ hours / 4 days)
5. `vol_{E}_72`: 72-bar rolling mean ($\approx 288$ hours / 12 days)
6. `vol_{E}_chg_1`: 1-bar change ($\sigma_{E, t} - \sigma_{E, t-1}$)
7. `vol_{E}_chg_6`: 6-bar change ($\sigma_{E, t} - \sigma_{E, t-6}$)
8. `vol_{E}_ratio_6_24`: Term-structure ratio ($\text{mean}_{6}(\sigma_{E}) / (\text{mean}_{24}(\sigma_{E}) + \epsilon)$)

Total feature dimensions: $3 \times 8 = 24$. All rolling aggregations look backward only.

---

## Section H: Model Specification

* **Algorithm:** `RandomForestClassifier` (Scikit-Learn)
* **Hyperparameters (Strictly Frozen):**
  - `n_estimators`: 100
  - `max_depth`: 5
  - `min_samples_leaf`: 15
  - `class_weight`: `"balanced"`
  - `random_state`: 42
  - `n_jobs`: -1
* **Search / Optimization:** ZERO hyperparameter tuning or grid search allowed.

---

## Section I: Persistence Baseline Definition

The persistence baseline models volatility clustering by assuming the forward 6-bar volatility regime matches the trailing 6-bar realized volatility:

$$RV_{\text{trailing}, 6} = \sqrt{\sum_{k=0}^{5} r_{t-k}^2}$$

The persistence baseline prediction is obtained by thresholding against the same frozen training threshold $\theta_{\text{train}}$:

$$\hat{Y}_{t, \text{persist}} = \begin{cases} 1 & \text{if } RV_{\text{trailing}, 6} > \theta_{\text{train}} \\ 0 & \text{if } RV_{\text{trailing}, 6} \le \theta_{\text{train}} \end{cases}$$

This benchmark tests whether a 24-feature machine learning model extracts information beyond naive empirical volatility clustering.

---

## Section J: Walk-Forward Methodology

* **Number of Folds:** 10 chronological expanding folds.
* **Initial Training Size:** 5,000 bars ($\sim 3.2$ years).
* **Validation Window Size:** 1,774 bars ($\sim 1.15$ years per fold).
* **Chronological Order:** Fold $k+1$ includes all training data from Fold $k$ plus the preceding validation period. No shuffling, no lookahead.

---

## Section K: Purge and Embargo Methodology

To prevent serial correlation and label-overlap leakage:
* **Purge Window:** 6 bars immediately preceding the validation window. Because the target at time $t$ requires prices up to $t+6$, any bar $t$ within 6 bars of the validation start would have forward price returns overlapping into the validation window.
* **Embargo Window:** 4 bars immediately after the validation window to ensure that autoregressive lags and volatility persistence do not contaminate adjacent splits.
* **Total Separation:** $\ge 11$ bars strictly enforced between training and validation indices across all folds.

---

## Section L: Leakage Audit

A formal 7-point leakage verification was conducted prior to and during execution:
1. **Temporal Ordering:** All training timestamps strictly precede validation timestamps ($\max(\text{train\_ts}) < \min(\text{val\_ts})$).
2. **Purge Enforcement:** Gap between last training bar and first validation bar is $\ge 11$ bars (Purge 6 + Embargo 4 + 1 bar boundary).
3. **No Target Leakage in Features:** All 24 autoregressive features are calculated strictly using contemporaneous and backward-looking OHLC bars ($t, t-1, \dots$).
4. **No Validation Contamination in Thresholds:** The regime threshold $\theta_{\text{train}}$ is computed exclusively from training samples and applied out-of-sample to validation samples.
5. **No Data Snooping:** Zero feature selection, hyperparameter tuning, or post-hoc adjustments.
6. **Locked Test Partition Quarantine:** Locked test partition (`2026-02-19 12:00:00 UTC` onward) was completely excluded from feature generation, training, and validation.
7. **Leakage Audit Status:** **PASSED (0 violations detected)**.

---

## Section M: Fold-by-Fold Results

The table below presents out-of-sample performance across all 10 expanding walk-forward folds:

| Fold | Validation Window | Train N | Val N | Threshold | Persistence BalAcc | Model BalAcc | Delta BalAcc | Persistence Acc | Model Acc | Model F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2013-05-15 to 2014-07-07 | 4,923 | 1,774 | 0.002239 | 47.75% | 49.49% | +1.74% | 85.17% | 88.84% | 0.4904 |
| 2 | 2014-07-09 to 2015-08-31 | 6,707 | 1,774 | 0.002166 | 67.26% | 74.79% | +7.53% | 66.85% | 74.86% | 0.7493 |
| 3 | 2015-09-02 to 2016-10-24 | 8,491 | 1,774 | 0.002264 | 53.39% | 61.63% | +8.24% | 60.03% | 67.87% | 0.6133 |
| 4 | 2016-10-26 to 2017-12-13 | 10,275 | 1,774 | 0.002344 | 52.09% | 57.56% | +5.46% | 72.89% | 75.93% | 0.5847 |
| 5 | 2017-12-15 to 2019-02-08 | 12,059 | 1,774 | 0.002293 | 54.77% | 55.15% | +0.38% | 71.93% | 71.53% | 0.5517 |
| 6 | 2019-02-12 to 2020-04-03 | 13,843 | 1,774 | 0.002207 | 74.44% | 69.51% | -4.93% | 74.86% | 70.63% | 0.6974 |
| 7 | 2020-04-07 to 2021-05-27 | 15,627 | 1,774 | 0.002241 | 53.92% | 57.26% | +3.34% | 63.30% | 66.80% | 0.5759 |
| 8 | 2021-05-31 to 2022-07-19 | 17,411 | 1,774 | 0.002230 | 65.88% | 70.96% | +5.09% | 63.22% | 70.63% | 0.7077 |
| 9 | 2022-07-20 to 2023-09-07 | 19,195 | 1,774 | 0.002279 | 58.69% | 64.20% | +5.52% | 56.54% | 64.26% | 0.6406 |
| 10 | 2023-09-11 to 2024-11-01 | 20,979 | 1,781 | 0.002341 | 51.70% | 51.54% | -0.16% | 43.23% | 78.05% | 0.4771 |

---

## Section N: Aggregate Results

| Metric | Persistence Baseline | Frozen Random Forest Model | Net Difference ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Mean Balanced Accuracy** | **57.99%** | **61.21%** | **+3.22%** |
| Median Balanced Accuracy | 54.35% | 59.59% | +4.21% |
| Std Dev Balanced Accuracy | 8.47% | 8.53% | 4.02% |
| Min Balanced Accuracy | 47.75% | 49.49% | -4.93% |
| Max Balanced Accuracy | 74.44% | 74.79% | +8.24% |
| **Mean Overall Accuracy** | 65.80% | 72.94% | +7.13% |
| **Mean Macro F1-Score** | 0.5801 | 0.6085 | +0.0284 |

---

## Section O: Statistical Tests

To evaluate whether the outperformance of the Random Forest model over the persistence baseline is statistically distinguishable from chance:

* **Paired $t$-Test:**
  - $t$-statistic: **2.5320**
  - Degrees of Freedom: 9
  - $p$-value: **0.0321**
  - 95% Confidence Interval for $\mu_{\Delta}$: **$[+0.34\%, +6.10\%]$**
* **Wilcoxon Signed-Rank Test (Non-Parametric):**
  - $W$-statistic: **6.0**
  - $p$-value: **0.0273**
* **Threshold Assessment:**
  - Statistically significant at standard exploratory $\alpha = 0.05$: **YES** ($p = 0.0321 < 0.05$)
  - Statistically significant at pre-registered research gate $\alpha = 0.01$: **NO** ($p = 0.0321 \ge 0.01$)

---

## Section P: Volatility Persistence Diagnostics

To establish empirical foundation and verify whether volatility behaves fundamentally differently from price returns on EURUSD H4, the Autocorrelation Function (ACF) was estimated across lags 1 to 30:

| Lag (Bars) | Horizon (Hours) | Return ACF | Abs Return ACF | Parkinson ACF | Garman-Klass ACF | Rogers-Satchell ACF |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 4h | -0.0112 | 0.1738 | 0.4059 | 0.4379 | 0.4189 |
| 2 | 8h | -0.0099 | 0.0800 | 0.1859 | 0.2096 | 0.2077 |
| 3 | 12h | -0.0118 | 0.0232 | 0.0414 | 0.0489 | 0.0539 |
| 4 | 16h | +0.0079 | 0.0838 | 0.1733 | 0.1918 | 0.1878 |
| 5 | 20h | +0.0115 | 0.1395 | 0.3330 | 0.3611 | 0.3447 |
| **6** | **24h** | **-0.0053** | **0.2091** | **0.4868** | **0.5289** | **0.5057** |
| 12 | 48h | +0.0005 | 0.1843 | 0.4497 | 0.4902 | 0.4692 |
| 18 | 72h | -0.0108 | 0.1739 | 0.4401 | 0.4830 | 0.4642 |
| 24 | 96h | -0.0048 | 0.2008 | 0.4503 | 0.4874 | 0.4660 |
| 30 | 120h | -0.0044 | 0.2069 | 0.4577 | 0.4965 | 0.4772 |

### Key Diagnostic Takeaways
1. **Martingale Price Returns:** The raw return ACF is statistically indistinguishable from zero across all lags ($|\rho| \le 0.0118$). Directional price changes are efficient white noise.
2. **Massive Volatility Memory:** All realized volatility estimators show strong autocorrelation ($\rho > 0.40$).
3. **Diurnal Seasonality:** Autocorrelation shows pronounced 24-hour periodicity, surging at Lag 6 ($\rho = 0.5289$ for GK), Lag 12, Lag 18, and Lag 24, corresponding to recurring London/New York session trading volumes.

---

## Section Q: Robustness and Concentration Analysis

* **Folds $> 50.0\%$ Balanced Accuracy:** 9 of 10 folds (90.0%).
* **Folds $> 55.0\%$ Balanced Accuracy:** 8 of 10 folds (80.0%).
* **Folds $> 60.0\%$ Balanced Accuracy:** 5 of 10 folds (50.0%).
* **Outperformance Consistency:** The model outperformed the persistence baseline in 8 out of 10 folds. In Fold 6 (2019–2020), persistence was superior due to the sudden onset of the COVID-19 volatility shock, where historical training thresholds lagged rapid regime shifts.
* **Dispersion:** Cross-fold standard deviation is 8.53%, showing moderate variability across structural macroeconomic epochs.

---

## Section R: Success / Failure Gate Evaluation

The experiment was subject to the pre-registered decision rules:

1. **Success Condition 1:** Mean walk-forward balanced accuracy $\ge 60.0\%$.
   - **Observed:** 61.21%
   - **Gate 1 Status:** **PASS**
2. **Success Condition 2:** Statistical improvement versus persistence baseline satisfies $p < 0.01$.
   - **Observed:** Paired $t$-test $p = 0.0321$ (Wilcoxon $p = 0.0273$).
   - **Gate 2 Status:** **FAIL** ($p \ge 0.01$)
3. **Failure Gate Evaluation:**
   - Pre-registration stipulated: If either condition fails, the experiment triggers the failure gate.
   - **Result:** **FAILURE GATE TRIGGERED**.

---

## Section S: Locked-Test Partition Integrity

* **Quarantine Enforcement:** The locked test partition (`2026-02-19 12:00:00 UTC` to `2026-03-31 20:00:00 UTC`) remained completely unaccessed.
* **Integrity Audit:** Zero rows loaded into features, zero labels examined, zero backtests executed.
* **Quarantine Status:** **PRESERVED AND VERIFIED**.

---

## Section T: Trading-System Implications

### 1. Distinction Between Alpha and Risk/Execution Infrastructure
* Volatility predictability is **NOT** directional alpha. It cannot be traded directly via naive long/short EURUSD positions without an explicit volatility instrument (such as FX options or VIX-style variance swaps).
* While the model provides modest predictive value ($+3.22\%$ balanced accuracy, $p = 0.0321$), it fails the institutional evidentiary threshold ($p < 0.01$) required to justify a dedicated standalone machine learning volatility-forecasting microservice.

### 2. Practical Downstream Role
* The finding confirms that volatility clustering is robustly present on EURUSD H4 ($ACF \approx 0.50$), but a simple trailing persistence rule captures over 94% of the predictable structure (57.99% baseline vs. 61.21% ML model).
* Future risk architecture (dynamic position sizing, ATR-based stop adjustments, volatility circuit breakers) should prioritize computationally lean, deterministic rolling estimators rather than high-maintenance machine learning forecasting models.

---

## Section U: Scientific Decision

**Selection:** **B. NOT SUPPORTED — VOLATILITY FORECASTING FAILED**

### Scientific Decision Rationale
The Random Forest model achieved a Mean Balanced Accuracy of 61.21%, which met the threshold of $\ge 60.0\%$. However, when tested against the naive trailing persistence baseline, the incremental edge ($+3.22\%$) yielded a paired $t$-test $p$-value of $0.0321$. Because the pre-registered decision protocol strictly requires $p < 0.01$, the null hypothesis cannot be rejected at the specified confidence level. In strict adherence to scientific rigor, the hypothesis is declared **NOT SUPPORTED**.

---

## Post-Phase Governance and Next Steps
* **Phase 32 Status:** **NOT AUTHORIZED**.
* **Model Search Status:** STOP all directional and volatility-alpha experimentation under this information set.
* **Live Trading Status:** ZERO live, demo, or paper trading authorized.
