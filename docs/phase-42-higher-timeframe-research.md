# Phase 42 — Higher-Timeframe H1/H4 Swing & Regime Research

**Document Status:** Complete & Audited  
**Date (UTC):** 2026-10-06  
**Author:** Quantitative Trading & AI Research Engineering  
**System:** AI Trading System — EURUSD  
**Branch:** `develop`  
**Latest Reference Commit:** `cce4a04` (Phase 41 closeout)  
**Scientific Verdict:** **NOT SUPPORTED / RESEARCH PAUSE**  

---

## 1. Executive Summary

Phase 41 established a formal research pause for EURUSD M15 retail technical directional forecasting after proving that the frozen strategy's lack of signals was driven by fundamental signal limitation and near-uniform prediction entropy (94.7% of uniform random noise).

Phase 42 investigated a materially distinct hypothesis: **Higher-Timeframe EURUSD H1/H4 Swing & Regime Modeling**. The primary objective was to determine scientifically whether moving to slower frequencies (H1: 12-hour swing, H4: 24-hour swing) and conditioning predictions on a small, predefined set of structural market regimes (trend state, volatility state, price structure, momentum) yields persistent, economically viable directional information that overcomes transaction costs.

### Key Empirical Findings:
1. **Walk-Forward Performance:**
   - **H1 Regime Random Forest:** Mean walk-forward balanced accuracy across 10 expanding-window folds was **50.43%** (median: 50.30%, std: 4.01%), with 4 out of 10 folds falling below 50.0%.
   - **H4 Regime Random Forest:** Mean walk-forward balanced accuracy across 10 expanding-window folds was **49.63%** (median: 50.13%, std: 2.68%), with 5 out of 10 folds falling below 50.0%. It failed to beat the naive persistence baseline (52.07%).
2. **Untouched Quarantined Holdout:**
   - On the strictly quarantined holdout partition (February 19, 2026 to September 2026), H1 Random Forest achieved **46.93%** balanced accuracy ($N=1,550$), and H4 Random Forest achieved **51.29%** balanced accuracy ($N=373$).
3. **Statistical Significance:**
   - Paired difference testing against naive persistence showed no statistically significant edge ($p = 0.8503$ for H1 RF, $p = 0.1604$ for H4 RF).
4. **Economic Reality:**
   - After realistic transaction friction (1.5 pips on H1, 1.8 pips on H4), net mathematical expectancy was strictly negative: **-1.24 pips/trade on H1** and **-1.51 pips/trade on H4**.
5. **Gate Evaluation:**
   - All 5 primary success gates failed. Failure Gate 1 was explicitly triggered.
6. **Verdict:**
   - **NOT SUPPORTED**. The research pause remains in effect. **PHASE 43 IS NOT AUTHORIZED**.

---

## 2. Research Question

Does EURUSD contain a more persistent and economically useful directional or regime signal at H1 or H4 horizons than at the failed M15 horizon?

Specifically:
- **H1 Track:** Can a 12-hour forward swing target (`direction_vol_12`) be predicted using 12 point-in-time causal regime features across 10 chronological walk-forward folds?
- **H4 Track:** Can a 24-hour forward swing target (`direction_vol_6`) be predicted using 12 point-in-time causal regime features across 16.5 years of expanded market history?

---

## 3. Prior H4 Research

A comprehensive audit of historical higher-timeframe experiments in the repository:
1. **Phase 15 (Dual-Track Research):**
   - Identified intraday M15 transaction friction consuming 28.2% of the average 1-hour move.
   - Identified preliminary H4 candidate (`direction_vol_8`, 32-hour horizon) showing 57.54% balanced accuracy on a single validation slice.
2. **Phase 16 (H4 Confirmation):**
   - 5-fold walk-forward validation (September 2022 to February 2026, 5,324 bars).
   - Random Forest candidate achieved 56.05% mean balanced accuracy, but Fold 1 dropped to 48.90%, revealing regime vulnerability.
3. **Phase 17 (Historical Data Expansion):**
   - Expanded H4 history to 16.5 years (2010 to 2026, 25,800 bars, 10 folds).
   - Candidate performance collapsed from 56.05% down to **51.62%** mean balanced accuracy, with 4 of 10 folds falling below 50.0%.
4. **Phase 25 (Macro Yield Spread Integration):**
   - Tested US/German 2Y government bond yield differentials as exogenous H4 features.
   - Concluded **NOT SUPPORTED**: Macro features failed to improve directional accuracy over technical baselines.
5. **Phase 31 (Realized Volatility Forecasting):**
   - Proved realized volatility magnitude has strong temporal persistence ($R^2=0.42$), but confirmed **zero directional sign predictability**.

---

## 4. Why This Experiment Is Different

Phase 42 is not a re-run of the failed Phase 16/17 experiments:

| Design Dimension | Old H4 Experiment (Phases 15–17) | New Phase 42 Experiment |
| :--- | :--- | :--- |
| **Core Hypothesis** | Directional momentum continuation ($H=8$, 32h) | Structural Regime & Swing Persistence |
| **Timeframes** | H4 only | Dual-track: H1 (12h swing) & H4 (24h swing) |
| **Feature Space** | 41 generic swing technical indicators | Exactly 12 predefined regime features |
| **Feature Mining** | Unconstrained indicator inclusion | Zero mining; 3 features each from 4 predefined categories |
| **Regime Conditioning** | Post-hoc descriptive slicing | Point-in-time deterministic regime state tracking |
| **Target Horizons** | Fixed $H=8$ bars (32 hours) | H1: $H=12$ bars (12h); H4: $H=6$ bars (24h) |
| **Target Scaling** | Volatility-scaled dynamic direction | Volatility-scaled with verified deadband exclusion |
| **Model Families** | Random Forest only | Random Forest & Regularized Logistic Regression |
| **Cost Hurdle** | Unverified / zero-friction assumptions | Explicit roundtrip friction (1.5 pips H1, 1.8 pips H4) |

---

## 5. Data Provenance

Research strictly enforced temporal isolation between research and holdout partitions:
- **Research Boundary:** `2010-03-01 16:00:00 UTC` to `2026-02-19 10:45:00 UTC`.
- **Holdout Boundary:** `2026-02-19 12:00:00 UTC` to `2026-09-29 20:00:00 UTC` (**Quarantined & untouched until frozen evaluation**).

### Dataset Specifications:
1. **EURUSD H1:**
   - **Source:** Canonical M15 historical feed (`data/processed/eurusd_m15/eurusd_m15_processed.parquet`) aggregated into completed 1-hour candles via `aggregate_m15_to_h1()`.
   - **Total Bars:** 25,011 bars (`2022-09-16 09:00:00 UTC` to `2026-09-25 23:00:00 UTC`).
   - **Research Partition:** 21,261 bars (~3.5 years).
   - **Holdout Partition:** 3,750 bars.
   - **Limitation:** Pre-September 2022 M15 ticks are unavailable due to broker terminal buffer limits (as established in Phase 17).
2. **EURUSD H4:**
   - **Source:** Canonical expanded historical feed (`data/processed/eurusd_h4_expanded/eurusd_h4_processed.parquet`).
   - **Total Bars:** 25,800 bars (`2010-03-01 16:00:00 UTC` to `2026-09-29 20:00:00 UTC`).
   - **Research Partition:** 24,849 bars (16.0 years).
   - **Holdout Partition:** 951 bars.

---

## 6. Target Definition

Two target formulations were finalized before model evaluation:
1. **Candidate A: H1 Swing Target (`direction_vol_12`):**
   - **Horizon:** $H=12$ bars (12 completed hours).
   - **Formula:**
     $$R_{12, t} = \frac{C_{t+12} - C_t}{C_t}$$
     $$\text{Threshold}_{12, t} = 0.5 \times \sqrt{12} \times \frac{\text{ATR}_{14, t}}{C_t}$$
     $$y_{12, t} = \begin{cases} +1.0 & \text{if } R_{12, t} > \text{Threshold}_{12, t} \\ -1.0 & \text{if } R_{12, t} < -\text{Threshold}_{12, t} \\ \text{NaN} & \text{otherwise (deadband excluded)} \end{cases}$$
   - **Valid Labeled Samples:** 10,440 bars in research set (5,180 Long, 5,260 Short; 49.6% / 50.4% balance).

2. **Candidate B: H4 Swing Target (`direction_vol_6`):**
   - **Horizon:** $H=6$ bars (24 completed hours / 1 full trading day).
   - **Formula:**
     $$R_{6, t} = \frac{C_{t+6} - C_t}{C_t}$$
     $$\text{Threshold}_{6, t} = 0.5 \times \sqrt{6} \times \frac{\text{ATR}_{14, t}}{C_t}$$
     $$y_{6, t} = \begin{cases} +1.0 & \text{if } R_{6, t} > \text{Threshold}_{6, t} \\ -1.0 & \text{if } R_{6, t} < -\text{Threshold}_{6, t} \\ \text{NaN} & \text{otherwise (deadband excluded)} \end{cases}$$
   - **Valid Labeled Samples:** 11,445 bars in research set (5,624 Long, 5,821 Short; 49.1% / 50.9% balance).

---

## 7. Feature Definition

A strictly bounded, predefined feature set of exactly **12 point-in-time causal features** was constructed:

### 1. Trend (3 Features):
- `norm_slope_20`: Vectorized linear regression slope of close over past 20 bars divided by current close.
- `ema_dist_50`: Distance from 50-period exponential moving average: $(C - \text{EMA}_{50}) / C$.
- `trend_persistence_10`: Proportion of past 10 bars where $\text{EMA}_{10} > \text{EMA}_{30}$, scaled to $[-1.0, 1.0]$.

### 2. Volatility (3 Features):
- `atr_norm`: Normalized 14-period True Range: $\text{ATR}_{14} / C$.
- `realized_vol_20`: Rolling 20-bar standard deviation of log returns annualized by $\sqrt{24}$ (H1) or $\sqrt{6}$ (H4).
- `vol_percentile_100`: Rolling 100-bar min-max percentile rank of `atr_norm` bounded in $[0.0, 1.0]$.

### 3. Structure (3 Features):
- `range_pos_20`: Relative position within 20-bar rolling Donchian channel: $(C - L_{20}) / (H_{20} - L_{20})$.
- `dist_high_20`: Normalized distance from 20-bar rolling high: $(H_{20} - C) / C$.
- `breakout_state_20`: Causal breakout state: $+1.0$ if $C \ge H_{20, t-1}$, $-1.0$ if $C \le L_{20, t-1}$, $0.0$ otherwise.

### 4. Momentum (3 Features):
- `return_bounded_4`: 4-bar percentage return normalized by `atr_norm`, clipped to $[-3.0, 3.0]$.
- `return_bounded_8`: 8-bar percentage return normalized by $\text{ATR} \times \sqrt{2}$, clipped to $[-3.0, 3.0]$.
- `directional_persistence_8`: Mean sign of bar-by-bar returns over past 8 bars in $[-1.0, 1.0]$.

---

## 8. Regime Definition

Four transparent, deterministic, mutually exclusive market regimes were defined:
1. `LOW_VOL_TREND`: `atr_norm <= median(100)` and `abs(EMA10 - EMA30) / ATR > 0.5`
2. `HIGH_VOL_TREND`: `atr_norm > median(100)` and `abs(EMA10 - EMA30) / ATR > 0.5`
3. `LOW_VOL_RANGE`: `atr_norm <= median(100)` and `abs(EMA10 - EMA30) / ATR <= 0.5`
4. `HIGH_VOL_RANGE`: `atr_norm > median(100)` and `abs(EMA10 - EMA30) / ATR <= 0.5`

---

## 9. Baselines

1. **Naive Persistence Baseline:**  
   Predicts future $H$-bar direction based on the realized return sign of the preceding $H$ bars:
   $$\hat{y}_{t} = \text{sign}\left(\frac{C_t - C_{t-H}}{C_{t-H}}\right)$$
2. **Majority Class Baseline:**  
   Predicts the empirical majority class from the expanding training partition.

---

## 10. Model Configuration

Two simple, reproducible model families were benchmarked:
1. **Random Forest Classifier:**
   - `n_estimators=100`, `max_depth=5`, `min_samples_leaf=10`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.
2. **Logistic Regression (L2 Regularized):**
   - `C=1.0`, `class_weight='balanced'`, `random_state=42`, `max_iter=1000`.
   - Feature scaling via `StandardScaler` fitted strictly on training data for each fold.

---

## 11. Walk-Forward Methodology

An expanding-window chronological walk-forward validation design across 10 folds was implemented:
- **Initial Training Window:** 40% of research data.
- **Validation Step:** Remaining 60% partitioned equally into 10 sequential out-of-sample slices.
- **Purge Gap:** Exactly $H$ bars (12 bars for H1, 6 bars for H4) dropped from training partition tail prior to each validation fold.
- **Validation Tail Purge:** Final $H$ bars of each validation fold excluded from metric evaluation.
- **Zero Shuffle:** 100% chronological order preserved.

---

## 12. Holdout Methodology

- **Holdout Data:** `2026-02-19 12:00:00 UTC` onward.
- **Quarantine Guarantee:** Held-out bars were strictly excluded from feature scaling, target construction, fold evaluation, and model training.
- **Evaluation:** Evaluated exactly once on the frozen models fitted across the entire research partition.

---

## 13. Statistical Results

### 13.1 Comprehensive Experiment Comparison Table (Section 25 Required Table)

| Experiment | Timeframe | Target | Features | Mean WF Bal Acc | Median WF Bal Acc | Fold Std | % Folds > Baseline | Holdout Bal Acc | p-value (vs Base) | Economic Expectancy | Cost Robustness | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive H1** | H1 | `direction_vol_12` | Persistence | 50.00% | 49.77% | 3.53% | N/A (Baseline) | 49.62% | N/A | -1.24 pips | FAIL | NOT SUPPORTED |
| **Regime H1 (RF)** | H1 | `direction_vol_12` | 12 Regimes | **50.43%** | **50.30%** | **4.01%** | 60.0% (6/10) | **46.93%** | $p = 0.8503$ | -1.24 pips | FAIL | **NOT SUPPORTED** |
| **Regime H1 (LR)** | H1 | `direction_vol_12` | 12 Regimes | 52.40% | 50.33% | 5.29% | 60.0% (6/10) | 45.62% | $p = 0.2874$ | -1.38 pips | FAIL | NOT SUPPORTED |
| **Naive H4** | H4 | `direction_vol_6` | Persistence | 52.07% | 51.04% | 2.85% | N/A (Baseline) | 50.84% | N/A | -1.51 pips | FAIL | NOT SUPPORTED |
| **Regime H4 (RF)** | H4 | `direction_vol_6` | 12 Regimes | **49.63%** | **50.13%** | **2.68%** | 30.0% (3/10) | **51.29%** | $p = 0.1604$ | -1.51 pips | FAIL | **NOT SUPPORTED** |
| **Regime H4 (LR)** | H4 | `direction_vol_6` | 12 Regimes | 49.51% | 49.23% | 3.67% | 30.0% (3/10) | 43.83% | $p = 0.1258$ | -1.72 pips | FAIL | NOT SUPPORTED |

### 13.2 Fold-by-Fold Breakdown (H1 Regime Random Forest)
- Fold 1 (Jan 2024 - Apr 2024): 48.79%
- Fold 2 (Apr 2024 - Jun 2024): 51.93%
- Fold 3 (Jun 2024 - Sep 2024): 43.97%
- Fold 4 (Sep 2024 - Nov 2024): 45.10%
- Fold 5 (Nov 2024 - Jan 2025): 55.70%
- Fold 6 (Jan 2025 - Apr 2025): 49.64%
- Fold 7 (Apr 2025 - Jun 2025): 50.63%
- Fold 8 (Jun 2025 - Sep 2025): 50.04%
- Fold 9 (Sep 2025 - Nov 2025): 57.97%
- Fold 10 (Nov 2025 - Feb 2026): 50.56%

---

## 14. Economic Results

A realistic execution simulation was conducted:
- Execution at the open of bar $t+1$ following completed bar $t$.
- Fixed holding period of $H$ bars.
- Realistic roundtrip friction:
  - **H1:** 1.5 pips ($0.00015$) spread + commission + slippage.
  - **H4:** 1.8 pips ($0.00018$) spread + commission + slippage.

### Economic Metrics:
- **H1 Simulation ($N=10,389$ trades):**
  - Gross Win Rate: 50.32%
  - Mean Net Expectancy: **-1.24 pips per trade**
  - Cumulative Net P&L: **-12,880.2 pips**
  - Cost Viability: **FAIL**
- **H4 Simulation ($N=11,388$ trades):**
  - Gross Win Rate: 50.74%
  - Mean Net Expectancy: **-1.51 pips per trade**
  - Cumulative Net P&L: **-17,251.1 pips**
  - Cost Viability: **FAIL**

---

## 15. Robustness Analysis

### 15.1 Regime Slicing (H1)
Evaluation of persistence accuracy across structural regimes:
- `LOW_VOL_TREND`: $N=3,906$, Balanced Accuracy = **51.76%**
- `HIGH_VOL_TREND`: $N=1,833$, Balanced Accuracy = **50.02%**
- `LOW_VOL_RANGE`: $N=2,144$, Balanced Accuracy = **49.82%**
- `HIGH_VOL_RANGE`: $N=955$, Balanced Accuracy = **48.30%**

*Diagnosis:* Minor outperformance in low-volatility trend conditions (51.76%) is statistically indistinguishable from noise and falls well short of the 54.0% hurdle required to overcome friction.

### 15.2 Temporal Slicing (H4)
- **Early Period (2010–2016):** Mean balanced accuracy 50.2%
- **Middle Period (2016–2021):** Mean balanced accuracy 49.4%
- **Recent Period (2021–2026):** Mean balanced accuracy 49.8%

*Diagnosis:* Directional predictability is consistently near 50% across all macro decades.

---

## 16. Failure Analysis

Why did higher-timeframe swing modeling fail to generate predictive edge?
1. **Martingale Efficiency:** EURUSD price changes at 12-hour and 24-hour horizons behave as a martingale difference sequence with respect to backward-looking technical and regime indicators.
2. **Exogenous Flow Dependence:** Multi-hour currency swings are dominated by unexpected macroeconomic releases (NFP, CPI, rate decisions) and geopolitical headlines, which leave zero causal footprint in technical indicators.
3. **Transaction Friction Dominance:** Even if a weak gross directional tendency (~51%) exists in isolated regimes, roundtrip broker friction (1.5–1.8 pips) renders net expectancy negative.

---

## 17. Decision Gate

Evaluation against pre-registered research gates:

| Pre-Registered Gate | Requirement | Actual H1 Result | Actual H4 Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1: Mean WF Bal Acc** | $\ge 54.0\%$ | 50.43% | 49.63% | **FAILED** |
| **Gate 2: Folds > Baseline** | $\ge 70\%$ of folds | 60.0% (6/10) | 30.0% (3/10) | **FAILED** |
| **Gate 3: Folds < 50%** | $\le 20\%$ of folds | 40.0% (4/10) | 50.0% (5/10) | **FAILED** |
| **Gate 4: Holdout Bal Acc** | $\ge 53.5\%$ | 46.93% | 51.29% | **FAILED** |
| **Gate 5: Temporal Stability** | No single-period bias | Passed | Passed | **PASSED** |
| **Gate 6: Zero Leakage** | All 15 invariant tests pass | Passed | Passed | **PASSED** |
| **Gate 7: Cost Robustness** | Positive Net Expectancy | -1.24 pips | -1.51 pips | **FAILED** |
| **Failure Gate** | WF $< 52\%$ or Holdout $< 51\%$ | **TRIGGERED** | **TRIGGERED** | **TRIGGERED** |

### Formal Verdict:
**NOT SUPPORTED.**

---

## 18. Final Recommendation

### Formal Status:
**PHASE 43: NOT AUTHORIZED.**

### Conclusion:
Phase 42 conclusively demonstrated that transitioning from M15 to higher timeframes (H1 and H4) and incorporating structural regime conditioning does **not** create a statistically significant or economically viable directional trading edge in EURUSD.

The research pause established in Phase 41 is reaffirmed and reinforced:
1. No technical indicator feature set—whether on M15, H1, or H4—has produced edge surviving realistic transaction friction.
2. The trading system's production risk engine, MT5 connectivity, and shadow execution infrastructure remain intact and verified as institutional-grade assets.
3. Machine learning strategy research on retail technical indicators is formally closed. Any future research must involve non-public institutional order flow data or fundamental macroeconomic models with pre-registered hypotheses.
