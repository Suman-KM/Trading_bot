# Phase 29 — Pre-Registered CFTC COT Positioning Experiment Protocol

**Date:** 2026-10-02  
**Project:** AI Autonomous Trading System  
**Repository:** `https://github.com/Suman-KM/Trading_bot.git`  
**Branch:** `develop`  
**Pre-Registration Status:** **LOCKED & FROZEN PRIOR TO MODEL EXECUTION**  
**Base Commit:** [`2a3138a`](https://github.com/Suman-KM/Trading_bot/commit/2a3138af62b205c3febb6e8e7719e1594abe7fd4) (`research: establish CFTC COT data foundation`)  

---

## 1. Hypothesis

Extreme weekly speculative positioning and commercial hedging imbalances in CME Euro FX futures contain incremental, statistically significant directional information about the medium-term directional movement of EURUSD exchange rates beyond the information already captured by EURUSD price-derived technical features. Specifically, when speculative positioning reaches historical multi-year extremes, speculative inventory exhaustion drives multi-week mean reversion in spot EURUSD.

---

## 2. Null Hypothesis ($H_0$)

Weekly CFTC Commitments of Traders (COT) institutional positioning features provide no statistically meaningful incremental directional predictive information beyond the frozen EURUSD D1 price-only benchmark ($p \ge 0.05$ or $\Delta \text{Balanced Accuracy} \le 0.0\%$).

---

## 3. Data Source & Provenance

All positioning data is derived from the official regulatory archives of the United States Commodity Futures Trading Commission (CFTC):
- **Source Portal:** `https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm`
- **Report Families:** COT Legacy Futures Only (`deacot`) and Traders in Financial Futures (`fut_fin_txt`).
- **Cryptographic Provenance:** 34 annual archives (2010–2026) verified with SHA-256 digests in [`reports/cftc_cot_provenance.json`](file:///home/cino/projects/ai-trading-system/reports/cftc_cot_provenance.json).
- **Canonical Parquet:** [`data/exogenous/cftc/cftc_eurofx_cot.parquet`](file:///home/cino/projects/ai-trading-system/data/exogenous/cftc/cftc_eurofx_cot.parquet) (873 weekly reports, zero missing reports, zero accounting identity errors).

---

## 4. COT Contract Identification

- **CFTC Contract Market Code:** `099741`
- **Market Name:** `EURO FX - CHICAGO MERCANTILE EXCHANGE`
- **Underlying Instrument:** CME Euro FX Futures (contract size: €125,000)

---

## 5. Candidate Features (Strictly 3 Features)

In strict accordance with Phase 26 candidate specification and Phase 29 instructions, exactly three COT features are evaluated:

1. `speculative_net_zscore_3y`:
   $$\text{Net Speculative} = \text{NonCommercial\_Long} - \text{NonCommercial\_Short}$$
   Standardized as a rolling 3-year (156-week, $\min=52$) point-in-time Z-score using only historical weekly reports available at that time:
   $$Z_{\text{spec}, t} = \frac{\text{Net Spec}_{t} - \mu_{156, t}}{\sigma_{156, t}}$$
2. `commercial_position_zscore_3y`:
   $$\text{Net Commercial} = \text{Commercial\_Long} - \text{Commercial\_Short}$$
   Standardized as a rolling 3-year (156-week, $\min=52$) point-in-time Z-score:
   $$Z_{\text{comm}, t} = \frac{\text{Net Comm}_{t} - \mu_{156, t}}{\sigma_{156, t}}$$
3. `speculative_net_4w_change`:
   4-week delta in net speculative positioning:
   $$\Delta \text{Net Spec}_{t, 4\text{w}} = \text{Net Spec}_{t} - \text{Net Spec}_{t-4}$$

> [!IMPORTANT]
> - Zero additional COT features.
> - Zero additional technical features.
> - Zero automated feature selection.
> - Candidate feature set size: exactly $30 + 3 = 33$ features.

---

## 6. Target Definition

The target variable is the **volatility-adjusted directional return** across multi-day holding horizons:
$$\text{return}_h(t) = \frac{\text{close}(t + h) - \text{close}(t)}{\text{close}(t)}$$
$$\text{vol\_threshold}_h(t) = 0.5 \times \sqrt{h} \times \text{ATR\_norm\_14}(t)$$

$$\text{direction\_vol}_h(t) = \begin{cases} 
+1.0 & \text{if } \text{return}_h(t) > \text{vol\_threshold}_h(t) \\
-1.0 & \text{if } \text{return}_h(t) < -\text{vol\_threshold}_h(t) \\
\text{NaN (unclassified)} & \text{if } |\text{return}_h(t)| \le \text{vol\_threshold}_h(t)
\end{cases}$$

---

## 7. Pre-Registered Horizons

Two pre-registered horizons identified by the Phase 26 research design:
- **Horizon 1:** $H = 10$ business days (~2 calendar weeks, 10 D1 bars)
- **Horizon 2:** $H = 20$ business days (~4 calendar weeks, 20 D1 bars)

Both horizons are reported in full. Model selection across horizons is strictly prohibited.

---

## 8. Baseline Benchmark

- **Baseline Dataset:** Canonical EURUSD Daily (D1) OHLCV aggregated causally from verified EURUSD H4 bars (`2010-03-01` to `2024-11-04`).
- **Baseline Feature Matrix:** Exactly the canonical 30 D1 technical swing features (`EURUSD_BASELINE_32_COLS[:30]`), covering return momentum, moving average relationships, trend regime, RSI, volatility, and cyclical day-of-week encoding.
- **Baseline Model:** Identical model family, identical hyperparameters, identical walk-forward folds.

---

## 9. Model Family

- **Classifier:** `RandomForestClassifier` (Scikit-Learn).
- **Single Model Family:** No comparison across XGBoost, LightGBM, SVM, Logistic Regression, or Neural Networks.

---

## 10. Fixed Hyperparameters

The model hyperparameters are frozen to the established `FROZEN_RF_CONFIG`:
```python
FROZEN_RF_CONFIG = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
```
Hyperparameter sweeps, grid searches, and tree depth tuning are strictly prohibited.

---

## 11. Walk-Forward Methodology

- **Validation Structure:** Expanding chronological walk-forward validation.
- **Historical Research Window:** `2010-03-01` through `2024-11-04` (3,816 D1 bars).
- **Number of Folds:** 10 forward chronological folds.
- **Initial Warmup Window:** 1,000 D1 bars (~3.8 years, covering 2010 to early 2014).
- **Fold Size:** ~281 D1 bars per fold (~1.1 calendar years).
- **Temporal Ordering:** Strict non-overlapping forward progression ($t_{\text{train, end}} < t_{\text{val, start}}$).

---

## 12. Purge Window

To prevent target lookahead between the training set and the validation set:
- For $H = 10$: `purge_bars = 10` daily bars.
- For $H = 20$: `purge_bars = 20` daily bars.
$$\text{train\_end\_index} = \text{val\_boundary} - \text{purge\_bars}$$

---

## 13. Embargo Window

To eliminate autoregressive feature memory (e.g. 20-day rolling volatilities and momentum):
- `embargo_bars = 5` daily bars.
$$\text{val\_start\_index} = \text{val\_boundary} + \text{embargo\_bars}$$

---

## 14. Publication Lag & Point-in-Time Rules

1. **Tuesday Observation Close:** Position snapshot taken at Tuesday 21:00 UTC.
2. **Official Release:** Friday 15:30 US Eastern Time (`America/New_York`).
3. **UTC Conversion:** 19:30 UTC during EDT (summer) / 20:30 UTC during EST (winter).
4. **Actionable Effective Timestamp:** Monday `00:00:00 UTC` post-release.
5. **No Lookahead:** Tuesday positions are never accessible to Tuesday–Friday price bars.

---

## 15. Leakage Prevention Verification

Before executing any fold:
- Verify zero contemporaneous Tuesday alignment.
- Verify backward as-of merge direction.
- Verify zero train/validation index overlap.
- Verify strict monotonicity of timestamps.

---

## 16. Primary Evaluation Metric

- **Primary Metric:** **Balanced Accuracy** (arithmetic mean of sensitivity and specificity):
$$\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right)$$
- Evaluated out-of-sample on each of the 10 expanding validation folds.
- Secondary metrics tracked: Macro F1, Accuracy, ROC-AUC.

---

## 17. Statistical Testing

Paired hypothesis test comparing Candidate vs. Baseline balanced accuracies across the 10 folds:
- Two-tailed Paired Student's t-test (`scipy.stats.ttest_rel`).
- Wilcoxon Signed-Rank Test (`scipy.stats.wilcoxon`).
- 95% Confidence Interval for the mean difference $\Delta = \text{BalAcc}_{\text{COT}} - \text{BalAcc}_{\text{Baseline}}$.
- Threshold for significance: $\alpha = 0.05$ ($p < 0.05$).

---

## 18. Pre-Registered Success Gate

The experiment is classified as **`SUPPORTED`** if and only if:
1. Mean Walk-Forward Balanced Accuracy $\ge 55.0\%$ across all 10 folds.
2. Statistically significant improvement over the Baseline ($p < 0.05$ with $\Delta > 0$).
3. Results are consistent across both horizons or strongly supported on at least one without degradation on the other.

---

## 19. Pre-Registered Failure Gate

The experiment is classified as **`NOT SUPPORTED`** if:
1. Mean Walk-Forward Balanced Accuracy $< 53.0\%$ on both horizons.
2. OR net improvement over the D1 price-only baseline is zero or negative ($\Delta \le 0.0\%$).
3. OR $p \ge 0.05$ (difference indistinguishable from noise).

The experiment is classified as **`INCONCLUSIVE`** if mean balanced accuracy is between $53.0\%$ and $55.0\%$ with $p \ge 0.05$.

---

## 20. Locked Data Policy

- Phase 11 Test Partition (`2026-02-19 12:00:00 UTC` onward) remains **PERMANENTLY LOCKED & UNTOUCHED**.
- No test partition data or labels may be accessed for training, validation, or model selection.
- Research concludes with the 10-fold walk-forward validation results.
- No trading backtest or P&L optimization may be run unless the predictive success gate is passed.
