# EURUSD M15 Label and Target Engineering Specification

**Phase:** Phase 6 — Supervised Learning Target Engineering  
**Instrument:** EURUSD  
**Timeframe:** M15 (15-Minute Candles)  
**Input Source:** `data/processed/eurusd_m15/eurusd_m15_processed.parquet` (100,000 candles)  
**Output Target Dataset:** `data/labels/eurusd_m15/eurusd_m15_labels.parquet` (100,000 rows $\times$ 22 columns)  
**Metadata Spec:** `reports/label_metadata.json` (Machine-readable label contract)  
**Environment:** Canonical Ubuntu Linux x86_64, Python 3.13.15  

---

## 1. Prediction Problem Definition

In supervised quantitative machine learning for trading systems, learning models require well-defined future targets $y_t$. 

At candle completion timestamp $t$:
1. **Input Feature Vector $X_t$:** Evaluated strictly using historical and concurrent market data available at or before $t$ (from the Phase 5 80-feature library).
2. **Prediction Output $\hat{y}_t$:** An algorithm forecasts future market dynamics over a forward horizon $H$.
3. **Realized Target $y_t$:** Observed ex-post at timestamp $t + H$, measuring the actual price evolution.

$$\text{Information at } t: \quad X_t \longrightarrow f_\theta(X_t) = \hat{y}_t \approx y_t = g\left(\{C_{t+k}\}_{k=1}^H, C_t\right)$$

While the target $y_t$ intentionally utilizes future price points $\{C_{t+1}, \dots, C_{t+H}\}$, the input features $X_t$ **never** contain any forward-looking data. The target labels exist strictly as the training ground truth for model optimization and out-of-sample evaluation.

---

## 2. Future-Return Definitions

For any forward lookahead horizon $H \in \mathbb{N}^+$, future returns are formulated as:

### 2.1 Simple Forward Percentage Return
$$R_{t, H} = \frac{C_{t+H} - C_t}{C_t} = \frac{C_{t+H}}{C_t} - 1$$
Measures the proportional capital return if an un-leveraged position opened at the close of candle $t$ is closed at the close of candle $t+H$.

### 2.2 Continuous Forward Log Return
$$r_{t, H} = \ln\left(\frac{C_{t+H}}{C_t}\right) = \ln(C_{t+H}) - \ln(C_t)$$
Additively scalable across multi-period intervals and symmetrical for positive and negative percentage changes.

### 2.3 Volatility-Adjusted Forward Return
$$r^{\text{vol}}_{t, H} = \frac{R_{t, H}}{\text{ATR}^{\text{norm}}_{14}(t) + 10^{-12}}, \quad \text{where } \text{ATR}^{\text{norm}}_{14}(t) = \frac{\text{ATR}_{14}(t)}{C_t}$$
Expresses forward price displacement in units of local trailing volatility. Crucially, $\text{ATR}_{14}(t)$ uses **only** historical information observable at candle $t$. Future volatility is strictly excluded from normalization.

---

## 3. Candidate Horizons

EURUSD M15 market dynamics operate across distinct microstructure and session intervals. We define four candidate prediction horizons:

| Horizon ($H$) | Equivalent Duration | Market Interpretation | Primary Use Case |
| :---: | :---: | :--- | :--- |
| **1 bar** | 15 minutes | Immediate next-bar price impulse | High-frequency execution filters, immediate drift |
| **4 bars** | 1 hour (60 min) | Intra-session momentum leg | Short-term tactical trade setups |
| **8 bars** | 2 hours (120 min) | Session development / overlap horizon | Intermediate swing setups (London/NY overlap) |
| **16 bars** | 4 hours (240 min) | Half-day multi-session trend | Macro regime and directional trend following |

*No single horizon is assumed to be superior.* All four horizons are generated and documented to enable objective empirical model evaluation in later phases.

---

## 4. Direction Labels & Class Structure

To support directional classification algorithms, ternary target labels are constructed:

$$y_{t, H} \in \{-1.0 \text{ (SHORT)}, \; 0.0 \text{ (NEUTRAL)}, \; +1.0 \text{ (LONG)}\}$$

### 4.1 Fixed-Threshold Formulation (`direction_H`)
$$y_{t, H} = \begin{cases} 
+1.0 \text{ (LONG)}, & R_{t, H} > +\tau_H \\ 
-1.0 \text{ (SHORT)}, & R_{t, H} < -\tau_H \\ 
0.0 \text{ (NEUTRAL)}, & -\tau_H \le R_{t, H} \le +\tau_H 
\end{cases}$$

### 4.2 Volatility-Scaled Formulation (`direction_vol_H`)
To account for time-varying volatility clustering (identified in Phase 4 EDA), dynamic thresholds scale with local volatility and horizon root-time:
$$\tau^{\text{vol}}_{t, H} = 0.5 \times \sqrt{H} \times \text{ATR}^{\text{norm}}_{14}(t)$$

$$y^{\text{vol}}_{t, H} = \begin{cases} 
+1.0 \text{ (LONG)}, & R_{t, H} > +\tau^{\text{vol}}_{t, H} \\ 
-1.0 \text{ (SHORT)}, & R_{t, H} < -\tau^{\text{vol}}_{t, H} \\ 
0.0 \text{ (NEUTRAL)}, & -\tau^{\text{vol}}_{t, H} \le R_{t, H} \le +\tau^{\text{vol}}_{t, H} 
\end{cases}$$

---

## 5. Threshold Methodology & Empirical Justification

Rather than picking arbitrary thresholds or artificially balancing classes, thresholds were determined through empirical analysis of the 100,000-candle EURUSD M15 dataset:

| Horizon ($H$) | Return Std Dev ($\sigma_H$) | Fixed Threshold ($\tau_H$) | Threshold in BPS / Pips | Threshold / $\sigma_H$ |
| :---: | :---: | :---: | :---: | :---: |
| **1 bar (15m)** | 4.90 bps | $0.00025$ | 2.50 bps (2.50 pips) | $0.51 \times \sigma_1$ |
| **4 bars (60m)** | 9.59 bps | $0.00050$ | 5.00 bps (5.00 pips) | $0.52 \times \sigma_4$ |
| **8 bars (120m)** | 13.46 bps | $0.00075$ | 7.50 bps (7.50 pips) | $0.56 \times \sigma_8$ |
| **16 bars (240m)** | 18.95 bps | $0.00100$ | 10.00 bps (10.00 pips) | $0.53 \times \sigma_{16}$ |

### Empirical Rationale:
1. **Statistical Consistency:** Across all horizons, the threshold represents approximately **$0.5 \times$ standard deviation** of the forward return distribution.
2. **Economic Significance:** The threshold movement is **10 to 40 times larger than the average observed spread** (0.24 pips), ensuring that classified directional moves represent meaningful structural displacements rather than bid-ask bounce.
3. **Neutral Noise Filtration:** The central band ($[-\tau, +\tau]$) filters out ~57%–60% of small drift intervals where directional trading would incur transaction costs without statistical expectation.

---

## 6. Cost-Aware & Spread Considerations

Phase 4 EDA documented that while the MetaQuotes-Demo feed has an average recorded spread of 2.44 points (0.24 pips / 0.22 bps), spreads widen significantly during rollover (averaging 14.5 points at 00:00 UTC).

| Horizon ($H$) | Mean Forward Return Magnitude ($E[|R_H|]$) | Mean Spread ($S_{\text{avg}}$) | Ratio $E[|R_H|] / S_{\text{avg}}$ |
| :---: | :---: | :---: | :---: |
| **1 bar** | 3.24 bps (3.24 pips) | 0.24 pips | **13.5x** |
| **4 bars** | 6.46 bps (6.46 pips) | 0.24 pips | **26.9x** |
| **8 bars** | 9.17 bps (9.17 pips) | 0.24 pips | **38.2x** |
| **16 bars** | 13.25 bps (13.25 pips) | 0.24 pips | **55.2x** |

*Crucial Caveat:* Raw forward return is not net profit. In live execution, round-trip trading costs include:
- Entry spread + Exit spread
- Execution slippage (especially during high-volatility news releases)
- Broker commissions
- Financing overnight swap rates (for holds crossing 21:00 UTC)

Therefore, single-bar targets ($H=1$) have an expected movement of ~3.2 pips, leaving a narrower margin against institutional execution costs than longer horizons ($H=8, 16$).

---

## 7. Volatility-Adjusted Target Diagnostics

Because EURUSD volatility clusters into high-variance regimes (macro announcements, geopolitical crises) and low-variance regimes (holiday periods, summer doldrums), a fixed 5-pip move has very different meanings in low vs. high volatility.

- In low volatility ($\text{ATR} \approx 3$ pips), a 5-pip move is an unusual excursion ($>1.6 \times \text{ATR}$).
- In high volatility ($\text{ATR} \approx 12$ pips), a 5-pip move is sub-bar noise ($<0.4 \times \text{ATR}$).

The continuous target `future_vol_adj_return_H` normalizes this effect. Furthermore, the volatility-scaled classification `direction_vol_H` dynamically expands thresholds in turbulent regimes and contracts them in quiet regimes, maintaining an invariant class distribution across market cycles.

---

## 8. Natural Class Distributions

The empirical class distributions across the full 100,000 candles show near-perfect directional symmetry:

### 8.1 Fixed Thresholds (`direction_H`)
| Horizon ($H$) | Threshold | SHORT (-1) | NEUTRAL (0) | LONG (+1) | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1 bar (15m)** | 2.5 bps | 20,615 (20.62%) | 58,661 (58.66%) | 20,723 (20.72%) | 2.85 : 1 |
| **4 bars (60m)** | 5.0 bps | 20,581 (20.58%) | 58,611 (58.61%) | 20,804 (20.80%) | 2.85 : 1 |
| **8 bars (120m)** | 7.5 bps | 19,719 (19.72%) | 60,435 (60.44%) | 19,838 (19.84%) | 3.06 : 1 |
| **16 bars (240m)** | 10.0 bps | 21,459 (21.46%) | 56,669 (56.68%) | 21,856 (21.86%) | 2.64 : 1 |

### 8.2 Volatility-Scaled Thresholds (`direction_vol_H`)
| Horizon ($H$) | Threshold Rule | SHORT (-1) | NEUTRAL (0) | LONG (+1) | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1 bar (15m)** | $0.5 \times \text{ATR}$ | 19,327 (19.33%) | 61,214 (61.22%) | 19,445 (19.45%) | 3.17 : 1 |
| **4 bars (60m)** | $1.0 \times \text{ATR}$ | 19,242 (19.25%) | 61,139 (61.15%) | 19,602 (19.61%) | 3.18 : 1 |
| **8 bars (120m)** | $1.41 \times \text{ATR}$ | 19,229 (19.23%) | 60,823 (60.84%) | 19,927 (19.93%) | 3.16 : 1 |
| **16 bars (240m)** | $2.0 \times \text{ATR}$ | 20,194 (20.20%) | 58,955 (58.97%) | 20,822 (20.83%) | 2.92 : 1 |

*Observation:* LONG and SHORT frequencies are almost identical (~20% vs ~20%), reflecting the empirical martingale-like symmetry of foreign exchange price discovery.

---

## 9. End-of-Data Boundary Handling

For any horizon $H$, the final $H$ candles in the dataset do not possess forward prices $C_{t+H}$.

- **Boundary Policy:** The final $H$ candles are strictly assigned `NaN` / unavailable.
- **Unavailable Counts:**
  - $H=1$: Exactly 1 row at index 99,999 is `NaN`.
  - $H=4$: Exactly 4 rows at indices 99,996–99,999 are `NaN`.
  - $H=8$: Exactly 8 rows at indices 99,992–99,999 are `NaN`.
  - $H=16$: Exactly 16 rows at indices 99,984–99,999 are `NaN`.
- **Zero Fabrication:** No arbitrary imputation, forward-fill, or placeholder class (such as NEUTRAL) is applied to the final rows.

---

## 10. Overlapping Labels & Autocorrelation

For horizons $H > 1$ (e.g. $H=4, 8, 16$), consecutive target samples overlap in time:
- $R_{t, 4}$ and $R_{t+1, 4}$ share 3 candles ($C_{t+2}, C_{t+3}, C_{t+4}$).
- This induces serial correlation in the target sequence $y_t$.

### Protocol for Subsequent Phases:
1. **No Data Deletion in Phase 6:** In accordance with prompt instructions, no observations are dropped or purged during target generation.
2. **Purged Cross-Validation in Phase 9/10:** Subsequent train/test splits must enforce an embargo/purge buffer equal to horizon $H$ between training folds and validation folds to eliminate leakage from target overlap.

---

## 11. Feature / Label Architectural Separation

The codebase strictly enforces structural segregation between feature extraction and target extraction:

```
                          Validated Market Data (Parquet)
                                         |
                 +-----------------------+-----------------------+
                 |                                               |
                 v                                               v
    Phase 5 Feature Pipeline                         Phase 6 Label Pipeline
      (ai.features.pipeline)                           (ai.labels.pipeline)
                 |                                               |
                 | (Strictly Causal: t-k to t)                   | (Forward-Looking: t+1 to t+H)
                 v                                               v
          Feature Matrix X(t)                             Target Matrix y(t)
         [100,000 x 80 features]                        [100,000 x 20 targets]
                 |                                               |
                 +-----------------------+-----------------------+
                                         |
                                         v
                         Supervised Training Assembler
                              (Model Training Phase)
```

- Features $X_t$ are generated independently and know nothing of future returns.
- Target labels $y_t$ are stored in a separate dataset `data/labels/`.
- Training pipelines load $X_t$ and $y_t$, drop the initial 80-bar warm-up and final $H$-bar boundary NaNs, and construct $(X, y)$ pairs safely.

---

## 12. Automated Leakage Tests

A dedicated test suite in `tests/test_labels.py` validates all six core safety requirements:
1. **TEST 1 (Future Dependency):** Changing $C_{t+H}$ directly changes $y_t$, while changing historical candle $C_{t-1}$ does not affect $R_{t, H}$.
2. **TEST 2 (Feature Immutability):** Appending or altering future bars does not change feature values at or before $t$.
3. **TEST 3 (Horizon Alignment):** Verifies analytical equivalence of $R_{t, H} \equiv \frac{C_{t+H}}{C_t} - 1$ across all bars.
4. **TEST 4 (End-of-Data Handling):** Exactly the final $H$ rows receive `NaN`, with zero fabricated classes.
5. **TEST 5 (Class Label Correctness):** Verifies that LONG, NEUTRAL, and SHORT classifications strictly follow documented threshold criteria.
6. **TEST 6 (Label Absence in Features):** Verifies that no target column names or forward prefixes exist in the feature matrix.

---

## 13. Limitations of the Target Design

1. **Discrete Fixed Horizons:** Fixed-horizon targets evaluate price strictly at $t+H$, ignoring path dynamics (e.g. intra-horizon drawdown or maximum adverse excursion). Advanced triple-barrier labeling may be evaluated in subsequent research.
2. **Demo Spread Modeling:** Spreads reflect MetaQuotes-Demo snapshot feeds. Fills at the exact mid/close price are theoretical approximations.
3. **Unadjusted Weekend Boundary Overlaps:** Bars immediately preceding weekend closures have forward returns spanning the 48-hour weekend gap. Downstream models should account for weekend gap risks.

---

## 14. Why No Predictive Claims Are Made

Phase 6 is strictly **target construction and empirical documentation**:
- Establishing target distributions does not demonstrate predictable edge.
- Symmetrical class balances (~20% / ~60% / ~20%) are natural characteristics of Brownian-like price diffusions with symmetric thresholds.
- Evaluating whether features $X_t$ can predict $y_t$ with statistically significant edge is reserved for formal machine learning baselines (Phase 7) and walk-forward validation (Phases 9–10).
