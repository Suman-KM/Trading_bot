# EURUSD M15 Dataset Assembly and Leakage-Safe Time-Series Splitting Specification

**Phase:** Phase 7 — Dataset Assembly & Leakage-Safe Time-Series Splitting  
**Instrument:** EURUSD  
**Timeframe:** M15 (15-Minute Candles)  
**Input Features Source:** `data/features/eurusd_m15/eurusd_m15_features.parquet` (100,000 candles $\times$ 89 columns)  
**Input Labels Source:** `data/labels/eurusd_m15/eurusd_m15_labels.parquet` (100,000 rows $\times$ 22 columns)  
**Metadata Spec:** `reports/dataset_split_metadata.json` (Machine-readable splitting contract)  
**Environment:** Canonical Ubuntu Linux x86_64, Python 3.13.15  

---

> [!IMPORTANT]
> **Research Boundary & Disclaimer:** Phase 7 defines a deterministic dataset assembly and chronological train/validation/test holdout methodology. It does **not** evaluate machine learning models, establish predictive performance, or demonstrate trading profitability.

---

## 1. Overview and Core Objectives

Quantitative machine learning on financial time series requires strict causal segregation between past inputs and future targets. Unlike standard cross-sectional machine learning:
1. Observations are chronologically ordered and autocorrelated.
2. Target labels evaluate price changes over forward horizons $H \in \{1, 4, 8, 16\}$ bars ($15$ to $240$ minutes).
3. Random shuffling or naive train-test splitting introduces catastrophic lookahead leakage.

Phase 7 constructs a deterministic dataset assembly and chronological holdout splitting pipeline located in `ai/dataset/`:
- `assembly.py`: Synchronizes point-in-time features with forward-looking labels on UTC timestamps, drops warm-up rows, and preserves end-of-dataset label NaNs.
- `splits.py`: Implements chronological 70% / 15% / 15% partitioning with forward-purge and optional embargo handling.
- `validation.py`: Enforces 10 automated leakage and integrity safety checks.

---

## 2. Source Datasets and Assembly Methodology

The pipeline integrates two independently verified artifacts:

```
+-------------------------------------------------------------+
|  Phase 5 Feature Library: 80 Derived Features + 9 Base Cols |
|  [100,000 rows x 89 cols; Max Lookback = 80 bars]           |
+-------------------------------------------------------------+
                              |
                              v  Inner Join on UTC Timestamp
+-------------------------------------------------------------+
|  Phase 6 Target Library: 20 Prediction Targets + 2 Time Cols |
|  [100,000 rows x 22 cols; Horizons H = 1, 4, 8, 16]         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|  Phase 7 Assembled Dataset (ai.dataset.assembly)            |
|  - Drops first 80 warm-up bars (features complete, 0 NaNs)  |
|  - Preserves 99,920 post-warmup candles                     |
|  - Preserves tail H rows with NaN targets                   |
|  - Segregates X (80 features) strictly from y (target)      |
+-------------------------------------------------------------+
```

### 2.1 Feature Warm-Up Exclusion
- In Phase 5, the maximum rolling lookback across all 80 features was 80 bars (e.g., `roc_80`, `vol_std_80`).
- The initial 80 rows (indices 0..79, timestamps `2022-09-16 09:30:00 UTC` to `2022-09-19 05:15:00 UTC`) contain incomplete rolling windows and are excluded during assembly.
- All remaining 99,920 rows contain **zero NaNs** across all 80 features.

### 2.2 Preservation of Forward-Horizon Boundary NaNs
- Because labels evaluate price at candle $t + H$, the final $H$ candles in the dataset lack future realizations.
- In accordance with the project specification, the assembly function preserves these boundary rows as `NaN` in `assembled.df`, while `assembled.usable_df` isolates the subset where both features and targets are valid.

---

## 3. Supported Horizons and Targets

The assembly pipeline supports all candidate targets engineered in Phase 6:

| Horizon ($H$) | Duration | Classification Targets | Continuous Return Targets | Volatility-Adjusted Targets |
| :---: | :---: | :--- | :--- | :--- |
| **1 bar** | 15 min | `direction_1` | `future_return_1`, `future_log_return_1` | `future_vol_adj_return_1`, `direction_vol_1` |
| **4 bars** | 60 min | `direction_4` | `future_return_4`, `future_log_return_4` | `future_vol_adj_return_4`, `direction_vol_4` |
| **8 bars** | 120 min | `direction_8` | `future_return_8`, `future_log_return_8` | `future_vol_adj_return_8`, `direction_vol_8` |
| **16 bars** | 240 min | `direction_16` | `future_return_16`, `future_log_return_16` | `future_vol_adj_return_16`, `direction_vol_16` |

---

## 4. Chronological Splitting Methodology (70 / 15 / 15)

The baseline partition allocates the usable dataset chronologically without shuffling:
- **TRAIN:** Earliest 70% of usable observations.
- **VALIDATION:** Subsequent 15% of usable observations.
- **TEST:** Final 15% of usable observations.

The partition boundary indices are calculated dynamically from the actual usable dataset size $N = 99,920 - H$:
$$\text{raw\_train\_end} = \lfloor N \times 0.70 \rfloor, \quad \text{raw\_val\_end} = \text{raw\_train\_end} + \lfloor N \times 0.15 \rfloor, \quad \text{raw\_test\_end} = N$$

---

## 5. Purge and Embargo Logic

### 5.1 The Overlapping Label Leakage Problem
When predicting returns over horizon $H > 1$, the label for candle $t$ depends on price at $t + H$. If candle $t$ is in the Training set, but $t + H$ falls within the Validation set:
$$t \in \text{Train}, \quad t + H \in \text{Validation}$$
The training label $y_t$ directly observes market prices from the validation period. A model trained on candle $t$ would learn from future validation outcomes, resulting in data leakage.

### 5.2 Purge Implementation
To eliminate this leakage:
- For any partition ending at index $K$, observations with $t + H \ge K \iff t \ge K - H$ must be removed.
- **Training Purge:** The last $H$ bars of the raw training segment are removed. Train ends at $\text{raw\_train\_end} - H$.
- **Validation Purge:** The last $H$ bars of the raw validation segment are removed. Validation ends at $\text{raw\_val\_end} - H$.
- The resulting gap between Train and Validation (and Validation and Test) spans at least $H$ bars ($H \times 15$ minutes).

### 5.3 Embargo Implementation
While purging prevents label lookahead, short-term autoregressive autocorrelation in market features can still carry information across the split boundary.
- An optional **embargo buffer** $E$ (bars) delays the start of the subsequent set.
- Validation starts at $\text{raw\_train\_end} + E$.
- Test starts at $\text{raw\_val\_end} + E$.
- When $E = H$, the separation between sets expands to $2H$ bars ($2H \times 15$ minutes), eliminating both forward-label leakage and immediate feature autocorrelation.

```
Chronological Timeline:
[================= TRAIN =================] [PURGE H] [EMBARGO E] [========= VAL =========] [PURGE H] [EMBARGO E] [========= TEST =========]
0                                    raw_train_end               raw_val_end                                                N
```

---

## 6. Exact Partition Sizes and Timestamp Boundaries

### 6.1 Baseline Holdout (Purge Only, $H$ bars gap, Embargo = 0)

| Horizon ($H$) | Target | Usable Rows ($N$) | Train Rows (70%) | Validation Rows (15%) | Test Rows (15%) | Total Purged |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$H=1$** | `direction_1` | 99,919 | 69,942 (70.00%) | 14,986 (15.00%) | 14,989 (15.00%) | 2 bars |
| **$H=4$** | `direction_4` | 99,916 | 69,937 (70.00%) | 14,983 (15.00%) | 14,988 (15.00%) | 8 bars |
| **$H=8$** | `direction_8` | 99,912 | 69,930 (69.99%) | 14,978 (14.99%) | 14,988 (15.00%) | 16 bars |
| **$H=16$** | `direction_16` | 99,904 | 69,916 (69.98%) | 14,969 (14.98%) | 14,987 (15.00%) | 32 bars |

### 6.2 Timestamp Boundaries for Primary Horizon ($H=4$, Target: `direction_4`)

- **Train Partition (69,937 observations):**
  - Start: `2022-09-19 05:30:00 UTC`
  - End: `2025-07-14 04:30:00 UTC`
- **Train $\to$ Validation Purged Gap (4 bars / 60 minutes):**
  - Purged bars: `04:45`, `05:00`, `05:15`, `05:30` on `2025-07-14`
- **Validation Partition (14,983 observations):**
  - Start: `2025-07-14 05:45:00 UTC`
  - End: `2026-02-19 10:45:00 UTC`
- **Validation $\to$ Test Purged Gap (4 bars / 60 minutes):**
  - Purged bars: `11:00`, `11:15`, `11:30`, `11:45` on `2026-02-19`
- **Test Partition (14,988 observations):**
  - Start: `2026-02-19 12:00:00 UTC`
  - End: `2026-09-25 22:45:00 UTC`

### 6.3 Enhanced Holdout (Purge = $H$, Embargo = $H$)

| Horizon ($H$) | Target | Train Rows | Validation Rows | Test Rows | Total Omitted Gap |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **$H=1$** | `direction_1` | 69,942 | 14,985 | 14,988 | 4 bars (60 min) |
| **$H=4$** | `direction_4` | 69,937 | 14,979 | 14,984 | 16 bars (240 min) |
| **$H=8$** | `direction_8` | 69,930 | 14,970 | 14,980 | 32 bars (480 min) |
| **$H=16$** | `direction_16` | 69,916 | 14,953 | 14,971 | 64 bars (960 min) |

---

## 7. Automated Leakage Controls & Integrity Gates

The `ai.dataset.validation.validate_dataset_splits()` engine enforces 10 automated safety gates:

1. **Target Segregation:** The target column is strictly excluded from feature matrix $X$ (`target_not_in_features`).
2. **Forward Label Exclusion:** No label or future column (`future_*`, `direction_*`) can exist in $X$ (`no_label_columns_in_features`).
3. **Internal Alignment:** Observation counts and index mappings between $X$, $y$, and timestamps match exactly within all partitions.
4. **Zero Feature NaNs:** Feature matrices contain zero NaNs post-assembly.
5. **Zero Target NaNs:** Target vectors in Train, Validation, and Test contain zero NaNs.
6. **Monotonic Ordering:** Timestamps within each partition are strictly monotonically increasing.
7. **Strict Temporal Sequence:** $\max(\text{Train}) < \min(\text{Validation})$ and $\max(\text{Validation}) < \min(\text{Test})$.
8. **Disjoint Timestamp Sets:** The intersection of timestamps across Train, Validation, and Test is strictly empty.
9. **Purge Boundary Correctness:** At least $H$ bars are omitted between partitions.
10. **Embargo Boundary Correctness:** When configured, at least $H + E$ bars are omitted between partitions.

---

## 8. Known Limitations of the Holdout Design

1. **Single Fixed Holdout:** While 70/15/15 chronological splitting prevents leakage, it evaluates model performance on a single fixed test regime (February 2026 to September 2026). Walk-forward expanding or rolling splits may be evaluated in subsequent research phases.
2. **Non-Uniform Weekend Boundaries:** Purging 16 bars over a Friday market close spans a 48-hour calendar weekend. The purge logic operates in bar-index space, correctly eliminating the 16 bars preceding the boundary regardless of calendar elapsed time.
3. **Regime Distribution Shift:** Macroeconomic regimes differ across 2022–2024 (monetary tightening) vs. 2025–2026 (neutral/easing). Chronological splitting naturally tests generalization across these regimes.
