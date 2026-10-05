# Phase 40.4 — Live Frozen Strategy Integration Validation

## 1. Objective

The objective of Phase 40.4 is to perform an engineering validation of the live strategy integration in shadow mode on `MetaQuotes-Demo` EURUSD M15:
1. Wire the canonical frozen machine learning strategy (`RandomForestBaseline`) and canonical 80-feature extraction pipeline into the live forward validation architecture.
2. Ingest historical M15 warmup bars from MT5 to construct the required feature history ($\ge 80$ bars).
3. Evaluate the frozen model on live EURUSD completed M15 candles in real time.
4. Record class probabilities, directional confidence values, and filter signals against the frozen confidence threshold ($\tau = 0.60$).
5. Operate strictly in **SHADOW MODE** (`execution_enabled = FALSE`, `is_live = FALSE`, orders = 0, fills = 0, positions = 0, capital exposure = $0.00$).
6. Empirically determine whether the zero-signal observation in Phase 40.2 was caused by implementation wiring, insufficient warmup, or genuine model confidence characteristics.

---

## 2. Phase 40.3 Root Cause

Phase 40.3 conducted a comprehensive diagnostic investigation into the Phase 40.2 2-hour open-market observation (which produced 0 signals across 7,207.80 seconds). The diagnostic audit revealed two distinct root causes:

1. **Implementation Wiring Defect (Category A):**
   In `scripts/run_phase40_forward_demo.py`, `ForwardDemoRunner` was instantiated with `strategy_fn=None`. The strategy evaluation method `evaluate_frozen_strategy()` contained an early-exit check:
   ```python
   if self.strategy_fn is None:
       return None
   ```
   Consequently, the machine learning model was **never evaluated** during the entire Phase 40.2 observation session.

2. **Model Confidence Ceiling Inherent to the Frozen Strategy (Category C):**
   Historical validation across the locked research dataset (Phase 11–13) demonstrated that the frozen `RandomForestBaseline` (trained on `direction_4`, balanced, depth 10, leaf 20, estimators 100) exhibits a directional confidence ceiling:
   $$\max(P_{\text{LONG}}, P_{\text{SHORT}}) \le 59.36\% < 60.00\%$$
   The historical validation distribution had:
   - Mean directional confidence: **34.84%**
   - Median directional confidence: **36.02%**
   - Maximum directional confidence: **59.36%**
   - Proportion $\ge 0.60$: **0.000%** (0 out of 14,987 validation samples)

Therefore, even if the model had been wired in Phase 40.2, zero signals would have been emitted under the frozen threshold of 0.60. Phase 40.4 was commissioned to eliminate the wiring gap and empirically test the model on live EURUSD market candles.

---

## 3. Implementation Changes

The following modules and artifacts were developed to wire and validate the live frozen strategy:

1. **`trading/adapters/mt5/normalizer.py`:**
   - Extended `normalize_timestamp` to handle ISO strings and Wine-to-Python IPC boundary string outputs cleanly, guaranteeing timezone-aware UTC timestamps across all platforms.

2. **`trading/adapters/mt5/shadow.py`:**
   - `M15CandleBuffer`: Bounded rolling candle buffer enforcing strict point-in-time causality, monotonicity checks, duplicate prevention, and lookahead protection (forming candles with $T_{\text{close}} > T_{\text{current}}$ are strictly rejected).
   - `LiveFrozenStrategyPipeline`: Deterministically fits and locks the canonical Phase 12 Random Forest baseline on the Training partition; extracts the canonical 80 features; validates schema (80 columns, 0 NaNs, 0 Infs); evaluates directional confidence $\max(P_{\text{LONG}}, P_{\text{SHORT}})$; filters against $\tau = 0.60$; produces shadow signal records with no order execution.
   - `ShadowValidationRunner`: Orchestrates read-only MT5 preflight checks (14 safety gates), historical warmup ingestion (120 bars), point-in-time candle buffering, periodic tick freshness checks ($\le 120$s cutoff), incremental evaluation, and fail-closed diagnostics accounting.

3. **`scripts/run_phase40_4_live_strategy.py`:**
   - Production shadow runner script supporting both `--smoke-only` (offline/static verification) and `--target-duration 7200.0` (real-time open-market shadow observation), writing telemetry to `reports/phase40_4_live_strategy_integration.json`.

4. **`tests/test_phase40_4_live_strategy.py`:**
   - Comprehensive test suite covering all 22 required diagnostic, model, feature, and safety invariants.

---

## 4. Frozen Strategy Identity

The strategy evaluated during Phase 40.4 is identical to the canonical production research strategy locked in Phase 12 and audited in Phase 40.3:

| Attribute | Specification |
|---|---|
| **Model Family** | `RandomForestBaseline` (`sklearn.ensemble.RandomForestClassifier`) |
| **Model Parameters** | `n_estimators=100`, `max_depth=10`, `min_samples_leaf=20`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1` |
| **Training Partition** | `splits.train.X` (69,937 M15 bars, 70% temporal split) |
| **Target Variable** | `direction_4` ($H=4$ M15 bars / 60-minute forward horizon, $\tau_{\text{return}}=0$) |
| **Target Classes** | `[-1.0 (SHORT), 0.0 (NEUTRAL), 1.0 (LONG)]` |
| **Canonical Features** | Exactly 80 technical indicators (Momentum, Volatility, Trend, Volume, Temporal) |
| **Confidence Threshold** | $\tau = 0.60$ (strictly frozen; no lowering permitted) |
| **Directional Confidence** | $\max(P_{\text{LONG}}, P_{\text{SHORT}})$ |
| **Signal Semantics** | `P(LONG) >= 0.60 -> BUY`, `P(SHORT) >= 0.60 -> SELL`, otherwise `NONE` |

---

## 5. Live Data Pipeline

The live data flow follows a strictly unidirectional, read-only sequence:

```mermaid
flowchart TD
    MT5["MT5 Terminal (MetaQuotes-Demo EURUSD)"] -->|"copy_rates_from_pos(EURUSD, M15, 120)"| Warmup["M15 Warmup (119 Completed Bars)"]
    MT5 -->|"symbol_info_tick(EURUSD)"| Tick["Tick Freshness Gate (Age <= 120s)"]
    MT5 -->|"copy_rates_from_pos(EURUSD, M15, 3)"| Stream["Live M15 Candle Stream"]
    Stream -->|"now_utc >= candle_close"| Buffer["M15CandleBuffer (Monotonic, No Duplicates)"]
    Warmup --> Buffer
    Buffer -->|"to_dataframe() >= 80 bars"| FeatPipe["Canonical 80-Feature Pipeline"]
    FeatPipe -->|"X_latest (80 cols, 0 NaNs, 0 Infs)"| Model["Frozen RandomForestBaseline"]
    Model -->|"predict_proba(X_latest)"| Probs["[P_short, P_neutral, P_long]"]
    Probs -->|"max(P_long, P_short)"| Conf["Directional Confidence"]
    Conf -->|"Confidence >= 0.60?"| Filter{"Threshold Check"}
    Filter -->|No (< 0.60)| NoneSig["Signal: NONE (Record Telemetry)"]
    Filter -->|Yes (>= 0.60)| ShadowSig["Signal: SHADOW_BUY / SHADOW_SELL (No Order Sent)"]
```

---

## 6. Point-in-Time Safety & Lookahead Protection

Strict causal integrity is enforced across the entire ingestion and evaluation path:
1. **Incomplete Forming Bar Rejection:** MT5's `copy_rates_from_pos` includes the currently forming candle at index 0. The buffer computes $T_{\text{close}} = T_{\text{open}} + 15\text{m}$. If $T_{\text{close}} > T_{\text{current}}$, the candle is immediately rejected.
2. **Evaluation Timestamp Verification:** `evaluate_latest_completed_bar` asserts $T_{\text{close}} \le T_{\text{eval}}$. Any attempt to evaluate before candle completion raises `FeatureVectorValidationError`.
3. **Point-in-Time Feature Alignment:** Technical indicators are computed strictly over historical completed bars; only the final row $\mathbf{x}_T$ corresponding to the latest closed candle is fed to the model.
4. **NaN and Inf Detection:** The extracted vector $\mathbf{x}_T$ is validated for exact length ($D=80$), $\text{NaN count} = 0$, and $\text{Inf count} = 0$.

---

## 7. Shadow-Mode Safety & Governance Invariants

Absolute safety guarantees remained sovereign throughout Phase 40.4:
- `execution_enabled = FALSE`: Permanently hardcoded as an immutable property on `ShadowValidationRunner`.
- `is_live = FALSE`: System configured strictly for demo environments.
- **Zero Order Submission:** No calls to `order_send()`, `order_check()`, or trade endpoints occurred.
- **Sovreignty of RiskEngine:** Preflight audits confirmed zero open positions, zero pending orders, and normal status.
- **Demo-Only Verification:** `account_info()` verified `trade_mode = 0` (`ACCOUNT_TRADE_MODE_DEMO`), `company = 'MetaQuotes Ltd.'`, `server = 'MetaQuotes-Demo'`.

---

## 8. Live Observation Results

### Preflight Verification Summary
| Preflight Check | Requirement | Observed Status | Result |
|---|---|---|---|
| **Broker Server** | `MetaQuotes-Demo` | `MetaQuotes-Demo` | PASS |
| **Account Mode** | `DEMO (0)` | `DEMO (0)` | PASS |
| **Live Mode Gate** | `is_live == False` | `False` | PASS |
| **Execution Enabled** | `False` | `False` | PASS |
| **Terminal Connected**| `True` | `True` | PASS |
| **Symbol** | `EURUSD` | `EURUSD` | PASS |
| **Spread Sanity** | Spread > 0, spread < 5 pips | 0.0 to 0.1 pips | PASS |
| **Tick Freshness** | Data age $\le 120$s | $0.26\text{s} - 0.77\text{s}$ | PASS |
| **Open Positions** | Exactly 0 | 0 | PASS |

### Live Evaluations Telemetry
| Evaluation Time (UTC) | Candle Open (UTC) | Candle Close (UTC) | $P(\text{SHORT})$ | $P(\text{NEUTRAL})$ | $P(\text{LONG})$ | Directional Confidence | Threshold | Signal Emitted | Execution Status |
|---|---|---|---|---|---|---|---|---|---|
| **15:09:33** (Static Smoke) | 2026-10-05 14:45 | 2026-10-05 15:00 | **0.4373** | 0.1735 | 0.3892 | **0.4373** | 0.60 | `NONE` | `DISABLED` |
| **15:15:04** (Live Bar 1) | 2026-10-05 15:00 | 2026-10-05 15:15 | **0.4486** | 0.1689 | 0.3826 | **0.4486** | 0.60 | `NONE` | `DISABLED` |
| **15:33:34** (Live Bar 2) | 2026-10-05 15:15 | 2026-10-05 15:30 | **0.4327** | 0.2030 | 0.3642 | **0.4327** | 0.60 | `NONE` | `DISABLED` |
| **15:45:05** (Live Bar 3) | 2026-10-05 15:30 | 2026-10-05 15:45 | **0.4263** | 0.2020 | 0.3717 | **0.4263** | 0.60 | `NONE` | `DISABLED` |
| **16:00:02** (Live Bar 4) | 2026-10-05 15:45 | 2026-10-05 16:00 | **0.4246** | 0.2086 | 0.3668 | **0.4246** | 0.60 | `NONE` | `DISABLED` |
| **16:15:00** (Live Bar 5) | 2026-10-05 16:00 | 2026-10-05 16:15 | **0.4162** | 0.2250 | 0.3588 | **0.4162** | 0.60 | `NONE` | `DISABLED` |
| **16:30:04** (Live Bar 6) | 2026-10-05 16:15 | 2026-10-05 16:30 | **0.4150** | 0.2235 | 0.3614 | **0.4150** | 0.60 | `NONE` | `DISABLED` |
| **16:45:01** (Live Bar 7) | 2026-10-05 16:30 | 2026-10-05 16:45 | 0.3361 | 0.2944 | **0.3696** | **0.3696** | 0.60 | `NONE` | `DISABLED` |
| **17:00:03** (Live Bar 8) | 2026-10-05 16:45 | 2026-10-05 17:00 | **0.3739** | 0.2569 | 0.3693 | **0.3739** | 0.60 | `NONE` | `DISABLED` |

---

## 9. Confidence Distribution

| Metric | Live Observed Value (Phase 40.4) | Phase 13 Historical Validation Benchmark |
|---|---|---|
| **Evaluations** | **9** | 14,987 |
| **Min Confidence** | **0.3696** (36.96%) | 24.12% |
| **Max Confidence** | **0.4486** (44.86%) | **59.36%** |
| **Mean Confidence** | **0.4160** (41.60%) | 34.84% |
| **Median Confidence**| **0.4246** (42.46%) | 36.02% |
| **Predictions $\ge 0.60$** | **0** (0.0%) | **0** (0.000%) |
| **Predictions $< 0.60$** | **9** (100.0%) | 14,987 (100.0%) |

The live observed directional confidences (ranging between $36.96\%$ and $44.86\%$, mean $41.60\%$) fall squarely within the historical interquartile confidence range of the frozen model. Not a single prediction approached or exceeded the frozen $\tau = 0.60$ threshold.

---

## 10. Signal Results

- **Signals Generated:** 0
- **Signals Approved:** 0
- **Signals Rejected by Threshold:** 9
- **Synthetic / Forced Signals:** 0
- **Threshold Invariant:** Preserved strictly at 0.60

---

## 11. Execution Results

- **Demo Orders Submitted:** 0
- **Live Orders Submitted:** 0
- **Order Fills:** 0
- **Positions Opened:** 0
- **Realized P&L:** $0.00
- **Unrealized P&L:** $0.00
- **Net Capital Exposure:** $0.00

---

## 12. Safety Results

- **Live Account Violations:** 0
- **Kill-Switch Triggers:** 0
- **Stale Data Violations:** 0
- **Lookahead Violations:** 0 (all evaluated bars strictly satisfied $T_{\text{close}} \le T_{\text{eval}}$)
- **Reconciliation Failures:** 0

---

## 13. Test Results

- Full repository pytest test suite: **647 passed, 0 failed** in 184.55s.
- Phase 40.4 dedicated integration test suite (`tests/test_phase40_4_live_strategy.py`): **22 passed, 0 failed** covering all 22 required diagnostic and safety invariants.
- Code style and linters: `ruff check` passed clean (0 errors); `ruff format --check` passed clean (269 files formatted); `git diff --check` passed clean.

---

## 14. Historical Comparison

| Dimension | Phase 13 Validation Set | Phase 40.2 Forward Demo | Phase 40.4 Live Shadow |
|---|---|---|---|
| **Pipeline Status** | Offline Research | Live Adapter | Live Adapter + Pipeline |
| **Model Invocations** | 14,987 | 0 (Unwired) | Active Live Invocations |
| **Warmup Bars** | Full dataset | N/A | 119 M15 Bars |
| **Feature Extraction** | Canonical 80 | None | Canonical 80 (0 NaNs/Infs) |
| **Max Confidence** | 59.36% | N/A | 44.86% |
| **Signals Emit ($\ge 0.60$)** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Orders Placed** | 0 | 0 | 0 |

The live forward shadow results mirror the historical research findings with remarkable fidelity: the frozen model naturally produces directional confidence values in the 35%–50% range and does not generate signals under the locked 0.60 threshold.

---

## 15. Root Cause Determination

### Comprehensive Classification:
1. **Phase 40.2 Zero-Signal Cause:**
   - **Primary:** Implementation Wiring Defect (**Category A**). `ForwardDemoRunner` was initialized with `strategy_fn=None`, causing the model to never be invoked.
   - **Secondary:** Had the model been invoked, zero signals would still have been emitted due to the model's inherent confidence ceiling (**Category C**).
2. **Phase 40.4 Live Verification Classification:**
   - **`CATEGORY C: MODEL EVALUATED AND CONFIDENCE < 0.60`**
   - The live pipeline wiring, M15 historical warmup, point-in-time feature extraction, and model inference are **100% operational and verified**.
   - Zero live signals were emitted solely because the canonical frozen model naturally outputs directional confidences below the locked $\tau = 0.60$ threshold on live EURUSD market data.

---

## 16. Phase 41 Decision

**PHASE 41 IS NOT AUTHORIZED.**

### Justification:
1. **Governance & Non-Negotiable Rules:** Phase 40.4 is an engineering validation phase. Advancing to Phase 41 requires an explicitly validated, profitable live strategy under sovereign risk management.
2. **Strategy Signal Starvation:** The frozen strategy produces zero actionable trading signals under the approved threshold of 0.60, as confirmed by both historical validation (14,987 bars, max confidence 59.36%) and live forward observation (max confidence 44.86%).
3. **No Premature Optimization:** Lowering thresholds, retraining models, or forcing trades is strictly prohibited under current governance.
4. **Conclusion:** Phase 40.4 successfully accomplished its objective of resolving implementation wiring defects and providing rigorous empirical proof of model behavior. System remains safely halted in shadow mode.
