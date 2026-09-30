# Phase 21.1 — Tick Data Integrity Repair & Affected Result Revalidation

> **Executive Statement**:
> "Phase 21 discovered a tick-data coverage defect affecting the Phase 20 execution dataset. Phase 21.1 repairs the lookup behavior, establishes a strict coverage model with gap detection, eliminates unsafe candle-extreme fallbacks, extracts missing historical tick windows from MT5, and re-evaluates the affected research scenario."

---

## 1. Discovery

During Phase 21 execution robustness analysis, an audit of individual trade outcomes and P&L concentration identified anomalous outcomes in the Phase 20 tick-realistic backtest for the EURUSD M15 strategy sensitivity scenario ($\tau=0.50$, Stop Loss = $1.0 \times \text{ATR}_{14}$, Take Profit = $1.5 \times \text{ATR}_{14}$).

Specifically, five trades generated between 2025-07-25 and 2025-07-28 produced anomalous, uncharacteristic P&L amounts exceeding 300 pips:
- **Trade 5**: $+\$561.06$
- **Trade 6**: $-\$592.01$
- **Trade 7**: $+\$447.80$
- **Trade 8**: $+\$440.05$
- **Trade 9**: $+\$409.50$

For a strategy configured with an ATR-based target of $1.5 \times \text{ATR}_{14}$ (typically 12–18 pips on EURUSD M15, or approximately $\$12.00$ to $\$18.00$ on 0.1-lot sizing), single-trade gains of $+\$561.06$ represent a $35\times$ target overshoot. An investigation into the raw tick quotes associated with these fills revealed that signals emitted in late July 2025 were being filled at Bid/Ask quotes of $\sim 1.14120$, even though the prevailing EURUSD spot market throughout late July 2025 was trading near $1.17400$–$1.17600$.

---

## 2. Original Phase 20 Issue

The Phase 20 tick-realistic backtest concluded that tick execution produced a dramatic performance divergence from candle-level simulation:
- **Candle Execution**: 60 trades, Net P&L $= -\$258.10$, Profit Factor $= 0.5453$
- **Original Phase 20 Tick Execution**: 60 trades, Net P&L $= +\$1,545.43$, Profit Factor $= 2.3839$

Phase 20 attributed this $+\$1,803.53$ divergence almost entirely to chronological intrabar execution dynamics. However, Phase 21.1 demonstrates that **$+\$1,868.76$ of the apparent tick P&L was purely artificial**, created by two interacting bugs:
1. **Unbounded Future Quote Lookup**: When searching for the first tick at or after a bar's open timestamp, `get_first_tick_at_or_after()` jumped across an unpopulated data gap spanning $190.5$ hours (7.9 days), pairing a July 25 signal with an August 1 quote.
2. **Unsafe Candle-Extreme Fallback**: When evaluating open positions inside bars where tick data was absent, the engine fell back to candle OHLC extremes. Mixing an entry price from August ($1.14120$) with candle extremes from July ($1.17450$) manufactured artificial $300$-pip gains and losses.

---

## 3. Root Cause Analysis

### A. Exporter Window Omission
In Phase 20, the tick extraction script [`scripts/export_tick_execution_data.py`](file:///home/cino/projects/ai-trading-system/scripts/export_tick_execution_data.py) utilized a hardcoded list of trade windows rather than an automated scan across the entire validation partition. A manual omission in the window list created a complete void in the exported parquet file [`data/raw/microstructure_audit/validation_trades_ticks.parquet`](file:///home/cino/projects/ai-trading-system/data/raw/microstructure_audit/validation_trades_ticks.parquet):
- **Last Tick Before Gap**: `2025-07-24 15:29:59.842 UTC`
- **First Tick After Gap**: `2025-08-01 14:00:00.864 UTC`
- **Gap Duration**: $685,801.02$ seconds ($190.50$ hours / $7.94$ days)

Within this $190.5$-hour window, 14 trade signals occurred (Trades 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17) for which **zero historical ticks existed** in the repository.

### B. Unbounded Binary Search Lookup
The original implementation of `get_first_tick_at_or_after(timestamp)` in [`ai/backtest/tick_data.py`](file:///home/cino/projects/ai-trading-system/ai/backtest/tick_data.py):
```python
# DEFECTIVE IMPLEMENTATION (Phase 20)
def get_first_tick_at_or_after(self, timestamp: datetime) -> TickQuote | None:
    target_ms = int(timestamp.timestamp() * 1000)
    idx = int(np.searchsorted(self.time_msc, target_ms, side="left"))
    if idx >= len(self.time_msc):
        return None
    return self._build_quote(idx)
```
When queried for Trade 5 entry at `2025-07-25 00:15:00 UTC` ($1,753,402,500,000$ ms), `searchsorted` found index $34,506$, which corresponded to `2025-08-01 14:00:00.864 UTC` ($1,754,056,800,864$ ms). The query silently returned a quote from 7.5 days in the future at $1.14120$.

### C. Unsafe Fallback to Candle Extremes
In [`ai/backtest/tick_engine.py`](file:///home/cino/projects/ai-trading-system/ai/backtest/tick_engine.py), if `len(t_slice_ms) == 0` during holding bars, the engine executed:
```python
# DEFECTIVE FALLBACK (Phase 20)
else:
    if position.direction == TradeDirection.LONG:
        touched_sl = lows[i] <= position.stop_loss
        touched_tp = highs[i] >= position.take_profit
        if touched_tp:
            exited = True
            exit_reason = TradeExitReason.TAKE_PROFIT
            exit_bid = max(open_i, position.take_profit)
```
For Trade 5 (LONG):
- Entry Price (from August 1 tick): $1.14130$
- Take Profit Target: $1.14130 + 1.5 \times \text{ATR} \approx 1.14280$
- Candle High on July 25: $1.17650$
- Because $1.17650 \ge 1.14280$, the engine declared an immediate `TAKE_PROFIT` fill at $1.17650$, manufacturing a synthetic $+352$ pip profit.

---

## 4. Tick Coverage Model

Phase 21.1 formalizes a deterministic temporal coverage model in [`ai/backtest/tick_data.py`](file:///home/cino/projects/ai-trading-system/ai/backtest/tick_data.py):

```python
@dataclass(frozen=True, slots=True)
class TickCoverageInterval:
    """Continuous temporal interval of verified tick coverage."""

    start_time_msc: int
    end_time_msc: int
    start_dt: datetime
    end_dt: datetime
    tick_count: int


@dataclass(frozen=True, slots=True)
class TickDataGap:
    """Detected unpopulated gap between tick coverage intervals."""

    gap_start_msc: int
    gap_end_msc: int
    gap_start_dt: datetime
    gap_end_dt: datetime
    duration_seconds: float
    duration_hours: float
    ticks_before: int
    ticks_after: int
```

Every `TickDataRepository` automatically partitions its time series into continuous intervals separated by gaps where consecutive ticks exceed `max_allowed_gap_ms = 300_000` (5 minutes).

---

## 5. Gap Detection

During repository initialization, `_detect_coverage_and_gaps()` scans `np.diff(self.time_msc)`:
1. Any inter-tick duration $> 300$ seconds is cataloged as a `TickDataGap`.
2. Segments between gaps are cataloged as `TickCoverageInterval`.
3. The repository provides `is_window_covered(start_time, end_time)`:
   - Confirms that `[start_time, end_time]` does NOT intersect any detected `TickDataGap`.
   - Confirms that the entire execution window falls strictly within a single `TickCoverageInterval` (within `max_entry_delay_ms = 60_000` tolerance).

---

## 6. Lookup Repair & Fallback Elimination

### A. 5-Point Strict Tick Query
In [`ai/backtest/tick_data.py`](file:///home/cino/projects/ai-trading-system/ai/backtest/tick_data.py), `get_first_tick_at_or_after(timestamp)` enforces 5 sequential invariant checks:
1. **Earliest/Latest Bound Check**: Rejects timestamps strictly before earliest tick or strictly after latest tick.
2. **Gap Containment Check**: If `timestamp` falls inside any known `TickDataGap`, returns `None`.
3. **Binary Search**: Locates candidate index via `np.searchsorted`.
4. **Delay Cap Check**: If candidate tick timestamp exceeds query timestamp by $> 60,000$ ms (60 seconds), returns `None` (preventing future-interval jump).
5. **Coverage Interval Coherence**: Verifies candidate tick belongs to the same continuous interval as query timestamp.

### B. Fallback Elimination in Execution Engine
In [`ai/backtest/tick_engine.py`](file:///home/cino/projects/ai-trading-system/ai/backtest/tick_engine.py):
- **Entry**: If `entry_tick is None`, the engine **does NOT fall back to `open_i`**. Instead, the signal is recorded as `DATA_UNAVAILABLE`, logged in `unavailable_trades`, and cleared.
- **Exit**: Lines 451–484 (candle-extreme fallback) were completely eliminated. If `n_ticks_in_bar == 0` during holding, the position is immediately liquidated at entry price with `net_pnl = 0.0` and marked with `TradeExitReason.DATA_UNAVAILABLE`.

---

## 7. Data Extraction Repair

To restore valid historical tick execution across all validation windows without compromising historical integrity:
1. **Extraction Script**: Built [`scripts/export_tick_execution_data_v2.py`](file:///home/cino/projects/ai-trading-system/scripts/export_tick_execution_data_v2.py) with a decoupled architecture (Linux host controller + headless Wine MT5 worker).
2. **Bounded Retrieval**: Executed bounded tick extractions from the MetaQuotes demo account for all 14 missing trade windows in July 2025.
3. **Performance**: Retrieval completed in $7.82$ seconds (well within the 2-minute hard timeout limit).
4. **Dataset Versioning**:
   - Original `data/raw/microstructure_audit/validation_trades_ticks.parquet` ($449,077$ ticks, $7.2$ MB) remains **completely untouched**.
   - Created `data/raw/microstructure_audit/validation_trades_ticks_v2.parquet` ($537,065$ ticks, $8.6$ MB, $+87,988$ verified ticks).
   - Invariants verified: monotonic timestamps, zero negative spreads, holdout boundary respected.

---

## 8. Affected Trades & Coverage Classification

Every one of the 60 trades in the Phase 12 $\tau=0.50$ sensitivity scenario was audited across both dataset versions:

| Classification | Definition | v1 Original Count | v2 Corrected Count |
| :--- | :--- | :---: | :---: |
| **FULL_TICK_COVERAGE** | Entry tick found $\le 60$s, unbroken ticks across all holding bars | 35 | **50** |
| **PARTIAL_TICK_COVERAGE** | Entry tick found, but horizon crosses into missing data | 1 | **1** |
| **GAP_CROSSED** | Trade window intersects a detected data gap ($>5$ min) | 10 | **9** |
| **NO_TICK_COVERAGE** | 0 ticks exist in entry bar or execution window | 14 | **0** |
| **TOTAL TRADES AUDITED** | All sensitivity signals | 60 | **60** |

- **v1 Original Defect**: 14 trades had zero ticks and were executing via future jumps into August 1 quotes.
- **v2 Corrected Dataset**: All 14 missing trade windows now possess true millisecond tick coverage. 59 of 60 trades execute validly. Only Trade 4 (signal on 2025-07-24 23:45 UTC, entry at midnight during MT5 broker maintenance) remains `DATA_UNAVAILABLE`.

---

## 9. Specific Inspection of July Outlier Trades

Explicit verification of Trades 5, 6, 7, 8, 9 proves that the artificial August 1 quote ($1.14120$) has been completely purged:

| Trade ID | Signal Time (UTC) | Phase 20 Erroneous P&L | Phase 21.1 v1 Status | Corrected v2 Entry Quote | Corrected v2 Exit Status | Corrected v2 Net P&L |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | 2025-07-25 00:00 | $+\$561.06$ | `DATA_UNAVAILABLE` | $1.17524$ Ask | `MAX_HOLD` | **$+\$1.02$** |
| **6** | 2025-07-25 09:30 | $-\$592.01$ | `DATA_UNAVAILABLE` | $1.17409$ Ask | `TAKE_PROFIT` | **$+\$15.12$** |
| **7** | 2025-07-28 10:45 | $+\$447.80$ | `DATA_UNAVAILABLE` | $1.17066$ Ask | `MAX_HOLD` | **$+\$5.30$** |
| **8** | 2025-07-28 12:30 | $+\$440.05$ | `DATA_UNAVAILABLE` | $1.17173$ Bid | `STOP_LOSS` | **$-\$22.77$** |
| **9** | 2025-07-28 14:45 | $+\$409.50$ | `DATA_UNAVAILABLE` | $1.17246$ Ask | `MAX_HOLD` | **$+\$12.00$** |

### Confirmation of Anomaly Elimination:
- **Trade 5**: Artificial $+\$561.06$ ($+352$ pips) collapsed to **$+\$1.02$** (realistic 1-pip max-hold exit).
- **Trade 6**: Artificial $-\$592.01$ ($-372$ pips) collapsed to **$+\$15.12$** (normal TP fill at real July Bid).
- **Trade 7**: Artificial $+\$447.80$ collapsed to **$+\$5.30$**.
- **Trade 8**: Artificial $+\$440.05$ collapsed to **$-\$22.77$** (normal SL fill).
- **Trade 9**: Artificial $+\$409.50$ collapsed to **$+\$12.00$**.

All 300-pip distortions have vanished. Every trade executed against authentic $1.1700$–$1.1755$ July quotes.

---

## 10. Original vs Corrected Comparison

Execution of [`scripts/run_phase21_1_revalidation.py`](file:///home/cino/projects/ai-trading-system/scripts/run_phase21_1_revalidation.py) generates the canonical Section 19 comparison table:

| Metric | Original Phase 20 Result | Corrected Phase 21.1 Result | Absolute Change |
| :--- | :---: | :---: | :---: |
| **Total Trades Emitted** | 60 | 60 | $0$ |
| **Valid Tick Trades** | 60 | **59** | $-1$ |
| **Unavailable Trades** | 0 | **1** | $+1$ |
| **Net P&L (USD)** | **$+\$1,545.43$** | **$-\$323.33$** | **$-\$1,868.76$** |
| **Profit Factor** | **2.3839** | **0.4786** | **$-1.9053$** |
| **Win Rate (%)** | 46.67% | **35.59%** | $-11.08\%$ |
| **Max Drawdown (%)** | 0.54% | **0.36%** | $-0.18\%$ |
| **Mean Trade (USD)** | $+\$25.76$ | **$-\$5.48$** | $-\$31.24$ |
| **Median Trade (USD)** | $-\$18.23$ | **$-\$11.06$** | $+\$7.17$ |
| **Top 1 Trade (USD)** | $+\$561.06$ | **$+\$35.21$** | $-\$525.85$ |
| **Top 5 Contribution (%)** | 153.22% | **N/A (Net Negative)** | Purged |
| **Affected July Trades** | 0 (undetected) | **14 trades repaired** | Identified & Fixed |

### Key Reconciliation Insight:
- Candle Engine Benchmark: Net P&L $= -\$258.10$, Profit Factor $= 0.5453$
- Corrected Tick Engine: Net P&L $= -\$323.33$, Profit Factor $= 0.4786$
- Divergence between Candle and Tick: **$-\$65.23$**

When tick execution is performed on authentic market data without quote jumping or candle fallback, **the tick-level result closely mirrors the candle-level result**. The tick execution is slightly more negative ($-\$323.33$ vs $-\$258.10$) due to realistic Bid/Ask spread drag and entry slippage, which is standard for microstructure modeling.

---

## 11. Re-Run Robustness Analysis on Corrected Valid Trades

All Phase 21 robustness metrics were recalculated strictly across the 59 valid corrected tick trades:

### A. Temporal Stability (5 Equal-Duration Blocks)
| Block ID | Period (UTC) | Valid Trades | Win Rate | Net P&L (USD) | Profit Factor | Average Trade | Max DD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2025-07-14 to 2025-08-27 | 18 | 38.89% | $-\$165.98$ | 0.3542 | $-\$9.22$ | $\$138.96$ |
| **2** | 2025-08-27 to 2025-10-10 | 23 | 26.09% | $-\$140.69$ | 0.3816 | $-\$6.12$ | $\$138.96$ |
| **3** | 2025-10-10 to 2025-11-23 | 3 | 33.33% | $+\$1.44$ | 1.0644 | $+\$0.48$ | $\$8.29$ |
| **4** | 2025-11-23 to 2026-01-06 | 6 | 16.67% | $-\$49.80$ | 0.1536 | $-\$8.30$ | $\$44.12$ |
| **5** | 2026-01-06 to 2026-02-19 | 9 | 66.67% | $+\$31.72$ | 1.5837 | $+\$3.52$ | $\$39.38$ |

4 out of 5 temporal blocks are negative or near break-even. The strategy demonstrates no persistent positive alpha.

### B. Monthly Breakdown
- **2025-07**: 12 trades, Win Rate: 50.0%, Net P&L: $-\$103.34$, PF: 0.4256
- **2025-08**: 7 trades, Win Rate: 14.3%, Net P&L: $-\$70.51$, PF: 0.1703
- **2025-09**: 19 trades, Win Rate: 31.6%, Net P&L: $-\$96.61$, PF: 0.4733
- **2025-10**: 5 trades, Win Rate: 20.0%, Net P&L: $-\$26.48$, PF: 0.4735
- **2025-11**: 1 trade, Win Rate: 0.0%, Net P&L: $-\$8.29$, PF: 0.0000
- **2025-12**: 6 trades, Win Rate: 16.7%, Net P&L: $-\$49.80$, PF: 0.1536
- **2026-01**: 7 trades, Win Rate: 71.4%, Net P&L: $+\$28.11$, PF: 1.6961
- **2026-02**: 2 trades, Win Rate: 50.0%, Net P&L: $+\$3.61$, PF: 1.2586

### C. Exit Reasons Breakdown
- **STOP_LOSS**: 35 trades (59.3%) — $-\$562.88$ gross loss
- **MAX_HOLD**: 16 trades (27.1%) — $-\$56.88$ gross loss
- **TAKE_PROFIT**: 8 trades (13.6%) — $+\$296.43$ gross win
- **DATA_UNAVAILABLE**: 1 trade (1.7%) — $\$0.00$

### D. Slippage Stress Testing (0.5 pip / 5.0 pts per fill)
- **Baseline (0.0 pts slippage)**: Net P&L $= -\$323.33$, Profit Factor $= 0.4786$
- **Stressed (5.0 pts slippage)**: Net P&L $= -\$388.50$, Profit Factor $= 0.4104$
- Adverse impact of $\$65.17$ confirms consistent linear cost degradation.

---

## 12. Test Results

A comprehensive unit test suite in [`tests/test_phase21_1_data_integrity.py`](file:///home/cino/projects/ai-trading-system/tests/test_phase21_1_data_integrity.py) verifies all 10 required data integrity invariants:

1. `test_query_inside_valid_coverage_succeeds`: Confirms query inside valid interval succeeds.
2. `test_query_inside_gap_fails_safely`: Confirms query inside gap returns `None`.
3. `test_query_before_earliest_tick_fails`: Confirms query before earliest tick returns `None`.
4. `test_query_after_latest_tick_fails`: Confirms query after latest tick returns `None`.
5. `test_query_cannot_jump_across_gap`: Confirms query cannot bridge across unpopulated gap.
6. `test_regression_july_25_future_tick_substitution_prevented`: **Explicit regression test proving July 25 2025 query returns `None` and NEVER substitutes August 1 2025 quote**.
7. `test_exit_lookup_cannot_cross_gap`: Confirms `is_window_covered` returns `False` when crossing gaps.
8. `test_candle_fallback_cannot_manufacture_tick_execution`: Confirms uncovered windows yield `DATA_UNAVAILABLE` with $\$0.00$ P&L rather than synthetic candle fills.
9. `test_test_partition_timestamps_rejected`: Confirms timestamps $\ge 2026-02-19\text{ 10:45:00 UTC}$ raise hard governance errors.
10. `test_tick_repository_determinism`: Confirms bit-identical intervals and gaps across runs.

**Suite Status**: 10 passed in 0.44s. Full test suite: 339 passed in 184s.

---

## 13. Governance Compliance

- **Phase 11 Test Set Locked**: The 14,988 test rows starting `2026-02-19 12:00:00 UTC` remained completely untouched and unreferenced.
- **Phase 15 / Phase 18 Holdouts**: Untouched.
- **No Optimization**: Zero threshold tuning, zero SL/TP adjustment, zero model retraining, zero filtering.
- **Untracked Design Preservation**: `docs/mt5-demo-integration-design.md` was preserved in its untracked state and was not modified or staged.

---

## 14. Limitations

1. **Broker Maintenance Intervals**: Midnight rollovers (e.g. 2025-07-24 23:45 UTC) exhibit quote freezes on the MetaQuotes demo server; such intervals are properly tagged as `DATA_UNAVAILABLE`.
2. **Execution Latency**: Simulation assumes immediate next-tick fill ($\le 60$s) without broker queuing latency.
3. **Market Depth**: Only top-of-book (Bid/Ask) was modeled, which is sufficient for standard retail sizes (0.1–1.0 lots) but does not model order book depth exhaustion.

---

## 15. Final Interpretation

The corrected Phase 21.1 result **BECOMES NEGATIVE**:

$$\text{Net P\&L} = -\$323.33 \quad (\text{Profit Factor} = 0.4786, \quad \text{Win Rate} = 35.59\%)$$

The apparent profitability reported in Phase 20 ($+\$1,545.43$, PF $2.3839$) was entirely an artifact of a data-coverage gap in the exported parquet file and unsafe candle-extreme fallback execution. When execution is modeled with chronological data integrity using true historical Bid and Ask quotes, the EURUSD M15 model is non-profitable, closely tracking its negative candle-level benchmark ($-\$258.10$, PF $0.5453$).

**DO NOT START PHASE 22.**
**RESEARCH CONCLUDES AT PHASE 21.1.**
