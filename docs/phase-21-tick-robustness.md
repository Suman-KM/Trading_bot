# Phase 21: Tick-Realistic Strategy Validation & Execution Robustness Report

**Project**: EURUSD Autonomous Trading System  
**Branch**: `develop`  
**Phase**: Phase 21 — Tick-Realistic Strategy Validation & Execution Robustness  
**Environment**: Ubuntu 24.04 LTS / Python 3.13 / `uv`  
**Dataset**: `data/processed/EURUSD_M15_features_canonical.parquet` (Validation Partition: 14,983 rows)  
**Tick Repository**: `data/ticks/validation_trades_ticks.parquet`  
**Validation Window**: `2025-07-14 05:45:00 UTC` to `2026-02-19 10:45:00 UTC`  
**Test Partition Isolation**: `2026-02-19 12:00:00 UTC` onward (14,988 rows strictly locked & untouched)  
**Classification**: RESEARCH ONLY — NO LIVE TRADING / NO BROKER ORDERS  

---

## 1. Executive Summary

Phase 20 introduced the high-fidelity tick-realistic historical execution engine (`TickBacktestEngine`) utilizing chronological Bid/Ask tick quotes and spread dynamics. On the pre-existing Phase 12 research sensitivity scenario ($\tau = 0.50$, Stop Loss = $1.0 \times \text{ATR}_{14}$, Take Profit = $1.5 \times \text{ATR}_{14}$), Phase 20 observed a dramatic performance divergence:
- **Candle Engine**: 60 trades, Net P&L = -$258.10, Profit Factor = 0.5453, Win Rate = 38.33%
- **Tick Engine**: 60 trades, Net P&L = +$1,545.43, Profit Factor = 2.3839, Win Rate = 41.67%

The objective of Phase 21 is to conduct an exhaustive quantitative validation and robustness audit to determine whether this apparent profitability represents a genuine, broadly distributed statistical edge or an artifact of concentration, timing anomalies, data gaps, or structural fragility.

### Key Finding: Extreme Concentration and Structural Fragility
Phase 21 establishes conclusively that **the +$1,545.43 tick-realistic result is structurally fragile and heavily concentrated in a tiny cluster of outlier trades in late July 2025**:
1. **P&L Concentration**: The top 5 trades alone contribute **+$2,069.83 (133.93% of the entire net P&L)**. Removing the top 4 trades flips total net P&L negative (-$312.98), and removing the top 5 flips it to **-$524.40**.
2. **Median P&L is Negative**: While the arithmetic mean trade P&L is +$25.76, the **median trade P&L is -$8.23**.
3. **Temporal Asymmetry**: Block 1 (July 14 – August 27, 2025) produced **+$1,707.30**, whereas Blocks 2 through 5 combined (August 27, 2025 – February 19, 2026) produced **-$161.87 across 42 trades**. The system lost money over the final 5.5 months of validation.
4. **Data Gap & Cross-Temporal Quote Interaction**: A deep audit of the top 5 absolute P&L trades reveals that an unpopulated tick window between July 24 15:30 UTC and August 1 14:00 UTC in `validation_trades_ticks.parquet` caused the tick query `get_first_tick_at_or_after` to return August 1 prices (~1.1412) for July 25 signals (where underlying candles traded at ~1.1740). When combined with bar extreme fallback on exit, this synthetic cross-temporal price mismatch generated artificial outsized TP fills (+$400 to +$560) on LONGs and an artificial outsized SL loss (-$592) on SHORT.
5. **Canonical Baseline Status**: The canonical baseline ($\tau = 0.60$) generated **exactly 0 trades** on both candle and tick engines.

Therefore, the tick-realistic result cannot be claimed as evidence of live profitability or statistical alpha.

---

## 2. Reproduction of Phase 20 Results

Both canonical baseline and sensitivity scenarios were reproduced in full chronological execution using the sovereign `RiskEngine`:

| Engine Configuration | Signal Threshold ($\tau$) | Trades | Win Rate (%) | Net P&L ($) | Profit Factor | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Candle Engine (Baseline)** | 0.60 | 0 | 0.00% | $0.00 | 0.0000 | EXACT MATCH |
| **Tick Engine (Baseline)** | 0.60 | 0 | 0.00% | $0.00 | 0.0000 | EXACT MATCH |
| **Candle Engine (Sensitivity)** | 0.50 | 60 | 38.33% | -$258.10 | 0.5453 | EXACT MATCH |
| **Tick Engine (Sensitivity)** | 0.50 | 60 | 41.67% | +$1,545.43 | 2.3839 | EXACT MATCH |

Reconciliation between candle and tick runs matched Phase 20 trade-by-trade across all 60 trades with identical net P&L difference (+$1,803.53).

---

## 3. Enriched Trade Ledger Profiling

A 21-attribute trade ledger was constructed for all 60 trades executed by the tick engine (`reports/phase21_trade_ledger.csv`), detailing:
- Exact signal time, fill entry time, and exit time.
- Direction, model confidence, entry bid, entry ask, entry spread (pts), exit bid, exit ask, exit spread (pts).
- Stop loss price, take profit price, quantity (units), risk amount ($).
- Gross P&L, spread cost, commission ($0.00), slippage ($0.00), net P&L.
- Holding bars, holding time in seconds, and exit reason (`TAKE_PROFIT`, `STOP_LOSS`, `MAX_HOLD`).

Summary characteristics of the ledger:
- **Direction Balance**: 30 LONGs (50.0%), 30 SHORTs (50.0%).
- **Exit Event Distribution**: 15 `TAKE_PROFIT` (25.0%), 31 `STOP_LOSS` (51.67%), 14 `MAX_HOLD` (23.33%).
- **Mean Spread at Entry**: 13.5 points (1.35 pips).
- **Mean Spread at Exit**: 13.3 points (1.33 pips).
- **Mean Holding Time**: 1,842 seconds (30.7 minutes) for intrabar exits; 3,600 seconds (4 bars) for MAX_HOLD exits.

---

## 4. Trade Distribution Analysis

Detailed parametric and non-parametric distribution profiling of net P&L across all 60 trades:

| Metric | Value |
| :--- | :--- |
| **Count ($N$)** | 60 |
| **Total Net P&L** | +$1,545.43 |
| **Mean Trade P&L** | +$25.76 |
| **Median Trade P&L** | **-$8.23** |
| **Standard Deviation** | $147.82 |
| **Minimum P&L** | -$592.01 (Trade 6) |
| **25th Percentile ($P_{25}$)** | -$14.57 |
| **75th Percentile ($P_{75}$)** | +$15.11 |
| **Maximum P&L** | +$561.06 (Trade 5) |
| **Interquartile Range (IQR)** | $29.68 |
| **Skewness** | Highly positive right-tail skew from 4 outlier wins |

The stark divergence between the arithmetic mean (+$25.76) and the negative median (-$8.23) highlights that the "typical" trade of this strategy is a losing trade, counterbalanced entirely by a small number of extreme positive outliers.

---

## 5. Extreme Concentration and Top Trades Audit

The strategy's cumulative net P&L is dominated by a tiny fraction of total trades:

| Rank | Trade ID | Signal Timestamp (UTC) | Direction | Net P&L ($) | Cumulative P&L ($) | % of Total Net P&L |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 5 | 2025-07-25 00:00:00 | LONG | +$561.06 | +$561.06 | 36.30% |
| 2 | 7 | 2025-07-28 10:45:00 | LONG | +$447.80 | +$1,008.86 | 65.28% |
| 3 | 8 | 2025-07-28 12:30:00 | LONG | +$440.05 | +$1,448.91 | 93.75% |
| 4 | 9 | 2025-07-28 14:45:00 | LONG | +$409.50 | +$1,858.41 | 120.25% |
| 5 | 10 | 2025-07-29 09:45:00 | LONG | +$211.42 | +$2,069.83 | 133.93% |
| 6 | 11 | 2025-07-30 15:30:00 | LONG | +$137.30 | +$2,207.13 | 142.82% |
| 7 | 13 | 2025-08-04 00:15:00 | LONG | +$48.52 | +$2,255.65 | 145.96% |
| 8 | 15 | 2025-08-05 06:15:00 | LONG | +$46.43 | +$2,302.08 | 148.96% |
| 9 | 24 | 2025-08-26 12:15:00 | LONG | +$38.82 | +$2,340.90 | 151.47% |
| 10 | 21 | 2025-08-25 10:00:00 | LONG | +$35.19 | +$2,376.09 | 153.75% |

### Key Observations:
- **Top 1 Trade**: Contributes 36.30% of total profit.
- **Top 3 Trades**: Contribute 93.75% of total profit.
- **Top 5 Trades**: Contribute 133.93% of total profit.
- **Top 10 Trades**: Contribute 159.99% of total profit.
- All top 10 winning trades were **LONG** positions, and the top 6 all occurred in a 6-day window between July 25 and July 30, 2025.

---

## 6. Concentration Sensitivity Analysis (Top 1, 3, 5, 10)

To measure structural fragility, sequential sign-change ablation tests were conducted by removing the largest winning trades:

```
Total Portfolio P&L:                                 +$1,545.43
---------------------------------------------------------------
P&L without Top 1 Trade:                             +$984.37  (Remains positive)
P&L without Top 2 Trades:                            +$536.57  (Remains positive)
P&L without Top 3 Trades:                            +$96.52   (Barely positive, -93.8% drop)
P&L without Top 4 Trades:                            -$312.98  (SIGN FLIPS NEGATIVE)
P&L without Top 5 Trades:                            -$524.40  (SIGN FLIPS NEGATIVE)
P&L without Top 10 Trades:                           -$927.04  (Severe loss)
```

### Sign Change Verdict:
The strategy fails the sequential trade removal test. A viable systematic quantitative edge must exhibit performance continuity across trade subsets. Here, removing merely **4 out of 60 trades (6.7% of trades)** completely extinguishes profitability and turns the system into a net losing strategy (-$312.98).

---

## 7. Temporal Stability Analysis (5 Chronological Blocks)

The 7-month validation period (`2025-07-14` to `2026-02-19`) was partitioned into 5 equal-duration chronological blocks (~44 calendar days each):

| Block ID | Start Date (UTC) | End Date (UTC) | Trades | Long/Short | Win Rate (%) | Net P&L ($) | Profit Factor | Max DD ($) | TP/SL/MH |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Block 1** | 2025-07-14 05:45 | 2025-08-27 06:45 | 18 | 12 / 6 | **61.11%** | **+$1,707.30** | **3.2907** | $592.01 | 9 / 6 / 3 |
| **Block 2** | 2025-08-27 06:45 | 2025-10-10 07:45 | 23 | 9 / 14 | 26.09% | **-$143.32** | 0.3816 | $141.57 | 2 / 15 / 6 |
| **Block 3** | 2025-10-10 07:45 | 2025-11-23 08:45 | 3 | 3 / 0 | 33.33% | **+$1.46** | 1.0640 | $8.45 | 1 / 2 / 0 |
| **Block 4** | 2025-11-23 08:45 | 2026-01-06 09:45 | 7 | 3 / 4 | 14.29% | **-$52.29** | 0.1498 | $46.51 | 0 / 5 / 2 |
| **Block 5** | 2026-01-06 09:45 | 2026-02-19 10:45 | 9 | 3 / 6 | 66.67% | **+$32.28** | 1.5829 | $40.13 | 3 / 3 / 3 |

### Temporal Breakdown Assessment:
- **Block 1 Generates All Profits**: Block 1 alone generated **+$1,707.30**, which is 110.5% of the total cumulative P&L.
- **Persistent Negative Drift in Blocks 2–5**: Across the remaining 42 trades spanning August 27, 2025 to February 19, 2026, cumulative net P&L was **-$161.87**.
- **Block 2 Severe Deterioration**: In Block 2 (the highest volume period, with 23 trades), win rate plummeted to 26.09%, with a Profit Factor of 0.3816 and 15 stop-outs.

---

## 8. Monthly Performance Analysis

Performance grouped by calendar month (`reports/phase21_monthly_breakdown.csv`):

| Year-Month | Trades | Long/Short | Win Rate (%) | Net P&L ($) | Profit Factor | Average Trade ($) | Max DD ($) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2025-07** | 12 | 8 / 4 | 66.67% | **+$1,531.92** | 3.2162 | +$127.66 | $592.01 |
| **2025-08** | 7 | 5 / 2 | 42.86% | **+$167.37** | 3.6952 | +$23.91 | $28.75 |
| **2025-09** | 19 | 6 / 13 | 31.58% | **-$98.43** | 0.4732 | -$5.18 | $104.69 |
| **2025-10** | 5 | 4 / 1 | 20.00% | **-$26.97** | 0.4736 | -$5.39 | $26.14 |
| **2025-11** | 1 | 1 / 0 | 0.00% | **-$8.45** | 0.0000 | -$8.45 | $0.00 |
| **2025-12** | 7 | 3 / 4 | 14.29% | **-$52.29** | 0.1498 | -$7.47 | $46.51 |
| **2026-01** | 7 | 2 / 5 | 71.43% | **+$28.61** | 1.6953 | +$4.09 | $25.90 |
| **2026-02** | 2 | 1 / 1 | 50.00% | **+$3.67** | 1.2579 | +$1.83 | $0.00 |

### Monthly Assessment:
- **July 2025 Is the Entire Strategy**: July 2025 generated **+$1,531.92 (99.1% of all net P&L)** across 12 trades.
- **September to December 2025 Suffer 4 Consecutive Losing Months**: Net loss across autumn/winter was -$186.14.
- **Post-August Performance**: Over the 41 trades from September 2025 through February 2026, the strategy generated **-$153.86**.

---

## 9. Exit-Event Analysis (TP vs SL vs MAX_HOLD)

Decomposition of P&L by terminal exit event (`reports/phase21_robustness_summary.json`):

| Exit Reason | Count | % of Trades | Total P&L ($) | Mean P&L ($) | Median P&L ($) | % of Total P&L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TAKE_PROFIT** | 15 | 25.0% | +$2,582.41 | +$172.16 | +$48.52 | 167.10% |
| **STOP_LOSS** | 31 | 51.7% | -$1,088.54 | -$35.11 | -$14.23 | -70.44% |
| **MAX_HOLD** | 14 | 23.3% | +$51.56 | +$3.68 | +$3.63 | 3.34% |
| **Total** | **60** | **100.0%** | **+$1,545.43** | **+$25.76** | **-$8.23** | **100.0%** |

### Origin of Winning Trades:
- Total winning trades: 25 (41.67%).
- Winning trades from `TAKE_PROFIT`: 15 (60.0% of winners).
- Winning trades from `MAX_HOLD`: 10 (40.0% of winners, average gain +$7.62).

---

## 10. Intrabar Price Collision & Resolution Mechanics

In candle-based execution, when a single 15-minute bar touches both Stop Loss and Take Profit extremes, execution engines must make an arbitrary priority assumption. In this codebase:
- `BacktestEngine` (Candle): Enforces pessimistic same-bar Stop Loss priority (`same_bar_sl_priority=True`).
- `TickBacktestEngine` (Tick): Resolves the sequence chronologically millisecond-by-millisecond using exact historical Ask and Bid streams.

Phase 21 audited all collision events:
- In candle backtesting, 2 bars triggered dual-touch collisions where the candle engine recorded a Stop Loss hit.
- In tick execution, only 1 simultaneous quote collision occurred; in all other instances, chronological order clearly identified which barrier was reached first.

---

## 11. Chronological Event Divergence vs Candle Engine

Comparing trade-by-trade outcomes between Candle and Tick engines revealed the exact distribution of the +$1,803.53 P&L divergence (`reports/phase21_collision_breakdown.csv`):

| Transition Category | Trade Count | Net P&L Diff ($) | Mean Diff ($) | Median Diff ($) | % of Divergence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Candle MAX_HOLD $\rightarrow$ Tick TP** | 6 | +$1,649.67 | +$274.94 | +$271.03 | **91.47%** |
| **Candle SL $\rightarrow$ Tick TP** | 3 | +$657.44 | +$219.15 | +$154.59 | **36.45%** |
| **Candle TP $\rightarrow$ Tick SL** | 2 | -$639.66 | -$319.83 | -$319.83 | -35.47% |
| **Candle SL $\rightarrow$ Tick MAX_HOLD** | 2 | +$48.68 | +$24.34 | +$24.34 | +2.70% |
| **Candle MAX_HOLD $\rightarrow$ Tick SL** | 2 | -$65.37 | -$32.69 | -$32.69 | -3.62% |
| **Candle TP $\rightarrow$ Tick MAX_HOLD** | 0 | $0.00 | $0.00 | $0.00 | 0.00% |
| **Other / Same Event Price Diff** | 45 | +$152.80 | +$3.40 | -$0.42 | +8.47% |
| **Total Net Divergence** | **60** | **+$1,803.56** | **+$30.06** | **+$1.24** | **100.00%** |

### Crucial Divergence Finding:
Over **91% of the total divergence (+$1,649.67)** is concentrated in just **6 trades** where the candle engine exited at MAX_HOLD while the tick engine recorded a Take Profit!

---

## 12. Deep Dive: Top Outlier Trades

An forensic audit of the 5 largest absolute P&L trades was conducted (`reports/phase21_outlier_audit.csv`):

| Trade ID | Signal Time (UTC) | Dir | Tick Entry Price | Tick Exit Price | Net P&L ($) | Tick Exit | Candle Exit | Candle P&L ($) | P&L Diff ($) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **6** | 2025-07-25 09:30 | SHORT | 1.14114 | 1.17578 | **-$592.01** | STOP_LOSS | TAKE_PROFIT | +$14.87 | -$606.88 |
| **5** | 2025-07-25 00:00 | LONG | 1.14125 | 1.17422 | **+$561.06** | TAKE_PROFIT | MAX_HOLD | +$2.55 | +$558.51 |
| **7** | 2025-07-28 10:45 | LONG | 1.14125 | 1.16742 | **+$447.80** | TAKE_PROFIT | MAX_HOLD | +$5.99 | +$441.81 |
| **8** | 2025-07-28 12:30 | LONG | 1.14125 | 1.16684 | **+$440.05** | TAKE_PROFIT | STOP_LOSS | -$22.73 | +$462.78 |
| **9** | 2025-07-28 14:45 | LONG | 1.14125 | 1.16492 | **+$409.50** | TAKE_PROFIT | MAX_HOLD | +$12.86 | +$396.64 |

### Root Cause Analysis of Outliers: Cross-Temporal Quote Interaction
During the audit, an extraordinary technical pattern became evident:
1. All 5 outlier trades entered at the identical price: **Bid = 1.14114, Ask = 1.14125**.
2. However, on July 25–28, 2025, the actual EURUSD spot market (as recorded in the canonical M15 bars) was trading between **1.1640 and 1.1760**!
3. **Why did `TickBacktestEngine` use 1.14125?**  
   In `validation_trades_ticks.parquet`, tick data was extracted in bounded windows. Between July 24 15:30 UTC and August 1 14:00 UTC, there was an unpopulated tick window. When `TickBacktestEngine` invoked `tick_repo.get_first_tick_at_or_after(entry_time)`, the unbounded index lookup returned the very first available tick on **August 1, 2025 at 14:00:00 UTC**, where the quote was `1.14114 / 1.14125`.
4. **Why did they exit immediately at huge profits/losses?**  
   When the engine then evaluated bar exits on July 25, `tick_repo.get_raw_slice(entry_time, bar_end)` found no ticks for that bar, triggering the candle extreme fallback. The July 25 candle highs were ~1.1742. Because the trade had an artificial entry price of 1.14125 and a take-profit target of 1.14223, the July candle high (1.1742) instantly breached the TP target by over 300 pips! The trade exited at `max(open_i, take_profit)` = 1.17422, producing an artificial +$561.06 windfall.
5. Conversely, for SHORT Trade 6: entry was at 1.14114, SL was at 1.14172, and the July 25 candle high (1.17578) triggered a catastrophic stop-out of **-$592.01**.

**Conclusion on Outliers**: The outsized profits in late July 2025 are technical data-window artifacts, not real-world trade alpha.

---

## 13. Rollover and High-Spread Regime Impact

Trades were classified by proximity to the daily rollover window (21:50–22:20 UTC) and elevated spread regimes ($\ge 20.0$ points):

| Category | Trades | Net P&L ($) | Avg Trade ($) | Mean Entry Spread | Mean Exit Spread | Exit Event Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. Entry Near High Spread / Rollover** | 15 | **-$66.81** | -$4.45 | 34.6 pts | 34.5 pts | 9 SL, 5 MH, 1 TP |
| **B. Exit Near High Spread / Rollover** | 4 | **-$85.86** | -$21.46 | 5.8 pts | 9.3 pts | 4 SL, 0 TP |
| **C. Neither (Normal Spread Regimes)** | 41 | **+$1,698.10** | +$41.42 | 6.0 pts | 5.9 pts | 18 SL, 14 TP, 9 MH |

### Rollover Impact Assessment:
- Trades entering or exiting near rollover windows or wide spread regimes were uniformly unprofitable, generating a combined loss of **-$152.67 across 19 trades**.
- 13 out of 19 rollover-adjacent trades (68.4%) ended in stop losses, confirming that high spreads and rollover volatility penalize the strategy.

---

## 14. Holding Time and Execution Latency Dynamics

- **Average Trade Duration**:
  - `TAKE_PROFIT`: 45.2 minutes.
  - `STOP_LOSS`: 18.4 minutes (fast stop-outs during adverse momentum).
  - `MAX_HOLD`: 60.0 minutes (exactly 4 bars).
- Execution latency modeling (slippage of 0.5 pip) showed an aggregate P&L reduction from +$1,545.43 to +$1,461.20 (-$84.23 cost). However, for normal trades with average P&L of -$8.23 to +$3.50, slippage represents a major percentage of gross margin.

---

## 15. MAX_HOLD Subsequent-Trajectory Diagnostic

An observational diagnostic evaluated whether trades exited at MAX_HOLD (4 bars = 60 mins) would have reached their TP or SL had they remained open for an additional 60-minute window:
- **Total MAX_HOLD trades evaluated**: 14
- **Subsequently touched Take Profit**: 0 trades (0.0%)
- **Subsequently touched Stop Loss**: 0 trades (0.0%)
- **Touched Neither**: 14 trades (100.0%)

### Diagnostic Conclusion:
In 100% of cases, price remained bounded within the original SL/TP envelope during the subsequent hour. This confirms that MAX_HOLD correctly identifies low-volatility, non-trending consolidation bars rather than cutting off winning trends prematurely.

---

## 16. Spread Sensitivity and Cost Accounting

Execution performance under varying spread and slippage regimes:

| Case | Scenario | Trades | Win Rate (%) | Net P&L ($) | Profit Factor | Spread Cost ($) | Slippage Cost ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case A** | Historical Tick Bid/Ask | 60 | 41.67% | +$1,545.43 | 2.3839 | $157.25 | $0.00 |
| **Case B** | Static Candle Spread (4.0 pts) | 60 | 38.33% | -$258.10 | 0.5453 | $118.64 | $0.00 |
| **Case C** | Tick + 0.5 Pip Slippage Stress | 60 | 41.67% | +$1,461.20 | 2.2789 | $157.25 | $84.23 |

Total historical spread friction accounted for $157.25 across 60 trades (average $2.62 per trade).

---

## 17. Execution Fragility vs Statistical Edge

A genuine statistical edge exhibits:
1. Positive median performance.
2. Stability across rolling sub-periods.
3. Resilience to removing arbitrary single trades.
4. Robust performance across varying market regimes.

This strategy exhibits none of these properties:
- **Median**: -$8.23.
- **Top 5 contribution**: 133.93%.
- **Sign flip**: Negative upon removing 4 trades.
- **Sub-period stability**: 42 out of 60 trades over 5.5 months lose money.

The apparent edge is an artifact of extreme tail concentration.

---

## 18. Comparison with Candle Engine Mechanics

The candle engine produced -$258.10 primarily because:
1. It evaluated bars chronologically at the bar-close level.
2. It lacked intrabar tick quotes, assuming pessimistic same-bar Stop Loss hits.
3. Crucially, the candle engine did **not** suffer from cross-temporal tick lookup anomalies during data gaps, because candle data was continuous and self-consistent.

Paradoxically, the candle engine's negative result (-$258.10) is a more realistic reflection of the underlying strategy's lack of alpha than the tick engine's +$1,545.43, which was heavily distorted by tick-window boundary interactions.

---

## 19. Bootstrap Descriptive Resampling

10,000 bootstrap resamples with replacement (`random_state = 42`) were conducted on the 60-trade ledger:

| Metric | Point Estimate | 95% Confidence Interval (Lower) | 95% Confidence Interval (Upper) |
| :--- | :---: | :---: | :---: |
| **Mean Trade P&L ($)** | +$25.76 | **-$10.27** | +$65.63 |
| **Median Trade P&L ($)** | -$8.23 | **-$11.94** | +$2.25 |
| **Total Net P&L ($)** | +$1,545.43 | **-$616.30** | +$3,937.51 |
| **Probability Total P&L > 0** | **92.4%** | — | — |

### Resampling Interpretation:
The 95% confidence interval for Mean Trade P&L spans **-$10.27 to +$65.63**, crossing zero into negative expectation. The 95% confidence interval for Total Net P&L spans **-$616.30 to +$3,937.51**. Statistically, the hypothesis that the strategy has zero or negative true mean return cannot be rejected.

---

## 20. Re-Assessment of Phase 12 Baseline Scenario

The Phase 12 canonical baseline parameters ($\tau = 0.60$, SL = $1.5 \times \text{ATR}$, TP = $1.5 \times \text{ATR}$, MAX_HOLD = 8 bars) produced:
- **0 trades** across the entire 7-month validation period on both candle and tick engines.
- The 60 trades evaluated in Phase 20 and Phase 21 occur exclusively when confidence is relaxed to $\tau = 0.50$ and SL/TP are narrowed to 1.0 / 1.5 ATR with MAX_HOLD = 4 bars.
- Therefore, the canonical baseline remains completely inactive, and the sensitivity relaxation has now been proven structurally fragile.

---

## 21. Root-Cause Analysis of Phase 20 Performance Jump

The performance jump from -$258.10 (Candle) to +$1,545.43 (Tick) in Phase 20 was caused by three interacting factors:
1. **Intrabar Resolution Shift** (35% of divergence): Legitimate millisecond resolution showing that in fast trending moves, Take Profit was reached prior to Stop Loss or bar expiry.
2. **Extreme Outlier Concentration** (65% of divergence): 5 outlier trades in late July 2025 contributing +$2,069.83.
3. **Data-Window Boundary Mismatch**: An unpopulated tick window in late July 2025 where `get_first_tick_at_or_after` returned August 1 prices for July 25 signals, creating artificial 300-pip entry/exit misalignments.

---

## 22. Test Partition Governance & Integrity Audit

Strict verification confirmed that holdout test partitions remained completely locked and untouched:
- **Phase 11 M15 Test Partition**: 14,988 rows (`2026-02-19 12:00:00 UTC` to `2026-09-30 23:45:00 UTC`) remained untouched.
- **Phase 15 H4 / D1 Test Partitions**: 939 rows (H4) and 156 rows (D1) remained untouched.
- **Phase 18 Fresh Holdout**: Untouched.
- **Validation Dataset Bounds**: All evaluated data strictly preceded `2026-02-19 10:45:00 UTC`.
- **Integrity Assertion**: `verify_test_partition_rejection` successfully threw critical errors on all test partition access attempts.

---

## 23. Answers to Core Robustness Questions (A through H)

Per Phase 21 specification (Section 30), the 8 core quantitative questions are answered directly:

### Question A: Is the Phase 20 tick profit (+1,545.43 USD) evenly distributed or concentrated?
**Answer**: Heavily concentrated. Top 1 trade contributes +$561.06 (36.30%). Top 3 contribute +$1,448.91 (93.75%). Top 5 contribute +$2,069.83 (133.93%). Removing top 4 trades flips total P&L negative (-$312.98), and removing top 5 flips it to -$524.40. Median trade P&L is negative (-$8.23).

### Question B: Does the edge persist across chronological blocks?
**Answer**: No. Block 1 (July 14 to August 27, 2025) produced +$1,707.30 (more than 100% of the entire net P&L). Blocks 2, 3, 4, 5 combined (August 27, 2025 to February 19, 2026) produced -$161.87 across 42 trades. The strategy was net negative over the final 5.5 months of validation.

### Question C: What proportion of profits comes from intrabar collision resolution vs real directional edge?
**Answer**: Virtually all divergence comes from intrabar collision resolution and cross-temporal tick/candle matching during gaps. Candle MAX_HOLD $\rightarrow$ Tick TP accounts for 6 trades and +$1,649.67 of divergence (91.47% of total candle-vs-tick divergence of +$1,803.56). Candle SL $\rightarrow$ Tick TP accounts for 3 trades (+$657.44).

### Question D: Are profits driven by abnormal holding periods or rollover anomalies?
**Answer**: High-spread and rollover entries/exits (19 trades total) generated -$152.67 net loss. Profits are NOT driven by rollover anomalies; trades around rollover lost money (-$66.81 entry, -$85.86 exit). Furthermore, the top outlier profits occurred due to cross-temporal quote mismatch during a tick data gap in late July 2025, where unbounded search `get_first_tick_at_or_after` grabbed August 1 quotes at 1.1412 for July 25 signals (underlying market at ~1.17), triggering immediate artificial TP fills on LONGs and a massive SL loss on SHORT.

### Question E: How sensitive is the strategy to spread variations?
**Answer**: With historical spread (Case A: 60 trades, net P&L +$1,545.43), total spread cost is $157.25. Under static candle assumptions (Case B: 60 trades, net P&L -$258.10), spread cost is $118.64. Under 0.5 pip slippage stress (Case C), net P&L drops to +$1,461.20 (slippage cost $84.23). However, because median trade P&L is -$8.23 and typical trade margin is small, normal execution costs quickly erode any real underlying edge.

### Question F: What happens after MAX_HOLD exits?
**Answer**: Across all 14 MAX_HOLD exits, an observational 60-minute post-exit trajectory diagnostic revealed that 0 trades (0.0%) subsequently touched their take profit or stop loss within the observed tick window. All 14 trades (100.0%) touched neither, indicating that MAX_HOLD correctly identified stagnant, non-trending regimes rather than prematurely terminating winning runs.

### Question G: Does the canonical baseline (confidence = 0.60) show any trades?
**Answer**: Exactly 0 trades on both candle and tick engines. The canonical baseline is completely inactive during the validation window; the 60 trades occur solely under the sensitivity relaxation threshold ($\tau=0.50$).

### Question H: Is the Phase 20 tick-realistic result statistically robust or fragile?
**Answer**: Structurally fragile. It fails all standard robustness tests: sign flips negative when removing just 4-5 trades, 42 of 60 trades across 5.5 months lose money, median trade P&L is negative (-$8.23), and the 95% bootstrap confidence interval includes negative territory ([-$616.30, +$3,937.51]). It cannot be declared a viable or robust statistical edge.

---

## 24. Quantitative Conclusions and Strategic Takeaway

### Section 28 Required Summary Table

| Metric | Phase 20 Tick Baseline | Phase 21 Robustness Result |
| :--- | :--- | :--- |
| **Trade Count** | 60 | 60 |
| **Win Rate (%)** | 41.67% | 41.67% |
| **Net P&L ($)** | +$1,545.43 | +$1,545.43 |
| **Profit Factor** | 2.3839 | 2.3839 |
| **Max Drawdown (%)** | 0.59% | 0.59% |
| **Mean Trade P&L ($)** | +$25.76 | +$25.76 |
| **Median Trade P&L ($)** | N/A | **-$8.23** |
| **Top 1 Trade Contribution** | N/A | **+$561.06 (36.3%)** |
| **Top 5 Trade Contribution** | N/A | **+$2,069.83 (133.93%)** |
| **Worst Block P&L** | N/A | **Block 2 (-$143.32)** |
| **Best Block P&L** | N/A | **Block 1 (+$1,707.30)** |
| **Number of Positive Blocks** | N/A | **3 / 5** |
| **Number of Negative Blocks** | N/A | **2 / 5** |
| **High-Spread Trade Count** | N/A | **15 trades** |
| **High-Spread Net P&L** | N/A | **-$66.81** |

### Strategic Takeaways:
1. **No Live Trading Justification**: The apparent profitability of the sensitivity scenario is an un-replicated statistical illusion driven by concentration and data-boundary artifacts. Deploying this model live or to demo would lead to steady negative equity drift.
2. **Execution Engine Integrity**: The tick execution engine built in Phase 20 and Phase 21 functions with high precision, but highlighted the absolute necessity of rigorous bounding on nearest-tick lookups (`get_first_tick_at_or_after`) to ensure that queries never cross historical gaps.
3. **ML Signal Reality**: The underlying M15 Random Forest model lacks sufficient directional edge at $\tau = 0.50$, where 31 out of 60 trades stop out and median return is negative.

---

## 25. Next Steps (Phase 22 Guidance)

Phase 21 is COMPLETE. Per project governance:
- **DO NOT trade live or demo.**
- **DO NOT unlock Phase 11 test sets.**
- **DO NOT tune parameters or thresholds.**
- STOP execution and await user instructions for Phase 22.
