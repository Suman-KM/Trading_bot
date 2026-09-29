# Phase 12 — Research Trading Strategy & Historical Backtester

**Instrument:** EURUSD  
**Timeframe:** M15  
**Data Partition:** Validation (14,983 bars: 2025-07-14 05:45:00 UTC to 2026-02-19 10:45:00 UTC)  
**Holdout Test Partition:** 14,988 bars (2026-02-19 12:00:00 UTC to 2026-09-25 22:45:00 UTC) — **STRICTLY LOCKED & UNTOUCHED**  
**Model Architecture:** Random Forest (`class_weight="balanced"`, `max_depth=10`, `min_samples_leaf=20`, `n_estimators=100`, `random_state=42`)  
**Training Set:** 69,937 bars (2022-09-19 05:30:00 UTC to 2025-07-14 04:30:00 UTC)  
**Safety Core:** Sovereign, non-bypassable deterministic `RiskEngine` with hard capital limits  
**Execution Environment:** Pure in-memory historical backtesting simulation; zero live broker connectivity, zero MT5 order execution  

---

## 1. Objective

Phase 12 investigates the empirical behavior of an ML-assisted trading strategy under a realistic, event-driven, bar-by-bar historical backtesting simulation. The primary research question is:

> *"How does a strictly causal, ML-assisted directional strategy for EURUSD M15 behave when subjected to realistic market frictions (spread, commission, slippage), deterministic risk constraints, and conservative collision policies?"*

The objective is **not** to produce an artificially inflated equity curve or to curve-fit parameters to achieve profitability. Rather, the objective is to build a mathematically watertight, leakage-safe, reproducible backtesting engine and honestly report the strategy's historical performance and execution bottlenecks.

---

## 2. Architecture & System Flow

The backtesting architecture strictly preserves the project's Core Invariant: **AI/ML generates signals only; the deterministic Safety Core holds sovereign execution authority.**

```
+-----------------------------------------------------------------------------------+
|                              HISTORICAL SIMULATION                                |
|                                                                                   |
|  [Market Data Bar t]  ---> [Point-in-Time Features X_t]                          |
|                                       |                                           |
|                                       v                                           |
|                           [ML Model Inference]                                    |
|                      (Random Forest Balanced - H=4)                               |
|                                       |                                           |
|                                       v                                           |
|                       [Candidate Signal Generation]                               |
|                     (Direction, Confidence, SL/TP)                                |
|                                       |                                           |
|                                       v                                           |
|                        [Sovereign RiskEngine] <----+ [Account State & RiskLimits] |
|                     (Confidence, Loss Limit, Exposure)                            |
|                                       |                                           |
|                      +----------------+----------------+                          |
|                      |                                 |                          |
|                 [APPROVED]                        [REJECTED]                      |
|                      |                                 |                          |
|                      v                                 v                          |
|            [Queue Pending Order]              [Audit Log Rejection]               |
|                      |                                                            |
|                      v (Execute at bar t+1 Open)                                  |
|            [CostModel Execution]                                                  |
|        (Bid/Ask Spread + Slippage)                                                |
|                      |                                                            |
|                      v                                                            |
|       [Bar-by-Bar Position Tracking]                                              |
|   - Same-Bar SL/TP Priority (SL First)                                            |
|   - Max Holding Horizon (4 bars)                                                  |
|   - Cooldown Enforcement (1 bar)                                                  |
|   - Emergency Kill Switch Tripping                                                |
+-----------------------------------------------------------------------------------+
```

The system executes in a clean chronological loop. At bar $t$:
1. Check daily baseline equity and reset daily loss halt at 00:00:00 UTC.
2. If a pending order was approved at bar $t-1$, fill at bar $t$ Open price adjusted for Ask/Bid spread and slippage.
3. If a position is active, evaluate SL/TP collision and holding bars across bar $t$ High/Low/Close.
4. Mark portfolio equity to market and check daily drawdown limit ($1.0\%$).
5. If no position is open, generate point-in-time features for completed bar $t$, obtain model probabilities, evaluate confidence, size position within exposure limits, and pass to `RiskEngine`.
6. Record execution audit events.

---

## 3. Strategy Definition

The strategy operates as a multi-class directional trend-following / mean-reversion filter:
- **Prediction Target:** `direction_4` ($H=4$ bars / 60 minutes, fixed return threshold $\theta = 0.00050 = 5.0\text{ pips}$).
- **Output Classes:** $\{-1.0 \text{ (SHORT)}, 0.0 \text{ (NEUTRAL)}, 1.0 \text{ (LONG)}\}$.
- **Base Model:** Preselected Phase 10 candidate: Random Forest with `class_weight="balanced"`.
- **Confidence Threshold:** $\tau = 0.60$ (predefined research baseline, unoptimized).

---

## 4. Model Selection Provenance

The model artifact and hyperparameter configuration were locked in Phase 10 based exclusively on the pre-test Validation study (`docs/phase-10-model-improvement.md`).
- **Algorithm:** Scikit-learn `RandomForestClassifier` wrapped in `RandomForestBaseline`.
- **Hyperparameters:** `n_estimators=100`, `max_depth=10`, `min_samples_leaf=20`, `class_weight="balanced"`, `random_state=42`, `n_jobs=-1`.
- **Training Data:** Strictly fitted on the Training partition (69,937 bars, 2022-09-19 to 2025-07-14).
- **Holdout Integrity:** The model was never fitted or tuned on the Validation or Test partitions.

---

## 5. Data Period

- **Training Period:** 2022-09-19 05:30:00 UTC to 2025-07-14 04:30:00 UTC (69,937 observations). Used solely for fitting the ML classifier.
- **Validation Period:** 2025-07-14 05:45:00 UTC to 2026-02-19 10:45:00 UTC (14,983 observations). Used for historical backtesting and sensitivity analysis.
- **Test Period (Phase 11 Holdout):** 2026-02-19 12:00:00 UTC to 2026-09-25 22:45:00 UTC (14,988 observations). **Permanently locked and untouched.**

---

## 6. Entry Rules

A candidate entry signal is generated at the close of candle $t$ if and only if:
1. Model predicted class is directional: $\hat{y}_t \in \{-1.0, 1.0\}$.
2. Model confidence exceeds the threshold: $\max_c P(y_t = c \mid X_t) \ge \tau = 0.60$.
3. Portfolio is currently flat (no existing open position on EURUSD).
4. Cooldown requirement is satisfied ($\ge 1\text{ bar}$ since the previous position closed).
5. Emergency Kill Switch is not engaged.
6. The candidate order passes all deterministic `RiskEngine` constraints.

---

## 7. Exit Rules

An active position exits upon the first occurrence of any of the following:
1. **Stop Loss (SL):** High/Low touches the stop price.
2. **Take Profit (TP):** High/Low touches the profit target price.
3. **Maximum Holding Horizon:** Position reaches 4 completed bars ($H=4$ bars = 60 minutes) without touching SL or TP. Exit occurs at bar $t+4$ Close.
4. **End of Dataset:** Any position open on the final candle is flattened at market close with reason `END_OF_DATA`.

---

## 8. SL / TP Methodology

Stop loss and take profit targets are established dynamically at signal time using the 14-period Average True Range ($\text{ATR}_{14}$):
- **Stop Loss Distance:** $\Delta_{\text{SL}} = 1.0 \times \text{ATR}_{14}$
- **Take Profit Distance:** $\Delta_{\text{TP}} = 1.5 \times \text{ATR}_{14}$

For **LONG** positions:
$$\text{SL} = P_{\text{entry}} - \Delta_{\text{SL}}, \quad \text{TP} = P_{\text{entry}} + \Delta_{\text{TP}}$$
For **SHORT** positions:
$$\text{SL} = P_{\text{entry}} + \Delta_{\text{SL}}, \quad \text{TP} = P_{\text{entry}} - \Delta_{\text{TP}}$$

Upon actual execution at next-bar Open, the levels are referenced from the actual filled entry price $P_{\text{entry}}$, ensuring invariant ATR distance.

---

## 9. Spread Methodology

In MetaTrader 5 Forex conventions, OHLC candlestick data represents the **Bid** price.
The execution model applies the bid/ask spread directly:
- **Spread Points Conversion:** $\text{Spread Price} = \text{Spread Points} \times 10^{-5}$ (for 5-digit EURUSD).
- **Default/Fallback Spread:** If historical spread is missing or zero, a default of 10.0 points ($1.0\text{ pip} = 0.00010$) is applied.
- **LONG Positions:** Enter at **Ask** ($P_{\text{open}} + \text{Spread Price}$); exit at **Bid** ($P_{\text{exit}}$).
- **SHORT Positions:** Enter at **Bid** ($P_{\text{open}}$); exit at **Ask** ($P_{\text{exit}} + \text{Spread Price}$).

---

## 10. Commission Assumptions

- The raw historical MT5 dataset does not contain broker commission schedules.
- **Baseline Modeling Assumption:** `commission_per_lot = 0.0`. Zero commission is a baseline research assumption, not evidence that live execution incurs zero fees.
- **Configurable Frictions:** The engine supports arbitrary broker schedules (e.g. $\$7.00$ per round-turn standard lot of 100,000 units), evaluated in the sensitivity scenarios.

---

## 11. Slippage Assumptions

- **Baseline Modeling Assumption:** `slippage = 0.0 points`. Zero slippage is an optimistic baseline assumption.
- **Configurable Frictions:** The engine implements deterministic per-fill slippage in points:
  - BUY fills at $P_{\text{ask}} + \text{Slippage}$
  - SELL fills at $P_{\text{bid}} - \text{Slippage}$
- Evaluated under 5.0 points ($0.5\text{ pip}$) in the sensitivity scenarios.

---

## 12. Position Sizing

Position sizing follows the deterministic formula established in `trading/risk/limits.py`:
$$\text{Risk Capital} = \text{Equity} \times \frac{\text{MAX\_POSITION\_RISK\_PERCENT}}{100} = \text{Equity} \times 0.005$$
$$\text{Stop Distance} = |P_{\text{entry}} - \text{Stop Loss}|$$
$$\text{Raw Quantity} = \frac{\text{Risk Capital}}{\text{Stop Distance}}$$

To prevent violating the portfolio exposure constraint (`MAX_TOTAL_EXPOSURE_PERCENT = 20.0%`), quantity is safely constrained:
$$\text{Max Exposure Quantity} = \frac{\text{Equity} \times 0.20}{P_{\text{entry}}}$$
$$\text{Quantity} = \min(\text{Raw Quantity}, \text{Max Exposure Quantity})$$

This ensures both constraints are strictly satisfied:
1. Trade risk $\le 0.5\%$ of equity.
2. Gross notional exposure $\le 20.0\%$ of equity.

---

## 13. Safety Limits & Sovereign RiskEngine

All candidate trades are submitted to `RiskEngine.evaluate()`. The Safety Core enforces:
- `MAX_DAILY_LOSS_PERCENT = 1.0%`
- `MAX_POSITION_RISK_PERCENT = 0.5%`
- `MAX_OPEN_POSITIONS = 3` (strategy restricts to 1 for EURUSD)
- `MAX_TOTAL_EXPOSURE_PERCENT = 20.0%`
- `MIN_SIGNAL_CONFIDENCE = 0.60`

If the strategy threshold differs from `MIN_SIGNAL_CONFIDENCE`, the stricter constraint sovereignly prevails.

---

## 14. Cooldown

- `COOLDOWN_BARS = 1`.
- When a position exits at bar $t$, no new entry signal may be evaluated on bar $t$. The first bar eligible for strategy evaluation is bar $t+1$, executing at bar $t+2$.

---

## 15. Maximum Holding Period

- Baseline: $H=4$ M15 bars (60 minutes).
- If neither SL nor TP is hit during the 4 bars, the trade is liquidated at market close of bar $t+4$.

---

## 16. Same-Bar SL/TP Ambiguity Policy

When bar $t$ exhibits both $\text{Low} \le \text{SL}$ and $\text{High} \ge \text{TP}$, the intra-bar tick sequence is inherently unknown from M15 OHLC data.
- **Conservative Policy:** The engine **assumes Stop Loss occurred first**.
- **Exit Price:** Filled at the Stop Loss price (or gap open price if gapped beyond SL).
- **Rationale:** Prevents optimistic upward bias in backtest performance.

---

## 17. Baseline Backtest Performance Metrics

Evaluated on the full 14,983-bar Validation period (2025-07-14 to 2026-02-19):

| Metric | Baseline Value | Interpretation / Context |
| :--- | :--- | :--- |
| **Starting Equity** | $\$100,000.00$ | Standard test account capital |
| **Ending Equity** | $\$100,000.00$ | Exact capital preservation |
| **Total Net P&L** | $\$0.00$ | No capital lost or gained |
| **Total Return (%)** | $0.00\%$ | Unchanged equity |
| **Total Trades** | **0** | **All 14,983 bars filtered by confidence** |
| **Winning Trades** | 0 | N/A |
| **Losing Trades** | 0 | N/A |
| **Win Rate** | $0.00\%$ | N/A |
| **Profit Factor** | $0.0000$ | N/A |
| **Max Drawdown** | $\$0.00$ ($0.00\%$) | Zero equity drawdown |
| **Total Transaction Costs** | $\$0.00$ | Zero fees incurred |
| **Total Signals Evaluated** | 14,983 | Full validation coverage |
| **Signals Generated** | 0 | Directional confidence never reached 0.60 |
| **Confidence-Filtered Bars** | 14,983 | $100\%$ filtered at $\tau = 0.60$ |
| **Trades Approved by Risk** | 0 | N/A |
| **Total Rejections** | 0 | Filtered prior to order generation |

### Why Did the Baseline Execute 0 Trades?
This is a critical finding that validates the integrity of Phase 10:
In the 3-class distribution ($\text{SHORT}, \text{NEUTRAL}, \text{LONG}$), random/uninformed probability is $1/3 \approx 33.33\%$.
The Random Forest model's directional confidence for $\text{SHORT}$ and $\text{LONG}$ peaks at $55.0\%$, with a mean of $45.68\%$. At the predefined threshold $\tau = 0.60$, directional predictions are **never** emitted. The only class that ever achieved $\ge 0.60$ probability was $\text{NEUTRAL}$ ($0.0$), which by rule generates no trade (`SignalAction.HOLD`).

The 0.60 confidence / 1.00× ATR SL / 1.50× ATR TP configuration remains the Phase 12 baseline. It generated zero trades because the selected model did not produce directional confidence at or above the 0.60 threshold during the validation period. The zero-trade result is reported as observed and is not used to justify parameter optimization.

---

## 18. Integrity Tests

The test suite in `tests/test_backtest.py` contains 22 automated test functions verifying all 26 mandatory integrity checks:
1. `test_1_and_2_chronological_processing_and_no_future_data` (PASS)
2. `test_3_and_4_no_lookahead_and_next_bar_entry` (PASS)
3. `test_5_long_bid_ask_handling` (PASS)
4. `test_6_short_bid_ask_handling` (PASS)
5. `test_7_spread_conversion` (PASS)
6. `test_8_and_9_sl_and_tp_distance_calculation` (PASS)
7. `test_10_same_bar_sl_tp_collision_conservative_policy` (PASS)
8. `test_11_max_holding_period` (PASS)
9. `test_12_cooldown_enforcement` (PASS)
10. `test_13_max_open_positions` (PASS)
11. `test_14_and_15_daily_loss_limit_and_kill_switch` (PASS)
12. `test_16_position_sizing_respects_limits` (PASS)
13. `test_17_risk_engine_rejection_enforced` (PASS)
14. `test_18_and_19_commission_and_slippage_calculation` (PASS)
15. `test_20_pnl_reconciliation` (PASS)
16. `test_21_deterministic_repeated_runs` (PASS)
17. `test_22_no_target_columns_in_features` (PASS)
18. `test_23_test_partition_isolation_guard` (PASS)
19. `test_24_no_model_fitting_during_simulation` (PASS)
20. `test_25_end_of_data_behavior` (PASS)
21. `test_26_gap_opening_behavior` (PASS)
22. `test_27_anti_lookahead_leakage_test` (PASS)

---

## 19. Leakage Test (Section 31 Verification)

`test_27_anti_lookahead_leakage_test` proves strict anti-lookahead compliance:
1. Strategy is executed on dataset through time $T$ (10 bars).
2. Future candles ($T+1$ to $T+10$) are appended to the dataset.
3. The strategy is re-executed on the extended dataset.
4. **Result:** All predictions, trades, entry times, exit times, fill prices, and net PnLs through time $T$ remain **bit-for-bit identical**. Adding future bars does not alter past decisions.

---

## 20. Sensitivity Analysis

To investigate behavior when the confidence filter permits trade execution, 10 predefined scenarios were evaluated neutrally on the Validation set (without parameter optimization):

| Scenario | Conf ($\tau$) | SL ATR | TP ATR | Slip (pts) | Comm (\$/lot) | Trades | Win Rate | Profit Factor | Net P&L (\$) | Max DD (%) | Avg Trade (\$) | Costs (\$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.60 | 1.00 | 1.50 | 0.0 | 0.0 | 0 | 0.00% | 0.0000 | $\$0.00$ | 0.00% | $\$0.00$ | $\$0.00$ |
| **Conf = 0.50** | 0.50 | 1.00 | 1.50 | 0.0 | 0.0 | 60 | 38.33% | 0.5453 | $-\$258.10$ | 0.29% | $-\$4.30$ | $\$118.64$ |
| **Conf = 0.70** | 0.70 | 1.00 | 1.50 | 0.0 | 0.0 | 0 | 0.00% | 0.0000 | $\$0.00$ | 0.00% | $\$0.00$ | $\$0.00$ |
| **Tighter SL** | 0.50 | 0.75 | 1.50 | 0.0 | 0.0 | 60 | 33.33% | 0.5147 | $-\$240.94$ | 0.27% | $-\$4.02$ | $\$118.65$ |
| **Wider SL** | 0.50 | 1.25 | 1.50 | 0.0 | 0.0 | 59 | 57.63% | 1.1678 | $+\$59.28$ | 0.11% | $+\$1.00$ | $\$120.05$ |
| **Tighter TP** | 0.50 | 1.00 | 1.00 | 0.0 | 0.0 | 61 | 42.62% | 0.5711 | $-\$237.87$ | 0.26% | $-\$3.90$ | $\$120.36$ |
| **Wider TP** | 0.50 | 1.00 | 2.00 | 0.0 | 0.0 | 59 | 37.29% | 0.5527 | $-\$253.03$ | 0.32% | $-\$4.29$ | $\$116.91$ |
| **Slippage** | 0.50 | 1.00 | 1.50 | 5.0 | 0.0 | 60 | 35.00% | 0.4155 | $-\$364.55$ | 0.39% | $-\$6.08$ | $\$220.81$ |
| **Commission** | 0.50 | 1.00 | 1.50 | 0.0 | 7.0 | 60 | 36.67% | 0.4612 | $-\$329.63$ | 0.35% | $-\$5.49$ | $\$190.19$ |
| **Combined** | 0.50 | 1.00 | 1.50 | 5.0 | 7.0 | 60 | 31.67% | 0.3506 | $-\$436.00$ | 0.45% | $-\$7.27$ | $\$292.28$ |

### Quantitative Findings from Sensitivity:
1. **Low Win Rate / Net Losses at $\tau = 0.50$:** At $\tau = 0.50$, the strategy generates 60 trades over 7 months. The win rate is only $38.33\%$ and the profit factor is $0.5453$, losing $-\$258.10$ before commission or slippage.
2. **Impact of Frictions:** Adding realistic transaction frictions (0.5 pip slippage and $\$7/\text{lot}$ commission) increases total trading costs from $\$118.64$ to $\$292.28$, plunging the profit factor to $0.3506$ and net loss to $-\$436.00$.
3. **Sensitivity Analysis Findings (1.25× ATR Scenario):** The +$59.28 result observed under the 0.50 confidence, 1.25× ATR stop-loss, and 1.50× ATR take-profit sensitivity scenario is highly sensitive to the 4-bar maximum-holding-period rule and the small 59-trade validation sample. The trade-by-trade audit shows that changes from STOP_LOSS to MAX_HOLD and resulting position-occupancy/cooldown effects account for most of the P&L difference versus the 1.00× ATR scenario. This result is therefore not evidence of a persistent statistical trading edge or live profitability.

---

## 21. Realism Limitations

1. **Intra-Bar Tick Trajectory:** M15 OHLC bars cannot resolve intra-bar price paths. While the conservative policy (SL first) eliminates optimistic bias, it cannot capture microsecond quote dynamics.
2. **Spread Volatility:** Historical MT5 data contains discrete spread snapshots, which may underestimate spread widening during news releases or rollover.
3. **Queue & Market Impact:** In-memory simulation assumes immediate fills without queue positioning or market depth impact.
4. **Latency:** Zero latency is assumed between candle close and order execution at the next open.

---

## 22. Financial Interpretation Guidance

- **No Commercial Viability:** The directional accuracy of the ML baseline is insufficient to overcome bid-ask spread and transaction costs.
- **Empirical Execution Performance:** Empirical win rates ($31\%-38\%$) with $1.0\times\text{SL} / 1.5\times\text{TP}$ do not provide an edge in liquid FX markets under realistic transaction frictions.
- **Risk Core Functionality:** The RiskEngine, KillSwitch, and position sizer behaved flawlessly, strictly preventing excessive drawdown or outsized exposures.

---

## 23. Research-Only Disclaimer

> [!WARNING]
> **EXPLICIT RESEARCH DISCLAIMER:**  
> This backtesting simulation is strictly an offline academic research exercise. The results do not demonstrate trading profitability, commercial viability, or statistical edge. Under no circumstances should this strategy be deployed to live accounts, used with real capital, or connected to live market execution. All trading systems carry substantial risk of capital loss.

---

**PHASE 12 COMPLETE — PHASE 13 NOT STARTED**
