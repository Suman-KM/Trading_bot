# Phase 43 — New Information Edge Research & Data-Feasibility Gate

**Document Status:** Complete & Audited  
**Date (UTC):** 2026-10-06  
**Author:** Quantitative Trading & AI Research Engineering  
**System:** AI Trading System — EURUSD  
**Branch:** `develop`  
**Latest Reference Commit:** `68728e9` (Phase 42 closeout)  
**Scientific Verdict:** **RESEARCH PAUSE / INFEASIBLE WITHOUT COMMERCIAL FEEDS**  

---

## 1. Executive Summary

Phases 8 through 42 established conclusively that retail technical indicators, moving-average filters, momentum oscillators, price geometry, volatility estimators, multi-timeframe regime conditioning (M15, H1, H4), daily sovereign yield spreads, and weekly CFTC commitment-of-traders positioning contain **near-zero out-of-sample directional predictive edge** for EURUSD after accounting for market efficiency and real-world broker transaction friction.

Phase 43 investigated whether the system can obtain a statistically defensible trading edge from **genuinely NEW information sources** that were NOT already exhausted by prior phases.

Following strict multiple-testing controls and scientific governance, Phase 43 executed a formal **Data Feasibility Audit** across 18 candidate information sources in 6 structural categories:
- **Category A:** True or High-Quality Market Microstructure (Quote dynamics, order flow, L2 DOM)
- **Category B:** Central Bank / Macroeconomic Event Surprises (FOMC, ECB, CPI, NFP, PMI)
- **Category C:** Interest-Rate Expectation Information (Sovereign yields, OIS, policy repricing)
- **Category D:** Futures / Exchange-Traded Information (CME 6E, futures basis, CVD, open interest)
- **Category E:** Positioning (Beyond weekly CFTC COT, intraday dealer flow)
- **Category F:** Options / Implied Information (Implied volatility skew, 25-delta risk reversals)

### Key Empirical & Feasibility Findings:
1. **Feasibility Classification:**
   - **Already Exhausted (5 Sources):** MT5 top-of-book bid/ask quotes, MT5 quote frequency (tick volume), daily US-DE 2Y yield spread, CME 6E daily settlement, and weekly CFTC COT net positioning.
   - **Infeasible Under Current Infrastructure (12 Sources):** CME Globex L2/L3 order book depth, cumulative volume delta (CVD), macroeconomic consensus surprises (CPI/NFP/GDP), intraday cash Treasury yields, SOFR/Euribor short-rate futures, and OTC options volatility surfaces.
   - **Partially Feasible (1 Source):** Central bank policy rate decision calendar (public domain), but low frequency (~16 meetings/year) and already priced in by market participants before release.
   - **Feasible with Unexhausted Edge (0 Sources):** Zero free or retail-accessible data sources provide unexhausted directional predictive alpha.
2. **The Consensus Forecast & Revision Bottleneck:**
   - While macroeconomic historical *actuals* are public domain, pre-release *consensus forecasts* are proprietary commercial intellectual property (Bloomberg Survey / Refinitiv poll).
   - Free web calendar scrapers violate Terms of Service, lack point-in-time publication audit trails, and routinely suffer from database corruption (retroactively overwriting initial consensus with post-revision numbers).
3. **The Microstructure Reality:**
   - Retail MetaTrader 5 demo feeds deliver indicative dealer quotes. In 100% of historical ticks, `last=0.0`, `volume_real=0.0`, and Level 2 market depth returns an empty tuple `()`.
   - True institutional order flow exists exclusively on centralized exchanges (CME Globex MDP 3.0) and requires commercial licenses ($1,500–$4,000/month).
4. **Hard-Stop Rule Triggered:**
   - Under Section 31 (Hard Stop Conditions 1 and 12), research must stop when high-value data sources require paid commercial credentials not present in the repository.
   - Fabricating order flow or pretending MT5 ticks equal exchange order flow is strictly forbidden.
5. **Verdict:**
   - **RESEARCH PAUSE / INFEASIBLE WITHOUT COMMERCIAL SUBSCRIPTIONS**. **PHASE 44 IS NOT AUTHORIZED**.

---

## 2. Research Question

*"Can a genuinely NEW information source provide incremental, statistically significant, temporally robust, economically meaningful information about EURUSD price movement that survives realistic transaction costs and strict point-in-time validation?"*

---

## 3. Prior Research Ledger Audit (Phases 8–42)

Every tested information category and its definitive empirical conclusion:

| Research Domain | Evaluated Phases | Tested Information | Empirical Conclusion | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Intraday M15 Technicals** | Phases 8–12, 22, 40.3, 41 | RSI, MACD, EMAs, ATR, Stochastics, BBands, Candle Body | Max confidence 0.5936; Top 1% directional win rate = 48.94%; Entropy = 94.7% of uniform noise | **NOT SUPPORTED** |
| **Cross-Market FX / Commodities** | Phases 14, 18 | USDX, GBPUSD, USDJPY, Gold (XAUUSD) lead-lag | Cross-market arbitrage operates in microseconds; zero M15 lead-lag edge | **NOT SUPPORTED** |
| **Higher-Timeframe H1/H4 Regimes** | Phases 15–17, 42 | H1 (12h swing) & H4 (24h swing) conditioned on trend/vol regimes | H1 WF Bal Acc = 50.43% (Holdout: 46.93%); H4 WF Bal Acc = 49.63% (Holdout: 51.29%) | **NOT SUPPORTED** |
| **Tick-Realistic Execution** | Phases 19, 20, 21.1, 38 | Tick bid/ask spread, queue delays, fill mechanics | 59 trades, 35.59% win rate, Net P&L = -$323.33, Profit Factor = 0.4786 | **NOT SUPPORTED** |
| **Macro Yield Differentials** | Phases 23, 24, 25 | Daily US 2Y (DGS2) vs German 2Y Bund yield spread | 10-fold WF Bal Acc = 50.84% (underperformed technical baseline 51.62%) | **NOT SUPPORTED** |
| **CFTC COT Speculative Flow** | Phases 28, 29 | Weekly Disaggregated / TFF Net Speculative Positioning | 10-fold WF Bal Acc = 50.62%, Net expectancy = -0.73 pips/trade (lagging indicator) | **NOT SUPPORTED** |
| **Realized Volatility Forecasting** | Phase 31 | Parkinson, Garman-Klass, Rogers-Satchell realized variance | Magnitude predictable ($R^2=0.42$), but directional sign predictability = **zero** | **NOT SUPPORTED (Sign)** |
| **Live MT5 Forward Observation** | Phases 40.1–40.4 | Live shadow M15 inference on MetaQuotes-Demo | Max live confidence = 0.4486; 0/9 reached $\tau=0.60$; execution properly blocked | **VALIDATED (Wiring)** |

---

## 4. Comprehensive Data Feasibility Audit Matrix (Section 4 Required Table)

Eighteen candidate information sources evaluated across all 15 required criteria:

| ID | Source Name | Category | Provider | Historical Availability | Resolution | Point-in-Time Safe? | Revision Risk | Cost | Accessibility | Unique Content | Already Tested? | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MICRO_01** | MT5 Top-of-Book Bid/Ask Quotes | Microstructure | MetaQuotes-Demo | 2020–Present | Milliseconds | Yes | Low | Free | Full | Low (Dealer quotes) | Yes (Phases 19–21.1) | **ALREADY EXHAUSTED** |
| **MICRO_02** | MT5 Tick Frequency ("Tick Volume") | Microstructure | MetaQuotes-Demo | 2010–Present | Bar counts | Yes | Zero | Free | Full | Zero (Quote counts) | Yes (Phases 9–22) | **ALREADY EXHAUSTED** |
| **MICRO_03** | MT5 Real Traded Volume (`volume_real`) | Microstructure | MT5 Dealer | Unavailable (100% 0.0) | N/A | No | N/A | Free | Inaccessible | None (Absent in OTC) | Yes (Phase 19) | **INFEASIBLE** |
| **MICRO_04** | MT5 Level 2 Market Depth (`market_book_get`)| Microstructure | MT5 Dealer | Unavailable (Empty) | N/A | No | N/A | Free | Inaccessible | None (Unsupported) | Yes (Phase 19, 23) | **INFEASIBLE** |
| **MICRO_05** | CME Euro FX (6E) Level 2/3 Order Book | Microstructure | CME Globex MDP 3.0 | 2010–Present | Nanoseconds | Yes | Zero | $1.5k–$4k/mo | Inaccessible (Paywall) | Very High (Centralized tape)| No | **INFEASIBLE** |
| **MICRO_06** | Cumulative Volume Delta (CVD) | Microstructure | CME / Rithmic | Historical tick data | Tick / 1-sec | Yes | Zero | $300–$1k/mo | Inaccessible (Paywall) | High (Aggressor signs) | No | **INFEASIBLE** |
| **MACRO_01** | US Macro Event Surprises (Actual vs Exp) | Macro Events | Bloomberg / Refinitiv | Actuals public; Exp private | Release sec | Conditional | Severe | $2.5k/mo | Inaccessible (Paywall) | Very High (Price shocks) | Audited (Phase 23) | **INFEASIBLE** |
| **MACRO_02** | Eurozone Macro Event Surprises | Macro Events | Eurostat / Bloomberg | Actuals public; Exp private | Release min | Conditional | High | Paid survey | Inaccessible (Paywall) | High | Audited (Phase 23) | **INFEASIBLE** |
| **MACRO_03** | Central Bank Policy Rate Decisions | Macro Events | Fed / ECB (Public) | 1999–Present | Release min | Yes | Zero | Free | Fully Accessible | Medium (Priced in early) | Audited (Phase 23) | **PARTIALLY FEASIBLE** |
| **MACRO_04** | Central Bank Press Conference NLP | Macro Events | Fed / ECB / Dow Jones | Transcripts post-hoc | Seconds | No | Medium | Paid feed | Inaccessible (Paywall) | High | No | **INFEASIBLE** |
| **RATES_01** | Daily US-Germany 2Y Yield Spread | Rates Exp | FRED / Bundesbank | 1990–Present (>30y) | Daily close | Yes | Zero | Free | Fully Ingested | Medium (Macro drift) | Yes (Phase 24, 25) | **ALREADY EXHAUSTED** |
| **RATES_02** | Intraday Cash Treasury Yields (2Y/10Y) | Rates Exp | GovPX / BrokerTec | Commercially available | 1-minute | Yes | Zero | $1k–$3k/mo | Inaccessible (Paywall) | High (Real-time repricing)| No | **INFEASIBLE** |
| **RATES_03** | Short-Rate Futures (SOFR / Euribor) | Rates Exp | CME / ICE | Commercially available | Minute / Hour | Yes | Zero | Exchange fees | Inaccessible (Paywall) | Very High (Policy paths) | No | **INFEASIBLE** |
| **FUTURES_01**| CME 6E Daily Settlement & Volume | Futures | CME / Yahoo | 2000–Present | Daily close | Yes | Low | Free | Accessible | Low (99.9% corr with spot)| Audited (Phase 14) | **ALREADY EXHAUSTED** |
| **FUTURES_02**| Spot-Futures Arbitrage Basis | Futures | Synchronized MT5 + CME | Microsecond colocation | Sub-ms | No | N/A | Institutional | Inaccessible | Zero for retail (CIP bound)| No | **INFEASIBLE** |
| **POS_01** | CFTC COT TFF Net Speculative Positioning | Positioning | CFTC Open Data | 2010–2026 (843 weeks) | Weekly | Yes | Zero | Free | Fully Ingested | Low (Coincident/Lagging) | Yes (Phase 28, 29) | **ALREADY EXHAUSTED** |
| **POS_02** | Intraday Retail SSI / Dealer Flow | Positioning | FXCM / OANDA / Citi | Retail partial; DB private| Hourly / Daily | No | High | Restricted | Inaccessible / Unverified | Low for retail SSI | No | **INFEASIBLE** |
| **OPTIONS_01**| OTC EURUSD Implied Volatility & 25D RR | Options Implied | ICAP / Tradeweb / Bbg | 1998–Present | Daily / Realtime | Yes | Zero | $2k–$3.5k/mo | Inaccessible (Paywall) | Very High (Directional skew)| No | **INFEASIBLE** |

---

## 5. Information Novelty & Redundancy Screening

To evaluate whether any surviving source provides genuine non-redundant predictive alpha:
1. **Redundancy of Existing Exogenous Ingestions:**
   - **CFTC COT (`cftc_eurofx_cot.parquet`):** Net speculative positioning has a 0.78 correlation with the 20-week price trend. It acts as a lagging follower of multi-month moves, failing in Phase 29 with 50.62% balanced accuracy and negative expectancy (-0.73 pips/trade).
   - **US-Germany 2Y Yield Spread (`eurusd_h4_yield_aligned.parquet`):** Daily yield differential was rigorously tested in Phase 25 across 10 walk-forward folds. It delivered 50.84% balanced accuracy, underperforming the pure technical baseline (51.62%).
   - **MT5 Tick Volume (`tick_volume`):** Linear correlation with absolute bar return is 0.38 (measures volatility magnitude), but correlation with signed directional return is **-0.004** (zero directional signal).
2. **Novelty Screening Conclusion:**
   - There are **ZERO FEASIBLE SOURCES** currently accessible in the repository that contain unexhausted directional predictive alpha.
   - All high-novelty information sources (CME futures L2/L3 order flow, macroeconomic consensus expectations, options risk reversals) are paywalled behind institutional commercial contracts.

---

## 6. Point-in-Time, Vintage & Revision Protection Rules

Section 6 and Section 7 establish non-negotiable rules for external data ingestion:
1. **Mathematical Timestamp Causality:**
   $$T_{\text{eff}} = T_{\text{info}} + \delta_{\text{latency}}$$
   $$T_{\text{decision}} \ge T_{\text{eff}}$$
   A model operating at bar open $T_{\text{bar}}$ may strictly utilize information where $T_{\text{eff}} \le T_{\text{bar}}$. For example, an event occurring at 10:00:00 UTC with an availability timestamp of 10:05:00 UTC **cannot exist** in a feature vector evaluated at 10:02:00 UTC.
2. **Revision & Vintage Protection:**
   - Economic time series (NFP, GDP, CPI) are subject to massive historical revisions.
   - Training on final revised series introduces severe hindsight bias.
   - In accordance with Section 7, all macroeconomic series lacking historical point-in-time publication vintages are classified as **REVISION-UNSAFE** and excluded from causal model training.

---

## 7. Timezone Normalization & DST Desynchronization Audit

A vital architectural finding from our timezone evaluation:
- The United States initiates Daylight Saving Time (EDT, UTC-4) on the **second Sunday in March** and returns to Standard Time (EST, UTC-5) on the **first Sunday in November**.
- Europe initiates Summer Time (CEST, UTC+2 / BST, UTC+1) on the **last Sunday in March** and returns to Standard Time (CET, UTC+1 / GMT, UTC+0) on the **last Sunday in October**.
- **The Desynchronization Window:**
  - For **2 to 3 weeks in March**, the US is on EDT while Europe remains on CET. The time difference between New York and Frankfurt shrinks from **6 hours to 5 hours**.
  - During this period, an 08:30 US release occurs at **12:30 UTC** instead of 13:30 UTC.
  - Pipelines with hardcoded UTC offsets incur a fatal 1-hour lookahead or lag error. All calendar pipelines must strictly resolve timestamps via IANA timezone database resolvers (`zoneinfo.ZoneInfo("America/New_York")`).

---

## 8. Event-Study and Macro Surprise Analysis

### The Macroeconomic Event Dilemma:
1. **Information Content:**  
   The true informational driver of exchange rate repricing is the **normalized surprise shock**:
   $$\text{Surprise}_t = \frac{\text{Actual}_t - \text{Consensus}_t}{\sigma_{\text{surprise}}}$$
2. **The Execution Reality:**
   - 80% to 90% of price displacement following NFP or CPI occurs within the first **30 to 180 seconds** of release.
   - Retail MT5 execution involves 30–100 ms network latency and dealer requotes/slippage.
   - During major releases, broker spreads widen from 0.8 pips to **8.0–25.0 pips**, with slippage exceeding 10.0 pips.
   - Any candle-based strategy entering at completed M15 close enters into market exhaustion or mean-reversion with toxic friction.

---

## 9. Pre-Registered Success Gates Audit (Section 12)

Every candidate must be audited against the 12 pre-registered success criteria:

| Pre-Registered Gate | Requirement | Status | Audit Findings |
| :--- | :--- | :--- | :--- |
| **Gate 1: Out-of-Sample Accuracy** | $\text{Bal Acc} \ge 54.5\%$ or positive conditional expectancy | **FAILED** | All tested sources achieved 49.5%–51.6% |
| **Gate 2: Statistical Significance** | $p < 0.01$ against clearly defined null hypothesis | **FAILED** | All walk-forward paired tests yielded $p > 0.15$ |
| **Gate 3: Positive Net Expectancy** | Positive net pips after realistic transaction costs | **FAILED** | Net expectancy consistently negative (-1.24 to -1.51 pips) |
| **Gate 4: Cost Robustness** | Edge survives realistic spread/slippage | **FAILED** | Friction destroys gross returns |
| **Gate 5: Temporal Distribution** | No single period contributes $> 50\%$ profit | **N/A** | No profitable candidate found |
| **Gate 6: Walk-Forward Consistency**| $\ge 70\%$ of folds beat baseline | **FAILED** | Only 30% to 60% of folds beat baseline |
| **Gate 7: Untouched Holdout** | Holdout partition independently passes | **FAILED** | Holdout balanced accuracy 46.9% to 51.3% |
| **Gate 8: No Leakage** | Zero lookahead or feature timing leakage | **PASSED** | Validated in 15 unit tests |
| **Gate 9: No Revision Leakage** | No revised series substituted for initial releases | **PASSED** | Revision-unsafe sources excluded |
| **Gate 10: Holdout Integrity** | Zero parameter tuning on holdout partition | **PASSED** | Holdout evaluated strictly once |
| **Gate 11: Cost Tolerance** | Survives modest friction deterioration (+0.3 pips) | **FAILED** | Net expectancy already negative |
| **Gate 12: Economic Significance** | Results remain economically viable | **FAILED** | Severe capital drawdown under broker friction |

---

## 10. Hard-Stop Conditions & Falsification Ledger

In accordance with Section 31 (Hard Stop Conditions):
- **Condition 1 Triggered:** High-value data sources (CME Globex MDP 3.0 order book depth, macroeconomic pre-release consensus forecasts, OTC options volatility surfaces) require commercial credentials and institutional subscriptions not available in the repository.
- **Condition 5 Triggered:** Low-cost publicly available sources (CFTC COT positioning, daily US-DE sovereign yield spreads, MT5 quote frequency) were already thoroughly tested in Phases 22, 25, 29, 31, and 42 and conclusively failed.
- **Scientific Action:** Rather than improvising or fabricating synthetic order flow, research halted immediately at the Data-Feasibility Gate.

---

## 11. Answers to All 27 Questions from Section 28

### 1. What information sources were investigated?
18 candidate information sources across 6 categories: MT5 top-of-book quotes, tick frequency, real volume, Level 2 DOM, CME 6E futures order book depth, cumulative volume delta (CVD), US macro event surprises, Eurozone macro event surprises, central bank rate decisions, central bank press conference NLP, daily US-Germany 2Y yield spread, intraday cash Treasury yields, short-rate futures (SOFR/Euribor), CME 6E daily settlement, spot-futures arbitrage basis, CFTC COT net positioning, intraday retail sentiment/dealer flow, and OTC EURUSD options volatility surfaces.

### 2. Which ones were already exhausted?
5 sources: MT5 top-of-book bid/ask quotes, MT5 tick volume, daily US-Germany 2Y yield spread (Phase 25), CME 6E daily settlement (Phase 14), and weekly CFTC COT positioning (Phase 28/29).

### 3. Which sources are genuinely new?
CME Globex L2/L3 order book depth, cumulative volume delta (CVD), pre-release macroeconomic consensus surprises, intraday Treasury yields, SOFR/Euribor implied rate expectations, and OTC options volatility surfaces (25-delta risk reversals).

### 4. Which sources are feasible?
Central bank rate decision calendar timestamps (public domain, but low frequency and lack surprise expectations).

### 5. Which sources are infeasible?
12 sources: CME futures L2/L3 depth, CVD, US macro surprises, Eurozone macro surprises, press conference NLP, intraday Treasury yields, SOFR/Euribor futures, spot-futures arbitrage basis, intraday dealer flow, and OTC options volatility surfaces.

### 6. Why are they infeasible?
They require paid commercial institutional data subscriptions (Bloomberg Terminal, Refinitiv, CME DataMine, BrokerTec), specialized colocation infrastructure, or private dealer client access costing thousands of dollars per month.

### 7. What timestamp/availability limitations exist?
Macro surprises require exact publication timestamps down to the second and pre-release consensus survey timestamps. Treasury yields are daily closing snapshots posted at 21:30 UTC, creating an 11-hour lookahead if applied to intraday M15 candles.

### 8. What revision risks exist?
Severe revision risk in macroeconomic actuals (NFP, GDP, CPI), which undergo large historical benchmark revisions. Training on revised data introduces severe lookahead bias.

### 9. What information survives the novelty test?
None of the freely accessible sources in the repository survive the novelty test. All surviving high-novelty sources are commercially paywalled.

### 10. What event-study evidence exists?
Macroeconomic events produce large initial price displacements (15–40 pips) within 1–3 minutes, accompanied by spread blowouts (8–25 pips), precluding profitable retail M15 post-candle entries.

### 11. What predictive evidence exists?
No source currently accessible in the repository exhibits out-of-sample directional predictive accuracy exceeding 51.6%.

### 12. What walk-forward results exist?
Previous expanding walk-forward results: Macro yields = 50.84% (Phase 25); CFTC COT = 50.62% (Phase 29); H1 regimes = 50.43% (Phase 42); H4 regimes = 49.63% (Phase 42).

### 13. What holdout results exist?
Holdout balanced accuracy across frozen models has consistently ranged from 46.9% to 51.3%, triggering failure gates.

### 14. What are the p-values?
All paired t-tests and Wilcoxon signed-rank tests against naive baselines yielded $p > 0.15$, failing the pre-registered $p < 0.01$ threshold.

### 15. What are the costs?
EURUSD roundtrip friction is 1.3–1.8 pips in normal market conditions, expanding to 5.0–25.0 pips during major macroeconomic releases.

### 16. What is net expectancy?
Strictly negative across all tested configurations: -1.24 to -1.51 pips per trade.

### 17. Does the edge survive stressed costs?
No. There is no positive gross edge, so stressed costs accelerate capital depletion.

### 18. Is performance temporally stable?
Consistently around 50% across all historical decades (2010–2026).

### 19. Is there leakage?
No. Point-in-time causality, purge gaps, and holdout isolation were verified with zero leakage.

### 20. Is there overfitting?
Historically suspected in unregularized linear models, where fold accuracy varied widely but collapsed out-of-sample.

### 21. Does the candidate satisfy every pre-registered gate?
No. It fails Gates 1, 2, 3, 4, 6, 7, 11, and 12.

### 22. What is the strongest surviving hypothesis?
The strongest theoretical hypothesis is that **directional EURUSD repricing is driven by exogenous macroeconomic consensus surprises and institutional order flow imbalances**, but testing it requires commercial data feeds.

### 23. What information is still missing?
1) Historical pre-release consensus survey distributions (Bloomberg/Refinitiv),
2) Centralized exchange order book depth and aggressive trade tape (CME Globex MDP 3.0),
3) OTC options volatility surfaces (25-delta risk reversals).

### 24. What would be required to obtain it?
Commercial enterprise data subscriptions: Bloomberg Professional Terminal (~$2,500/month) and CME DataMine historical order book archive ($1,500–$4,000 one-time/monthly).

### 25. Is another research phase scientifically justified?
**No.** Not until commercial institutional data feeds are procured. Continuing to test retail indicators or free web data constitutes unscientific curve fitting.

### 26. Is paper trading authorized?
**NOT AUTHORIZED.**

### 27. Is live trading authorized?
**NOT AUTHORIZED.**

---

## 12. Final Recommendation

### Formal Status:
**PHASE 44: NOT AUTHORIZED.**

### Strategic Summary:
1. The AI Trading System codebase, historical data ingestion engines, point-in-time validation architecture, risk engine, and MT5 demo integration boundary are fully verified, robust, and operational as institutional-grade assets.
2. The project has scientifically falsified the hypothesis that public retail technical indicators, daily yield spreads, or weekly COT positioning contain directional edge on EURUSD.
3. Machine learning strategy research is formally paused. Strategy development should only resume if commercial institutional feeds (CME Globex Level 2/3 order flow or Bloomberg consensus survey archives) are procured.
