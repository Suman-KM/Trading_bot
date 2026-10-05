# Phase 44 — Low-Cost Institutional Data Discovery, Provider Comparison & Acquisition Decision Gate

**Document Status:** Complete & Audited  
**Date (UTC):** 2026-10-06  
**Author:** Quantitative Trading & AI Research Engineering  
**System:** AI Trading System — EURUSD  
**Branch:** `develop`  
**Latest Reference Commit:** `ef927bd` (Phase 43 closeout)  
**Scientific Verdict:** **DISCOVERY COMPLETE — TIER 0 FREE CREDIT & OPEN ACADEMIC PATH IDENTIFIED**  
**Budget Currency Conversion:** 1 USD = 83.50 INR (2025/2026 Baseline)  

---

## 1. Executive Summary

Phase 43 concluded with a definitive **RESEARCH PAUSE / INFEASIBLE WITHOUT COMMERCIAL FEEDS**, having proved across Phases 8–43 that public retail technical indicators, daily yield spreads, weekly CFTC COT positioning, and moving-average filters contain near-zero out-of-sample directional predictive edge on EURUSD after accounting for transaction costs.

Phase 44 investigated whether the project can obtain **genuinely NEW, legally usable, historically reconstructable, point-in-time institutional information at a realistic developer/student budget (₹0 to ₹10,000 INR) WITHOUT purchasing expensive enterprise terminals (e.g. Bloomberg at $2,500/month)**.

Following strict data provenance, revision safety, and multiple-testing rules, Phase 44 audited **12 candidate data providers** across 6 structural categories (Futures Order Flow, Macroeconomic Consensus Surprises, Macro Vintages, Interest Rate Expectations, FX Options Implied Skew, and Retail Positioning).

### Key Discovery Findings:
1. **The ₹0 Path (Zero Out-of-Pocket Expenditure):**
   - **Federal Reserve Bank of San Francisco (Bauer & Swanson 2023):** High-frequency 30-minute window monetary policy surprises (FOMC target rate and forward guidance shocks). Completely free, official, public domain, point-in-time safe, zero revision risk.
   - **European Central Bank (EA-MPD, Altavilla et al. 2019/2023):** High-frequency intraday OIS and sovereign bond repricing around ECB policy announcements and press conferences. Completely free, official ECB research database, open academic license.
   - **Federal Reserve Bank of St. Louis (ALFRED):** Official real-time publication vintages of initial releases (NFP, CPI, GDP, PCE). Free via `fredapi`, eliminates revision leakage.
   - **Academic Replication Repositories (Harvard Dataverse / Zenodo):** Peer-reviewed replication archives containing actual vs pre-release Bloomberg consensus survey numbers. Free, open access (CC-BY).
   - **Databento Free Signup Credits ($125 Platform Credit):** All new Databento accounts receive $125 in free platform credits. Under pay-as-you-go historical batch pricing ($0.50–$3.50/GB), $125 allows downloading **~2–3 years of raw CME 6E Euro FX trade ticks (with aggressor signs) and top-of-book quotes** at **₹0 cash outlay**.
2. **The ₹2,000 INR Path (~$24 USD):**
   - **Databento Pay-As-You-Go Historical Batch:** A developer can purchase ~15–25 GB of raw uncompressed CME Globex MDP 3.0 trade ticks with buy/sell aggressor flags and top-of-book quotes for 2 full historical years of continuous 6E futures for **~$15 USD (~₹1,252.50 INR)**. There is zero recurring monthly platform fee.
   - **Interactive Brokers (IBKR):** CME non-professional real-time/historical tick pass-through is only **~$2.25/month (~₹188 INR/mo)**, but requires maintaining a funded brokerage account ($500+ deposit) and API pacing is throttled.
3. **Redundant Sources Rejected:**
   - **FirstRate Data ($49.95 USD / ~₹4,170 INR):** Sells 1-minute OHLC bars. Phase 14 already proved 6E futures OHLC has 99.9% correlation with EURUSD spot OHLC. Without tick-level aggressor flags or depth, it provides zero incremental alpha.
   - **CBOE EVZ (Euro Currency Volatility Index):** Measures 30-day implied volatility magnitude (like VIX). Phase 31 proved volatility magnitude lacks directional sign. Discontinued by CBOE in March 2025.
4. **Acquisition Decision Gate:**
   - **NO MONEY SPENT. NO PURCHASES MADE. NO TRADING.**
   - The project should NOT purchase data immediately. The optimal path is to **first exploit the ₹0 Tier 0 pathway** (FRBSF + ECB EA-MPD monetary policy surprises + Databento $125 free signup credits) to formulate and test pre-registered event-driven and microstructure hypotheses.

---

## 2. Prior Research Ledger & Phase 43 Audit (Exhausted vs New)

Every domain investigated across the project's history:

| Domain | Phases | Information Type | Empirical Conclusion | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Retail Technical Indicators** | 8–12, 22, 40.3, 41 | RSI, MACD, EMAs, ATR, Stochastics, BBands, Candle Body | Max confidence 0.5936; Top 1% win rate = 48.94%; Entropy 94.7% of uniform noise | **ALREADY EXHAUSTED** |
| **Cross-Market Lead-Lag** | 14, 18 | Spot DXY, GBPUSD, USDJPY, Gold lead-lag | Microsecond arbitrage; zero M15 predictive edge | **ALREADY EXHAUSTED** |
| **Higher-Timeframe Swings** | 15–17, 42 | H1 (12h) & H4 (24h) trend/vol regimes | H1 WF Bal Acc = 50.43%; H4 WF Bal Acc = 49.63% | **ALREADY EXHAUSTED** |
| **MT5 Bid/Ask Top-of-Book** | 19–21.1, 38, 40.2 | Indicative dealer tick quotes | Spread 0.8–2.5 pips; necessary for friction, zero directional alpha | **ALREADY EXHAUSTED** |
| **MT5 Tick Volume** | 9–22 | Quote arrival counts per bar | Linear corr with absolute return = 0.38; corr with signed return = -0.004 | **ALREADY EXHAUSTED** |
| **Macro Yield Differentials** | 24, 25 | Daily US 2Y vs German 2Y yield spread | 10-fold WF Bal Acc = 50.84% (underperformed technical baseline 51.62%) | **ALREADY EXHAUSTED** |
| **CFTC COT Positioning** | 28, 29 | Weekly Disaggregated / TFF Speculative Net | 10-fold WF Bal Acc = 50.62%; Net expectancy = -0.73 pips (lagging follower) | **ALREADY EXHAUSTED** |
| **Realized Volatility** | 31 | Parkinson, Garman-Klass, Rogers-Satchell | Predicts volatility magnitude ($R^2=0.42$), zero directional sign predictability | **ALREADY EXHAUSTED** |
| **CME 6E Daily Settlement** | 14 | Daily closing settlement and volume | 99.9% correlated with spot daily closes | **ALREADY EXHAUSTED** |
| **CME 6E Intraday Order Flow** | **Phase 44 New** | Aggressive trade tape, CVD, L2/L3 order book | Microstructure order flow imbalance; never tested in prior phases | **GENUINELY NEW** |
| **High-Freq Macro Surprises** | **Phase 44 New** | 30-min window central bank policy shocks (FRBSF, ECB) | Normalized surprise shocks; never tested in prior phases | **GENUINELY NEW** |
| **Point-in-Time Vintages** | **Phase 44 New** | Initial unrevised macro actuals (ALFRED) | Prevents lookahead/revision leakage; never tested in prior phases | **GENUINELY NEW** |

---

## 3. Comprehensive Data Provider Audit Matrix (12 Providers)

All 12 candidate providers audited across all technical, financial, and licensing dimensions:

| Provider | Product | Category | Instrument | Resolution | Price Model | Total 1st Mo (USD / INR) | Price Tier | Point-in-Time? | Reliability | Status & Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Databento** | CME Globex MDP 3.0 (6E) | Futures Order Flow | CME 6E | Nanoseconds | Pay-As-You-Go ($0.50–$3.50/GB) | $15.00 / ₹1,252.50 | TIER 1 (<$25) | Yes (Immutable tape) | HIGH | **RECOMMENDED (#1 Futures)** |
| **FRB San Francisco** | Bauer-Swanson Surprises | Macro / Rates | FOMC Events | 30-min window | Free ($0.00) | $0.00 / ₹0.00 | TIER 0 (Free) | Yes (Event window) | HIGH | **RECOMMENDED (#1 Macro)** |
| **European Central Bank**| EA-MPD (Altavilla et al.)| Macro / Rates | ECB Events | 15/50-min window| Free ($0.00) | $0.00 / ₹0.00 | TIER 0 (Free) | Yes (Event window) | HIGH | **RECOMMENDED (#2 Macro)** |
| **FRB St. Louis (ALFRED)**| ALFRED Real-Time Vintages| Macro Vintages | US Indicators | Vintage date | Free ($0.00) | $0.00 / ₹0.00 | TIER 0 (Free) | Yes (Vintage tracking)| HIGH | **RECOMMENDED (#1 Vintages)** |
| **Harvard Dataverse** | Academic Surprise Archives| Macro Surprises | Macro Shocks | Event release | Free ($0.00) | $0.00 / ₹0.00 | TIER 0 (Free) | Yes (Published study) | HIGH | **RECOMMENDED (#2 Consensus)**|
| **Interactive Brokers** | IBKR API Historical Ticks | Futures Order Flow | CME 6E | Milliseconds | Pass-Through (~$2.25/mo) | $2.25 / ₹187.88 | TIER 1 (<$25) | Yes (Trade tape) | HIGH | **CONDITIONAL (Needs $500 dep)**|
| **FirstRate Data** | CME 6E 1-Min Continuous | Futures OHLC | CME 6E | 1-Minute | Bulk ($49.95 one-time) | $49.95 / ₹4,170.83 | TIER 2 ($25–$100)| Yes (OHLC bars) | MEDIUM | **REJECTED (Redundant OHLC)** |
| **OANDA** | REST v20 Order/Position | Retail Sentiment | EURUSD | 20-Minute | Free with account ($0.00) | $0.00 / ₹0.00 | TIER 0 (Free) | Yes (Snapshots) | MEDIUM | **REJECTED (Retail, No Archive)**|
| **CBOE / FRED** | CBOE EVZ Volatility Index | FX Options Vol | EURUSD / FXE | Daily close | Free via FRED ($0.00) | $0.00 / ₹0.00 | TIER 0 (Free) | Yes (Daily close) | HIGH | **REJECTED (Discontinued, Vol only)**|
| **Trading Economics** | Economic Calendar API | Macro Consensus | Global Macro | Release min | Subscription ($149/mo) | $149.00 / ₹12,441.50 | TIER 3 ($100–$500)| Conditional (Revisions) | MEDIUM | **REJECTED (Exceeds budget, $149/mo)**|
| **CME Group** | CME DataMine MBO / Depth | Futures Order Flow | CME 6E | Nanoseconds | Subscription ($275–$550/mo)| $275.00 / ₹22,962.50 | TIER 3 ($100–$500)| Yes (Official tape) | HIGH | **REJECTED (Exceeds budget, $275/mo)**|
| **Bloomberg L.P.** | Bloomberg Terminal | All Categories | Global FX/Fut | Real-time / Tick | Enterprise ($2,500/mo) | $2,600.00 / ₹2,17,100 | TIER 5 (>$1,000) | Yes (Gold standard) | HIGH | **REJECTED (Extreme enterprise cost)**|

---

## 4. Evaluation Framework & Information Value Scoring (0 to 50)

Each provider is scored across 10 objective criteria (0 to 5 each):
- **Novelty (NOV):** Genuinely new information vs previously exhausted retail indicators.
- **Historical Depth (DEP):** Multi-year availability suitable for walk-forward validation.
- **Timestamp Quality (TSQ):** Sub-second exchange matching or exact release second timestamps.
- **Point-in-Time Safety (PIT):** Guarantees zero future information leakage.
- **Data Provenance (PRV):** Official exchange or central bank origin vs unverified web scraping.
- **Resolution (RES):** Granularity (ticks/nanoseconds vs coarse daily snapshots).
- **Information Value (INF):** Theoretical link to market price discovery.
- **Cost Efficiency (CST):** Information delivered per dollar spent.
- **Ease of Integration (INT):** Clean API/download formats (Parquet/CSV) vs complex reverse engineering.
- **Licensing Clarity (LIC):** Clear terms for personal research without hidden redistribution liabilities.

### Scoring Table:

| Provider | Product | NOV | DEP | TSQ | PIT | PRV | RES | INF | CST | INT | LIC | Total Score (/50) | Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Databento** | CME Globex MDP 3.0 (6E) | 5 | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 5 | **48** | **#1** |
| **FRB San Francisco** | Bauer-Swanson Surprises | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | **49** | **#1** |
| **ECB** | EA-MPD Event Database | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | **49** | **#1** |
| **FRB St. Louis** | ALFRED Real-Time Vintages | 4 | 5 | 5 | 5 | 5 | 3 | 4 | 5 | 5 | 5 | **46** | **#4** |
| **Harvard Dataverse** | Academic Surprise Archives | 4 | 4 | 4 | 4 | 5 | 3 | 4 | 5 | 4 | 5 | **41** | **#5** |
| **Interactive Brokers** | IBKR API Historical Ticks | 4 | 3 | 4 | 4 | 5 | 5 | 4 | 4 | 2 | 4 | **39** | **#6** |
| **FirstRate Data** | CME 6E 1-Min History | 3 | 4 | 4 | 4 | 4 | 3 | 3 | 4 | 4 | 4 | **37** | **#7** |
| **Trading Economics** | Economic Calendar API | 4 | 4 | 3 | 3 | 4 | 3 | 4 | 2 | 4 | 4 | **35** | **#8** |
| **OANDA** | REST v20 Position Book | 3 | 2 | 3 | 3 | 4 | 2 | 2 | 4 | 4 | 4 | **31** | **#9** |
| **CBOE / FRED** | CBOE EVZ Volatility Index | 2 | 4 | 3 | 4 | 5 | 1 | 2 | 5 | 5 | 5 | **36** | **#10** |
| **CME Group** | CME DataMine MBO | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1 | 3 | 5 | **44** | **#11** |
| **Bloomberg L.P.** | Bloomberg Terminal | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1 | 3 | 5 | **44** | **#12** |

---

## 5. Top 5 Shortlist Ranked by Scientific Value / Acquisition Cost

### Rank 1 (Tied): Federal Reserve Bank of San Francisco — Bauer & Swanson Monetary Policy Surprises
- **Dataset:** High-frequency 30-minute window FOMC interest-rate and yield changes.
- **Why It Is New:** Measures the exact unpriced monetary policy surprise directly from money-market futures repricing during the announcement window. Never tested in prior phases.
- **Historical Coverage:** 1988–2024+ (36 years of FOMC announcements).
- **Resolution:** 30-minute event windows around policy releases.
- **Price:** **$0.00 / ₹0.00 (Completely Free).**
- **Licensing:** Public domain / Open academic citation.
- **Timestamp Quality:** Release-minute window timestamps.
- **Point-in-Time Safety:** 100% point-in-time safe (derived strictly from market prices during the window). Zero revision risk.
- **Future Hypothesis:** *"FOMC monetary policy surprise shocks (target rate surprise vs forward guidance factor) predict EURUSD multi-hour directional drift following the post-announcement window."*
- **Recommendation:** **RECOMMENDED FOR EVALUATION (BEST FREE MACRO SURPRISE SOURCE).**

### Rank 1 (Tied): European Central Bank (ECB) — Euro Area Monetary Policy Event-Study Database (EA-MPD)
- **Dataset:** Intraday OIS and sovereign bond yield changes around ECB Governing Council events.
- **Why It Is New:** Separates the immediate policy decision surprise (15-min window) from the press conference communication/forward guidance shock (50-min window).
- **Historical Coverage:** 1999–2024+ (25 years of ECB announcements).
- **Resolution:** Separate 15-minute and 50-minute intraday event windows.
- **Price:** **$0.00 / ₹0.00 (Completely Free).**
- **Licensing:** Open academic research citation (Altavilla et al. 2019/2023).
- **Timestamp Quality:** Intraday event-window timestamps.
- **Point-in-Time Safety:** 100% point-in-time safe. Zero revision risk.
- **Future Hypothesis:** *"ECB press conference communication factor (forward guidance surprise) produces persistent multi-hour post-event directional drift in EURUSD."*
- **Recommendation:** **RECOMMENDED FOR EVALUATION (BEST FREE ECB SURPRISE SOURCE).**

### Rank 3: Databento — CME Globex MDP 3.0 (6E Euro FX Futures)
- **Dataset:** CME 6E trade ticks (with buyer/seller aggressor signs), top-of-book (TBBO), and market depth (MBP-10).
- **Why It Is New:** True centralized exchange order flow from the primary EURUSD futures matching engine. Delivers genuine trade prints and aggressor volume delta (CVD).
- **Historical Coverage:** 2010–Present (~15 years).
- **Resolution:** Nanoseconds.
- **Price:** **Pay-As-You-Go ($0.50–$3.50/GB). No monthly subscription fee.**
  - **Free Tier:** **$125 platform credit** upon signup (covers ~2–3 years of 6E trades at ₹0 cash outlay).
  - **Out-of-Pocket Purchase:** ~$15 USD (~₹1,252.50 INR) for 2 years of continuous 6E trades.
- **Licensing:** Personal non-display research (internal use only, no redistribution).
- **Timestamp Quality:** Exchange matching engine nanoseconds.
- **Point-in-Time Safety:** 100% point-in-time safe (immutable exchange matching tape).
- **Future Hypothesis:** *"CME 6E trade aggressor volume imbalance (CVD) contains incremental directional predictive information about subsequent EURUSD 15-minute price changes beyond spot OHLC."*
- **Recommendation:** **RECOMMENDED FOR EVALUATION (BEST OVERALL FUTURES VALUE).**

### Rank 4: Federal Reserve Bank of St. Louis — ALFRED Real-Time Data Vintages
- **Dataset:** Historical publication vintages of initial releases and subsequent revisions for NFP, CPI, GDP, and PCE.
- **Why It Is New:** Eliminates historical revision lookahead bias by preserving the exact number originally published on release day.
- **Historical Coverage:** 1920s–Present (>100 years).
- **Resolution:** Publication date and vintage timestamps.
- **Price:** **$0.00 / ₹0.00 (Completely Free via `fredapi`).**
- **Licensing:** Public domain.
- **Timestamp Quality:** Exact release date and vintage tracking.
- **Point-in-Time Safety:** 100% vintage-safe.
- **Future Hypothesis:** *"Initial unrevised first-release macro actuals produce statistically distinct post-release volatility expansion compared to subsequent revised numbers."*
- **Recommendation:** **RECOMMENDED FOR EVALUATION (BEST FREE MACRO VINTAGE ARCHIVE).**

### Rank 5: Academic Repositories (Harvard Dataverse & Zenodo)
- **Dataset:** Published macroeconomic news surprise and high-frequency shock archives.
- **Why It Is New:** Contains standardized surprises (Actual minus Bloomberg Consensus divided by SD) compiled for peer-reviewed academic literature.
- **Historical Coverage:** 1990–2022+ (varies by study).
- **Resolution:** Announcement event timestamps.
- **Price:** **$0.00 / ₹0.00 (Open Access / CC-BY).**
- **Licensing:** Creative Commons CC-BY 4.0.
- **Timestamp Quality:** Announcement date and time.
- **Point-in-Time Safety:** Point-in-time safe as constructed for published replication.
- **Future Hypothesis:** *"Standardized macroeconomic announcement surprises provide statistically significant conditional directional edge when interacting with pre-release volatility regimes."*
- **Recommendation:** **RECOMMENDED FOR EVALUATION (BEST FREE CONSENSUS PROXY).**

---

## 6. Budget Scenario Analysis

### SCENARIO A: ₹0 ($0.00 USD)
- **Strongest Legitimate Datasets:**
  1. *Federal Reserve Bank of SF (Bauer-Swanson):* 36 years of FOMC high-frequency policy surprise shocks.
  2. *European Central Bank (EA-MPD):* 25 years of ECB intraday policy and press conference shocks.
  3. *Federal Reserve Bank of St. Louis (ALFRED):* Complete archive of initial unrevised macro vintages.
  4. *Academic Repositories (Harvard Dataverse):* Published Bloomberg consensus surprise replication datasets.
  5. *Databento $125 Free Platform Signup Credits:* Up to 2–3 years of CME 6E trade ticks and top-of-book quotes.
- **Finding:** Under a ₹0 budget, the project can obtain institutional-grade high-frequency central bank shocks, macro vintages, and 2+ years of CME 6E raw trade ticks at **zero out-of-pocket cash outlay**.

### SCENARIO B: ₹2,000 maximum (~$23.95 USD)
- **Strongest Legitimate Dataset:**
  - *Databento Pay-As-You-Go Historical Batch:* 2 full years of continuous CME 6E trade ticks with aggressor buy/sell flags and top-of-book quotes for **~$15.00 USD (~₹1,252.50 INR)**.
- **Finding:** A scientifically adequate, institutional-grade dataset of genuine exchange-traded order flow **CAN be obtained within ₹2,000 INR**.

### SCENARIO C: ₹5,000 maximum (~$59.88 USD)
- **Strongest Legitimate Dataset:**
  - *Databento Extended Batch:* 4–5 years of continuous CME 6E trade ticks + 1 year of 10-level Market Depth (MBP-10) for **~$45.00 USD (~₹3,757.50 INR)**.
- **Finding:** Within ₹5,000 INR, full multi-year trade tape and multi-level order book depth become accessible.

### SCENARIO D: ₹10,000 maximum (~$119.76 USD)
- **Strongest Legitimate Dataset:**
  - *Databento Dual-Market Dataset:* 5 years of CME 6E currency futures order flow plus CME SOFR short-rate futures ticks for **~$95.00 USD (~₹7,932.50 INR)**.
- **Finding:** Within ₹10,000 INR, the project can acquire simultaneous visibility into exchange liquidity and intraday short-rate repricing.

---

## 7. Data Provenance, Licensing & Revision Rules

1. **Strict Provenance Verification:**
   - Every candidate was traced to its primary matching engine or statistical authority.
   - Retail indicators and web scrapers (Forex Factory, Investing.com) were rejected due to Terms of Service prohibitions, absence of publication audit trails, and retroactive overwriting of consensus numbers.
2. **No Fake Order Flow:**
   - Retail MT5 dealer quote frequency (`tick_volume`) must never be relabeled as "volume delta".
   - Bid/ask quote jitter must never be relabeled as "institutional buying/selling".
   - Real trade aggressor classification requires exchange-printed trade records matched against prevailing quotes (e.g. CME MDP 3.0 via Databento).
3. **Point-in-Time Timestamp Causality:**
   - $T_{\text{decision}} \ge T_{\text{eff}} = T_{\text{event}} + \delta_{\text{latency}}$.
   - Event studies must use discrete post-announcement entry windows with realistic spread blowout models (5.0–25.0 pips friction).

---

## 8. Pre-Registered Future Success Gates (Section 29)

Any future experiment utilizing the shortlisted datasets must adhere strictly to these locked gates:
1. **Out-of-Sample Accuracy / Expectancy:** Walk-forward Balanced Accuracy $\ge 54.5\%$ (or statistically significant positive conditional expectancy for event-driven strategies).
2. **Statistical Significance:** $p < 0.01$ against naive baselines in paired non-parametric tests.
3. **Net Expectancy:** Positive net pips after realistic transaction costs (1.5 pips baseline; 5.0–15.0 pips during event windows).
4. **Stressed Cost Robustness:** Edge must remain positive when transaction costs are stressed by $+0.5$ pips.
5. **Untouched Holdout:** Out-of-sample holdout partition must independently pass without parameter adjustments.
6. **Walk-Forward Consistency:** At least $70\%$ of walk-forward folds must beat the naive baseline.
7. **Temporal Distribution:** No single event or period may contribute $>50\%$ of cumulative net return.
8. **No Information Leakage:** Zero lookahead in timestamps or feature calculations.
9. **No Revision Leakage:** Only initial unrevised vintages may be used.
10. **No Threshold Mining:** Decision thresholds must be set a priori.
11. **Multiple Testing Correction:** Family-wise error rate (FWER) or False Discovery Rate (FDR) control enforced.
12. **Economic Significance:** Annualized Sharpe Ratio $\ge 1.0$ with maximum drawdown $< 15\%$.

---

## 9. Final Acquisition Recommendation

1. **NO PURCHASES AUTHORIZED AT THIS STAGE.**
   - While Databento provides an institutional dataset under ₹2,000 INR, no capital should be spent immediately.
2. **Recommended Action Sequence:**
   - **Step 1 (₹0 Outlay):** Ingest and analyze the public, high-frequency central bank monetary policy shock series (FRBSF Bauer-Swanson and ECB EA-MPD) alongside ALFRED macro vintages.
   - **Step 2 (₹0 Outlay):** Formulate a formal event-driven hypothesis. If pre-registered criteria are satisfied, utilize Databento's $125 free signup credits to download a targeted sample of CME 6E trade ticks without spending funds.
   - **Step 3 (Conditional Purchase):** Only if the free credit sample demonstrates a statistically defensible edge should a modest cash purchase ($15 USD / ~₹1,250 INR) be considered.
3. **Governance Status:**
   - **TRADING AUTHORIZED:** NO.
   - **PAPER TRADING:** NOT AUTHORIZED.
   - **LIVE TRADING:** NOT AUTHORIZED.
   - **PHASE 45:** NOT AUTHORIZED.

---
