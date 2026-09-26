# AI Autonomous Trading System

An institutional-grade, research-driven automated trading platform built on statistical rigor, robust out-of-sample validation, and strict separation between quantitative signal generation and execution risk controls.

Canonical development environment: **Ubuntu Linux**.

---

## 1. Purpose and Philosophy

The primary objective of this project is to build a reproducible, testable, and statistically honest algorithmic trading system. We approach trading as an empirical research and engineering problem rather than speculative forecasting.

### Core Tenets
- **Empirical Validation First:** We do not assume that machine learning models possess predictive power in financial markets. A strategy is only deemed viable if it demonstrates a statistically significant edge over simple heuristic baselines after accounting for realistic transaction costs, slippage, spread, and latency.
- **Strict Separation of Concerns:** Research, feature calculation, and predictive modeling are strictly decoupled from broker execution and portfolio risk management.
- **Immutable Raw Data:** Market data is treated as immutable historical truth; data cleaning and transformations must be strictly point-in-time and point-to-point without look-ahead bias or data leakage.
- **Safety Over Returns:** Capital preservation and drawdown management dictate system constraints. The AI model suggests candidate signals; deterministic risk rules govern whether capital is allocated.

---

## 2. System Architecture

The end-to-end data and execution pipeline is structured as follows:

```text
Historical Market Data (MetaTrader 5 on Ubuntu)
         ↓
Data Quality Validation (Schema, Gaps, Outliers, Session Boundaries)
         ↓
Data Cleaning & Normalization (Point-in-Time, Zero Look-Ahead)
         ↓
Feature Engineering (Price Action, Trend, Volatility, Regimes)
         ↓
Baseline Heuristic Strategies (Buy & Hold, SMA Crossover, RSI Rules)
         ↓
Vectorized & Event-Driven Backtesting (Spreads, Slippage, Fees)
         ↓
Machine Learning (Linear Models → Random Forest → XGBoost)
         ↓
Chronological Out-of-Sample Testing & Stress Testing
         ↓
Prediction & Standardized Signal Generation
         ↓  [HTTP / REST Signal Interface]
Risk Engine & Pre-Trade Checks (Deterministic Safety Core)
         ↓
Demo / Paper Execution (MetaTrader 5)
         ↓
(Only after exhaustive paper validation) → Production / Live Execution
```

---

## 3. Package Structure & Responsibilities

The codebase brings together quantitative research, signal generation, and deterministic risk execution in one unified repository:

### `trading/` — Execution, Risk & Broker Gateway
- **Domain:** Execution, broker connectivity, risk controls, and order routing.
- **Components:**
  - MetaTrader 5 (MT5) gateway and Python bridge
  - Pre-trade RiskEngine (margin limits, maximum drawdown caps, position sizing)
  - Order execution and latency monitoring
  - PaperBroker and demo trading account operations
  - Production deployment (FastAPI service, monitoring)

### `ai/` — Research, Data Pipeline & Signal Engine
- **Domain:** Quantitative research, market data pipelines, machine learning, and signal generation.
- **Components:**
  - Historical market data acquisition, storage, and validation (`ai/data/`)
  - Point-in-time feature engineering and market regime classification (`ai/features/`)
  - Baseline strategy implementations and benchmark comparisons (`ai/backtesting/`)
  - Statistical time-series modeling (Linear, Tree-based, Gradient Boosted) (`ai/models/`)
  - Backtesting engine simulation with cost models (`ai/backtesting/`)
  - Standardized JSON signal payload generation (`ai/prediction/`)
  - Analytical dashboard (`dashboard/`)

---

## 4. Standardized Signal Interface

The research engine communicates trade intent to the execution server via a strictly typed, versioned JSON payload:

```json
{
  "symbol": "EURUSD",
  "timestamp": "2026-09-25T12:00:00Z",
  "action": "BUY",
  "confidence": 0.72,
  "model_version": "xgb_v1",
  "timeframe": "15m",
  "expected_return": 0.004,
  "feature_version": "features_v1"
}
```

> [!IMPORTANT]
> **Signal Confidence vs. Account Risk:**
> A signal confidence value of `0.72` reflects statistical model certainty under historical distributions. It does **NOT** indicate risking 72% of portfolio equity. Sizing, leverage, and stop-loss placement are strictly governed by the RiskEngine.

---

## 5. Local Setup and Environment Management

### Prerequisites
- Ubuntu Linux (`x86_64`)
- Python `>=3.12` (Canonical: `3.13`)
- `uv` package manager (`>= 0.6.x`)
- Git (`>= 2.40.x`)
- Wine 10.0 + MT5 (for live MT5 demo connectivity on Linux)

### Setup Instructions
1. **Clone the repository and switch to `develop`:**
   ```bash
   git clone https://github.com/Suman-KM/Trading_bot.git
   cd Trading_bot
   git checkout develop
   ```

2. **Synchronize dependencies into `.venv`:**
   ```bash
   uv sync
   ```

3. **Verify environment and test suite:**
   ```bash
   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   ```

---

## 6. Repository Layout

```text
Trading_bot/
│
├── README.md                  # Project documentation & architecture overview
├── pyproject.toml             # PEP 621 metadata & dependencies
├── .gitignore                 # Exclusion rules for virtual environments, data, caches
├── .env.example               # Template environment configuration (no secrets)
│
├── data/                      # Historical and processed market data (gitignored)
│   ├── raw/                   # Immutable raw OHLCV snapshots (.gitkeep)
│   ├── processed/             # Cleaned, aligned time-series datasets (.gitkeep)
│   └── external/              # Reference and calendar data (.gitkeep)
│
├── trading/                   # Execution, Risk Engine & Broker Gateway
│   ├── api/                   # Local FastAPI signal API
│   ├── execution/             # PaperBroker and execution service
│   ├── models/                # Pydantic schemas (Order, Position, Portfolio, Signal)
│   ├── portfolio/             # Portfolio manager and bookkeeping
│   └── risk/                  # Deterministic RiskEngine and Kill Switch
│
├── ai/                        # Research, Data Pipeline & Signal Engine
│   ├── data/                  # Ingestion, validation, gap analysis, schema
│   ├── features/              # Point-in-time feature extraction & indicators
│   ├── models/                # Model wrappers (Baseline, Random Forest, XGBoost)
│   ├── training/              # Chronological walk-forward training routines
│   ├── prediction/            # Inference engine & signal payload formatting
│   ├── evaluation/            # Statistical validation & out-of-sample metrics
│   └── backtesting/           # Vectorized & event simulation with cost models
│
├── dashboard/                 # Streamlit analytical dashboard (.gitkeep)
├── notebooks/                 # Exploratory research & environment checks
├── scripts/                   # Utility and export scripts
├── tests/                     # Automated test suites
└── docs/                      # Architectural specs & research documentation
```

---

## 7. Safety & Compliance Principles

1. **No Live Trading During Development:** Under no circumstances will real capital or live broker endpoints be connected during development and backtesting.
2. **Deterministic Risk Authority:** The AI subsystem has zero direct order placement authority. The RiskEngine acts as an immutable gatekeeper.
3. **No Look-Ahead or Data Snooping:** Validation and test splits are strictly chronological. Features are computed point-in-time without future leakage.
4. **Zero Secrets in Version Control:** Credentials, API keys, broker tokens, and proprietary trading records must never be committed to Git.
