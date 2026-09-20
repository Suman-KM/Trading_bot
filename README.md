# AI Autonomous Trading System

An institutional-grade, research-driven automated trading platform built on statistical rigor, robust out-of-sample validation, and strict separation between quantitative signal generation and execution risk controls.

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
Historical Market Data
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
Member 1 Risk Engine & Pre-Trade Checks (Linux / Server)
         ↓
Demo / Paper Execution (MetaTrader 5)
         ↓
(Only after exhaustive paper validation) → Production / Live Execution
```

---

## 3. Team Responsibilities & Scope Separation

The engineering effort is divided across two distinct operating environments to ensure clean boundaries between research and execution:

### Member 1 — Linux / Server Infrastructure
- **Domain:** Execution, broker connectivity, risk controls, and server operations.
- **Responsibilities:**
  - MetaTrader 5 (MT5) gateway and Python bridge
  - Pre-trade risk engine (margin limits, maximum drawdown caps, position sizing)
  - Order execution and latency monitoring
  - Paper / demo trading account operations
  - Production deployment (Docker, systemd, continuous monitoring)
- **Primary Codebase Ownership:**
  ```text
  trading/
  ├── mt5/
  ├── execution/
  ├── risk/
  └── portfolio/
  ```

### Member 2 — macOS (Apple Silicon) — Research & Signal Engine (Current Workspace)
- **Domain:** Quantitative research, market data pipelines, machine learning, and signal generation.
- **Responsibilities:**
  - Historical market data acquisition, storage, and validation
  - Point-in-time feature engineering and market regime classification
  - Baseline strategy implementations and benchmark comparisons
  - Statistical time-series modeling (Linear, Tree-based, Gradient Boosted)
  - Backtesting engine simulation (with realistic slippage and transaction costs)
  - Standardized JSON signal payload generation
  - Streamlit research and backtest analysis dashboard
- **Primary Codebase Ownership:**
  ```text
  data/
  ai/
  dashboard/
  notebooks/
  ```

### Shared Interfaces
- `api/`: Inter-service schemas and communication contracts
- `tests/`: End-to-end integration and smoke test suites
- `docs/`: System documentation and architecture specifications

---

## 4. Current Development Phase

**Milestone 1: Research Environment & Repository Foundation**

Current Status:
- [x] macOS Apple Silicon (arm64) development environment validated
- [x] Python 3.12 virtual environment established using `uv`
- [x] PEP 621 package metadata and PEP 735 dependency groups configured
- [x] Core research libraries installed (NumPy, Pandas, PyArrow, Scikit-Learn, XGBoost, Streamlit, FastAPI, Pytest, Ruff)
- [x] Security policies implemented (strict `.gitignore`, `.env.example`, zero credentials)
- [x] Structural scaffolding and deterministic environment smoke tests in place
- [ ] Historical market data acquisition (Awaiting Member 1 broker specification)

---

## 5. Standardized Signal Interface

The research engine communicates trade intent to Member 1's execution server via a strictly typed, versioned JSON payload:

```json
{
  "symbol": "SYMBOL_TBD",
  "timestamp": "2026-09-21T12:00:00Z",
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
> A signal confidence value of `0.72` reflects statistical model certainty under historical distributions. It does **NOT** indicate risking 72% of portfolio equity. Sizing, leverage, and stop-loss placement are strictly governed by Member 1's risk engine.

---

## 6. Local Setup and Environment Management

### Prerequisites
- macOS Apple Silicon (`arm64`)
- Python `3.12.x`
- `uv` package manager (`>= 0.12.x`)
- Git (`>= 2.50.x`)

### Setup Instructions
1. **Clone the repository and switch to `develop`:**
   ```bash
   git clone <repo-url> ai-trading-system
   cd ai-trading-system
   git checkout develop
   ```

2. **Synchronize dependencies into `.venv`:**
   ```bash
   uv sync --all-groups
   ```

3. **Configure local environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env to configure local API host/port (do not commit .env)
   ```

4. **Verify environment and test suite:**
   ```bash
   # Run automated test suite
   uv run pytest tests/

   # Run code linter
   uv run ruff check .
   ```

5. **Launch Jupyter for exploratory research:**
   ```bash
   uv run jupyter lab notebooks/00_environment_check.ipynb
   ```

---

## 7. Repository Layout

```text
ai-trading-system/
│
├── README.md                  # Project documentation & architecture overview
├── pyproject.toml             # PEP 621 metadata & PEP 735 dependency groups
├── .gitignore                 # Exclusion rules for virtual environments, data, caches
├── .env.example               # Template environment configuration (no secrets)
│
├── data/                      # Historical and processed market data (gitignored)
│   ├── raw/                   # Immutable raw OHLCV snapshots (.gitkeep)
│   ├── processed/             # Cleaned, aligned time-series datasets (.gitkeep)
│   └── external/              # Reference and calendar data (.gitkeep)
│
├── ai/                        # Member 2 Research & Signal Package
│   ├── __init__.py            # Root package definition & version
│   ├── features/              # Point-in-time feature extraction & indicators
│   ├── models/                # Model wrappers (Baseline, Random Forest, XGBoost)
│   ├── training/              # Chronological walk-forward training routines
│   ├── prediction/            # Inference engine & signal payload formatting
│   ├── evaluation/            # Statistical validation & out-of-sample metrics
│   └── backtesting/           # Vectorized & event simulation with cost models
│
├── dashboard/                 # Streamlit analytical dashboard (.gitkeep)
├── notebooks/                 # Exploratory research & environment checks
├── scripts/                   # Utility and data extraction scripts (.gitkeep)
├── tests/                     # Automated test suites & deterministic smoke tests
└── docs/                      # Architectural specs & research documentation
```

---

## 8. Safety & Compliance Principles

1. **No Live Trading During Research:** Under no circumstances will live capital or live broker endpoints be connected during development and backtesting.
2. **No Profit Guarantees:** Financial markets exhibit low signal-to-noise ratios and non-stationary regimes. No claim of future profitability is made.
3. **No Direct Account Access:** The AI subsystem has zero execution authority. Member 1's risk engine acts as an immutable gatekeeper.
4. **Instrument Neutrality:** The trading instrument is not pre-selected. It will be decided strictly based on Member 1's MT5 broker specifications (spread, liquidity, commission structure, and historical tick quality).
5. **No Data Snooping:** Validation and test splits are strictly chronological (e.g., Train: 2022-2024, Validation: 2025, Test: 2026). The final test partition remains quarantined until model architecture and hyperparameters are finalized.
6. **Zero Secrets in Version Control:** Credentials, API keys, broker tokens, and proprietary trading records must never be committed to Git.

---

## 9. Development Roadmap

| Phase | Milestone | Description | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Repository Foundation** | Setup macOS Python 3.12 env, uv sync, PEP 621, tests, git scaffolding | **Completed** |
| **Phase 2** | **Broker & Data Spec** | Receive MT5 symbol & contract spec from Member 1; select data provider | *Pending* |
| **Phase 3** | **Data Pipeline** | Ingest, validate (OHLC integrity, zero gaps), and store immutable parquet | *Pending* |
| **Phase 4** | **Baseline Models** | Implement naive baselines (Buy & Hold, SMA cross, RSI) with cost modeling | *Pending* |
| **Phase 5** | **Feature Pipeline** | Build point-in-time feature transformations with leakage diagnostics | *Pending* |
| **Phase 6** | **ML Signal Engine** | Train & cross-validate Logistic Regression, Random Forest, and XGBoost | *Pending* |
| **Phase 7** | **Paper / Integration** | Connect signal interface to Member 1 risk engine for paper forward testing | *Pending* |
