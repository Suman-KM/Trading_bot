# Phase 34 Engineering Report: Persistent Paper Trading, Market-Data Replay & Monitoring

AI Autonomous Trading System — Paper Trading Infrastructure Hardening
Ubuntu-Only | Engineering-Only | Paper-Only | Zero Real-Money | Zero Model Training

---

## 1. Governance & Role Clarification

Phase 32 permanently determined that the information set available in 2010–2026 EURUSD market data provides no statistically significant out-of-sample directional predictive edge. In strict adherence to that governance decision, **Phase 34 is exclusively an engineering phase**.

Under no circumstances does Phase 34:
- Train machine learning models
- Perform feature engineering or indicator mining
- Tune hyperparameters or optimize decision thresholds
- Run predictive alpha backtests
- Access the permanently locked test partition (`>= 2026-02-19 12:00:00 UTC`)
- Connect to live brokers, demo brokers, or execute real orders
- Place real-money capital at risk

The sole mission of Phase 34 is to harden the paper-trading platform created in Phase 33 into an enterprise-grade, deterministic, persistent, restart-safe, observable, and replayable paper execution system.

---

## 2. Architecture Overview & Component Diagram

The paper-trading platform separates execution safety, portfolio risk management, persistent storage, and market replay:

```
[ Market Observation / Replay Stream ]  -->  [ ReplayEngine ]
                                                    |
[ AI Signal (Synthetic / Paper) ]                   v
                 |                         [ PaperBroker ] <---> [ PaperTradingRepository ]
                 v                                  ^                      |
    [ TradingExecutionService ]                     |                      v
                 |                                  |          [ SQLite WAL Database ]
                 v                                  |            (paper_trading.db)
         [ RiskEngine ] ----------------------------+
                 |
                 +--> [ KillSwitch ]
                 +--> [ PortfolioManager ]
                 +--> [ AuditTrail (SHA-256 Chained) ]
```

---

## 3. Persistent Data Store Design

In accordance with Section 5 of the design prompt ("Prefer the simplest reliable architecture compatible with the existing Ubuntu environment"), the system utilizes Python's built-in `sqlite3` engine configured for enterprise production:

1. **Write-Ahead Logging (WAL Mode)**: Configured via `PRAGMA journal_mode = WAL;`. Readers never block writers, and writers never block readers.
2. **Foreign Key Enforcement**: Configured via `PRAGMA foreign_keys = ON;`. Cascades and parent-child constraints are strictly validated.
3. **Full Synchronous Durability**: Configured via `PRAGMA synchronous = FULL;` ensuring ACID persistence to physical storage before transaction completion.
4. **Immediate Transaction Locks**: Transactions execute using `BEGIN IMMEDIATE;` to serialize writes and eliminate database lock contention across threads.
5. **Thread Safety**: The connection manager (`PaperDatabaseManager`) wraps connection operations in a reentrant lock (`threading.RLock()`), enabling concurrent reader/writer safety.
6. **Isolated In-Memory Support**: Supports `:memory:` mode for sub-millisecond, hermetic test execution.

---

## 4. Database Schema Specification

The database schema defines 7 tables:

### 1. `paper_orders`
Stores simulated order specifications, state transitions, client request references, and fill details:
- `order_id TEXT PRIMARY KEY`: Unique identifier.
- `client_request_id TEXT`: Client idempotency key.
- `symbol TEXT NOT NULL`: Traded asset.
- `direction TEXT NOT NULL`: `BUY` or `SELL`.
- `order_type TEXT NOT NULL`: `MARKET` or `LIMIT`.
- `quantity REAL NOT NULL`: Trade volume.
- `requested_price REAL NOT NULL`: Price at submission.
- `stop_loss REAL`: Protective exit threshold.
- `take_profit REAL`: Profit-taking threshold.
- `status TEXT NOT NULL`: Order lifecycle state (`CREATED`, `VALIDATED`, `SUBMITTED`, `FILLED`, `REJECTED`, `CANCELLED`).
- `created_at TEXT NOT NULL`: Submission timestamp (ISO 8601 UTC).
- `updated_at TEXT NOT NULL`: Last state change timestamp.
- `fill_price REAL`: Executed fill price.
- `fill_timestamp TEXT`: Fill timestamp.
- `rejection_reason TEXT`: Reason code if rejected.

### 2. `paper_executions`
Auditable execution records detailing transaction cost analysis (TCA) and P&L:
- `execution_id TEXT PRIMARY KEY`
- `order_id TEXT NOT NULL REFERENCES paper_orders(order_id)`
- `client_request_id TEXT`
- `timestamp TEXT NOT NULL`
- `symbol TEXT NOT NULL`
- `side TEXT NOT NULL`
- `requested_price REAL NOT NULL`
- `executed_price REAL NOT NULL`
- `quantity REAL NOT NULL`
- `spread REAL NOT NULL DEFAULT 0.0`
- `slippage REAL NOT NULL DEFAULT 0.0`
- `commission REAL NOT NULL DEFAULT 0.0`
- `swap REAL NOT NULL DEFAULT 0.0`
- `gross_pnl REAL NOT NULL DEFAULT 0.0`
- `net_pnl REAL NOT NULL DEFAULT 0.0`
- `execution_status TEXT NOT NULL`
- `rejection_reason TEXT`

### 3. `paper_positions`
Tracks active and closed position lifecycles:
- `position_id TEXT PRIMARY KEY`
- `symbol TEXT NOT NULL`
- `direction TEXT NOT NULL`
- `quantity REAL NOT NULL`
- `entry_price REAL NOT NULL`
- `current_price REAL`
- `exit_price REAL`
- `stop_loss REAL`
- `take_profit REAL`
- `entry_timestamp TEXT NOT NULL`
- `exit_timestamp TEXT`
- `gross_pnl REAL NOT NULL DEFAULT 0.0`
- `costs REAL NOT NULL DEFAULT 0.0`
- `net_pnl REAL NOT NULL DEFAULT 0.0`
- `unrealized_pnl REAL NOT NULL DEFAULT 0.0`
- `status TEXT NOT NULL` (`OPEN`, `ACTIVE`, `CLOSED`)

### 4. `paper_audit_events`
Cryptographically hash-chained append-only event ledger:
- `sequence_id INTEGER PRIMARY KEY AUTOINCREMENT`
- `event_id TEXT UNIQUE NOT NULL`
- `timestamp TEXT NOT NULL`
- `event_type TEXT NOT NULL`
- `source TEXT NOT NULL`
- `order_id TEXT`
- `position_id TEXT`
- `symbol TEXT`
- `details_json TEXT NOT NULL`
- `prev_hash TEXT`
- `event_hash TEXT`

### 5. `paper_account_snapshots`
Point-in-time account equity and balance snapshots:
- `snapshot_id TEXT PRIMARY KEY`
- `timestamp TEXT NOT NULL`
- `initial_balance REAL NOT NULL`
- `cash_balance REAL NOT NULL`
- `equity REAL NOT NULL`
- `margin_used REAL NOT NULL`
- `realized_pnl REAL NOT NULL`
- `unrealized_pnl REAL NOT NULL`
- `total_costs REAL NOT NULL`
- `daily_pnl REAL NOT NULL`
- `current_exposure REAL NOT NULL`

### 6. `paper_risk_state`
Singleton table maintaining daily baseline equity and kill switch state across restarts:
- `singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1)`
- `date_utc TEXT NOT NULL`
- `daily_loss REAL NOT NULL`
- `baseline_equity REAL NOT NULL`
- `kill_switch_active INTEGER NOT NULL`
- `kill_switch_reason TEXT`
- `kill_switch_timestamp TEXT`
- `updated_at TEXT NOT NULL`

### 7. `paper_idempotency_records`
Persistent idempotency table preventing duplicate signal processing across restarts:
- `client_request_id TEXT PRIMARY KEY`
- `order_id TEXT NOT NULL`
- `execution_id TEXT NOT NULL`
- `status TEXT NOT NULL`
- `execution_report_json TEXT NOT NULL`
- `created_at TEXT NOT NULL`

---

## 5. Schema Migration Framework

The migration runner (`PaperSchemaMigrator`) implements versioned, idempotent schema execution:
- Tracks applied versions in `paper_schema_migrations (version, name, applied_at_utc)`.
- Verifies migration existence prior to execution.
- Runs inside an explicit atomic database transaction (`BEGIN IMMEDIATE`).
- Idempotent: Subsequent invocations detect applied versions and apply 0 migrations.

---

## 6. Market Data Replay Engine Design

The `ReplayEngine` provides event-driven replay of historical market observations into the paper-trading platform:
1. **Modes**:
   - `FAST_REPLAY`: Batch execution processing bars at maximum CPU speed for high-throughput determinism verification.
   - `REALTIME_SIMULATED_REPLAY`: Paced execution simulating wall-clock tick progression with configurable speedup multiplier.
2. **Event Streaming**: Streams validated `MarketEvent` instances sequentially.
3. **Scheduled Order Evaluation**: Injects pre-scheduled AI test signals at exact historical timestamps.

---

## 7. Replay Data Ingestion & Sanitization

The `MarketReplayDataLoader` validates candle and tick datasets prior to replay:
1. **Timestamp Monotonicity**: Rejects non-monotonic (`t_{i} < t_{i-1}`) or duplicate (`t_{i} == t_{i-1}`) records by raising `DataIntegrityError`.
2. **Price Sanitization**: Validates that all prices are strictly positive, finite numbers (`math.isfinite(p)` and `p > 0`). Rejects `NaN`, `Inf`, 0, or negative values.
3. **OHLC Consistency**: Enforces `Low <= Open <= High` and `Low <= Close <= High`.
4. **Volume Sanitization**: Normalizes volume and tick volume fields.

---

## 8. Pre-Registered Locked-Test Quarantine Enforcement

In accordance with the fundamental governance rules across Phases 11–34:
- The locked test partition (`2026-02-19 12:00:00 UTC` onward) remains strictly quarantined.
- `MarketReplayDataLoader` validates every bar timestamp against `LOCKED_TEST_CUTOFF = datetime(2026, 2, 19, 12, 0, 0, tzinfo=timezone.utc)`.
- Any historical record with `timestamp >= LOCKED_TEST_CUTOFF` immediately aborts loading and raises `QuarantinedPartitionError`.

---

## 9. Bracket Order Evaluation & Collision Rule

During each replay step, active positions are evaluated against the current market candle extremes:
- **Long Positions**: Stop-Loss hit if `candle.low <= pos.stop_loss`; Take-Profit hit if `candle.high >= pos.take_profit`.
- **Short Positions**: Stop-Loss hit if `candle.high >= pos.stop_loss`; Take-Profit hit if `candle.low <= pos.take_profit`.
- **Bracket Collision Rule (Capital Preservation Invariant)**: In high-volatility bars where both the Stop-Loss and Take-Profit price thresholds are breached within the same bar, the system enforces a strict capital preservation rule: **Stop-Loss executes first**. The position is liquidated at the Stop-Loss price, preserving capital and preventing optimistic fill bias.

---

## 10. Replay Checkpoint & Resume Architecture

The replay engine implements serializable checkpointing:
- `create_checkpoint(checkpoint_id)` serializes the current cursor index, broker account state, active position details, closed positions count, orders count, and execution counts.
- `resume_from_checkpoint(checkpoint)` restores cursor location, scheduled order execution history, and bracket execution counters.
- Uninterrupted replay vs checkpointed-and-resumed replay produces bit-identical account balances, realized P&L, and open positions.

---

## 11. Idempotency Architecture Across Process Restarts

Idempotency deduplication operates at two tiers:
1. **In-Memory Cache**: Fast hash lookup for deduplication within the active process lifetime.
2. **Persistent Storage**: Idempotency records stored in SQLite table `paper_idempotency_records`.
3. **Restart Deduplication**: When the process restarts and re-receives a signal with a previously processed `client_request_id`, the system loads the execution result from SQLite, marks `is_idempotent_replay = True`, returns the identical execution report, and prevents duplicate execution or double-spending.

---

## 12. Restart Recovery Verification

The recovery mechanism reconstructs complete portfolio state:
1. Reloads all orders and historical executions.
2. Reloads all active and closed positions.
3. Loads the latest account snapshot (`initial_balance`, `cash_balance`, `realized_pnl`, `total_costs`).
4. Reconstructs mark-to-market equity:
   $$\text{Reconstructed Equity} = \text{Initial Balance} + \text{Realized P&L} + \sum \text{Unrealized P&L}$$
5. Verifies reconstructed equity matches persisted snapshot equity within 0.05 tolerance. If discrepancy is detected, raises `InconsistentStateError`.

---

## 13. Kill-Switch State Persistence & Recovery

The emergency kill-switch state is persistently synchronized in `paper_risk_state`:
- Activation records `kill_switch_active = 1`, `kill_switch_reason`, and `kill_switch_timestamp`.
- Deactivation records `kill_switch_active = 0`.
- On restart, `KillSwitch.recover_from_repository()` reads the stored record and re-activates if previously tripped, rejecting all subsequent orders until explicitly cleared.

---

## 14. Daily-Loss Tracking & UTC Calendar Date Boundaries

Daily loss limits are managed across restarts according to UTC calendar days:
- `PortfolioManager` records `date_utc` and `baseline_equity`.
- **Same UTC Calendar Day**: Reconstructed manager preserves the start-of-day baseline equity and accurately accumulates intraday losses.
- **New UTC Calendar Day**: On the rollover to the next UTC day, the baseline equity rolls forward to current equity, resetting the daily loss accumulator to 0.0%.

---

## 15. Portfolio Exposure Reconstruction Across Restarts

Gross portfolio market exposure is reconstructed across restarts:
- Active positions are reloaded from `paper_positions`.
- Portfolio manager computes gross market exposure:
  $$\text{Gross Exposure \%} = \frac{\sum (\text{Quantity} \times \text{Current Price})}{\text{Account Equity}} \times 100$$
- Reconstructed exposure exactly matches pre-restart exposure.

---

## 16. Cryptographic Audit Trail & Tamper-Evident Hash Chaining

The audit trail provides cryptographic tamper-evidence:
- Each event is recorded with an incremental `sequence_id`, `event_id`, UTC timestamp, structured JSON details, `prev_hash`, and `event_hash`.
- The hash chain links records using SHA-256:
  $$\text{Event Hash}_n = \text{SHA256}(\text{Event Hash}_{n-1} \parallel \text{Event ID}_n \parallel \text{Timestamp}_n \parallel \text{Event Type}_n \parallel \text{Details JSON}_n)$$
- `verify_audit_trail_integrity()` verifies sequential continuity and recalculates SHA-256 hashes across all records.
- Any manual SQL modification, deletion, or insertion causes immediate verification failure.

---

## 17. REST API Endpoints & Observability Enhancements

The FastAPI operational endpoints provide complete system visibility:
- **`GET /health`**: Returns operational status, environment (`DEMO`), and trading backend (`PAPER`).
- **`GET /readiness`**: Returns 200 OK when ready, or **HTTP 503 Service Unavailable** with structured failure diagnostics if the database is unreachable, corrupted, or the kill-switch is active.
- **`GET /metrics`**: Exposes operational metrics including orders received, filled, cancelled, costs, daily loss, exposure, database connectivity, and counts of persisted records.
- **`GET /audit/events`**: Returns immutable structured audit events.
- **`GET /executions`**: Returns auditable execution reports with full TCA breakdown.

---

## 18. Fail-Closed Persistence Error Handling

The system strictly enforces fail-closed behavior:
- If a database failure occurs during signal ingestion, order submission, or state persistence, the operation is rejected (`approved = False`).
- In-memory portfolio balances and position registries remain untouched.
- The failure is logged and recorded in the audit trail.

---

## 19. Concurrency & Reentrancy Safety

- `TradingExecutionService` protects signal processing with an internal reentrant lock (`threading.RLock()`).
- Concurrent duplicate requests with the same `client_request_id` are serialized; the first executes, and the remaining 9 return idempotent replay copies with identical order IDs.
- Exactly 1 simulated order is placed in the broker.

---

## 20. Verification Suite Results

All 20 formal verification requirements (A through T) pass in `tests/test_phase34_persistence_replay.py`:

| Test | Verification Requirement | Status |
|------|--------------------------|--------|
| **A** | Database schema creation, version tracking, idempotency | **PASSED** |
| **B** | Order persistence, state transitions, query history | **PASSED** |
| **C** | Execution report persistence, TCA breakdown, swap | **PASSED** |
| **D** | Position persistence, active vs closed query isolation | **PASSED** |
| **E** | Audit persistence, cryptographic hash chaining, tamper detection | **PASSED** |
| **F** | Idempotency persistence and deduplication across process restarts | **PASSED** |
| **G** | Atomic transaction rollback on simulated crash | **PASSED** |
| **H** | Restart recovery: reconstructed equity matches persisted equity | **PASSED** |
| **I** | Kill-switch recovery across process restart | **PASSED** |
| **J** | Daily-loss recovery across restart (same day vs new UTC day) | **PASSED** |
| **K** | Portfolio gross exposure reconstruction across restarts | **PASSED** |
| **L** | Replay ordering and non-monotonic validation (`DataIntegrityError`) | **PASSED** |
| **M** | Replay determinism (bit-identical balances and orders across runs) | **PASSED** |
| **N** | Replay checkpoint and resume determinism | **PASSED** |
| **O** | Stop-loss replay trigger and execution at SL threshold | **PASSED** |
| **P** | Take-profit trigger and bracket collision rule (SL executes first) | **PASSED** |
| **Q** | Concurrent duplicate request deduplication (10 threads, 1 execution) | **PASSED** |
| **R** | Persistence failure fail-closed behavior | **PASSED** |
| **S** | Readiness failure detection on corrupt state (HTTP 503) | **PASSED** |
| **T** | Locked-test quarantine enforcement (`QuarantinedPartitionError`) | **PASSED** |

**Full Test Suite Summary**:
- Phase 34 dedicated tests: 20 passed / 20 total (100%)
- Total project tests: 490 passed / 490 total (100%)
- Ruff Linting: All checks passed (clean)
- Ruff Formatting: 231 files checked, 100% compliant

---

## 21. Executive Summary & Non-Trading Platform Declaration

Phase 34 has achieved all objectives. The paper-trading platform is now:
1. **Persistent**: All orders, executions, positions, audit events, account snapshots, and risk states are stored with ACID guarantees in WAL-enabled SQLite.
2. **Restart-Safe**: The system completely reconstructs its financial and operational state across restarts.
3. **Deterministically Replayable**: Replays historical candles with bracket triggers, collision safety, and checkpoint/resume capabilities.
4. **Auditable & Observable**: Features SHA-256 hash-chained audit logging and extended REST API observability endpoints.
5. **Fail-Closed & Secure**: Operates without external credentials, fails closed upon persistence faults, and strictly enforces the locked test partition quarantine.

**NON-TRADING DECLARATION**:
This platform remains paper-only. Zero machine learning models were trained. Zero live trades or broker demo trades were executed.
