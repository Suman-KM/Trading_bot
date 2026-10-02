# Phase 33 — Paper Trading, Risk & Execution Engineering Documentation

## 1. Architecture

Following the Phase 32 research synthesis and governance gate (Option B: Pivot to System Engineering / Risk / Execution), Phase 33 transitions the AI Trading System away from predictive alpha modeling into a hardened, deterministic, observable, and auditable paper-trading execution platform.

The system architecture consists of a layered, decoupled design:
- **API Layer (`trading/api`)**: FastAPI-based REST endpoints providing interfaces for paper trading interaction, health/readiness probing, execution query, audit inspection, and operational metrics.
- **Service Layer (`trading/execution/service.py`)**: `TradingExecutionService` coordinates signal intake, idempotency verification, risk screening, paper broker submission, execution report generation, and audit logging.
- **Risk Gate (`trading/risk/engine.py`, `trading/risk/kill_switch.py`)**: Authoritative risk management kernel enforcing hard invariant constraints (daily loss, single position risk, maximum open positions, aggregate portfolio exposure, confidence thresholds, and kill-switch state).
- **Execution Engine (`trading/execution/paper_broker.py`)**: Isolated in-memory broker maintaining account equity, open/closed position inventories, and order state transitions.
- **Broker Abstraction (`trading/execution/adapter.py`)**: Formal `BrokerAdapter` interface establishing strict component boundaries and isolating paper execution from broker protocols.
- **Cost Model (`trading/execution/costs.py`)**: Explicit, modular accounting for spread, slippage, commission, and swap fees.
- **Audit Trail (`trading/audit/trail.py`)**: Append-only, thread-safe in-memory audit ledger recording all system events with microsecond timestamps and structured context.

```
       [ Signal / REST Client ]
                  │
                  ▼
       [ TradingExecutionService ] ◄────► [ Idempotency Cache ]
                  │
                  ▼
           [ RiskEngine ] ◄─────────────► [ KillSwitch ]
                  │
             (If Approved)
                  ▼
          [ PaperBroker ] ◄─────────────► [ TransactionCostConfig ]
                  │
                  ├───► [ Position Lifecycle & Balance Accounting ]
                  ├───► [ ExecutionReport Generation ]
                  ▼
           [ AuditTrail ] (Append-only immutable event ledger)
```

---

## 2. Component Boundaries

Each subsystem adheres to strict interface boundaries:
- **Signals**: Intake models (`TradeSignal`) enforce frozen pydantic schemas. Signal producers cannot inject order parameters directly into the broker without routing through the `TradingExecutionService` and `RiskEngine`.
- **Risk Engine Boundary**: The `RiskEngine` is an immutable gatekeeper. The broker adapter and execution service cannot bypass risk checks under any circumstances. If `RiskEngine.evaluate()` rejects a signal, order creation is aborted and an auditable `RISK_REJECTED` event is emitted.
- **Broker Adapter Boundary**: Defined by `BrokerAdapter` (`trading/execution/adapter.py`). The concrete implementation `PaperBrokerAdapter` wraps `PaperBroker`. It guarantees `is_live == False`, zero network calls, zero external socket connections, and zero broker API bindings (e.g. MT5).
- **Audit Ledger Boundary**: The `AuditTrail` is decoupled and thread-safe. Other components emit events to it without depending on persistent database drivers, ensuring zero data leakage or execution blocking.

---

## 3. Risk Flow

Signal evaluation follows a deterministic 10-step validation pipeline in `RiskEngine`:
1. **Kill Switch Check**: If `KillSwitch.is_active` is True, reject with `KILL_SWITCH_ACTIVE`.
2. **Confidence Threshold**: Signal confidence must meet or exceed `MIN_SIGNAL_CONFIDENCE` (0.60).
3. **Daily Loss Check**: Cumulative daily loss must not equal or exceed `MAX_DAILY_LOSS_PERCENT` (1.0% of initial equity).
4. **Max Open Positions**: Active positions must be strictly less than `MAX_OPEN_POSITIONS` (3).
5. **Symbol Validation**: Symbol must be in allowed instrument universe (`SUPPORTED_SYMBOLS`, e.g. `EURUSD`).
6. **Price & Stop Validation**: Entry and stop-loss prices must be positive and non-zero. Stop-loss distance must be strictly positive (`abs(entry - stop_loss) > 0`).
7. **Directional Stop-Loss Consistency**: For BUY signals, `stop_loss < entry_price`. For SELL signals, `stop_loss > entry_price`.
8. **Take-Profit Consistency (if provided)**: For BUY, `take_profit > entry_price`. For SELL, `take_profit < entry_price`.
9. **Position Sizing & Single-Trade Risk**: Sizing calculates `risk_capital = equity * (risk_pct / 100)` and `quantity = risk_capital / stop_distance`. Position risk cannot exceed `MAX_POSITION_RISK_PERCENT` (0.5%).
10. **Aggregate Exposure Check**: Total notional exposure across all open positions plus the candidate order cannot exceed `MAX_TOTAL_EXPOSURE_PERCENT` (20.0% of equity).

Failure at any step halts evaluation and returns `RiskCheckResult(approved=False, reason=...)`.

---

## 4. Order State Machine

The `Order` model implements explicit, deterministic state transitions:

```
                    ┌─────────────┐
                    │   CREATED   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
        ┌───────────┤  VALIDATED  ├───────────┐
        │           └──────┬──────┘           │
        ▼                  │                  ▼
  ┌───────────┐            │            ┌───────────┐
  │ REJECTED  │            │            │ CANCELLED │
  └───────────┘            ▼            └───────────┘
                    ┌─────────────┐
         ┌──────────┤  SUBMITTED  ├──────────┐
         │          └──────┬──────┘          │
         ▼                 │                 ▼
   ┌───────────┐           ▼           ┌───────────┐
   │ REJECTED  │    ┌─────────────┐    │ CANCELLED │
   └───────────┘    │   FILLED    │    └───────────┘
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   CLOSED    │
                    └─────────────┘
```

- **Allowed Transitions**:
  - `CREATED` → `VALIDATED`, `REJECTED`, `CANCELLED`
  - `VALIDATED` → `SUBMITTED`, `FILLED`, `REJECTED`, `CANCELLED`
  - `SUBMITTED` → `FILLED`, `PARTIALLY_FILLED`, `REJECTED`, `CANCELLED`
  - `PARTIALLY_FILLED` → `FILLED`, `CANCELLED`
  - `FILLED` → `CLOSED`
  - Terminal states: `REJECTED`, `CANCELLED`, `CLOSED`.
- **Validation**: Any invalid transition (e.g. `FILLED` → `CREATED`, `CANCELLED` → `FILLED`, `CLOSED` → `SUBMITTED`) immediately raises `InvalidOrderStateTransitionError`.

---

## 5. Position Lifecycle

Positions advance through a rigorous lifecycle:
1. **Creation (`OPEN`)**: Generated upon order execution with initial parameters: `symbol`, `direction`, `entry_price`, `quantity`, `stop_loss`, `take_profit`, and `entry_timestamp`.
2. **Mark-to-Market (`ACTIVE`)**: As market prices update via `PaperBroker.update_market_price()`, unrealized P&L is recalculated. If status was `OPEN`, it transitions to `ACTIVE`.
3. **Automated Bracket Checks**: On each price update:
   - For `BUY`: If `current_price <= stop_loss`, triggers stop-loss closure; if `take_profit` is set and `current_price >= take_profit`, triggers take-profit closure.
   - For `SELL`: If `current_price >= stop_loss`, triggers stop-loss closure; if `take_profit` is set and `current_price <= take_profit`, triggers take-profit closure.
4. **Closure (`CLOSED`)**: When closed (either manually via `PaperBroker.close_position()` or automatically via SL/TP hit):
   - Computes `gross_pnl = (exit_price - entry_price) * quantity` for BUY, or `(entry_price - exit_price) * quantity` for SELL.
   - Computes closing transaction costs (spread, exit slippage, exit commission).
   - Computes `net_pnl = gross_pnl - total_costs`.
   - Records `exit_price` and `exit_timestamp`.
   - Updates account cash balance: `balance += (margin + net_pnl)`.
   - Atomically removes position from `_positions` dict and archives into `_closed_positions` list.

---

## 6. Cost Model

Implemented in `trading/execution/costs.py`:
- **Spread Cost**: Calculated as `half_spread_pips * pip_size * quantity` per leg (entry and exit).
- **Slippage Cost**: Configured as `slippage_pips * pip_size * quantity` per execution leg.
- **Commission**: Flat fee per trade plus per-unit commission (`commission_per_unit * quantity`).
- **Swap / Holding Cost**: Time-based financing fee: `swap_rate * holding_days * quantity`.
- **Default Policy**: All cost components default to `0.0` unless explicitly parameterized, avoiding silent fictitious estimates while allowing full fidelity when cost profiles are specified.
- **P&L Reporting**: Both `gross_pnl` and `net_pnl` are preserved in `Position`, `ExecutionReport`, and `PaperBroker.get_metrics()`.

---

## 7. Audit Events

The `AuditTrail` (`trading/audit/trail.py`) maintains an immutable record of system events. Each event captures:
- `event_id`: Unique identifier (`uuid4`).
- `timestamp`: UTC datetime with microsecond precision.
- `event_type`: Enumerated `AuditEventType`:
  1. `SIGNAL_RECEIVED`
  2. `RISK_ACCEPTED`
  3. `RISK_REJECTED`
  4. `ORDER_CREATED`
  5. `ORDER_REJECTED`
  6. `ORDER_FILLED`
  7. `POSITION_OPENED`
  8. `POSITION_UPDATED`
  9. `POSITION_CLOSED`
  10. `KILL_SWITCH_ACTIVATED`
  11. `KILL_SWITCH_DEACTIVATED`
  12. `EXECUTION_ERROR`
- `symbol`, `order_id`, `position_id`: Contextual linkages.
- `source`: Subsystem emitting the event (`trading_service`, `risk_engine`, `paper_broker`, `api`).
- `details`: Structured payload dictionary. Sensitive credentials and secrets are strictly excluded.

---

## 8. Idempotency

`TradingExecutionService` implements deterministic client request deduplication:
- Requests may provide a `client_request_id`.
- The service maintains an in-memory execution cache mapping `client_request_id` to its completed `ExecutionReport`.
- If an order with an existing `client_request_id` is re-submitted, the service immediately returns the cached `ExecutionReport` without re-evaluating risk, placing a second order, or creating a duplicate position.

---

## 9. Kill Switch

`KillSwitch` (`trading/risk/kill_switch.py`):
- Acts as a master safety cutoff across the application.
- When activated (`activate(reason=...)`):
  - Emits `KILL_SWITCH_ACTIVATED` audit event.
  - Causes `RiskEngine.evaluate()` to reject all incoming signals immediately.
  - Leaves existing open positions visible and untouched for controlled inspection.
- When deactivated (`deactivate()`):
  - Emits `KILL_SWITCH_DEACTIVATED` audit event.
  - Normal risk evaluation resumes.
- Accessible via thread-safe methods and REST endpoints (`POST /risk/kill-switch/activate`, `POST /risk/kill-switch/deactivate`).

---

## 10. Failure Handling & Atomic Consistency

- **Exception Capture**: In `TradingExecutionService.process_signal()`, execution exceptions are caught safely, recorded as `EXECUTION_ERROR` audit events, and returned as rejected `ExecutionReport` objects rather than unhandled server crashes.
- **Atomicity**:
  - If risk rejects an order, no broker order or position is created.
  - If order creation fails, account balance is untouched.
  - If a position is closed, it is instantaneously removed from active exposure calculations.
  - Exposure and open position counts are calculated directly from active positions, preventing phantom exposure or state drift.

---

## 11. API Behavior

The REST API (`trading/api/routes.py`) provides safe, read-only and paper-execution endpoints:
- `GET /health`: Liveness probe verifying the service process is running (`status: healthy`).
- `GET /readiness`: Deep readiness check validating `PaperBroker`, `RiskEngine`, `KillSwitch`, and portfolio invariants.
- `GET /account`: Account balance, equity, margin, realized and unrealized P&L.
- `GET /positions`: List of active paper trading positions.
- `GET /orders`: List of paper orders.
- `POST /signals`: Signal intake and paper execution pipeline.
- `GET /executions`: Structured history of execution reports.
- `GET /audit/events`: Queryable audit event trail with optional filtering by `event_type` and pagination.
- `GET /metrics`: Operational metrics counters.
- `GET /risk/status`: Risk limits, current exposure, daily loss, and kill-switch status.
- `POST /risk/kill-switch/activate` & `POST /risk/kill-switch/deactivate`: Emergency control endpoints.

---

## 12. Health vs Readiness

Phase 33 cleanly separates liveness from operational readiness:
- **Liveness (`GET /health`)**:
  - Simple ping probe confirming process responsiveness.
  - Returns `{"status": "healthy", "service": "trading-service"}`.
- **Readiness (`GET /readiness`)**:
  - Detailed state probe answering: *Is the paper execution engine internally ready to accept and safely process orders?*
  - Checks:
    - Broker initialized and instance of `PaperBroker`.
    - RiskEngine online.
    - KillSwitch status (if active, `ready: false` with reason).
    - Daily loss status (if exceeded, `ready: false`).
    - Portfolio consistency (open positions count, exposure within limits).
    - Unrecovered execution error status.

---

## 13. Security

- **Localhost Binding**: All endpoints and servers are designed strictly for loopback execution (`127.0.0.1`).
- **Zero Credentials**: No API keys, broker tokens, accounts, or passwords exist in codebase or configuration.
- **Log Sanitation**: Audit logging and execution reports strictly log order metadata and execution figures; sensitive fields are not captured.
- **No External Outbound Connectivity**: The execution engine has no network transport dependencies (no sockets, no HTTP client calls to external broker APIs, no MT5 terminal connection).

---

## 14. Testing

Phase 33 introduces a dedicated, comprehensive test suite in `tests/test_phase33_paper_trading.py` containing 18 rigorous test cases:
1. `test_signal_validation`: Schema validation, frozen properties, forbidden extra fields.
2. `test_risk_engine_limits_and_position_sizing`: Position sizing calculations and exposure caps.
3. `test_risk_engine_invalid_stop_loss_and_take_profit`: Directional validation for stops and targets.
4. `test_order_state_machine_valid_transitions`: Verification of legal state progressions.
5. `test_order_state_machine_invalid_transitions`: Rejection of illegal transitions.
6. `test_position_lifecycle_long_and_short`: Mark-to-market and closure accounting.
7. `test_position_stop_loss_and_take_profit_triggers`: Automated threshold trigger verification.
8. `test_paper_broker_deterministic_execution`: Balance, equity, margin, and position management.
9. `test_transaction_cost_accounting`: Separate accounting for spread, slippage, commission, swap.
10. `test_execution_service_idempotency`: Prevention of duplicate executions.
11. `test_kill_switch_behavior`: Immediate rejection while active; clean recovery on deactivation.
12. `test_daily_loss_protection`: Enforcement of 1% maximum cumulative daily loss limit.
13. `test_max_open_positions_and_exposure_limits`: Strict enforcement of 3-position and 20% exposure limits.
14. `test_failure_injection_and_atomic_state`: Graceful failure recovery without portfolio corruption.
15. `test_audit_event_logging`: Comprehensive verification of event creation and field capture.
16. `test_broker_adapter_boundary`: Verification that adapter is paper-only with `is_live == False`.
17. `test_api_endpoints_health_readiness_executions_audit`: Full FastAPI endpoint coverage.
18. `test_security_and_credential_sanitization`: Codebase and test environment credential audit.

All 18 new tests and 452+ existing tests pass cleanly (100% pass rate).

---

## 15. Explicit Limitations

- **Simulated Fill Assumption**: Fills in `PaperBroker` execute instantaneously at the requested price (or price adjusted by explicit slippage config). It does not model a limit order book (LOB), queue position dynamics, or partial fill latency.
- **In-Memory Volatility**: The audit trail, order ledger, and position inventories currently reside in memory and reset upon process termination.
- **Single Instrument Testing**: Current test suite focuses on `EURUSD` consistent with the research project specifications.
- **Zero Real Alpha**: In accordance with Phase 32 findings, no predictive alpha exists in the technical or macro feature sets. Execution testing does not indicate market profitability.

---

## 16. Confirmation of Zero Live/Demo Trades

- **No Live Trading**: The system contains zero live broker integration.
- **No Broker Demo Trading**: The system does not connect to MetaTrader 5, FIX, cTrader, or any broker demo environment.
- **No Real Money Execution**: 100% of execution is simulated in-memory within `PaperBroker`.
- **Locked Test Quarantine**: The locked test partition (`2026-02-19 12:00:00 UTC` onwards) remains completely untouched and quarantined.
- **Models Trained**: 0.
- **Predictive Backtests**: 0.
