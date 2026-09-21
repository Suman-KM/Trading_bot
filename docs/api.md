# FastAPI Paper Trading API Specification

## 1. Overview & Security Mandate

The Paper Trading API exposes the deterministic Safety Core to local clients (e.g. Member 2's AI signal generator) over HTTP.

### Mandatory Operational Boundaries:
- **DEMO / PAPER TRADING ONLY**: Real capital, broker API keys, and live market connections are completely absent.
- **Localhost Binding**: The application is strictly bound to `127.0.0.1`. It must NEVER be bound to `0.0.0.0` or exposed to the LAN/Internet without authorization and TLS.
- **Non-Bypassable Execution Invariant**:
  ```
  HTTP Request (POST /signals)
        ↓
  Signal Schema Validation (FastAPI + Pydantic)
        ↓
  TradingExecutionService
        ↓
  RiskEngine Evaluation (Sovereign Authority)
        ↓ (Approved Only)
  Position Sizing
        ↓
  Paper Broker (In-Memory Simulator)
        ↓
  Simulated Portfolio
  ```
  **If the Risk Engine rejects a signal or the Kill Switch is active, `PaperBroker.submit_order` is NEVER invoked.**

---

## 2. API Endpoints

### `GET /health`
Returns the operational health status and explicit confirmation of DEMO mode.

**Response (`200 OK`)**:
```json
{
  "status": "ok",
  "environment": "DEMO",
  "trading_backend": "PAPER"
}
```

---

### `GET /account`
Returns the simulated paper account equity, cash balance, and profit/loss metrics.

**Response (`200 OK`)**:
```json
{
  "balance": 100000.0,
  "equity": 100000.0,
  "realized_pnl": 0.0,
  "unrealized_pnl": 0.0,
  "available_cash": 100000.0
}
```

---

### `GET /positions`
Returns all currently open simulated paper positions.

**Response (`200 OK`)**:
```json
[
  {
    "symbol": "BNBUSDT",
    "side": "BUY",
    "quantity": 100.0,
    "entry_price": 100.0,
    "current_price": 100.0,
    "unrealized_pnl": 0.0,
    "realized_pnl": 0.0,
    "stop_loss": 95.0,
    "take_profit": null
  }
]
```

---

### `POST /signals`
Receives an AI trading signal, validates its schema, submits it to the Risk Engine, and executes via the Paper Broker if approved.

**Request Body**:
```json
{
  "symbol": "BNBUSDT",
  "action": "BUY",
  "confidence": 0.85,
  "timestamp": "2026-09-21T01:25:00Z",
  "model_version": "xgb_v1",
  "timeframe": "15m",
  "expected_return": 0.015,
  "feature_version": "features_v2",
  "suggested_entry_price": 100.0,
  "suggested_stop_loss": 95.0,
  "suggested_take_profit": 110.0
}
```

**Approved Response (`200 OK`)**:
```json
{
  "approved": true,
  "reason": null,
  "order": {
    "order_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
    "symbol": "BNBUSDT",
    "side": "BUY",
    "quantity": 100.0,
    "order_type": "MARKET",
    "price": 100.0,
    "stop_loss": 95.0,
    "take_profit": 110.0,
    "status": "FILLED",
    "fill_price": 100.0,
    "fill_timestamp": "2026-09-21T01:25:01Z"
  },
  "signal_symbol": "BNBUSDT",
  "metadata": {
    "quantity": 100.0,
    "entry_price": 100.0,
    "stop_loss": 95.0,
    "exposure_percent": 10.0
  }
}
```

**Rejected Response (`409 Conflict`)**:
```json
{
  "approved": false,
  "reason": "CONFIDENCE_TOO_LOW",
  "order": null,
  "signal_symbol": "BNBUSDT",
  "metadata": {
    "confidence": 0.45,
    "min_required": 0.60
  }
}
```

---

### `GET /orders`
Returns the audit history of all simulated orders placed during the session.

**Response (`200 OK`)**:
```json
[
  {
    "order_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
    "symbol": "BNBUSDT",
    "side": "BUY",
    "quantity": 100.0,
    "order_type": "MARKET",
    "price": 100.0,
    "status": "FILLED",
    "fill_price": 100.0,
    "fill_timestamp": "2026-09-21T01:25:01Z"
  }
]
```

---

### `GET /risk/status`
Returns the active safety state, configured risk limits, gross market exposure, and kill switch status.

**Response (`200 OK`)**:
```json
{
  "kill_switch_active": false,
  "kill_switch_reason": null,
  "daily_loss_percent": 0.0,
  "open_positions": 0,
  "gross_exposure_percent": 0.0,
  "configured_limits": {
    "MAX_DAILY_LOSS_PERCENT": 1.0,
    "MAX_POSITION_RISK_PERCENT": 0.5,
    "MAX_OPEN_POSITIONS": 3,
    "MAX_TOTAL_EXPOSURE_PERCENT": 20.0,
    "MIN_SIGNAL_CONFIDENCE": 0.60
  },
  "trading_mode": "DEMO/PAPER"
}
```

---

### `POST /risk/kill-switch/activate`
Engages the emergency kill switch, rejecting all subsequent orders.

**Request Body**:
```json
{
  "reason": "MANUAL_OPERATOR_HALT"
}
```

**Response (`200 OK`)**:
```json
{
  "status": "activated",
  "is_active": true,
  "reason": "MANUAL_OPERATOR_HALT"
}
```

---

### `POST /risk/kill-switch/deactivate`
Deactivates the emergency kill switch, restoring order execution.

**Request Body**: Empty or `{}`.

**Response (`200 OK`)**:
```json
{
  "status": "deactivated",
  "is_active": false,
  "reason": null
}
```

---

## 3. Interactive Documentation
When running locally on `127.0.0.1:8000`:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **OpenAPI JSON**: `http://127.0.0.1:8000/openapi.json`
