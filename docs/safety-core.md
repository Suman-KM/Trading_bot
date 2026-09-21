# Safety Core Architecture & Specification

## 1. Safety Boundary & Invariant

The core security and safety boundary of this AI Trading System separates intelligence from execution:

```
┌─────────────────────────┐
│       Member 2 AI       │  (Signal generation ONLY — Zero execution authority)
└────────────┬────────────┘
             │ Signal
             ▼
┌─────────────────────────┐
│    Signal Validation    │  (Strict Pydantic perimeter schema defense)
└────────────┬────────────┘
             │ Validated Signal
             ▼
┌─────────────────────────┐
│   Sovereign Risk Engine │  (Deterministic authorization authority)
└────────────┬────────────┘
             │ Approved Order ONLY
             ▼
┌─────────────────────────┐
│   Paper Broker Sim      │  (100% in-memory simulated execution)
└─────────────────────────┘
```

### Core Invariant
> **The AI generates trading signals ONLY. The Risk Engine is the final, non-bypassable authority before execution. The Paper Broker must NEVER be invoked if the Risk Engine rejects the signal or if the Kill Switch is active.**

---

## 2. Signal Validation
All incoming signals from Member 2 must validate against the `Signal` Pydantic model (`trading/models/signal.py`):

- **Model Constraints**:
  - `extra = "forbid"`: Prevents parameter injection or unrecognized fields.
  - `symbol`: Non-empty string, automatically sanitized and converted to uppercase.
  - `action`: Strict enum (`BUY`, `SELL`, `HOLD`).
  - `confidence`: Finite float in the closed interval `[0.0, 1.0]`.
  - `expected_return`: Finite float (rejects `NaN`, `+Inf`, `-Inf`).
  - `timestamp`: Valid ISO datetime.
  - `model_version`, `timeframe`, `feature_version`: Non-empty strings.
  - `suggested_entry_price`, `suggested_stop_loss`, `suggested_take_profit`: Optional positive finite floats.

---

## 3. Sovereign Risk Engine & Rules
The `RiskEngine` (`trading/risk/engine.py`) operates independently of the AI model. It enforces deterministic checks against `RiskLimits`:

| Parameter | Default Limit | Description |
|---|---|---|
| `MAX_DAILY_LOSS_PERCENT` | `1.0%` | Maximum allowable daily account drawdown before halting trading. |
| `MAX_POSITION_RISK_PERCENT` | `0.5%` | Maximum capital at risk per single trade. |
| `MAX_OPEN_POSITIONS` | `3` | Maximum number of concurrent open positions. |
| `MAX_TOTAL_EXPOSURE_PERCENT` | `20.0%` | Maximum gross portfolio market exposure relative to equity. |
| `MIN_SIGNAL_CONFIDENCE` | `0.60` | Minimum AI confidence score required for approval. |

### Evaluation Pipeline
1. **Kill Switch Verification**: If active, immediate rejection (`KILL_SWITCH_ACTIVE`).
2. **Action Check**: If `action == HOLD`, immediate rejection (`SIGNAL_ACTION_HOLD`).
3. **Confidence Check**: If `confidence < MIN_SIGNAL_CONFIDENCE`, rejected (`CONFIDENCE_TOO_LOW`).
4. **Daily Drawdown Check**: If `(daily_start_equity - current_equity) / daily_start_equity >= MAX_DAILY_LOSS_PERCENT`, activates kill switch and rejects (`DAILY_LOSS_LIMIT_EXCEEDED`).
5. **Position Count Check**: If new symbol and `len(open_positions) >= MAX_OPEN_POSITIONS`, rejected (`MAX_OPEN_POSITIONS_REACHED`).
6. **Price & Stop-Loss Check**: Requires valid positive entry and stop-loss (`MISSING_PRICE_OR_STOP_LOSS`).
7. **Position Sizing Calculation**: Computes quantity. If invalid or calculation fails, rejected (`POSITION_SIZING_FAILED` / `INVALID_QUANTITY`).
8. **Exposure Limit Check**: If `total_exposure > MAX_TOTAL_EXPOSURE_PERCENT`, rejected (`EXCESSIVE_EXPOSURE`).
9. **Order Staging**: Emits approved `Order` with `status = PENDING`.

---

## 4. Deterministic Position Sizing
Position sizing (`trading/risk/limits.py:calculate_position_size`) determines exact quantity based strictly on capital at risk:

$$\text{risk\_capital} = \text{account\_equity} \times \left(\frac{\text{risk\_percent}}{100}\right)$$
$$\text{stop\_distance} = |\text{entry\_price} - \text{stop\_loss\_price}|$$
$$\text{quantity} = \frac{\text{risk\_capital}}{\text{stop\_distance}}$$

### Safeguards:
- Rejects zero or negative equity (`account_equity <= 0`).
- Rejects zero stop distance (`|entry_price - stop_loss_price| <= 0`).
- Rejects risk percentages exceeding `MAX_POSITION_RISK_PERCENT` (0.5%).
- Zero leverage is assumed.

---

## 5. Emergency Kill Switch
The `KillSwitch` (`trading/risk/kill_switch.py`) provides an auditable emergency halt:
- **Methods**: `activate(reason: str)`, `deactivate()`, `is_active() -> bool`.
- **Automatic Triggers**: Trips automatically if the daily loss limit (1.0%) is breached.
- **Manual Triggers**: Can be activated by operators or automated safety monitors.
- **Audit Logging**: Retains an immutable timestamped event log of activations and deactivations.
- **Behavior**: When active, all incoming signals are rejected with `KILL_SWITCH_ACTIVE`.

---

## 6. In-Memory Paper Broker Simulator
The `PaperBroker` (`trading/execution/paper_broker.py`) simulates execution without external connectivity:
- **Zero Network Sockets**: Does not connect to MT5, Binance, or any network services.
- **Account Ledger**: Tracks `cash_balance`, `realized_pnl`, `unrealized_pnl`, and `equity`.
- **Order Handling**: Validates orders, checks cash requirements, and transitions states (`PENDING` -> `FILLED` or `REJECTED`).
- **Positions**: Tracks active holdings, computes mark-to-market unrealized P&L on price updates, and realizes P&L upon closing.

---

## 7. Standard Machine-Readable Rejection Reasons
Every rejection provides a structured reason code:
- `KILL_SWITCH_ACTIVE`: Emergency halt engaged.
- `SIGNAL_ACTION_HOLD`: Signal action is HOLD.
- `CONFIDENCE_TOO_LOW`: AI confidence score is below threshold.
- `DAILY_LOSS_LIMIT_EXCEEDED`: Daily drawdown limit breached.
- `MAX_OPEN_POSITIONS_REACHED`: Maximum simultaneous positions reached.
- `EXCESSIVE_EXPOSURE`: Order would exceed portfolio exposure limit.
- `INVALID_QUANTITY`: Calculated or specified quantity is non-positive.
- `POSITION_SIZING_FAILED`: Position sizing calculation failed.
- `MISSING_PRICE_OR_STOP_LOSS`: Missing entry or stop-loss price.
- `INVALID_SIGNAL`: Malformed signal schema.
- `INSUFFICIENT_CASH`: Paper broker cash balance insufficient.
- `INVALID_FILL_PRICE`: Market fill price missing or non-positive.

---

## 8. Structured Logging & Audit Trail
Structured JSON logs (`trading/logging_config.py`) record every trade decision:
```json
{
  "timestamp": "2026-09-21T01:10:00.000000Z",
  "level": "INFO",
  "logger": "trading.safety",
  "message": "Risk evaluation: APPROVED",
  "symbol": "EURUSD",
  "action": "BUY",
  "risk_decision": "APPROVED",
  "reason": "NONE",
  "order_id": "c1f7a149-a292-4f3b-b0be-a99f7d43b7e4",
  "confidence": 0.90
}
```
**Safety Mandate**: Credentials, keys, passwords, and tokens are never logged.
