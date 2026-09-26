"""Deterministic, non-bypassable Risk Engine."""

import math
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from trading.models.order import Order, OrderSide, OrderStatus, OrderType
from trading.models.portfolio import AccountInfo
from trading.models.signal import Signal, SignalAction
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits, calculate_position_size


class RiskReason(str, Enum):
    """Standardized machine-readable rejection reasons."""

    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"
    SIGNAL_ACTION_HOLD = "SIGNAL_ACTION_HOLD"
    CONFIDENCE_TOO_LOW = "CONFIDENCE_TOO_LOW"
    DAILY_LOSS_LIMIT_EXCEEDED = "DAILY_LOSS_LIMIT_EXCEEDED"
    MAX_OPEN_POSITIONS_REACHED = "MAX_OPEN_POSITIONS_REACHED"
    EXCESSIVE_EXPOSURE = "EXCESSIVE_EXPOSURE"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    POSITION_SIZING_FAILED = "POSITION_SIZING_FAILED"
    MISSING_PRICE_OR_STOP_LOSS = "MISSING_PRICE_OR_STOP_LOSS"
    INSUFFICIENT_EQUITY = "INSUFFICIENT_EQUITY"
    INVALID_SIGNAL = "INVALID_SIGNAL"


class RiskDecision(BaseModel):
    """Result of risk evaluation for a trade signal."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    reason: Optional[str] = None
    order: Optional[Order] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskEngine:
    """Deterministic risk authority independent of AI models."""

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        kill_switch: Optional[KillSwitch] = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.kill_switch = kill_switch or KillSwitch()

    def evaluate(
        self,
        signal: Signal,
        account: AccountInfo,
        daily_start_equity: float,
        current_market_price: Optional[float] = None,
        override_quantity: Optional[float] = None,
    ) -> RiskDecision:
        """Evaluate a signal against deterministic risk boundaries."""
        # 1. Kill switch check
        if self.kill_switch.is_active():
            return RiskDecision(
                approved=False,
                reason=RiskReason.KILL_SWITCH_ACTIVE.value,
                metadata={"kill_switch_reason": self.kill_switch.reason},
            )

        # 2. Action check
        if signal.action == SignalAction.HOLD:
            return RiskDecision(
                approved=False,
                reason=RiskReason.SIGNAL_ACTION_HOLD.value,
                metadata={"action": signal.action.value},
            )

        # 3. Confidence threshold check
        if signal.confidence < self.limits.MIN_SIGNAL_CONFIDENCE:
            return RiskDecision(
                approved=False,
                reason=RiskReason.CONFIDENCE_TOO_LOW.value,
                metadata={
                    "confidence": signal.confidence,
                    "min_required": self.limits.MIN_SIGNAL_CONFIDENCE,
                },
            )

        # 4. Daily loss limit check
        if daily_start_equity > 0:
            loss = ((daily_start_equity - account.equity) / daily_start_equity) * 100.0
            if loss >= self.limits.MAX_DAILY_LOSS_PERCENT:
                self.kill_switch.activate(reason="DAILY_LOSS_LIMIT")
                return RiskDecision(
                    approved=False,
                    reason=RiskReason.DAILY_LOSS_LIMIT_EXCEEDED.value,
                    metadata={
                        "daily_loss_percent": round(loss, 4),
                        "max_allowed": self.limits.MAX_DAILY_LOSS_PERCENT,
                    },
                )

        # 5. Max open positions check
        open_positions = account.positions
        is_new_symbol = signal.symbol not in open_positions
        if is_new_symbol and len(open_positions) >= self.limits.MAX_OPEN_POSITIONS:
            return RiskDecision(
                approved=False,
                reason=RiskReason.MAX_OPEN_POSITIONS_REACHED.value,
                metadata={
                    "open_positions_count": len(open_positions),
                    "max_allowed": self.limits.MAX_OPEN_POSITIONS,
                },
            )

        # 6. Price and stop-loss resolution
        entry_price = current_market_price or signal.suggested_entry_price
        if entry_price is None or entry_price <= 0 or not math.isfinite(entry_price):
            return RiskDecision(
                approved=False,
                reason=RiskReason.MISSING_PRICE_OR_STOP_LOSS.value,
                metadata={"error": "Valid positive entry price is required"},
            )

        stop_loss = signal.suggested_stop_loss
        if stop_loss is None or stop_loss <= 0 or not math.isfinite(stop_loss):
            return RiskDecision(
                approved=False,
                reason=RiskReason.MISSING_PRICE_OR_STOP_LOSS.value,
                metadata={"error": "Valid positive stop loss required for sizing"},
            )

        # 7. Quantity / Position Sizing
        if override_quantity is not None:
            if not math.isfinite(override_quantity) or override_quantity <= 0:
                return RiskDecision(
                    approved=False,
                    reason=RiskReason.INVALID_QUANTITY.value,
                    metadata={"quantity": override_quantity},
                )
            quantity = override_quantity
        else:
            try:
                quantity = calculate_position_size(
                    account_equity=account.equity,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    risk_percent=self.limits.MAX_POSITION_RISK_PERCENT,
                    max_risk_percent=self.limits.MAX_POSITION_RISK_PERCENT,
                )
            except ValueError as err:
                return RiskDecision(
                    approved=False,
                    reason=RiskReason.POSITION_SIZING_FAILED.value,
                    metadata={"error": str(err)},
                )

        if not math.isfinite(quantity) or quantity <= 0:
            return RiskDecision(
                approved=False,
                reason=RiskReason.INVALID_QUANTITY.value,
                metadata={"quantity": quantity},
            )

        # 8. Exposure Limit Check
        new_position_exposure = quantity * entry_price
        current_exposure = sum(pos.quantity * pos.current_price for pos in open_positions.values())
        total_exposure = current_exposure + new_position_exposure
        total_exposure_percent = (
            (total_exposure / account.equity) * 100.0 if account.equity > 0 else 100.0
        )

        if total_exposure_percent > self.limits.MAX_TOTAL_EXPOSURE_PERCENT:
            return RiskDecision(
                approved=False,
                reason=RiskReason.EXCESSIVE_EXPOSURE.value,
                metadata={
                    "total_exposure_percent": round(total_exposure_percent, 4),
                    "max_allowed": self.limits.MAX_TOTAL_EXPOSURE_PERCENT,
                },
            )

        # 9. All checks passed: Generate staged PENDING order
        order = Order(
            symbol=signal.symbol,
            side=OrderSide(signal.action.value),
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=entry_price,
            stop_loss=stop_loss,
            take_profit=signal.suggested_take_profit,
            source="risk_engine",
            status=OrderStatus.PENDING,
        )

        return RiskDecision(
            approved=True,
            order=order,
            metadata={
                "quantity": quantity,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "exposure_percent": round(total_exposure_percent, 4),
            },
        )
