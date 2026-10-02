"""Order data model for internal trade execution."""

import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrderSide(str, Enum):
    """Order side: BUY or SELL."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "MARKET"


class OrderStatus(str, Enum):
    """Internal lifecycle states for orders."""

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    CLOSED = "CLOSED"


# Deterministic state transition mapping
ALLOWED_ORDER_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.CREATED: {
        OrderStatus.VALIDATED,
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.VALIDATED: {
        OrderStatus.SUBMITTED,
        OrderStatus.PENDING,
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.PENDING: {
        OrderStatus.VALIDATED,
        OrderStatus.SUBMITTED,
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.SUBMITTED: {
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.FILLED: {
        OrderStatus.CLOSED,
    },
    OrderStatus.REJECTED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.CLOSED: set(),
}


class Order(BaseModel):
    """Internal order representation."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_request_id: Optional[str] = None
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "risk_engine"
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    def transition_to(self, new_status: OrderStatus, reason: Optional[str] = None) -> None:
        """Explicitly validate and execute deterministic state transitions."""
        if new_status == self.status:
            return
        allowed = ALLOWED_ORDER_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid order transition: cannot transition from "
                f"{self.status.value} to {new_status.value}"
            )
        self.status = new_status
        if reason:
            self.rejection_reason = reason

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty")
        return cleaned

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"Quantity must be a positive finite number, got {v}")
        return float(v)

    @field_validator("price", "stop_loss", "take_profit", "fill_price")
    @classmethod
    def validate_prices(cls, v: Optional[float], info) -> Optional[float]:
        if v is not None:
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{info.field_name} must be a positive finite number, got {v}")
            return float(v)
        return None
