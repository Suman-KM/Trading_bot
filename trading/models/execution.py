"""Structured execution report model for paper trading auditability."""

import math
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading.models.order import OrderSide, OrderStatus


class ExecutionReport(BaseModel):
    """Structured and auditable record of order execution or rejection."""

    model_config = ConfigDict(extra="forbid")

    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str
    client_request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str
    side: OrderSide
    requested_price: float
    executed_price: Optional[float] = None
    quantity: float
    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    execution_status: OrderStatus
    rejection_reason: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty")
        return cleaned

    @field_validator("quantity", "requested_price")
    @classmethod
    def validate_finite_numbers(cls, v: float, info) -> float:
        if not math.isfinite(v):
            raise ValueError(f"{info.field_name} must be a finite number, got {v}")
        return float(v)

    @field_validator("executed_price")
    @classmethod
    def validate_optional_executed_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (not math.isfinite(v) or v <= 0):
            raise ValueError(f"executed_price must be a positive finite number, got {v}")
        return float(v) if v is not None else None
