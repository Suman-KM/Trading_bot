"""Position data model for active paper portfolio holdings."""

import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from trading.models.order import OrderSide


class Position(BaseModel):
    """Represents an active or historical position."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty")
        return cleaned

    @field_validator("quantity", "entry_price", "current_price")
    @classmethod
    def validate_positive_floats(cls, v: float, info) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{info.field_name} must be a positive finite number, got {v}")
        return float(v)

    @field_validator("stop_loss", "take_profit")
    @classmethod
    def validate_optional_prices(cls, v: Optional[float], info) -> Optional[float]:
        if v is not None:
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{info.field_name} must be a positive finite number, got {v}")
            return float(v)
        return None

    def update_price(self, new_price: float) -> None:
        """Update current price and recalculate unrealized P&L mark-to-market."""
        if not math.isfinite(new_price) or new_price <= 0:
            raise ValueError(f"Market price must be a positive finite number, got {new_price}")
        self.current_price = float(new_price)
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity
