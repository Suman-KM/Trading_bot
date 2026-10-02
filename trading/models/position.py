"""Position data model for active paper portfolio holdings."""

import math
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    entry_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    status: str = "OPEN"

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
        self.status = "ACTIVE"
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity

    def is_stop_loss_hit(self, price: float) -> bool:
        """Check if market price breaches stop-loss boundary."""
        if self.stop_loss is None:
            return False
        if self.side == OrderSide.BUY:
            return price <= self.stop_loss
        else:
            return price >= self.stop_loss

    def is_take_profit_hit(self, price: float) -> bool:
        """Check if market price breaches take-profit boundary."""
        if self.take_profit is None:
            return False
        if self.side == OrderSide.BUY:
            return price >= self.take_profit
        else:
            return price <= self.take_profit

    def close(
        self,
        exit_price: float,
        exit_timestamp: Optional[datetime] = None,
        costs: float = 0.0,
    ) -> None:
        """Finalize position closure with explicit realized P&L and costs."""
        if not math.isfinite(exit_price) or exit_price <= 0:
            raise ValueError(f"Exit price must be a positive finite number, got {exit_price}")
        self.exit_price = float(exit_price)
        self.exit_timestamp = exit_timestamp or datetime.now(timezone.utc)
        self.current_price = float(exit_price)
        self.costs = float(costs)

        if self.side == OrderSide.BUY:
            self.gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            self.gross_pnl = (self.entry_price - self.exit_price) * self.quantity

        self.net_pnl = self.gross_pnl - self.costs
        self.realized_pnl = self.net_pnl
        self.unrealized_pnl = 0.0
        self.status = "CLOSED"
