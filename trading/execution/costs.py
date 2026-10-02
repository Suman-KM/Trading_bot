"""Deterministic transaction cost model for paper trading execution."""

import math
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CostBreakdown(NamedTuple):
    """Detailed breakdown of transaction costs for an execution."""

    spread_cost: float
    slippage_cost: float
    commission: float
    swap: float
    total_cost: float


class TransactionCostConfig(BaseModel):
    """Configurable transaction cost parameters for paper trading.

    All parameters default to 0.0 unless explicitly configured.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spread_points: float = Field(
        default=0.0,
        description="Fixed spread cost in price units (points/pips).",
    )
    slippage_points: float = Field(
        default=0.0,
        description="Execution slippage in price units.",
    )
    commission_per_unit: float = Field(
        default=0.0,
        description="Fixed commission per traded unit/share/contract.",
    )
    swap_per_day: float = Field(
        default=0.0,
        description="Overnight financing / swap cost per unit per day.",
    )

    @field_validator(
        "spread_points",
        "slippage_points",
        "commission_per_unit",
        "swap_per_day",
    )
    @classmethod
    def validate_non_negative_costs(cls, v: float, info) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{info.field_name} must be a non-negative finite number, got {v}")
        return float(v)

    def calculate_entry_costs(self, quantity: float, price: float) -> CostBreakdown:
        """Calculate costs incurred on order entry."""
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError(f"Quantity must be a positive finite number, got {quantity}")
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Price must be a positive finite number, got {price}")

        spread = round(self.spread_points * quantity, 4)
        slippage = round(self.slippage_points * quantity, 4)
        commission = round(self.commission_per_unit * quantity, 4)
        swap = 0.0
        total = round(spread + slippage + commission + swap, 4)

        return CostBreakdown(
            spread_cost=spread,
            slippage_cost=slippage,
            commission=commission,
            swap=swap,
            total_cost=total,
        )

    def calculate_exit_costs(
        self, quantity: float, price: float, holding_days: float = 0.0
    ) -> CostBreakdown:
        """Calculate costs incurred on order exit (spread, slippage, commission, swap)."""
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError(f"Quantity must be a positive finite number, got {quantity}")
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Price must be a positive finite number, got {price}")

        spread = round(self.spread_points * quantity, 4)
        slippage = round(self.slippage_points * quantity, 4)
        commission = round(self.commission_per_unit * quantity, 4)
        swap = round(self.swap_per_day * quantity * max(0.0, holding_days), 4)
        total = round(spread + slippage + commission + swap, 4)

        return CostBreakdown(
            spread_cost=spread,
            slippage_cost=slippage,
            commission=commission,
            swap=swap,
            total_cost=total,
        )
