"""Portfolio and Account data models."""

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field

from trading.models.position import Position


class AccountInfo(BaseModel):
    """Snapshot of account capital and portfolio state."""

    model_config = ConfigDict(extra="forbid")

    initial_balance: float
    cash_balance: float
    margin_used: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    equity: float
    positions: Dict[str, Position] = Field(default_factory=dict)
