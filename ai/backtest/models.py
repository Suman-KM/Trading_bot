"""Data models and configuration for historical backtesting simulation.

Defines schemas for backtest configuration, trade records, active positions,
and equity tracking.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trading.risk.limits import RiskLimits


class TradeDirection(str, Enum):
    """Direction of a simulated trade."""

    LONG = "LONG"
    SHORT = "SHORT"


class TradeExitReason(str, Enum):
    """Standardized deterministic exit reasons."""

    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    MAX_HOLD = "MAX_HOLD"
    DAILY_RISK_POLICY = "DAILY_RISK_POLICY"
    KILL_SWITCH = "KILL_SWITCH"
    END_OF_DATA = "END_OF_DATA"


class BacktestConfig(BaseModel):
    """Deterministic configuration specification for backtest execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_equity: float = Field(
        default=100_000.0, description="Initial starting portfolio equity in USD."
    )
    confidence_threshold: float = Field(
        default=0.60, description="Minimum ML prediction confidence required to emit entry signal."
    )
    atr_period: int = Field(default=14, description="Lookback window for ATR calculation.")
    stop_loss_atr_multiple: float = Field(
        default=1.0, description="ATR multiple for stop loss distance."
    )
    take_profit_atr_multiple: float = Field(
        default=1.5, description="ATR multiple for take profit distance."
    )
    max_holding_bars: int = Field(
        default=4, description="Maximum holding duration in bars (H=4 M15 bars = 60 minutes)."
    )
    cooldown_bars: int = Field(
        default=1, description="Bars to wait after a position closes before next entry."
    )
    point_value: float = Field(
        default=1e-5, description="Price unit per point for 5-digit quote (EURUSD: 0.00001)."
    )
    default_spread_points: float = Field(
        default=10.0,
        description="Default spread in points (10 pts = 1 pip) when historical spread is 0.",
    )
    use_historical_spread: bool = Field(
        default=True,
        description="Whether to use historical spread from market data when available.",
    )
    min_spread_points: float = Field(default=0.0, description="Minimum floor on spread points.")
    slippage_points: float = Field(
        default=0.0, description="Deterministic fixed slippage in points applied per fill."
    )
    commission_per_lot: float = Field(
        default=0.0,
        description="Round-turn broker commission in USD per standard lot (100,000 units).",
    )
    lot_size: float = Field(default=100_000.0, description="Base currency units per standard lot.")
    same_bar_sl_priority: bool = Field(
        default=True,
        description="Conservative collision policy: if both touched in bar, assume SL first.",
    )
    constrain_exposure: bool = Field(
        default=True,
        description="Cap position size to satisfy MAX_TOTAL_EXPOSURE_PERCENT limit.",
    )
    max_open_positions: int = Field(
        default=1, description="Maximum simultaneous open positions allowed (1 for EURUSD)."
    )
    risk_limits: RiskLimits = Field(
        default_factory=RiskLimits, description="Deterministic Safety Core RiskLimits instance."
    )


class SimulatedTrade(BaseModel):
    """Immutable record of an executed and closed trade in the backtest."""

    model_config = ConfigDict(extra="forbid")

    trade_id: int
    symbol: str = "EURUSD"
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    direction: TradeDirection
    confidence: float
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    risk_amount: float
    gross_pnl: float
    spread_cost: float
    commission: float
    slippage: float
    net_pnl: float
    holding_bars: int
    exit_reason: TradeExitReason
    risk_decision: str = "APPROVED"
    model_version: str = "RandomForest_balanced_H4"

    def to_dict(self) -> dict[str, Any]:
        """Convert trade record to dictionary for export."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "signal_time": self.signal_time.isoformat(),
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "direction": self.direction.value,
            "confidence": round(self.confidence, 4),
            "entry_price": round(self.entry_price, 5),
            "exit_price": round(self.exit_price, 5),
            "stop_loss": round(self.stop_loss, 5),
            "take_profit": round(self.take_profit, 5),
            "quantity": round(self.quantity, 2),
            "risk_amount": round(self.risk_amount, 2),
            "gross_pnl": round(self.gross_pnl, 2),
            "spread_cost": round(self.spread_cost, 2),
            "commission": round(self.commission, 2),
            "slippage": round(self.slippage, 2),
            "net_pnl": round(self.net_pnl, 2),
            "holding_bars": self.holding_bars,
            "exit_reason": self.exit_reason.value,
            "risk_decision": self.risk_decision,
            "model_version": self.model_version,
        }


class EquityPoint(BaseModel):
    """Point-in-time portfolio mark-to-market state."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    equity: float
    cash: float
    drawdown: float
    drawdown_pct: float
    open_positions_count: int
    exposure: float


class ActivePosition(BaseModel):
    """Internal state tracking for an currently open position."""

    model_config = ConfigDict(extra="forbid")

    trade_id: int
    symbol: str = "EURUSD"
    direction: TradeDirection
    quantity: float
    entry_price: float
    entry_time: datetime
    signal_time: datetime
    stop_loss: float
    take_profit: float
    confidence: float
    stop_distance: float
    risk_amount: float
    spread_cost_entry: float
    commission_entry: float
    slippage_entry: float
    holding_bars: int = 0
