"""Historical backtesting and strategy simulation package.

Provides deterministic, leakage-safe historical simulation with realistic market frictions,
strict RiskEngine gating, conservative collision handling, and comprehensive reporting.
"""

from ai.backtest.costs import CostModel
from ai.backtest.engine import BacktestEngine, BacktestResult
from ai.backtest.metrics import PerformanceMetrics, calculate_performance_metrics
from ai.backtest.models import (
    ActivePosition,
    BacktestConfig,
    EquityPoint,
    SimulatedTrade,
    TradeDirection,
    TradeExitReason,
)
from ai.backtest.strategy import MLAssistedStrategy

__all__ = [
    "ActivePosition",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "CostModel",
    "EquityPoint",
    "MLAssistedStrategy",
    "PerformanceMetrics",
    "SimulatedTrade",
    "TradeDirection",
    "TradeExitReason",
    "calculate_performance_metrics",
]
