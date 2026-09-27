"""Backtesting engine module.

Simulates strategy execution against historical market data with realistic market
frictions including spreads, slippage, commission, and latency. Evaluates equity curves,
drawdowns, Sharpe ratios, and profit factors against baselines.
"""

from ai.backtest import (
    ActivePosition,
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    CostModel,
    EquityPoint,
    MLAssistedStrategy,
    PerformanceMetrics,
    SimulatedTrade,
    TradeDirection,
    TradeExitReason,
    calculate_performance_metrics,
)

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
