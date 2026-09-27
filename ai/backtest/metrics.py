"""Comprehensive performance and risk metric calculation for backtest evaluation.

Calculates trade statistics, return distributions, drawdowns, transaction frictions,
holding periods, and execution rejection diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai.backtest.engine import BacktestResult
from ai.backtest.models import TradeDirection


@dataclass(frozen=True)
class PerformanceMetrics:
    """Comprehensive performance and execution summary."""

    # Returns
    starting_equity: float
    ending_equity: float
    total_net_pnl: float
    total_return_pct: float
    average_trade_return: float

    # Trade counts & Win rates
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    loss_rate: float

    # Trade PnL statistics
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    win_loss_ratio: float
    profit_factor: float
    expectancy_per_trade: float

    # Trade Quality / Direction Breakdown
    long_trades: int
    short_trades: int
    long_winning_trades: int
    short_winning_trades: int
    long_win_rate: float
    short_win_rate: float
    average_holding_bars: float
    median_holding_bars: float

    # Risk & Drawdown
    max_drawdown: float
    max_drawdown_pct: float
    daily_loss_events_count: int
    largest_exposure: float
    average_exposure: float

    # Costs
    total_spread_cost: float
    total_commission: float
    total_slippage: float
    total_transaction_costs: float

    # Execution & Safety
    total_bars: int
    total_signals_evaluated: int
    signals_generated: int
    confidence_filtered_signals: int
    trades_approved: int
    trades_executed: int
    total_rejections: int
    rejections_by_reason: dict[str, int]

    # Annualized Metrics (clearly documented)
    annualized_sharpe_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return asdict(self)


def calculate_performance_metrics(result: BacktestResult) -> PerformanceMetrics:
    """Compute all required Phase 12 performance, risk, and execution metrics.

    Parameters
    ----------
    result : BacktestResult
        Result container returned by BacktestEngine.run().

    Returns
    -------
    PerformanceMetrics
        Structured immutable dataclass containing all evaluated metrics.
    """
    trades = result.trade_ledger
    equity_curve = result.equity_curve
    starting_equity = float(result.config.initial_equity)
    ending_equity = equity_curve[-1].equity if equity_curve else starting_equity

    total_net_pnl = ending_equity - starting_equity
    total_return_pct = (total_net_pnl / starting_equity) * 100.0 if starting_equity > 0 else 0.0

    total_trades = len(trades)
    if total_trades == 0:
        return PerformanceMetrics(
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            total_net_pnl=total_net_pnl,
            total_return_pct=total_return_pct,
            average_trade_return=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            loss_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            win_loss_ratio=0.0,
            profit_factor=0.0,
            expectancy_per_trade=0.0,
            long_trades=0,
            short_trades=0,
            long_winning_trades=0,
            short_winning_trades=0,
            long_win_rate=0.0,
            short_win_rate=0.0,
            average_holding_bars=0.0,
            median_holding_bars=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            daily_loss_events_count=len(result.daily_loss_events),
            largest_exposure=0.0,
            average_exposure=0.0,
            total_spread_cost=0.0,
            total_commission=0.0,
            total_slippage=0.0,
            total_transaction_costs=0.0,
            total_bars=result.total_bars,
            total_signals_evaluated=result.total_signals_evaluated,
            signals_generated=result.signals_generated,
            confidence_filtered_signals=result.confidence_filtered,
            trades_approved=result.trades_approved,
            trades_executed=0,
            total_rejections=sum(result.rejections.values()),
            rejections_by_reason=result.rejections,
            annualized_sharpe_ratio=None,
        )

    # Net PnL series
    net_pnls = np.array([t.net_pnl for t in trades], dtype=np.float64)
    holding_bars_arr = np.array([t.holding_bars for t in trades], dtype=np.float64)

    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls < 0]
    breakevens = net_pnls[net_pnls == 0]

    winning_trades = len(wins)
    losing_trades = len(losses)
    breakeven_trades = len(breakevens)

    win_rate = (winning_trades / total_trades) * 100.0
    loss_rate = (losing_trades / total_trades) * 100.0

    average_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    average_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    largest_win = float(np.max(wins)) if len(wins) > 0 else 0.0
    largest_loss = float(np.min(losses)) if len(losses) > 0 else 0.0

    total_gross_gain = float(np.sum(wins)) if len(wins) > 0 else 0.0
    total_gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0

    if total_gross_loss > 1e-9:
        profit_factor = total_gross_gain / total_gross_loss
    elif total_gross_gain > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    win_loss_ratio = abs(average_win / average_loss) if abs(average_loss) > 1e-9 else 0.0

    # Expectancy = (Win_Rate * Avg_Win) - (Loss_Rate * |Avg_Loss|)
    expectancy = (win_rate / 100.0 * average_win) + (loss_rate / 100.0 * average_loss)
    average_trade_return = float(np.mean(net_pnls))

    # Direction Quality
    long_trades_list = [t for t in trades if t.direction == TradeDirection.LONG]
    short_trades_list = [t for t in trades if t.direction == TradeDirection.SHORT]

    long_trades = len(long_trades_list)
    short_trades = len(short_trades_list)

    long_winning_trades = sum(1 for t in long_trades_list if t.net_pnl > 0)
    short_winning_trades = sum(1 for t in short_trades_list if t.net_pnl > 0)

    long_win_rate = (long_winning_trades / long_trades * 100.0) if long_trades > 0 else 0.0
    short_win_rate = (short_winning_trades / short_trades * 100.0) if short_trades > 0 else 0.0

    average_holding_bars = float(np.mean(holding_bars_arr))
    median_holding_bars = float(np.median(holding_bars_arr))

    # Drawdown & Exposure from equity curve
    max_drawdown = max((ep.drawdown for ep in equity_curve), default=0.0)
    max_drawdown_pct = max((ep.drawdown_pct for ep in equity_curve), default=0.0)
    largest_exposure = max((ep.exposure for ep in equity_curve), default=0.0)
    average_exposure = float(np.mean([ep.exposure for ep in equity_curve])) if equity_curve else 0.0

    # Costs
    total_spread_cost = sum(t.spread_cost for t in trades)
    total_commission = sum(t.commission for t in trades)
    total_slippage = sum(t.slippage for t in trades)
    total_transaction_costs = total_spread_cost + total_commission + total_slippage

    # Annualized Sharpe ratio on M15 equity returns (approx 252 days * 96 bars/day = 24,192 bars/yr)
    if len(equity_curve) > 10:
        eq_series = pd.Series([ep.equity for ep in equity_curve])
        bar_returns = eq_series.pct_change().dropna()
        mean_ret = bar_returns.mean()
        std_ret = bar_returns.std()
        if std_ret > 1e-9:
            # 24,192 M15 bars per year
            sharpe = (mean_ret / std_ret) * math.sqrt(24192)
        else:
            sharpe = 0.0
    else:
        sharpe = None

    return PerformanceMetrics(
        starting_equity=round(starting_equity, 2),
        ending_equity=round(ending_equity, 2),
        total_net_pnl=round(total_net_pnl, 2),
        total_return_pct=round(total_return_pct, 4),
        average_trade_return=round(average_trade_return, 2),
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        win_rate=round(win_rate, 2),
        loss_rate=round(loss_rate, 2),
        average_win=round(average_win, 2),
        average_loss=round(average_loss, 2),
        largest_win=round(largest_win, 2),
        largest_loss=round(largest_loss, 2),
        win_loss_ratio=round(win_loss_ratio, 4),
        profit_factor=round(profit_factor, 4),
        expectancy_per_trade=round(expectancy, 2),
        long_trades=long_trades,
        short_trades=short_trades,
        long_winning_trades=long_winning_trades,
        short_winning_trades=short_winning_trades,
        long_win_rate=round(long_win_rate, 2),
        short_win_rate=round(short_win_rate, 2),
        average_holding_bars=round(average_holding_bars, 2),
        median_holding_bars=round(median_holding_bars, 2),
        max_drawdown=round(max_drawdown, 2),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        daily_loss_events_count=len(result.daily_loss_events),
        largest_exposure=round(largest_exposure, 2),
        average_exposure=round(average_exposure, 2),
        total_spread_cost=round(total_spread_cost, 2),
        total_commission=round(total_commission, 2),
        total_slippage=round(total_slippage, 2),
        total_transaction_costs=round(total_transaction_costs, 2),
        total_bars=result.total_bars,
        total_signals_evaluated=result.total_signals_evaluated,
        signals_generated=result.signals_generated,
        confidence_filtered_signals=result.confidence_filtered,
        trades_approved=result.trades_approved,
        trades_executed=total_trades,
        total_rejections=sum(result.rejections.values()),
        rejections_by_reason=result.rejections,
        annualized_sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
    )
