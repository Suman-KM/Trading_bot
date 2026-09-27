"""Visualizations and plotting routines for Phase 12 backtesting results.

Generates chronological equity curves, drawdown trajectories, and trade PnL distributions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from ai.backtest.engine import BacktestResult

matplotlib.use("Agg")  # Headless non-interactive backend


def plot_equity_curve(
    result: BacktestResult,
    output_path: Path | str,
    title: str = "EURUSD M15 — Portfolio Equity Curve",
) -> Path:
    """Generate and save chronological equity curve."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df_eq = result.to_equity_dataframe()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    if not df_eq.empty:
        ax.plot(
            df_eq["timestamp"],
            df_eq["equity"],
            label="Portfolio Equity ($)",
            color="#1f77b4",
            linewidth=1.5,
        )
        ax.axhline(
            y=result.config.initial_equity,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Initial Equity (${result.config.initial_equity:,.0f})",
        )
        ax.set_ylabel("Equity (USD)", fontsize=11)
        ax.set_xlabel("Date (UTC)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "No Equity Data Available", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_drawdown_curve(
    result: BacktestResult,
    output_path: Path | str,
    title: str = "EURUSD M15 — Portfolio Drawdown Trajectory",
) -> Path:
    """Generate and save portfolio underwater drawdown curve."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df_eq = result.to_equity_dataframe()

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    if not df_eq.empty:
        ax.fill_between(
            df_eq["timestamp"],
            -df_eq["drawdown_pct"],
            0,
            color="#d62728",
            alpha=0.3,
            label="Drawdown (%)",
        )
        ax.plot(
            df_eq["timestamp"],
            -df_eq["drawdown_pct"],
            color="#d62728",
            linewidth=1.0,
        )
        max_dd = df_eq["drawdown_pct"].max()
        ax.set_ylabel("Drawdown (%)", fontsize=11)
        ax.set_xlabel("Date (UTC)", fontsize=11)
        ax.set_title(
            f"{title} (Max Drawdown: {max_dd:.2f}%)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left")
    else:
        ax.text(0.5, 0.5, "No Drawdown Data Available", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_trade_pnl_distribution(
    result: BacktestResult,
    output_path: Path | str,
    title: str = "EURUSD M15 — Trade Net P&L Distribution",
) -> Path:
    """Generate and save trade net PnL histogram and breakdown."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    trades = result.trade_ledger
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    if trades:
        pnls = [t.net_pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        bins = min(max(len(trades) // 2, 10), 40)
        ax.hist(
            pnls,
            bins=bins,
            color="#2ca02c",
            alpha=0.7,
            edgecolor="black",
            label=f"Net PnL (Wins: {len(wins)}, Losses: {len(losses)})",
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Trade Net PnL (USD)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")
    else:
        ax.text(
            0.5,
            0.5,
            "Zero Trades Executed (All Filtered by Confidence)",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
