"""Exploratory Data Analysis (EDA) module for quantitative market datasets.

Computes comprehensive descriptive price statistics, return dynamics, volatility profiles,
temporal/session distributions, volume/spread mechanics, and timestamp gap characteristics.
Preserves UTC timestamps without look-ahead bias and never alters raw market data.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ai.data.gaps import analyze_gaps


def compute_price_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute exhaustive descriptive price action and return statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'open', 'high', 'low', 'close', and either 'timestamp' or 'time'.

    Returns
    -------
    dict[str, Any]
        Dictionary of descriptive statistics.
    """
    close = df["close"]
    open_p = df["open"]
    high = df["high"]
    low = df["low"]

    # Range and Candle Geometry
    hl_range = high - low
    hl_range_pct = (hl_range / close) * 100.0
    body = (close - open_p).abs()
    body_pct = (body / close) * 100.0
    upper_wick = high - np.maximum(open_p, close)
    lower_wick = np.minimum(open_p, close) - low

    bullish_count = int((close > open_p).sum())
    bearish_count = int((close < open_p).sum())
    doji_count = int((close == open_p).sum())

    # Returns
    simple_returns = close.pct_change().dropna()
    log_returns = np.log(close / close.shift(1)).dropna()

    # Volatility metrics
    bar_std = float(log_returns.std())
    # Nominal annualization factor for M15: 252 days * 24 hours * 4 bars = 24,192 bars
    annualized_vol = float(bar_std * math.sqrt(24192))

    # Average True Range (14-period)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = float(true_range.rolling(14).mean().dropna().mean())

    def _quantiles(s: pd.Series) -> dict[str, float]:
        q = s.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()
        return {f"p{int(k * 100):02d}": float(v) for k, v in q.items()}

    def _basic_stats(s: pd.Series) -> dict[str, Any]:
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "quantiles": _quantiles(s),
        }

    return {
        "count": int(len(df)),
        "open": _basic_stats(open_p),
        "high": _basic_stats(high),
        "low": _basic_stats(low),
        "close": _basic_stats(close),
        "range_points": _basic_stats(hl_range),
        "range_pct": {
            "min": float(hl_range_pct.min()),
            "max": float(hl_range_pct.max()),
            "mean": float(hl_range_pct.mean()),
            "median": float(hl_range_pct.median()),
            "std": float(hl_range_pct.std()),
            "quantiles": _quantiles(hl_range_pct),
        },
        "body_points": {
            "mean": float(body.mean()),
            "median": float(body.median()),
            "std": float(body.std()),
            "max": float(body.max()),
            "quantiles": _quantiles(body),
            "pct_quantiles": _quantiles(body_pct),
        },
        "wicks": {
            "upper_wick_mean": float(upper_wick.mean()),
            "lower_wick_mean": float(lower_wick.mean()),
            "upper_wick_max": float(upper_wick.max()),
            "lower_wick_max": float(lower_wick.max()),
        },
        "candle_types": {
            "bullish": bullish_count,
            "bearish": bearish_count,
            "doji": doji_count,
            "bullish_pct": float(bullish_count / len(df) * 100.0),
            "bearish_pct": float(bearish_count / len(df) * 100.0),
            "doji_pct": float(doji_count / len(df) * 100.0),
        },
        "simple_returns": {
            "mean": float(simple_returns.mean()),
            "std": float(simple_returns.std()),
            "min": float(simple_returns.min()),
            "max": float(simple_returns.max()),
            "skewness": float(stats.skew(simple_returns)),
            "kurtosis": float(stats.kurtosis(simple_returns)),
            "quantiles": _quantiles(simple_returns),
        },
        "log_returns": {
            "mean": float(log_returns.mean()),
            "std": bar_std,
            "min": float(log_returns.min()),
            "max": float(log_returns.max()),
            "skewness": float(stats.skew(log_returns)),
            "kurtosis": float(stats.kurtosis(log_returns)),
            "quantiles": _quantiles(log_returns),
        },
        "volatility": {
            "bar_std": bar_std,
            "annualized_vol": annualized_vol,
            "atr_14_mean": atr_14,
        },
    }


def compute_temporal_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze activity, volatility, and volume patterns by hour, weekday, month, and year.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' or 'time', 'close', 'open', 'high', 'low',
        'spread', 'tick_volume'.

    Returns
    -------
    dict[str, Any]
        Temporal breakdown statistics.
    """
    work_df = df.copy()
    if "timestamp" not in work_df.columns:
        work_df["timestamp"] = pd.to_datetime(work_df["time"], unit="s", utc=True)
    elif not pd.api.types.is_datetime64_any_dtype(work_df["timestamp"]):
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)

    ts = work_df["timestamp"]
    work_df["hour"] = ts.dt.hour
    work_df["day_name"] = ts.dt.day_name()
    work_df["day_of_week"] = ts.dt.dayofweek
    work_df["month"] = ts.dt.month
    work_df["month_name"] = ts.dt.month_name()
    work_df["year"] = ts.dt.year
    work_df["log_ret"] = np.log(work_df["close"] / work_df["close"].shift(1))

    # Hourly metrics (0..23 UTC)
    hourly_records: list[dict[str, Any]] = []
    for h in range(24):
        sub = work_df[work_df["hour"] == h]
        if sub.empty:
            continue
        rets = sub["log_ret"].dropna()
        hourly_records.append(
            {
                "hour_utc": h,
                "candle_count": int(len(sub)),
                "return_mean": float(rets.mean()) if not rets.empty else 0.0,
                "return_std": float(rets.std()) if not rets.empty else 0.0,
                "spread_mean": float(sub["spread"].mean()),
                "tick_volume_mean": float(sub["tick_volume"].mean()),
            }
        )

    # Weekday metrics (0=Mon .. 6=Sun)
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_records: list[dict[str, Any]] = []
    for dow, name in enumerate(weekday_order):
        sub = work_df[work_df["day_of_week"] == dow]
        if sub.empty:
            continue
        rets = sub["log_ret"].dropna()
        weekday_records.append(
            {
                "day_of_week": dow,
                "day_name": name,
                "candle_count": int(len(sub)),
                "return_mean": float(rets.mean()) if not rets.empty else 0.0,
                "return_std": float(rets.std()) if not rets.empty else 0.0,
                "spread_mean": float(sub["spread"].mean()),
                "tick_volume_mean": float(sub["tick_volume"].mean()),
            }
        )

    # Monthly metrics (1..12)
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    monthly_records: list[dict[str, Any]] = []
    for m, m_name in enumerate(month_names, start=1):
        sub = work_df[work_df["month"] == m]
        if sub.empty:
            continue
        rets = sub["log_ret"].dropna()
        monthly_records.append(
            {
                "month": m,
                "month_name": m_name,
                "candle_count": int(len(sub)),
                "return_mean": float(rets.mean()) if not rets.empty else 0.0,
                "return_std": float(rets.std()) if not rets.empty else 0.0,
                "spread_mean": float(sub["spread"].mean()),
                "tick_volume_mean": float(sub["tick_volume"].mean()),
            }
        )

    # Yearly metrics
    yearly_records: list[dict[str, Any]] = []
    for yr in sorted(work_df["year"].unique()):
        sub = work_df[work_df["year"] == yr]
        rets = sub["log_ret"].dropna()
        open_yr = float(sub["open"].iloc[0])
        close_yr = float(sub["close"].iloc[-1])
        high_yr = float(sub["high"].max())
        low_yr = float(sub["low"].min())
        ann_ret = (close_yr - open_yr) / open_yr * 100.0
        yearly_records.append(
            {
                "year": int(yr),
                "candle_count": int(len(sub)),
                "open": open_yr,
                "close": close_yr,
                "high": high_yr,
                "low": low_yr,
                "return_pct": float(ann_ret),
                "return_mean": float(rets.mean()) if not rets.empty else 0.0,
                "volatility_std": float(rets.std()) if not rets.empty else 0.0,
                "spread_mean": float(sub["spread"].mean()),
                "tick_volume_mean": float(sub["tick_volume"].mean()),
            }
        )

    # Trading Sessions (standard UTC conventions)
    # Asian: 00:00 - 08:00 UTC
    # European / London: 07:00 - 16:00 UTC
    # US / New York: 12:00 - 21:00 UTC
    # Rollover / Off-hours: 21:00 - 00:00 UTC
    def _session_metrics(mask: pd.Series, name: str) -> dict[str, Any]:
        sub = work_df[mask]
        rets = sub["log_ret"].dropna()
        return {
            "session": name,
            "candle_count": int(len(sub)),
            "return_mean": float(rets.mean()) if not rets.empty else 0.0,
            "return_std": float(rets.std()) if not rets.empty else 0.0,
            "spread_mean": float(sub["spread"].mean()),
            "tick_volume_mean": float(sub["tick_volume"].mean()),
        }

    session_records = [
        _session_metrics(work_df["hour"].between(0, 7), "Asian (00:00-08:00 UTC)"),
        _session_metrics(work_df["hour"].between(7, 15), "London (07:00-16:00 UTC)"),
        _session_metrics(work_df["hour"].between(12, 20), "New York (12:00-21:00 UTC)"),
        _session_metrics(work_df["hour"].between(21, 23), "Rollover (21:00-00:00 UTC)"),
    ]

    return {
        "hourly": hourly_records,
        "weekday": weekday_records,
        "monthly": monthly_records,
        "yearly": yearly_records,
        "sessions": session_records,
    }


def compute_volume_spread_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute detailed distribution metrics for tick_volume, spread, and real_volume.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'tick_volume', 'spread', 'real_volume'.

    Returns
    -------
    dict[str, Any]
        Volume and spread metrics.
    """
    tv = df["tick_volume"]
    sp = df["spread"]
    rv = df["real_volume"]

    def _stats(s: pd.Series) -> dict[str, Any]:
        q = s.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()
        quantiles = {f"p{int(k * 100):02d}": float(v) for k, v in q.items()}
        zero_count = int((s == 0).sum())
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "quantiles": quantiles,
            "zero_count": zero_count,
            "zero_pct": float(zero_count / len(s) * 100.0),
        }

    return {
        "tick_volume": _stats(tv),
        "spread_points": _stats(sp),
        "real_volume": _stats(rv),
    }


def compute_gap_statistics(df: pd.DataFrame, time_col: str = "time") -> dict[str, Any]:
    """Analyze inter-candle timestamp delta distributions and classify gaps.

    Reuses and integrates with ai.data.gaps.analyze_gaps for standard classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'time' or 'timestamp'.
    time_col : str, default 'time'
        Name of timestamp column.

    Returns
    -------
    dict[str, Any]
        Gap classification and statistics.
    """
    work_df = df.copy()
    if time_col not in work_df.columns:
        if "timestamp" in work_df.columns:
            work_df[time_col] = (
                pd.to_datetime(work_df["timestamp"], utc=True).astype("int64") // 10**9
            )
        else:
            raise KeyError(f"Neither '{time_col}' nor 'timestamp' found in DataFrame.")

    times = work_df[time_col].sort_values().to_numpy()
    deltas = np.diff(times)
    total_steps = len(deltas)
    normal_steps = int((deltas == 900).sum())

    gap_result = analyze_gaps(work_df, time_col=time_col, expected_interval_seconds=900)

    # Durations frequency summary (rounded to hours)
    duration_hours_counter = Counter(round(g.gap_duration_hours, 1) for g in gap_result.gaps)
    duration_freq = [
        {"duration_hours": k, "count": v}
        for k, v in sorted(duration_hours_counter.items(), key=lambda x: x[1], reverse=True)
    ]

    # Convert gaps to list of dicts sorted by duration
    sorted_gaps = sorted(
        [g.to_dict() for g in gap_result.gaps],
        key=lambda x: x["gap_duration_seconds"],
        reverse=True,
    )

    max_gap_sec = int(deltas.max()) if total_steps > 0 else 0
    max_gap_hrs = round(max_gap_sec / 3600.0, 2) if total_steps > 0 else 0.0

    return {
        "total_intervals": total_steps,
        "normal_intervals_900s": normal_steps,
        "normal_interval_pct": (
            float(normal_steps / total_steps * 100.0) if total_steps > 0 else 0.0
        ),
        "total_gaps": gap_result.total_gaps,
        "expected_weekend_count": gap_result.expected_weekend_count,
        "expected_holiday_count": gap_result.expected_holiday_count,
        "unexpected_gap_count": gap_result.unexpected_gap_count,
        "unclassified_gap_count": gap_result.unclassified_gap_count,
        "max_gap_seconds": max_gap_sec,
        "max_gap_hours": max_gap_hrs,
        "gap_duration_frequencies": duration_freq[:10],
        "top_10_largest_gaps": sorted_gaps[:10],
    }


def generate_eda_plots(df: pd.DataFrame, output_dir: Path | str) -> list[Path]:
    """Generate high-resolution non-interactive EDA charts.

    Produces 8 standard research charts:
    1. eurusd_close_price.png: Close price over time
    2. eurusd_return_distribution.png: Return distribution with normal overlay & Q-Q plot
    3. eurusd_return_timeseries.png: Log returns time series (volatility clustering)
    4. eurusd_rolling_volatility.png: Rolling annualized volatility (20-bar & 96-bar)
    5. eurusd_spread_distribution.png: Spread distribution (histogram & ECDF)
    6. eurusd_volume_distribution.png: Tick volume distribution (histogram & ECDF)
    7. eurusd_hourly_activity.png: Intraday activity by UTC hour (volume & spread)
    8. eurusd_weekday_activity.png: Day of week activity (volume & volatility)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with market data.
    output_dir : Path | str
        Target directory to save figures.

    Returns
    -------
    list[Path]
        Paths of generated figure files.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    work_df = df.copy()
    if "timestamp" not in work_df.columns:
        work_df["timestamp"] = pd.to_datetime(work_df["time"], unit="s", utc=True)
    elif not pd.api.types.is_datetime64_any_dtype(work_df["timestamp"]):
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)

    work_df["log_ret"] = np.log(work_df["close"] / work_df["close"].shift(1))
    work_df["hour"] = work_df["timestamp"].dt.hour
    work_df["day_name"] = work_df["timestamp"].dt.day_name()
    work_df["day_of_week"] = work_df["timestamp"].dt.dayofweek

    generated_files: list[Path] = []

    # 1. Close Price Series
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        work_df["timestamp"],
        work_df["close"],
        color="#1f77b4",
        linewidth=0.8,
        label="EURUSD Close",
    )
    ax.set_title(
        "EURUSD M15 Historical Close Price (2022-2026)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    p1 = out_path / "eurusd_close_price.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    generated_files.append(p1)

    # 2. Return Distribution & Q-Q vs Normal
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rets = work_df["log_ret"].dropna()
    ax1.hist(
        rets,
        bins=150,
        density=True,
        alpha=0.6,
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.3,
        label="M15 Log Returns",
    )
    mu, sigma = float(rets.mean()), float(rets.std())
    x_grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax1.plot(
        x_grid,
        stats.norm.pdf(x_grid, mu, sigma),
        "r--",
        linewidth=1.2,
        label=f"Normal Fit (μ={mu:.5f}, σ={sigma:.4f})",
    )
    ax1.set_title("Return Density vs. Normal Distribution", fontweight="bold")
    ax1.set_xlabel("Log Return")
    ax1.set_ylabel("Density")
    ax1.set_xlim(mu - 5 * sigma, mu + 5 * sigma)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    stats.probplot(rets, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot vs Normal (Fat Tails Diagnostic)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    p2 = out_path / "eurusd_return_distribution.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    generated_files.append(p2)

    # 3. Return Time Series (Volatility Clustering)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        work_df["timestamp"],
        work_df["log_ret"],
        color="#4b72b0",
        linewidth=0.4,
        alpha=0.7,
        label="M15 Log Return",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_title(
        "EURUSD M15 Log Return Time Series (Volatility Clustering)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Log Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    p3 = out_path / "eurusd_return_timeseries.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    generated_files.append(p3)

    # 4. Rolling Volatility
    fig, ax = plt.subplots(figsize=(12, 5))
    roll_vol_20 = work_df["log_ret"].rolling(20).std() * math.sqrt(24192) * 100.0
    roll_vol_96 = work_df["log_ret"].rolling(96).std() * math.sqrt(24192) * 100.0
    ax.plot(
        work_df["timestamp"],
        roll_vol_20,
        color="#ff7f0e",
        linewidth=0.5,
        alpha=0.6,
        label="Rolling Vol (20 bars / 5h, Ann %)",
    )
    ax.plot(
        work_df["timestamp"],
        roll_vol_96,
        color="#d62728",
        linewidth=0.9,
        label="Rolling Vol (96 bars / 1d, Ann %)",
    )
    ax.set_title(
        "EURUSD M15 Rolling Annualized Volatility",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    p4 = out_path / "eurusd_rolling_volatility.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    generated_files.append(p4)

    # 5. Spread Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    spread_data = work_df["spread"]
    span = int(spread_data.max() - spread_data.min())
    spread_bins = max(1, min(50, span + 1))
    ax1.hist(
        spread_data,
        bins=spread_bins,
        color="#e377c2",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.75,
    )
    ax1.axvline(
        spread_data.mean(),
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean: {spread_data.mean():.1f} pts",
    )
    ax1.axvline(
        spread_data.median(),
        color="green",
        linestyle="-",
        linewidth=1.2,
        label=f"Median: {spread_data.median():.1f} pts",
    )
    ax1.set_title("EURUSD M15 Spread Distribution", fontweight="bold")
    ax1.set_xlabel("Spread (Points)")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ECDF of spread
    sorted_spread = np.sort(spread_data.to_numpy())
    ecdf_y = np.arange(1, len(sorted_spread) + 1) / len(sorted_spread)
    ax2.plot(sorted_spread, ecdf_y, color="#9467bd", linewidth=1.5)
    ax2.set_title("Empirical CDF of Spread", fontweight="bold")
    ax2.set_xlabel("Spread (Points)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.grid(True, alpha=0.3)
    p5 = out_path / "eurusd_spread_distribution.png"
    fig.tight_layout()
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    generated_files.append(p5)

    # 6. Tick Volume Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    vol_data = work_df["tick_volume"]
    ax1.hist(
        vol_data,
        bins=min(80, max(1, len(vol_data))),
        color="#17becf",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.75,
    )
    ax1.axvline(
        vol_data.mean(),
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean: {vol_data.mean():.1f}",
    )
    ax1.axvline(
        vol_data.median(),
        color="green",
        linestyle="-",
        linewidth=1.2,
        label=f"Median: {vol_data.median():.1f}",
    )
    ax1.set_title("EURUSD M15 Tick Volume Distribution", fontweight="bold")
    ax1.set_xlabel("Tick Volume")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    sorted_vol = np.sort(vol_data.to_numpy())
    ecdf_vol_y = np.arange(1, len(sorted_vol) + 1) / len(sorted_vol)
    ax2.plot(sorted_vol, ecdf_vol_y, color="#1f77b4", linewidth=1.5)
    ax2.set_title("Empirical CDF of Tick Volume", fontweight="bold")
    ax2.set_xlabel("Tick Volume")
    ax2.set_ylabel("Cumulative Probability")
    ax2.grid(True, alpha=0.3)
    p6 = out_path / "eurusd_volume_distribution.png"
    fig.tight_layout()
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    generated_files.append(p6)

    # 7. Hourly Activity (Volume & Spread)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    hourly_vol = (
        work_df.groupby("hour")["tick_volume"].mean().reindex(np.arange(24), fill_value=0.0)
    )
    hourly_sp = work_df.groupby("hour")["spread"].mean().reindex(np.arange(24), fill_value=0.0)
    hours = np.arange(24)

    ax1.bar(
        hours,
        hourly_vol,
        color="#1f77b4",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Avg Tick Volume")
    ax1.set_title("EURUSD Market Mechanics by Hour of Day (UTC)", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(hours, hourly_sp, color="#d62728", marker="o", linewidth=1.5)
    ax2.set_ylabel("Avg Spread (Points)")
    ax2.set_xlabel("Hour of Day (UTC)")
    ax2.set_xticks(hours)
    ax2.grid(True, alpha=0.3)
    p7 = out_path / "eurusd_hourly_activity.png"
    fig.tight_layout()
    fig.savefig(p7, dpi=150)
    plt.close(fig)
    generated_files.append(p7)

    # 8. Weekday Activity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Sunday",
    ]
    sub_days = work_df[work_df["day_name"].isin(day_order)]
    weekday_vol = (
        sub_days.groupby("day_name")["tick_volume"].mean().reindex(day_order, fill_value=0.0)
    )
    weekday_vol_std = (
        sub_days.groupby("day_name")["log_ret"].std().reindex(day_order, fill_value=0.0).fillna(0.0)
        * 10000.0
    )

    ax1.bar(
        day_order,
        weekday_vol,
        color="#9467bd",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_title("Average Tick Volume by Weekday", fontweight="bold")
    ax1.set_ylabel("Avg Tick Volume")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(True, alpha=0.3)

    ax2.bar(
        day_order,
        weekday_vol_std,
        color="#8c564b",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_title("Return Volatility by Weekday (bps)", fontweight="bold")
    ax2.set_ylabel("Log Return Std (Basis Points)")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, alpha=0.3)
    p8 = out_path / "eurusd_weekday_activity.png"
    fig.tight_layout()
    fig.savefig(p8, dpi=150)
    plt.close(fig)
    generated_files.append(p8)

    return generated_files


def run_full_eda(df: pd.DataFrame, figures_dir: Path | str | None = None) -> dict[str, Any]:
    """Execute complete EDA suite and return comprehensive dictionary of results."""
    price_stats = compute_price_statistics(df)
    temp_stats = compute_temporal_statistics(df)
    vol_spread_stats = compute_volume_spread_statistics(df)
    gap_stats = compute_gap_statistics(df)

    figures: list[str] = []
    if figures_dir is not None:
        fig_paths = generate_eda_plots(df, figures_dir)
        figures = [str(p) for p in fig_paths]

    return {
        "price": price_stats,
        "temporal": temp_stats,
        "volume_spread": vol_spread_stats,
        "gaps": gap_stats,
        "figures": figures,
    }
