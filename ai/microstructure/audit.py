"""Phase 19: Pure Market Microstructure Analytical and Validation Engine.

Provides descriptive metrics, intrabar path reconstruction, tick intensity,
spread dynamics, and quality checks for MT5 tick data without generating ML features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_tick_data_quality(ticks_df: pd.DataFrame) -> dict[str, Any]:
    """Validate quality of raw tick data (monotonicity, duplicates, spreads, gaps).

    Args:
        ticks_df: DataFrame with ['time_msc', 'bid', 'ask', 'flags'].

    Returns:
        Dictionary of quality audit metrics and pass/fail indicators.
    """
    if len(ticks_df) == 0:
        return {"status": "EMPTY", "tick_count": 0}

    # Timestamp monotonicity
    is_monotonic = bool(ticks_df["time_msc"].is_monotonic_increasing)

    # Duplicate detection
    dup_timestamp_count = int(ticks_df.duplicated(subset=["time_msc"]).sum())
    dup_full_count = int(ticks_df.duplicated(subset=["time_msc", "bid", "ask"]).sum())

    # Spread checks
    spread_series = ticks_df["ask"] - ticks_df["bid"]
    negative_spread_count = int((spread_series < 0.0).sum())
    zero_spread_count = int((spread_series == 0.0).sum())

    # Price validity
    invalid_bid_count = int((ticks_df["bid"] <= 0.0).sum() + ticks_df["bid"].isna().sum())
    invalid_ask_count = int((ticks_df["ask"] <= 0.0).sum() + ticks_df["ask"].isna().sum())

    # Inter-tick intervals in milliseconds
    time_diffs_ms = ticks_df["time_msc"].diff().dropna()
    median_interval_ms = float(time_diffs_ms.median()) if len(time_diffs_ms) > 0 else 0.0
    p95_interval_ms = float(time_diffs_ms.quantile(0.95)) if len(time_diffs_ms) > 0 else 0.0
    max_gap_seconds = float(time_diffs_ms.max() / 1000.0) if len(time_diffs_ms) > 0 else 0.0

    return {
        "tick_count": len(ticks_df),
        "is_monotonic_increasing": is_monotonic,
        "duplicate_timestamps": dup_timestamp_count,
        "duplicate_quotes": dup_full_count,
        "negative_spread_count": negative_spread_count,
        "zero_spread_count": zero_spread_count,
        "invalid_bid_count": invalid_bid_count,
        "invalid_ask_count": invalid_ask_count,
        "median_inter_tick_ms": median_interval_ms,
        "p95_inter_tick_ms": p95_interval_ms,
        "max_gap_seconds": max_gap_seconds,
        "quality_pass": bool(
            is_monotonic
            and negative_spread_count == 0
            and invalid_bid_count == 0
            and invalid_ask_count == 0
        ),
    }


def compute_tick_spread_metrics(ticks_df: pd.DataFrame, point: float = 1e-5) -> dict[str, Any]:
    """Compute descriptive spread statistics from tick-level Bid/Ask quotes.

    Args:
        ticks_df: DataFrame containing 'bid' and 'ask'.
        point: Instrument point size (1e-5 for 5-digit EURUSD).

    Returns:
        Dictionary containing spread statistics in points and pips.
    """
    if len(ticks_df) == 0:
        return {}

    spread_price = ticks_df["ask"] - ticks_df["bid"]
    spread_points = spread_price / point
    spread_pips = spread_points / 10.0

    return {
        "mean_spread_points": float(spread_points.mean()),
        "median_spread_points": float(spread_points.median()),
        "p90_spread_points": float(spread_points.quantile(0.90)),
        "p95_spread_points": float(spread_points.quantile(0.95)),
        "p99_spread_points": float(spread_points.quantile(0.99)),
        "min_spread_points": float(spread_points.min()),
        "max_spread_points": float(spread_points.max()),
        "std_spread_points": float(spread_points.std()),
        "mean_spread_pips": float(spread_pips.mean()),
        "median_spread_pips": float(spread_pips.median()),
        "p95_spread_pips": float(spread_pips.quantile(0.95)),
        "missing_spread_pct": float(spread_price.isna().mean() * 100.0),
    }


def compute_tick_intensity_metrics(
    ticks_df: pd.DataFrame,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Compute tick arrival intensity across various bar durations.

    Args:
        ticks_df: DataFrame with 'timestamp' (tz-aware UTC datetime).
        start_ts: Optional explicit window start.
        end_ts: Optional explicit window end.

    Returns:
        Dictionary of intensity metrics across seconds, minutes, and bars.
    """
    if len(ticks_df) == 0:
        return {}

    ts = ticks_df["timestamp"]
    t_min = start_ts or ts.iloc[0]
    t_max = end_ts or ts.iloc[-1]
    total_duration_sec = max(1.0, (t_max - t_min).total_seconds())

    ticks_per_sec = float(len(ticks_df) / total_duration_sec)

    # 1-minute aggregation
    m1_counts = ticks_df.set_index("timestamp").resample("1min").size().loc[lambda s: s > 0]
    # 5-minute aggregation
    m5_counts = ticks_df.set_index("timestamp").resample("5min").size().loc[lambda s: s > 0]
    # 15-minute aggregation
    m15_counts = ticks_df.set_index("timestamp").resample("15min").size().loc[lambda s: s > 0]

    return {
        "ticks_per_second_mean": ticks_per_sec,
        "ticks_per_minute_mean": float(m1_counts.mean()) if len(m1_counts) > 0 else 0.0,
        "ticks_per_minute_median": float(m1_counts.median()) if len(m1_counts) > 0 else 0.0,
        "ticks_per_minute_p95": float(m1_counts.quantile(0.95)) if len(m1_counts) > 0 else 0.0,
        "ticks_per_minute_min": int(m1_counts.min()) if len(m1_counts) > 0 else 0,
        "ticks_per_minute_max": int(m1_counts.max()) if len(m1_counts) > 0 else 0,
        "ticks_per_5m_mean": float(m5_counts.mean()) if len(m5_counts) > 0 else 0.0,
        "ticks_per_5m_median": float(m5_counts.median()) if len(m5_counts) > 0 else 0.0,
        "ticks_per_15m_mean": float(m15_counts.mean()) if len(m15_counts) > 0 else 0.0,
        "ticks_per_15m_median": float(m15_counts.median()) if len(m15_counts) > 0 else 0.0,
        "ticks_per_15m_min": int(m15_counts.min()) if len(m15_counts) > 0 else 0,
        "ticks_per_15m_max": int(m15_counts.max()) if len(m15_counts) > 0 else 0,
    }


def compute_tick_direction_proxy(ticks_df: pd.DataFrame) -> dict[str, Any]:
    """Examine tick price direction proxies and audit availability of true signed flow.

    Args:
        ticks_df: DataFrame containing 'bid', 'ask', 'flags', 'last', 'volume'.

    Returns:
        Dictionary classifying direction distribution and signed flow status.
    """
    if len(ticks_df) == 0:
        return {}

    bid = ticks_df["bid"]
    ask = ticks_df["ask"]
    mid = (bid + ask) / 2.0

    delta_bid = bid.diff()
    delta_ask = ask.diff()
    delta_mid = mid.diff()

    n = max(1, len(ticks_df) - 1)
    bid_up = int((delta_bid > 0.0).sum())
    bid_down = int((delta_bid < 0.0).sum())
    bid_same = int((delta_bid == 0.0).sum())

    ask_up = int((delta_ask > 0.0).sum())
    ask_down = int((delta_ask < 0.0).sum())
    ask_same = int((delta_ask == 0.0).sum())

    mid_up = int((delta_mid > 0.0).sum())
    mid_down = int((delta_mid < 0.0).sum())
    mid_same = int((delta_mid == 0.0).sum())

    # Check for actual trade flags (TICK_FLAG_BUY=32, TICK_FLAG_SELL=64)
    flags = ticks_df["flags"] if "flags" in ticks_df.columns else pd.Series(0, index=ticks_df.index)
    has_buy_flag = bool(((flags & 32) > 0).any())
    has_sell_flag = bool(((flags & 64) > 0).any())
    has_last_trade = bool(("last" in ticks_df.columns and (ticks_df["last"] > 0.0).any()))

    true_signed_flow = bool(has_buy_flag or has_sell_flag or has_last_trade)

    return {
        "bid_upticks": bid_up,
        "bid_downticks": bid_down,
        "bid_unchanged": bid_same,
        "ask_upticks": ask_up,
        "ask_downticks": ask_down,
        "ask_unchanged": ask_same,
        "mid_upticks": mid_up,
        "mid_downticks": mid_down,
        "mid_unchanged": mid_same,
        "mid_uptick_share": float(mid_up / n),
        "mid_downtick_share": float(mid_down / n),
        "mid_unchanged_share": float(mid_same / n),
        "has_buy_flag": has_buy_flag,
        "has_sell_flag": has_sell_flag,
        "has_last_price": has_last_trade,
        "true_signed_order_flow_available": true_signed_flow,
        "classification_note": (
            "True signed order flow is unavailable from the inspected data. "
            "All tick direction metrics are strictly PRICE-DERIVED PROXIES."
            if not true_signed_flow
            else "True signed order flow available."
        ),
    }


def reconstruct_ohlc_from_ticks(
    ticks_df: pd.DataFrame, freq: str = "15min", price_col: str = "bid"
) -> pd.DataFrame:
    """Reconstruct bar OHLC from tick prices aligned to bar boundaries.

    Args:
        ticks_df: DataFrame with 'timestamp' and price_col.
        freq: Resampling frequency (e.g. '15min', '4h').
        price_col: Column to use for OHLC ('bid' or 'ask').

    Returns:
        DataFrame containing reconstructed open, high, low, close, and volume (tick count).
    """
    if len(ticks_df) == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "tick_count"])

    df = ticks_df.set_index("timestamp")
    grouped = df[price_col].resample(freq, label="left", closed="left")

    open_s = grouped.first()
    high_s = grouped.max()
    low_s = grouped.min()
    close_s = grouped.last()
    vol_s = grouped.count()

    # Intrabar spread statistics
    if "ask" in df.columns and "bid" in df.columns:
        spread = (df["ask"] - df["bid"]) / 1e-5
        spread_mean = spread.resample(freq, label="left", closed="left").mean()
        spread_max = spread.resample(freq, label="left", closed="left").max()
    else:
        spread_mean = pd.Series(0.0, index=open_s.index)
        spread_max = pd.Series(0.0, index=open_s.index)

    reconstructed = (
        pd.DataFrame(
            {
                "timestamp": open_s.index,
                "open": open_s.values,
                "high": high_s.values,
                "low": low_s.values,
                "close": close_s.values,
                "tick_count": vol_s.values,
                "spread_mean_points": spread_mean.values,
                "spread_max_points": spread_max.values,
            }
        )
        .dropna(subset=["open"])
        .reset_index(drop=True)
    )

    return reconstructed


def compute_intrabar_path_metrics(ticks_df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Compute intrabar price path geometry (path length, realized range, efficiency).

    Args:
        ticks_df: DataFrame with 'timestamp' and 'bid'.
        freq: Bar duration.

    Returns:
        DataFrame with intrabar geometric path measures per bar.
    """
    if len(ticks_df) == 0:
        return pd.DataFrame()

    df = ticks_df.copy()
    df["bar_ts"] = df["timestamp"].dt.floor(freq)

    metrics_list = []
    for bar_ts, group in df.groupby("bar_ts"):
        n_ticks = len(group)
        if n_ticks < 2:
            continue

        bids = group["bid"].to_numpy()
        abs_diffs = np.abs(np.diff(bids))
        path_length = float(np.sum(abs_diffs))
        realized_range = float(np.max(bids) - np.min(bids))
        net_displacement = float(np.abs(bids[-1] - bids[0]))
        efficiency = float(net_displacement / path_length) if path_length > 1e-12 else 0.0

        # Direction changes (zero crossings of diffs)
        diffs = np.diff(bids)
        signs = np.sign(diffs[diffs != 0.0])
        dir_changes = int(np.sum(signs[:-1] != signs[1:])) if len(signs) > 1 else 0

        metrics_list.append(
            {
                "timestamp": bar_ts,
                "tick_count": n_ticks,
                "path_length": path_length,
                "realized_range": realized_range,
                "net_displacement": net_displacement,
                "path_efficiency": efficiency,
                "direction_changes": dir_changes,
            }
        )

    return pd.DataFrame(metrics_list)


def compute_realized_volatility_metrics(
    ticks_df: pd.DataFrame, freq: str = "15min"
) -> pd.DataFrame:
    """Compute high-frequency realized volatility measures per bar.

    Args:
        ticks_df: DataFrame with 'timestamp' and 'bid'.
        freq: Bar duration.

    Returns:
        DataFrame with realized variance, realized volatility, and return dispersion.
    """
    if len(ticks_df) == 0:
        return pd.DataFrame()

    df = ticks_df.copy()
    df["bar_ts"] = df["timestamp"].dt.floor(freq)

    metrics_list = []
    for bar_ts, group in df.groupby("bar_ts"):
        if len(group) < 3:
            continue

        bids = group["bid"].to_numpy()
        # Tick percentage returns
        ret = np.diff(bids) / bids[:-1]
        realized_var = float(np.sum(ret**2))
        realized_vol = float(np.sqrt(realized_var))
        realized_abs = float(np.sum(np.abs(ret)))
        tick_ret_std = float(np.std(ret))

        metrics_list.append(
            {
                "timestamp": bar_ts,
                "tick_count": len(group),
                "realized_variance": realized_var,
                "realized_volatility": realized_vol,
                "realized_abs_return": realized_abs,
                "tick_return_std": tick_ret_std,
            }
        )

    return pd.DataFrame(metrics_list)


def analyze_session_microstructure(ticks_df: pd.DataFrame) -> dict[str, Any]:
    """Examine spread, intensity, and volatility variations across global market sessions.

    Session partitions (UTC):
        - Asia: 00:00 to 07:00 UTC
        - London Morning: 07:00 to 12:00 UTC
        - London/NY Overlap: 12:00 to 16:00 UTC
        - NY Afternoon: 16:00 to 21:00 UTC
        - Rollover / Off-Hours: 21:00 to 24:00 UTC

    Args:
        ticks_df: DataFrame with 'timestamp', 'bid', 'ask'.

    Returns:
        Dictionary of session characteristics.
    """
    if len(ticks_df) == 0:
        return {}

    df = ticks_df.copy()
    hour = df["timestamp"].dt.hour
    spread_pts = (df["ask"] - df["bid"]) / 1e-5

    sessions = {
        "Asia (00-07 UTC)": (hour >= 0) & (hour < 7),
        "London Morning (07-12 UTC)": (hour >= 7) & (hour < 12),
        "London_NY_Overlap (12-16 UTC)": (hour >= 12) & (hour < 16),
        "NY Afternoon (16-21 UTC)": (hour >= 16) & (hour < 21),
        "Rollover (21-24 UTC)": (hour >= 21) & (hour < 24),
    }

    out: dict[str, Any] = {}
    total_ticks = len(df)

    for sess_name, mask in sessions.items():
        sess_ticks = df[mask]
        n_ticks = len(sess_ticks)
        if n_ticks == 0:
            continue

        sess_spreads = spread_pts[mask]
        # Session duration in minutes represented in data
        if n_ticks > 1:
            dur_mins = max(
                1.0,
                (sess_ticks["timestamp"].max() - sess_ticks["timestamp"].min()).total_seconds()
                / 60.0,
            )
            ticks_per_min = float(n_ticks / dur_mins)
            time_diffs_ms = sess_ticks["time_msc"].diff().dropna()
            med_interval_ms = float(time_diffs_ms.median()) if len(time_diffs_ms) > 0 else 0.0
        else:
            ticks_per_min = 0.0
            med_interval_ms = 0.0

        out[sess_name] = {
            "tick_count": n_ticks,
            "tick_share_pct": float(n_ticks / total_ticks * 100.0),
            "median_spread_points": float(sess_spreads.median()),
            "mean_spread_points": float(sess_spreads.mean()),
            "p95_spread_points": float(sess_spreads.quantile(0.95)),
            "max_spread_points": float(sess_spreads.max()),
            "ticks_per_minute": ticks_per_min,
            "median_inter_tick_ms": med_interval_ms,
        }

    return out
