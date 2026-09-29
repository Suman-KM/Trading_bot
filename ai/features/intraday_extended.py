"""Controlled feature extension for Phase 15 Track A intraday research.

Calculates exactly 25 new point-in-time, strictly causal quantitative features
capturing market structure, range compression/expansion, momentum acceleration,
trend alignment, session transitions, and candle conviction without lookahead.

Strictly preserves the existing 80-feature production pipeline without modification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EXTENDED_FEATURE_NAMES: list[str] = [
    # A. Price & Market Structure (8)
    "dist_high_20",
    "dist_low_20",
    "dist_high_80",
    "dist_low_80",
    "channel_width_20",
    "channel_width_80",
    "breakout_high_20",
    "breakout_low_20",
    # B. Momentum & Acceleration (4)
    "return_persistence_5",
    "return_persistence_20",
    "momentum_acceleration_5",
    "roc_acceleration_20",
    # C. Volatility & Range Expansion/Compression (5)
    "range_expansion_ratio",
    "range_compression_5",
    "vol_regime_ratio_10_80",
    "high_low_wick_imbalance",
    "body_to_range_ratio",
    # D. Trend & Multi-Timeframe Alignment (2)
    "trend_regime_score",
    "dist_daily_pivot",
    # E. Session Transitions (3)
    "session_transition_london_open",
    "session_transition_ny_open",
    "session_transition_london_close",
    # F. Volume & Activity Dynamics (3)
    "volume_momentum_5",
    "volume_price_trend_5",
    "consecutive_direction_run",
]


def compute_intraday_extended_features(
    df: pd.DataFrame,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute 25 controlled causal features from M15 OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close', 'tick_volume', and 'timestamp'.
    eps : float, default 1e-12
        Epsilon constant to avoid division by zero.

    Returns
    -------
    pd.DataFrame
        DataFrame containing exactly the 25 extended features.
    """
    open_p = df["open"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    vol = df["tick_volume"].astype(np.float64)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    feats = pd.DataFrame(index=df.index)

    # 1. Price & Market Structure
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    high_80 = high.rolling(80).max()
    low_80 = low.rolling(80).min()

    feats["dist_high_20"] = (high_20 - close) / (close + eps)
    feats["dist_low_20"] = (close - low_20) / (close + eps)
    feats["dist_high_80"] = (high_80 - close) / (close + eps)
    feats["dist_low_80"] = (close - low_80) / (close + eps)
    feats["channel_width_20"] = (high_20 - low_20) / (close + eps)
    feats["channel_width_80"] = (high_80 - low_80) / (close + eps)

    # Breakouts: strictly uses past rolling max/min (shift 1)
    prev_high_20 = high.shift(1).rolling(20).max()
    prev_low_20 = low.shift(1).rolling(20).min()
    feats["breakout_high_20"] = (high >= prev_high_20).astype(np.float64)
    feats["breakout_low_20"] = (low <= prev_low_20).astype(np.float64)

    # 2. Momentum & Acceleration
    up_bar = (close > close.shift(1)).astype(np.float64)
    feats["return_persistence_5"] = up_bar.rolling(5).mean()
    feats["return_persistence_20"] = up_bar.rolling(20).mean()

    # Second derivative of price: (P[t] - P[t-5]) - (P[t-5] - P[t-10])
    mom_5 = close - close.shift(5)
    mom_prev_5 = close.shift(5) - close.shift(10)
    feats["momentum_acceleration_5"] = (mom_5 - mom_prev_5) / (close + eps)

    roc_10 = (close - close.shift(10)) / (close.shift(10) + eps)
    roc_20 = (close - close.shift(20)) / (close.shift(20) + eps)
    feats["roc_acceleration_20"] = roc_10 - roc_20

    # 3. Volatility & Range Expansion/Compression
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()

    hl_range = high - low
    feats["range_expansion_ratio"] = hl_range / (atr_14 + eps)

    range_5 = high.rolling(5).max() - low.rolling(5).min()
    range_20 = high_20 - low_20
    feats["range_compression_5"] = range_5 / (range_20 + eps)

    ret_1 = close.pct_change()
    vol_std_10 = ret_1.rolling(10).std()
    vol_std_80 = ret_1.rolling(80).std()
    feats["vol_regime_ratio_10_80"] = vol_std_10 / (vol_std_80 + eps)

    candle_body = (close - open_p).abs()
    feats["body_to_range_ratio"] = candle_body / (hl_range + eps)

    upper_wick = high - np.maximum(open_p, close)
    lower_wick = np.minimum(open_p, close) - low
    feats["high_low_wick_imbalance"] = (upper_wick - lower_wick) / (hl_range + eps)

    # 4. Trend & Daily Pivot Alignment
    sma_20 = close.rolling(20).mean()
    sma_80 = close.rolling(80).mean()
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()

    # Trend alignment: continuous score in [-1.0, 1.0]
    score = (
        (close > sma_20).astype(np.float64)
        + (sma_20 > sma_80).astype(np.float64)
        + (ema_10 > ema_20).astype(np.float64)
        - 1.5
    ) / 1.5
    feats["trend_regime_score"] = score

    # Multi-session pivot using past 80 bars (20 hours)
    pivot_80 = (high.rolling(80).max() + low.rolling(80).min() + close.shift(1)) / 3.0
    feats["dist_daily_pivot"] = (close - pivot_80) / (close + eps)

    # 5. Session Transitions
    hour = ts.dt.hour
    feats["session_transition_london_open"] = ((hour >= 7) & (hour < 8)).astype(np.float64)
    feats["session_transition_ny_open"] = ((hour >= 12) & (hour < 13)).astype(np.float64)
    feats["session_transition_london_close"] = ((hour >= 15) & (hour < 17)).astype(np.float64)

    # 6. Volume & Activity Dynamics
    vol_sma_20 = vol.rolling(20).mean()
    feats["volume_momentum_5"] = vol.rolling(5).mean() / (vol_sma_20 + eps)
    feats["volume_price_trend_5"] = close.pct_change(5) * feats["volume_momentum_5"]

    # Consecutive direction run (bounded in [-5, 5])
    direction = np.sign(close - open_p)
    run_counts = np.zeros(len(df), dtype=np.float64)
    dir_arr = direction.to_numpy()
    current_run = 0.0
    for i in range(1, len(dir_arr)):
        if dir_arr[i] == 0.0:
            current_run = 0.0
        elif dir_arr[i] == dir_arr[i - 1]:
            current_run = np.clip(current_run + dir_arr[i], -5.0, 5.0)
        else:
            current_run = dir_arr[i]
        run_counts[i] = current_run
    feats["consecutive_direction_run"] = run_counts

    return feats[EXTENDED_FEATURE_NAMES]


def get_extended_feature_catalog() -> list[dict[str, Any]]:
    """Return descriptive catalog of the 25 extended intraday features."""
    return [
        {
            "name": name,
            "category": (
                "price/market structure"
                if "high" in name or "low" in name or "channel" in name or "breakout" in name
                else (
                    "momentum"
                    if "persistence" in name or "acceleration" in name
                    else (
                        "volatility"
                        if "range" in name or "vol_" in name or "wick" in name or "body" in name
                        else (
                            "trend"
                            if "trend" in name or "pivot" in name
                            else "session"
                            if "session" in name
                            else "activity"
                        )
                    )
                )
            ),
        }
        for name in EXTENDED_FEATURE_NAMES
    ]
