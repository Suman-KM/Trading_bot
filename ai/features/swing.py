"""Swing feature engineering pipeline for H4 and D1 timeframes.

Calculates a focused set of ~30 point-in-time, causal features tailored for
multi-hour and multi-day swing holding periods across trend, momentum, volatility,
market structure, and calendar cycles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14, eps: float = 1e-12) -> pd.Series:
    """Compute standard Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing / exponential moving average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + eps)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_swing_features(
    df: pd.DataFrame,
    timeframe: str = "H4",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute controlled causal swing features for H4 or D1 timeframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close', 'tick_volume', 'timestamp'.
    timeframe : str, default 'H4'
        Timeframe identifier ('H4' or 'D1').
    eps : float, default 1e-12
        Epsilon constant to avoid division by zero.

    Returns
    -------
    pd.DataFrame
        DataFrame of swing features aligned with df.index.
    """
    open_p = df["open"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    vol = df["tick_volume"].astype(np.float64)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    feats = pd.DataFrame(index=df.index)

    # 1. Price Structure & Multi-Bar Returns
    feats["return_1"] = close.pct_change(1)
    feats["return_2"] = close.pct_change(2)
    feats["return_4"] = close.pct_change(4)
    feats["return_8"] = close.pct_change(8)

    hl_range = high - low
    feats["hl_range_norm"] = hl_range / (close + eps)

    candle_body = (close - open_p).abs()
    feats["candle_body_norm"] = candle_body / (close + eps)
    feats["candle_direction"] = np.sign(close - open_p)

    upper_wick = high - np.maximum(open_p, close)
    lower_wick = np.minimum(open_p, close) - low
    feats["upper_wick_ratio"] = upper_wick / (hl_range + eps)
    feats["lower_wick_ratio"] = lower_wick / (hl_range + eps)

    # 2. Trend & Moving Average Relationships
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()

    feats["dist_sma_10"] = (close - sma_10) / (close + eps)
    feats["dist_sma_20"] = (close - sma_20) / (close + eps)
    feats["dist_sma_50"] = (close - sma_50) / (close + eps)
    feats["sma_slope_10"] = (sma_10 - sma_10.shift(3)) / (close + eps)
    feats["sma_slope_20"] = (sma_20 - sma_20.shift(5)) / (close + eps)
    feats["ema_spread_10_20"] = (ema_10 - ema_20) / (close + eps)

    # Trend regime: alignment score in [-1.0, 1.0]
    trend_score = (
        (close > sma_20).astype(np.float64)
        + (sma_20 > sma_50).astype(np.float64)
        + (ema_10 > ema_20).astype(np.float64)
        - 1.5
    ) / 1.5
    feats["trend_regime"] = trend_score

    # 3. Momentum & Oscillators
    feats["roc_5"] = (close - close.shift(5)) / (close.shift(5) + eps)
    feats["roc_10"] = (close - close.shift(10)) / (close.shift(10) + eps)
    feats["roc_20"] = (close - close.shift(20)) / (close.shift(20) + eps)
    feats["rsi_14"] = compute_rsi(close, period=14)

    # Momentum acceleration: 2nd derivative over 5 bars
    mom_diff = (close - close.shift(5)) - (close.shift(5) - close.shift(10))
    feats["momentum_acceleration"] = mom_diff / (close + eps)

    # 4. Volatility & Channel Dynamics
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    feats["atr_14"] = atr_14
    feats["atr_norm_14"] = atr_14 / (close + eps)

    ret_1 = close.pct_change()
    feats["vol_std_10"] = ret_1.rolling(10).std()
    feats["vol_std_20"] = ret_1.rolling(20).std()
    feats["vol_ratio_5_20"] = ret_1.rolling(5).std() / (ret_1.rolling(20).std() + eps)

    # 5. Market Structure & Support/Resistance Extremes
    high_10 = high.rolling(10).max()
    low_10 = low.rolling(10).min()
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()

    feats["dist_rolling_high_10"] = (high_10 - close) / (close + eps)
    feats["dist_rolling_low_10"] = (close - low_10) / (close + eps)
    feats["dist_rolling_high_20"] = (high_20 - close) / (close + eps)
    feats["dist_rolling_low_20"] = (close - low_20) / (close + eps)

    # Stochastic-style range position
    feats["rolling_hl_ratio_20"] = (close - low_20) / (high_20 - low_20 + eps)
    feats["channel_width_20"] = (high_20 - low_20) / (close + eps)

    # Breakouts: strictly uses past highs/lows (shift 1)
    prev_h10 = high.shift(1).rolling(10).max()
    prev_l10 = low.shift(1).rolling(10).min()
    feats["breakout_high_10"] = (high >= prev_h10).astype(np.float64)
    feats["breakout_low_10"] = (low <= prev_l10).astype(np.float64)

    # 6. Volume & Activity
    vol_sma_10 = vol.rolling(10).mean()
    feats["vol_ratio_10"] = vol / (vol_sma_10 + eps)

    # 7. Time & Seasonality
    dow = ts.dt.dayofweek
    feats["day_of_week"] = dow.astype(np.float64)
    feats["sin_day_of_week"] = np.sin(2.0 * np.pi * dow / 5.0)
    feats["cos_day_of_week"] = np.cos(2.0 * np.pi * dow / 5.0)

    if timeframe == "H4":
        hour = ts.dt.hour
        feats["hour"] = hour.astype(np.float64)
        feats["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
        feats["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)

    return feats
