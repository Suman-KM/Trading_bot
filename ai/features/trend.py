"""Trend and moving average feature computation.

Calculates multi-scale moving average spreads, normalized price-to-trend distances,
and trend slope metrics strictly point-in-time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend, moving average spreads, and slope features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing calculated trend features.
    """
    close = df["close"].astype(np.float64)
    features = pd.DataFrame(index=df.index)

    windows = [10, 20, 40, 80]

    # 1. Simple Moving Averages and Normalized Price Distance
    for w in windows:
        sma = close.rolling(w).mean()
        features[f"sma_{w}"] = sma

    # 2. Exponential Moving Averages and Normalized Price Distance
    for w in windows:
        ema = close.ewm(span=w, adjust=False).mean()
        features[f"ema_{w}"] = ema
        features[f"dist_ema_{w}"] = (close - ema) / ema

    # 3. Moving Average Spread (Fast vs Slow EMAs)
    features["ema_spread_10_40"] = (features["ema_10"] - features["ema_40"]) / features["ema_40"]
    features["ema_spread_20_80"] = (features["ema_20"] - features["ema_80"]) / features["ema_80"]

    # 4. Safe Moving Average Slope (5-bar lookback velocity)
    features["ema_slope_10"] = (features["ema_10"] - features["ema_10"].shift(5)) / (
        5.0 * features["ema_10"].shift(5)
    )
    features["ema_slope_20"] = (features["ema_20"] - features["ema_20"].shift(5)) / (
        5.0 * features["ema_20"].shift(5)
    )

    return features
