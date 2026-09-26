"""Momentum and oscillator feature computation.

Calculates multi-scale rate-of-change, price-to-moving-average distance,
and normalized momentum oscillators strictly point-in-time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum, rate of change, and oscillator features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing calculated momentum features.
    """
    close = df["close"].astype(np.float64)
    features = pd.DataFrame(index=df.index)

    # 1. Rate of Change (ROC) across 5, 10, 20, 40, 80 bars
    windows = [5, 10, 20, 40, 80]
    for w in windows:
        features[f"roc_{w}"] = (close - close.shift(w)) / close.shift(w)

    # 2. Distance from Simple Moving Average (SMA)
    sma_windows = [10, 20, 40, 80]
    for w in sma_windows:
        sma = close.rolling(w).mean()
        features[f"dist_sma_{w}"] = (close - sma) / sma

    # 3. Relative Strength Index (RSI, 14-period Wilder convention)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    return features
