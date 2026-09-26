"""Price action and candle geometry feature computation.

Calculates strictly point-in-time single-candle and rolling price action features
without look-ahead bias or target leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price action and candle geometry features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'open', 'high', 'low', 'close'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing calculated price features.
    """
    open_p = df["open"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)

    features = pd.DataFrame(index=df.index)

    # 1. Single-bar returns
    features["return_1"] = close.pct_change(1)
    features["log_return_1"] = np.log(close / close.shift(1))
    features["open_to_close_return"] = (close - open_p) / open_p

    # 2. Candle geometry
    hl_range = high - low
    features["hl_range"] = hl_range
    features["hl_range_norm"] = hl_range / close

    body = (close - open_p).abs()
    features["candle_body"] = body
    features["candle_body_norm"] = body / close

    upper_wick = high - np.maximum(open_p, close)
    lower_wick = np.minimum(open_p, close) - low
    features["upper_wick"] = upper_wick
    features["upper_wick_ratio"] = upper_wick / (hl_range + 1e-8)
    features["lower_wick"] = lower_wick
    features["lower_wick_ratio"] = lower_wick / (hl_range + 1e-8)

    features["candle_direction"] = np.sign(close - open_p)

    # 3. Rolling return statistics (5 and 20 bars)
    features["return_mean_5"] = features["log_return_1"].rolling(5).mean()
    features["return_mean_20"] = features["log_return_1"].rolling(20).mean()

    return features
