"""Volatility, range, and dispersion feature computation.

Calculates multi-scale rolling return standard deviations, normalized Average True Range,
volatility term-structure ratios, and rolling price envelopes strictly point-in-time.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Nominal M15 annualization factor: 252 days * 24 hours * 4 bars = 24,192 bars
ANNUALIZATION_FACTOR_M15: float = math.sqrt(24192)


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility, ATR, and dispersion features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'high', 'low', 'close'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing calculated volatility features.
    """
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)

    features = pd.DataFrame(index=df.index)

    log_return = np.log(close / close.shift(1))

    # 1. Rolling return standard deviation across 10, 20, 40, 80 bars
    vol_windows = [10, 20, 40, 80]
    for w in vol_windows:
        features[f"vol_std_{w}"] = log_return.rolling(w).std()

    # 2. Annualized volatility (nominal 24,192 bars/year)
    features["vol_ann_20"] = features["vol_std_20"] * ANNUALIZATION_FACTOR_M15
    features["vol_ann_80"] = features["vol_std_80"] * ANNUALIZATION_FACTOR_M15

    # 3. Average True Range (ATR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    features["atr_14"] = true_range.rolling(14).mean()
    features["atr_norm_14"] = features["atr_14"] / close

    features["atr_40"] = true_range.rolling(40).mean()
    features["atr_norm_40"] = features["atr_40"] / close

    # 4. Volatility term structure ratios (short vs long term)
    features["vol_ratio_10_40"] = features["vol_std_10"] / (features["vol_std_40"] + 1e-8)
    features["vol_ratio_20_80"] = features["vol_std_20"] / (features["vol_std_80"] + 1e-8)

    # 5. Rolling high-low envelope ratio (20 bars)
    roll_max_high = high.rolling(20).max()
    roll_min_low = low.rolling(20).min()
    features["rolling_hl_ratio_20"] = (roll_max_high - roll_min_low) / close

    return features
