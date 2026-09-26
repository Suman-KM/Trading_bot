"""Market activity, tick volume, and spread feature computation.

Calculates normalized volume intensity, volume z-scores, and spread mechanics.
Explicitly excludes real_volume which is 100% unpopulated in retail OTC feeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market activity, volume, and spread features.

    Note: 'real_volume' is intentionally excluded because Phase 4 EDA confirmed
    it is 100% zero in this MetaQuotes-Demo OTC forex feed.
    Raw 'tick_volume' and 'spread' exist in the base dataset, so derived
    features are uniquely named to prevent column collision.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'tick_volume', 'spread', and optionally 'close'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing calculated activity features.
    """
    tick_vol = df["tick_volume"].astype(np.float64)
    spread = df["spread"].astype(np.float64)

    features = pd.DataFrame(index=df.index)

    # 1. Tick Volume Dynamics
    features["log_tick_volume"] = np.log1p(tick_vol)

    vol_sma_20 = tick_vol.rolling(20).mean()
    vol_std_20 = tick_vol.rolling(20).std()
    features["tick_vol_sma_20"] = vol_sma_20
    features["tick_vol_ratio_20"] = tick_vol / (vol_sma_20 + 1e-8)
    features["tick_vol_zscore_20"] = (tick_vol - vol_sma_20) / (vol_std_20 + 1e-8)

    vol_sma_80 = tick_vol.rolling(80).mean()
    vol_std_80 = tick_vol.rolling(80).std()
    features["tick_vol_zscore_80"] = (tick_vol - vol_sma_80) / (vol_std_80 + 1e-8)

    # 2. Spread Dynamics
    if "close" in df.columns:
        close = df["close"].astype(np.float64)
        features["spread_norm"] = (spread * 1e-5) / close

    spread_sma_20 = spread.rolling(20).mean()
    spread_std_20 = spread.rolling(20).std()
    features["spread_sma_20"] = spread_sma_20
    features["spread_ratio_20"] = spread / (spread_sma_20 + 1e-8)
    features["spread_zscore_20"] = (spread - spread_sma_20) / (spread_std_20 + 1e-8)

    return features
