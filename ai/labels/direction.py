"""Directional multi-class label generation across candidate horizons.

Assigns ternary directional labels (+1: LONG, 0: NEUTRAL, -1: SHORT) based on
well-defined empirical thresholds and point-in-time volatility scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZONS: list[int] = [1, 4, 8, 16]

# Standard fixed thresholds aligned with empirical distribution and spread
# H=1: 2.5 pips (25 pts), H=4: 5.0 pips (50 pts), H=8: 7.5 pips (75 pts), H=16: 10.0 pips (100 pts)
DEFAULT_FIXED_THRESHOLDS: dict[int, float] = {
    1: 0.00025,
    4: 0.00050,
    8: 0.00075,
    16: 0.00100,
}


def compute_direction_labels(
    df: pd.DataFrame,
    returns_df: pd.DataFrame,
    horizons: list[int] | None = None,
    thresholds: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Compute ternary directional classification labels (+1: LONG, 0: NEUTRAL, -1: SHORT).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and optionally 'high', 'low'.
    returns_df : pd.DataFrame
        DataFrame containing computed future returns.
    horizons : list[int] | None, default None
        List of forward horizons. Defaults to [1, 4, 8, 16].
    thresholds : dict[int, float] | None, default None
        Mapping of horizon to fixed absolute return threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ternary directional labels.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    if thresholds is None:
        thresholds = DEFAULT_FIXED_THRESHOLDS

    close = df["close"].astype(np.float64)
    labels = pd.DataFrame(index=df.index)

    # Point-in-time ATR-14 for volatility-scaled directional thresholds
    atr_norm: pd.Series | None = None
    if "high" in df.columns and "low" in df.columns:
        high = df["high"].astype(np.float64)
        low = df["low"].astype(np.float64)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        atr_norm = atr_14 / close

    for h in horizons:
        ret_col = f"future_return_{h}"
        if ret_col not in returns_df.columns:
            continue
        ret = returns_df[ret_col]

        # 1. Fixed Threshold Direction (+1: LONG, 0: NEUTRAL, -1: SHORT)
        th = thresholds.get(h, 0.00025 * np.sqrt(h))
        dir_series = pd.Series(np.nan, index=df.index, dtype=np.float64)
        valid_ret = ret.notna()

        dir_series[valid_ret & (ret > th)] = 1.0
        dir_series[valid_ret & (ret < -th)] = -1.0
        dir_series[valid_ret & (ret >= -th) & (ret <= th)] = 0.0
        labels[f"direction_{h}"] = dir_series

        # 2. Volatility-Scaled Dynamic Direction (0.5 * sqrt(h) * ATR_14)
        if atr_norm is not None:
            vol_th = 0.5 * np.sqrt(h) * atr_norm
            vol_dir = pd.Series(np.nan, index=df.index, dtype=np.float64)
            valid_vol = valid_ret & vol_th.notna()

            vol_dir[valid_vol & (ret > vol_th)] = 1.0
            vol_dir[valid_vol & (ret < -vol_th)] = -1.0
            vol_dir[valid_vol & (ret >= -vol_th) & (ret <= vol_th)] = 0.0
            labels[f"direction_vol_{h}"] = vol_dir

    return labels
