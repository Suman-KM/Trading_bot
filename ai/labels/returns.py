"""Forward return target calculation across multiple horizons.

Calculates future simple returns, future continuous log returns, and point-in-time
volatility-normalized forward movements. Strictly for target engineering (y),
never to be included as input features (X).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZONS: list[int] = [1, 4, 8, 16]


def compute_future_returns(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute forward-looking returns across candidate prediction horizons.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and optionally 'high', 'low'.
    horizons : list[int] | None, default None
        List of forward horizons in bars. Defaults to [1, 4, 8, 16].

    Returns
    -------
    pd.DataFrame
        DataFrame containing forward return targets.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    close = df["close"].astype(np.float64)
    targets = pd.DataFrame(index=df.index)

    # Point-in-time ATR-14 for volatility adjustment (strictly uses past/current data at t)
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
        # 1. Simple Forward Return: (close[t + h] - close[t]) / close[t]
        future_close = close.shift(-h)
        future_return = (future_close / close) - 1.0
        targets[f"future_return_{h}"] = future_return

        # 2. Continuous Forward Log Return: ln(close[t + h] / close[t])
        future_log_return = np.log(future_close / close)
        targets[f"future_log_return_{h}"] = future_log_return

        # 3. Volatility-Adjusted Forward Return: future_return / atr_norm(t)
        # Note: atr_norm(t) uses only information available at time t
        if atr_norm is not None:
            targets[f"future_vol_adj_return_{h}"] = future_return / (atr_norm + 1e-12)

    return targets
