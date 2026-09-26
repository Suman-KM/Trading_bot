"""Gap and market-state continuity feature computation.

Calculates timestamp deltas, gap boundary flags, and market re-opening states
strictly point-in-time without synthetic interpolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gap, session transition, and market-state continuity features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'time' (int64) or 'timestamp' (datetime64).

    Returns
    -------
    pd.DataFrame
        DataFrame containing calculated gap features.
    """
    if "time" in df.columns:
        times = df["time"].astype(np.int64)
    elif "timestamp" in df.columns:
        times = pd.to_datetime(df["timestamp"], utc=True).astype("int64") // 10**9
    else:
        raise KeyError("DataFrame must contain 'time' or 'timestamp' column.")

    features = pd.DataFrame(index=df.index)

    # 1. Delta between consecutive candles in seconds
    delta_seconds = times.diff()
    features["delta_seconds"] = delta_seconds

    # 2. Binary indicators for normal M15 interval and post-gap bar
    is_normal = delta_seconds == 900
    is_post_gap = delta_seconds > 900
    features["is_normal_m15"] = is_normal.astype(np.float64)
    features["is_post_gap"] = is_post_gap.astype(np.float64)

    # 3. Weekend re-open indicator (Friday close to Sunday/Monday open, delta >= 36 hours)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
    else:
        ts = pd.to_datetime(times, unit="s", utc=True)
    dow = ts.dt.dayofweek

    is_weekend_reopen = is_post_gap & (delta_seconds >= 129600) & (dow.isin([6, 0]))
    features["is_weekend_reopen"] = is_weekend_reopen.astype(np.float64)

    # 4. Bars elapsed since the most recent gap (clamped at 96 bars = 24 hours)
    gap_group = is_post_gap.cumsum()
    bars_since = gap_group.groupby(gap_group).cumcount()
    features["bars_since_last_gap"] = bars_since.clip(upper=96).astype(np.float64)

    return features
