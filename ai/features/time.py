"""Calendar, session, and diurnal cyclical feature computation.

Transforms UTC timestamps into continuous harmonic sine/cosine projections
and trading session indicator flags without future leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic calendar and cyclical time features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' (datetime64) or 'time' (int64 epoch).

    Returns
    -------
    pd.DataFrame
        DataFrame containing calculated time features.
    """
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
    elif "time" in df.columns:
        ts = pd.to_datetime(df["time"], unit="s", utc=True)
    else:
        raise KeyError("DataFrame must contain 'timestamp' or 'time' column.")

    features = pd.DataFrame(index=df.index)

    hour = ts.dt.hour
    minute = ts.dt.minute
    dow = ts.dt.dayofweek

    # 1. Raw calendar components
    features["hour"] = hour.astype(np.float64)
    features["minute"] = minute.astype(np.float64)
    features["day_of_week"] = dow.astype(np.float64)

    # 2. Cyclical harmonic encodings (continuous circle mappings)
    # Hour of day (period 24)
    features["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    features["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)

    # Minute of hour (period 60)
    features["sin_minute"] = np.sin(2.0 * np.pi * minute / 60.0)
    features["cos_minute"] = np.cos(2.0 * np.pi * minute / 60.0)

    # Day of week (period 7)
    features["sin_day_of_week"] = np.sin(2.0 * np.pi * dow / 7.0)
    features["cos_day_of_week"] = np.cos(2.0 * np.pi * dow / 7.0)

    # 3. Standard global FX session indicator regimes (UTC)
    features["is_asian_session"] = hour.between(0, 7).astype(np.float64)
    features["is_london_session"] = hour.between(7, 15).astype(np.float64)
    features["is_ny_session"] = hour.between(12, 20).astype(np.float64)
    features["is_rollover_window"] = hour.between(21, 23).astype(np.float64)

    return features
