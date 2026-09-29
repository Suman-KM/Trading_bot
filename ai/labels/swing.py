"""Swing prediction target generation for H4 and D1 timeframes.

Calculates multi-bar forward returns, unthresholded binary direction labels,
and volatility-adjusted direction labels across multi-bar holding horizons:
- H4 horizons: 4 bars (16h), 8 bars (32h), 12 bars (48h)
- D1 horizons: 3 bars (3d), 5 bars (1w), 10 bars (2w)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_H4_HORIZONS: list[int] = [4, 8, 12]
DEFAULT_D1_HORIZONS: list[int] = [3, 5, 10]


def compute_swing_returns(
    df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Compute future percentage returns across specified multi-bar horizons.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' price series.
    horizons : list[int]
        List of forward horizons in bars.

    Returns
    -------
    pd.DataFrame
        DataFrame with forward return columns 'future_return_{h}'.
    """
    close = df["close"].astype(np.float64)
    returns_df = pd.DataFrame(index=df.index)

    for h in horizons:
        future_close = close.shift(-h)
        returns_df[f"future_return_{h}"] = (future_close / close) - 1.0

    return returns_df


def compute_swing_targets(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    timeframe: str = "H4",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute binary direction and volatility-adjusted direction targets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close', 'high', 'low'.
    horizons : list[int] | None, default None
        List of lookahead horizons. Defaults to [4, 8, 12] for H4, [3, 5, 10] for D1.
    timeframe : str, default 'H4'
        Timeframe string.
    eps : float, default 1e-12
        Epsilon to avoid division by zero.

    Returns
    -------
    pd.DataFrame
        DataFrame containing target series.
    """
    if horizons is None:
        horizons = DEFAULT_H4_HORIZONS if timeframe == "H4" else DEFAULT_D1_HORIZONS

    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)

    # Point-in-time ATR-14 for volatility scaling
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    atr_norm = atr_14 / (close + eps)

    targets = pd.DataFrame(index=df.index)
    returns_df = compute_swing_returns(df, horizons)

    for h in horizons:
        ret = returns_df[f"future_return_{h}"]
        targets[f"future_return_{h}"] = ret

        # 1. Unthresholded Binary Direction (exact 0 excluded)
        dir_series = pd.Series(np.nan, index=df.index, dtype=np.float64)
        dir_series[ret > 0.0] = 1.0
        dir_series[ret < 0.0] = -1.0
        targets[f"direction_{h}"] = dir_series

        # 2. Volatility-Scaled Dynamic Direction (0.5 * sqrt(h) * ATR_norm)
        vol_th = 0.5 * np.sqrt(h) * atr_norm
        vol_dir = pd.Series(np.nan, index=df.index, dtype=np.float64)
        valid_vol = ret.notna() & vol_th.notna()
        vol_dir[valid_vol & (ret > vol_th)] = 1.0
        vol_dir[valid_vol & (ret < -vol_th)] = -1.0
        targets[f"direction_vol_{h}"] = vol_dir

    return targets


def summarize_swing_target_distribution(
    targets_df: pd.DataFrame,
    horizons: list[int],
    total_bars: int,
) -> dict[str, Any]:
    """Compute summary statistics for swing target distributions."""
    summary = {}
    for h in horizons:
        # Binary target stats
        s_bin = targets_df[f"direction_{h}"]
        ret = targets_df[f"future_return_{h}"]

        valid_bin = s_bin.notna()
        n_bin = int(valid_bin.sum())
        l_bin = int((s_bin == 1.0).sum())
        s_cnt_bin = int((s_bin == -1.0).sum())

        # Volatility-adjusted target stats
        s_vol = targets_df[f"direction_vol_{h}"]
        valid_vol = s_vol.notna()
        n_vol = int(valid_vol.sum())
        l_vol = int((s_vol == 1.0).sum())
        s_cnt_vol = int((s_vol == -1.0).sum())
        n_excl_vol = total_bars - n_vol

        ret_valid = ret[valid_bin]
        abs_ret = ret_valid.abs()

        summary[f"horizon_{h}"] = {
            "binary_direction": {
                "sample_size": n_bin,
                "long_count": l_bin,
                "short_count": s_cnt_bin,
                "long_pct": float(l_bin / n_bin * 100.0) if n_bin > 0 else 0.0,
                "short_pct": float(s_cnt_bin / n_bin * 100.0) if n_bin > 0 else 0.0,
            },
            "volatility_adjusted": {
                "sample_size": n_vol,
                "excluded_count": n_excl_vol,
                "pct_excluded": float(n_excl_vol / total_bars * 100.0) if total_bars > 0 else 0.0,
                "long_count": l_vol,
                "short_count": s_cnt_vol,
                "long_pct": float(l_vol / n_vol * 100.0) if n_vol > 0 else 0.0,
                "short_pct": float(s_cnt_vol / n_vol * 100.0) if n_vol > 0 else 0.0,
            },
            "return_statistics": {
                "mean_return_bps": float(ret_valid.mean() * 10000.0) if n_bin > 0 else 0.0,
                "median_return_bps": float(ret_valid.median() * 10000.0) if n_bin > 0 else 0.0,
                "mean_abs_pips": float(abs_ret.mean() * 10000.0) if n_bin > 0 else 0.0,
                "median_abs_pips": float(abs_ret.median() * 10000.0) if n_bin > 0 else 0.0,
            },
        }
    return summary
