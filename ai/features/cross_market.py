"""Phase 18: Cross-market feature engineering and quality filtering engine.

Constructs 40 point-in-time cross-market features across 6 functional groups:
Group A: Cross-Market Returns (12 features)
Group B: Relative USD Strength Proxies (5 features)
Group C: Cross-Market Momentum (9 features)
Group D: Cross-Market Volatility (5 features)
Group E: Cross-Market Correlation (6 features)
Group F: Relative Volatility (3 features)

Guarantees strict point-in-time alignment with EURUSD H4 decision moments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CROSS_DATA_DIR = Path("data/processed/cross_market_h4")

# Feature group definitions for controlled ablation
CROSS_MARKET_FEATURE_GROUPS: dict[str, list[str]] = {
    "Group_A_returns": [
        "gbpusd_return_1",
        "gbpusd_return_3",
        "gbpusd_return_6",
        "gbpusd_return_12",
        "usdjpy_return_1",
        "usdjpy_return_3",
        "usdjpy_return_6",
        "usdjpy_return_12",
        "eurgbp_return_1",
        "eurgbp_return_3",
        "eurgbp_return_6",
        "eurgbp_return_12",
    ],
    "Group_B_usd_proxy": [
        "usd_strength_proxy_1",
        "usd_strength_proxy_4",
        "usd_strength_proxy_8",
        "usd_strength_proxy_12",
        "usd_strength_proxy_norm_8",
    ],
    "Group_C_momentum": [
        "gbpusd_mom_4",
        "gbpusd_mom_8",
        "gbpusd_mom_12",
        "usdjpy_mom_4",
        "usdjpy_mom_8",
        "usdjpy_mom_12",
        "eurgbp_mom_4",
        "eurgbp_mom_8",
        "eurgbp_mom_12",
    ],
    "Group_D_volatility": [
        "gbpusd_atr_norm_14",
        "usdjpy_atr_norm_14",
        "eurgbp_atr_norm_14",
        "gbpusd_vol_std_20",
        "usdjpy_vol_std_20",
    ],
    "Group_E_correlation": [
        "corr_eurusd_gbpusd_20",
        "corr_eurusd_gbpusd_40",
        "corr_eurusd_usdjpy_20",
        "corr_eurusd_usdjpy_40",
        "corr_eurusd_eurgbp_20",
        "corr_eurusd_eurgbp_40",
    ],
    "Group_F_relative_vol": [
        "rel_vol_eur_gbp",
        "rel_vol_eur_jpy",
        "rel_vol_eur_eurgbp",
    ],
}

ALL_CROSS_MARKET_FEATURES: list[str] = [
    feat for group in CROSS_MARKET_FEATURE_GROUPS.values() for feat in group
]

# Canonical 32 EURUSD swing features (Phase 17/18 baseline definition)
EURUSD_BASELINE_32_COLS: list[str] = [
    "return_1",
    "return_2",
    "return_4",
    "return_8",
    "hl_range_norm",
    "candle_body_norm",
    "dist_sma_10",
    "dist_sma_20",
    "dist_sma_50",
    "sma_slope_10",
    "sma_slope_20",
    "ema_spread_10_20",
    "trend_regime",
    "roc_5",
    "roc_10",
    "roc_20",
    "rsi_14",
    "momentum_acceleration",
    "atr_14",
    "atr_norm_14",
    "vol_std_10",
    "vol_std_20",
    "vol_ratio_5_20",
    "dist_rolling_high_10",
    "dist_rolling_low_10",
    "dist_rolling_high_20",
    "dist_rolling_low_20",
    "channel_width_20",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_hour",
    "cos_hour",
]


def get_eurusd_baseline_features(df_h4: pd.DataFrame) -> pd.DataFrame:
    """Compute the canonical 32 EURUSD swing features."""
    from ai.features.swing import compute_swing_features

    feats_all = compute_swing_features(df_h4, timeframe="H4")
    return feats_all[EURUSD_BASELINE_32_COLS].copy()


def _compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (Wilder smoothing)."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def load_and_align_cross_market_data(
    eurusd_h4: pd.DataFrame,
    data_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load and align verified cross-market H4 datasets to EURUSD H4 timestamps."""
    if data_dir is None:
        data_dir = CROSS_DATA_DIR

    symbols = ["gbpusd", "usdjpy", "eurgbp", "usdchf", "audusd", "usdcad"]
    aligned_dfs: dict[str, pd.DataFrame] = {}

    eur_ts = eurusd_h4[["timestamp"]].copy().reset_index(drop=True)

    for sym in symbols:
        p = data_dir / f"{sym}_h4_processed.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing cross-market processed data: {p}")
        df_sym = pd.read_parquet(p)
        df_sym["timestamp"] = pd.to_datetime(df_sym["timestamp"], utc=True)
        # Sort and deduplicate
        df_sym = (
            df_sym.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Merge with EURUSD on timestamp (outer alignment, then forward-fill any rare missing bars)
        # Note: forward-fill only carries the last known completed bar from the past
        m = pd.merge(eur_ts, df_sym, on="timestamp", how="left")
        # Forward fill price fields if any single bar was missing
        for col in ["open", "high", "low", "close"]:
            m[col] = m[col].ffill()
        for col in ["tick_volume", "spread", "real_volume"]:
            m[col] = m[col].fillna(0)

        aligned_dfs[sym] = m

    return aligned_dfs


def compute_cross_market_features(
    eurusd_h4: pd.DataFrame,
    cross_dfs: dict[str, pd.DataFrame] | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute exactly 40 point-in-time cross-market features aligned with EURUSD H4.

    Args:
        eurusd_h4: EURUSD H4 DataFrame with at least ['timestamp', 'close', 'high', 'low'].
        cross_dfs: Pre-loaded cross-market DataFrames dictionary (optional).
        data_dir: Directory containing processed cross-market parquet files (optional).

    Returns:
        DataFrame containing exactly 40 cross-market feature columns.
    """
    if cross_dfs is None:
        cross_dfs = load_and_align_cross_market_data(eurusd_h4, data_dir=data_dir)

    feats = pd.DataFrame(index=eurusd_h4.index)

    # Pre-extract closes and returns
    c_eur = eurusd_h4["close"].to_numpy()
    ret_eur_1 = pd.Series(c_eur).pct_change(1)

    c_gbp = cross_dfs["gbpusd"]["close"]
    c_jpy = cross_dfs["usdjpy"]["close"]
    c_egb = cross_dfs["eurgbp"]["close"]
    c_chf = cross_dfs["usdchf"]["close"]
    c_aud = cross_dfs["audusd"]["close"]
    c_cad = cross_dfs["usdcad"]["close"]

    ret_gbp_1 = c_gbp.pct_change(1)
    ret_jpy_1 = c_jpy.pct_change(1)
    ret_egb_1 = c_egb.pct_change(1)
    ret_chf_1 = c_chf.pct_change(1)
    ret_aud_1 = c_aud.pct_change(1)
    ret_cad_1 = c_cad.pct_change(1)

    # -------------------------------------------------------------------------
    # GROUP A: CROSS-MARKET RETURNS (12 features)
    # -------------------------------------------------------------------------
    for k in [1, 3, 6, 12]:
        feats[f"gbpusd_return_{k}"] = c_gbp.pct_change(k)
        feats[f"usdjpy_return_{k}"] = c_jpy.pct_change(k)
        feats[f"eurgbp_return_{k}"] = c_egb.pct_change(k)

    # -------------------------------------------------------------------------
    # GROUP B: RELATIVE USD STRENGTH PROXIES (5 features)
    # -------------------------------------------------------------------------
    # Synthetic basket of USD strength across 6 major pairs:
    # Quote USD (EUR, GBP, AUD): inverse return
    # Base USD (JPY, CHF, CAD): direct return
    for k in [1, 4, 8, 12]:
        r_eur_k = pd.Series(c_eur).pct_change(k)
        r_gbp_k = c_gbp.pct_change(k)
        r_aud_k = c_aud.pct_change(k)
        r_jpy_k = c_jpy.pct_change(k)
        r_chf_k = c_chf.pct_change(k)
        r_cad_k = c_cad.pct_change(k)

        # Average USD performance
        usd_proxy = (-r_eur_k - r_gbp_k - r_aud_k + r_jpy_k + r_chf_k + r_cad_k) / 6.0
        feats[f"usd_strength_proxy_{k}"] = usd_proxy

    # Volatility-normalized 8-bar USD proxy
    usd_ret_1 = (-ret_eur_1 - ret_gbp_1 - ret_aud_1 + ret_jpy_1 + ret_chf_1 + ret_cad_1) / 6.0
    usd_vol_20 = usd_ret_1.rolling(20, min_periods=10).std()
    feats["usd_strength_proxy_norm_8"] = feats["usd_strength_proxy_8"] / (usd_vol_20 + 1e-8)

    # -------------------------------------------------------------------------
    # GROUP C: CROSS-MARKET MOMENTUM (9 features)
    # -------------------------------------------------------------------------
    # Normalized distance to rolling SMA: (Close - SMA_k) / SMA_k
    for k in [4, 8, 12]:
        sma_gbp = c_gbp.rolling(k, min_periods=k).mean()
        sma_jpy = c_jpy.rolling(k, min_periods=k).mean()
        sma_egb = c_egb.rolling(k, min_periods=k).mean()

        feats[f"gbpusd_mom_{k}"] = (c_gbp - sma_gbp) / sma_gbp
        feats[f"usdjpy_mom_{k}"] = (c_jpy - sma_jpy) / sma_jpy
        feats[f"eurgbp_mom_{k}"] = (c_egb - sma_egb) / sma_egb

    # -------------------------------------------------------------------------
    # GROUP D: CROSS-MARKET VOLATILITY (5 features)
    # -------------------------------------------------------------------------
    # Normalized ATR: ATR_14 / Close
    atr_gbp = _compute_atr_series(cross_dfs["gbpusd"], period=14)
    atr_jpy = _compute_atr_series(cross_dfs["usdjpy"], period=14)
    atr_egb = _compute_atr_series(cross_dfs["eurgbp"], period=14)

    feats["gbpusd_atr_norm_14"] = atr_gbp / c_gbp
    feats["usdjpy_atr_norm_14"] = atr_jpy / c_jpy
    feats["eurgbp_atr_norm_14"] = atr_egb / c_egb

    # Return standard deviation
    feats["gbpusd_vol_std_20"] = ret_gbp_1.rolling(20, min_periods=10).std()
    feats["usdjpy_vol_std_20"] = ret_jpy_1.rolling(20, min_periods=10).std()

    # -------------------------------------------------------------------------
    # GROUP E: CROSS-MARKET CORRELATION (6 features)
    # -------------------------------------------------------------------------
    # Rolling Pearson correlation with EURUSD returns
    for w in [20, 40]:
        feats[f"corr_eurusd_gbpusd_{w}"] = ret_eur_1.rolling(w, min_periods=w).corr(ret_gbp_1)
        feats[f"corr_eurusd_usdjpy_{w}"] = ret_eur_1.rolling(w, min_periods=w).corr(ret_jpy_1)
        feats[f"corr_eurusd_eurgbp_{w}"] = ret_eur_1.rolling(w, min_periods=w).corr(ret_egb_1)

    # -------------------------------------------------------------------------
    # GROUP F: RELATIVE VOLATILITY (3 features)
    # -------------------------------------------------------------------------
    # EURUSD normalized ATR
    atr_eur = _compute_atr_series(eurusd_h4, period=14)
    atr_eur_norm = atr_eur / pd.Series(c_eur)

    feats["rel_vol_eur_gbp"] = atr_eur_norm / (feats["gbpusd_atr_norm_14"] + 1e-8)
    feats["rel_vol_eur_jpy"] = atr_eur_norm / (feats["usdjpy_atr_norm_14"] + 1e-8)
    feats["rel_vol_eur_eurgbp"] = atr_eur_norm / (feats["eurgbp_atr_norm_14"] + 1e-8)

    # Ensure column order matches ALL_CROSS_MARKET_FEATURES
    feats = feats[ALL_CROSS_MARKET_FEATURES].copy()
    return feats


def generate_feature_quality_report(
    features_df: pd.DataFrame,
    timestamps: pd.Series,
) -> list[dict[str, Any]]:
    """Generate detailed audit metadata for every feature to enforce quality filter."""
    report: list[dict[str, Any]] = []

    for col in features_df.columns:
        s = features_df[col]
        n_total = len(s)
        n_nan = int(s.isna().sum())
        nan_pct = float(n_nan / n_total * 100.0) if n_total > 0 else 0.0
        n_inf = int(np.isinf(s.to_numpy()).sum())

        valid_mask = s.notna() & ~np.isinf(s)
        variance = float(s[valid_mask].var()) if valid_mask.sum() > 1 else 0.0

        earliest_valid_ts = (
            timestamps[valid_mask].iloc[0].isoformat() if valid_mask.sum() > 0 else "NONE"
        )
        latest_valid_ts = (
            timestamps[valid_mask].iloc[-1].isoformat() if valid_mask.sum() > 0 else "NONE"
        )

        # Source instrument determination
        if col.startswith("gbpusd"):
            source = "GBPUSD"
        elif col.startswith("usdjpy"):
            source = "USDJPY"
        elif col.startswith("eurgbp"):
            source = "EURGBP"
        elif "usd_strength" in col:
            source = "USD_BASKET_6"
        elif "corr_eurusd" in col:
            source = "EURUSD_CROSS"
        elif "rel_vol" in col:
            source = "EURUSD_CROSS_RATIO"
        else:
            source = "EURUSD"

        # Lookback estimation
        lookback = 1
        for part in col.split("_"):
            if part.isdigit():
                lookback = max(lookback, int(part))

        # Rejection criteria: constant, >25% NaN, infs present, or no variance
        rejected = bool(nan_pct > 25.0 or n_inf > 0 or variance < 1e-12)

        report.append(
            {
                "name": col,
                "source_instrument": source,
                "lookback_bars": lookback,
                "missing_count": n_nan,
                "nan_percentage": nan_pct,
                "infinite_count": n_inf,
                "variance": variance,
                "earliest_valid_timestamp": earliest_valid_ts,
                "latest_valid_timestamp": latest_valid_ts,
                "status": "REJECTED" if rejected else "APPROVED",
            }
        )

    return report
