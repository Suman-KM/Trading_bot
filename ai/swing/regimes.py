"""Descriptive market regime diagnostics for H4 swing candidate.

Analyzes candidate out-of-sample walk-forward predictions across volatility,
trend, range, session, and day-of-week regimes without strategy optimization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def analyze_h4_regimes(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    oof_indices: list[int] | np.ndarray,
    oof_preds: list[float] | np.ndarray,
    oof_probs: list[list[float]] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate candidate out-of-sample performance across descriptive market regimes.

    Parameters
    ----------
    features_df : pd.DataFrame
        H4 features DataFrame.
    target_series : pd.Series
        Target labels (e.g. direction_vol_8).
    oof_indices : list[int] or np.ndarray
        Indices of out-of-sample evaluated H4 bars.
    oof_preds : list[float] or np.ndarray
        Out-of-sample predicted labels.
    oof_probs : list or np.ndarray, optional
        Out-of-sample predicted probabilities.

    Returns
    -------
    dict[str, Any]
        Dictionary of regime evaluations.
    """
    oof_idx = np.asarray(oof_indices)
    preds = np.asarray(oof_preds)
    y_true = target_series.iloc[oof_idx].to_numpy()
    feats = features_df.iloc[oof_idx].reset_index(drop=True)

    probs = np.asarray(oof_probs) if oof_probs is not None else None

    def evaluate_slice(mask: np.ndarray | pd.Series, name: str) -> dict[str, Any]:
        m = np.asarray(mask, dtype=bool)
        n = int(m.sum())
        if n == 0:
            return {"name": name, "sample_count": 0, "status": "NO_SAMPLES"}

        sub_y = y_true[m]
        sub_pred = preds[m]

        long_cnt = int((sub_y == 1.0).sum())
        short_cnt = int((sub_y == -1.0).sum())
        long_pct = float(long_cnt / n * 100.0) if n > 0 else 0.0

        # Unique classes present in slice
        classes_present = np.unique(sub_y)
        if len(classes_present) < 2:
            acc = float(accuracy_score(sub_y, sub_pred))
            return {
                "name": name,
                "sample_count": n,
                "long_count": long_cnt,
                "short_count": short_cnt,
                "long_pct": long_pct,
                "accuracy": acc,
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
                "roc_auc": float("nan"),
                "status": "SINGLE_CLASS_PRESENT",
            }

        acc = float(accuracy_score(sub_y, sub_pred))
        bal_acc = float(balanced_accuracy_score(sub_y, sub_pred))
        macro_f1 = float(f1_score(sub_y, sub_pred, average="macro", zero_division=0))

        auc = float("nan")
        if probs is not None and probs.shape[1] == 2:
            try:
                sub_prob_long = probs[m, 1]
                y_bin = np.where(sub_y == 1.0, 1, 0)
                auc = float(roc_auc_score(y_bin, sub_prob_long))
            except Exception:
                auc = float("nan")

        return {
            "name": name,
            "sample_count": n,
            "long_count": long_cnt,
            "short_count": short_cnt,
            "long_pct": long_pct,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "roc_auc": auc,
            "status": "VALID",
        }

    results: dict[str, Any] = {}

    # 1. Volatility Regime (ATR normalized)
    atr_norm = feats["atr_norm_14"]
    atr_med = float(atr_norm.median())
    results["volatility_regime"] = {
        "median_atr_norm": atr_med,
        "low_volatility": evaluate_slice(atr_norm <= atr_med, "Low ATR Volatility (<= median)"),
        "high_volatility": evaluate_slice(atr_norm > atr_med, "High ATR Volatility (> median)"),
    }

    # 2. Volatility Compression / Expansion (vol_ratio_5_20)
    if "vol_ratio_5_20" in feats.columns:
        vr = feats["vol_ratio_5_20"]
        results["volatility_expansion_regime"] = {
            "compression": evaluate_slice(vr <= 1.0, "Volatility Compression (ratio <= 1.0)"),
            "expansion": evaluate_slice(vr > 1.0, "Volatility Expansion (ratio > 1.0)"),
        }

    # 3. Trend Regime (trend_regime)
    if "trend_regime" in feats.columns:
        tr = feats["trend_regime"]
        results["trend_regime"] = {
            "strong_bullish": evaluate_slice(tr == 1.0, "Strong Bullish Trend (+1.0)"),
            "strong_bearish": evaluate_slice(tr == -1.0, "Strong Bearish Trend (-1.0)"),
            "mixed_transition": evaluate_slice(
                tr.abs() < 1.0, "Mixed / Transition (|trend| < 1.0)"
            ),
        }

    # 4. Donchian Channel Width (Ranging vs Breakout)
    if "channel_width_20" in feats.columns:
        cw = feats["channel_width_20"]
        cw_med = float(cw.median())
        results["channel_width_regime"] = {
            "narrow_channel": evaluate_slice(cw <= cw_med, "Narrow Channel (<= median)"),
            "wide_channel": evaluate_slice(cw > cw_med, "Wide Channel (> median)"),
        }

    # 5. Session / Hour of Day Context
    if "hour" in feats.columns:
        hour_results = {}
        for h in [0, 4, 8, 12, 16, 20]:
            h_mask = feats["hour"] == float(h)
            hour_results[f"hour_{h:02d}"] = evaluate_slice(h_mask, f"H4 Bar {h:02d}:00 UTC")
        results["session_hour_regime"] = hour_results

    # 6. Reopen / Day-of-Week
    if "day_of_week" in feats.columns:
        dow = feats["day_of_week"]
        results["day_of_week_regime"] = {
            "monday": evaluate_slice(dow == 0, "Monday (Market Reopen)"),
            "midweek": evaluate_slice((dow >= 1) & (dow <= 3), "Midweek (Tue-Thu)"),
            "friday": evaluate_slice(dow == 4, "Friday (Pre-Weekend)"),
        }

    return results
