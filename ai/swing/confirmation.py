"""M15 entry confirmation filter research engine for H4 swing signals.

Implements point-in-time alignment between H4 decisions and strictly completed M15
candles (latest completed M15 timestamp <= H4 close - 15m < H4 close).
Evaluates 4 controlled confirmation rules without hyperparameter tuning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def align_m15_with_h4_decisions(
    h4_df: pd.DataFrame,
    m15_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align each H4 decision point with its latest completed M15 candle.

    At H4 bar labeled T_start, the candle closes at T_close = T_start + 4h.
    The latest completed M15 bar has timestamp T_close - 15m.
    This function strictly guarantees:
    M15 timestamp <= T_close - 15m < T_close.

    Parameters
    ----------
    h4_df : pd.DataFrame
        H4 DataFrame containing 'timestamp'.
    m15_df : pd.DataFrame
        M15 DataFrame containing 'timestamp', 'candle_direction', 'rsi_14', 'dist_ema_20'.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with H4 timestamps, matched M15 timestamps, and M15 confirmation features.
    """
    h4_work = h4_df[["timestamp"]].copy()
    h4_work["t_h4_start"] = pd.to_datetime(h4_work["timestamp"], utc=True).astype(
        "datetime64[us, UTC]"
    )
    h4_work["t_h4_close"] = h4_work["t_h4_start"] + pd.Timedelta(hours=4)
    h4_work["cutoff_m15"] = h4_work["t_h4_close"] - pd.Timedelta(minutes=15)
    h4_work = h4_work.sort_values("cutoff_m15").reset_index(drop=True)

    m15_work = m15_df[["timestamp", "candle_direction", "rsi_14", "dist_ema_20"]].copy()
    m15_work["timestamp"] = pd.to_datetime(m15_work["timestamp"], utc=True).astype(
        "datetime64[us, UTC]"
    )
    m15_work = m15_work.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        h4_work,
        m15_work,
        left_on="cutoff_m15",
        right_on="timestamp",
        direction="backward",
        suffixes=("_h4", "_m15"),
    )

    # Verification of zero lookahead for all matched bars
    # Any matched M15 bar must have timestamp <= cutoff_m15 < t_h4_close
    matched_mask = merged["timestamp_m15"].notna()
    valid_timing = (
        merged.loc[matched_mask, "timestamp_m15"] <= merged.loc[matched_mask, "cutoff_m15"]
    ) & (merged.loc[matched_mask, "timestamp_m15"] < merged.loc[matched_mask, "t_h4_close"])
    if not valid_timing.all():
        invalid_count = int((~valid_timing).sum())
        raise ValueError(
            f"M15 alignment timing violation: {invalid_count} bars violated causality."
        )

    # Rename and select output columns
    result = pd.DataFrame(
        {
            "h4_timestamp": merged["t_h4_start"],
            "h4_decision_time": merged["t_h4_close"],
            "cutoff_m15": merged["cutoff_m15"],
            "matched_m15_timestamp": merged["timestamp_m15"],
            "m15_candle_direction": merged["candle_direction"],
            "m15_rsi_14": merged["rsi_14"],
            "m15_dist_ema_20": merged["dist_ema_20"],
        }
    )
    return result


def evaluate_m15_confirmations(
    y_true: np.ndarray,
    h4_preds: np.ndarray,
    aligned_m15: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate candidate H4 signals with and without M15 confirmation filters.

    Evaluates:
    - BASE: H4 signal alone
    - CONFIRMATION A: Direction agreement (candle_direction agrees)
    - CONFIRMATION B: Momentum agreement (rsi_14 agrees: >50 for Long, <50 for Short)
    - CONFIRMATION C: Trend agreement (dist_ema_20 agrees: >0 for Long, <0 for Short)
    - CONFIRMATION D: 2-of-3 agreement (at least 2 of A, B, C agree)

    Parameters
    ----------
    y_true : np.ndarray
        True target labels.
    h4_preds : np.ndarray
        H4 model predicted directional labels (+1.0 or -1.0).
    aligned_m15 : pd.DataFrame
        Aligned M15 features for the corresponding evaluation bars.

    Returns
    -------
    dict[str, Any]
        Dictionary of performance metrics for base and filtered configurations.
    """
    n_base = len(h4_preds)
    if n_base == 0:
        raise ValueError("No predictions to evaluate.")

    m15_dir = aligned_m15["m15_candle_direction"].to_numpy()
    m15_rsi = aligned_m15["m15_rsi_14"].to_numpy()
    m15_ema = aligned_m15["m15_dist_ema_20"].to_numpy()

    # Rule A: Direction agreement
    mask_a = h4_preds == m15_dir

    # Rule B: Momentum agreement (RSI-14: >50 for Long, <50 for Short)
    mask_b = np.where(h4_preds == 1.0, m15_rsi > 50.0, m15_rsi < 50.0)

    # Rule C: Trend agreement (dist_ema_20: >0 for Long, <0 for Short)
    mask_c = np.where(h4_preds == 1.0, m15_ema > 0.0, m15_ema < 0.0)

    # Rule D: 2 of 3 agreement
    mask_d = (mask_a.astype(int) + mask_b.astype(int) + mask_c.astype(int)) >= 2

    rules = [
        ("Base (H4 Signal Alone)", np.ones(n_base, dtype=bool)),
        ("Confirmation A (M15 Direction)", mask_a),
        ("Confirmation B (M15 Momentum)", mask_b),
        ("Confirmation C (M15 Trend)", mask_c),
        ("Confirmation D (Two-of-Three)", mask_d),
    ]

    results: dict[str, Any] = {}

    for rule_name, mask in rules:
        n_rem = int(mask.sum())
        pct_filtered = float((1.0 - n_rem / n_base) * 100.0) if n_base > 0 else 0.0

        if n_rem == 0:
            results[rule_name] = {
                "signals_count": 0,
                "pct_filtered": pct_filtered,
                "status": "ZERO_SIGNALS_REMAINING",
            }
            continue

        sub_y = y_true[mask]
        sub_pred = h4_preds[mask]

        acc = float(accuracy_score(sub_y, sub_pred))
        bal_acc = float(balanced_accuracy_score(sub_y, sub_pred))
        macro_f1 = float(f1_score(sub_y, sub_pred, average="macro", zero_division=0))
        prec_short = float(precision_score(sub_y, sub_pred, pos_label=-1.0, zero_division=0))
        prec_long = float(precision_score(sub_y, sub_pred, pos_label=1.0, zero_division=0))
        rec_short = float(recall_score(sub_y, sub_pred, pos_label=-1.0, zero_division=0))
        rec_long = float(recall_score(sub_y, sub_pred, pos_label=1.0, zero_division=0))

        cm = confusion_matrix(sub_y, sub_pred, labels=[-1.0, 1.0])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        results[rule_name] = {
            "signals_count": n_rem,
            "pct_filtered": pct_filtered,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "precision_short": prec_short,
            "precision_long": prec_long,
            "recall_short": rec_short,
            "recall_long": rec_long,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    return results
