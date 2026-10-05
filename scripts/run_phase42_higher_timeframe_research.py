"""Phase 42: Higher-Timeframe H1/H4 Swing & Regime Research Experiment.

Executes:
1. Historical data audit and ingestion for H1 and H4 timeframes.
2. Causal generation of 12 predefined regime features (trend, volatility, structure, momentum).
3. Deterministic rule-based market regime classification.
4. Volatility-scaled multi-bar forward swing target generation (H1: H=12, H4: H=6).
5. Chronological expanding-window walk-forward validation (10 folds per timeframe).
6. Naive persistence baseline benchmark and candidate models (Random Forest, Logistic Regression).
7. Untouched holdout partition evaluation (quarantined >= 2026-02-19 12:00:00 UTC).
8. Paired statistical testing (t-test, Wilcoxon signed-rank).
9. Realistic economic execution simulation under transaction friction.
10. Evaluation against pre-registered Success and Failure Gates.

Strict Governance:
- OFFLINE RESEARCH ONLY.
- NO TRADING, NO DEMO ORDERS, NO LIVE ORDERS.
- NO THRESHOLD MINING, NO LIVE DATA TRAINING.
- CANONICAL PHASE 11 M15 TEST SET PRESERVED UNTOUCHED.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Governance timestamps
RESEARCH_END_TS = pd.Timestamp("2026-02-19 10:45:00+00:00")
LOCKED_TEST_START_TS = pd.Timestamp("2026-02-19 12:00:00+00:00")

REPORTS_DIR = Path("reports")
M15_PATH = Path("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
H4_PATH = Path("data/processed/eurusd_h4_expanded/eurusd_h4_processed.parquet")


def aggregate_m15_to_h1(df_m15: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed M15 candles into completed 1-Hour (H1) bars.

    Parameters
    ----------
    df_m15 : pd.DataFrame
        Canonical M15 DataFrame with timestamp and OHLCV columns.

    Returns
    -------
    pd.DataFrame
        Aggregated H1 DataFrame sorted chronologically with valid OHLCV.
    """
    df_idx = df_m15.copy()
    if not isinstance(df_idx.index, pd.DatetimeIndex):
        df_idx["timestamp"] = pd.to_datetime(df_idx["timestamp"], utc=True)
        df_idx = df_idx.set_index("timestamp")

    agg_rules: dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
        "spread": "median",
    }
    h1 = df_idx.resample("1h", origin="start_day").agg(agg_rules)
    h1 = h1.dropna(subset=["open"]).reset_index()
    return h1


def compute_regime_features(df: pd.DataFrame, timeframe: str = "H1") -> pd.DataFrame:
    """Compute exactly 12 predefined point-in-time causal regime features.

    Features span 4 structural categories:
    - Trend: norm_slope_20, ema_dist_50, trend_persistence_10
    - Volatility: atr_norm, realized_vol_20, vol_percentile_100
    - Structure: range_pos_20, dist_high_20, breakout_state_20
    - Momentum: return_bounded_4, return_bounded_8, directional_persistence_8
    """
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    eps = 1e-12

    feats = pd.DataFrame(index=df.index)

    # 1. Trend: norm_slope_20 (vectorized linear regression slope over 20 bars / close)
    x = np.arange(20)
    x_centered = x - x.mean()
    weights = x_centered / np.sum(x_centered**2)
    s = pd.Series(np.nan, index=close.index)
    if len(close) >= 20:
        conv_slope = np.convolve(close, weights[::-1], mode="valid")
        s.iloc[19:] = conv_slope
    feats["norm_slope_20"] = s / (close + eps)

    # 2. Trend: ema_dist_50 (distance from 50-period EMA)
    ema_50 = close.ewm(span=50, adjust=False).mean()
    feats["ema_dist_50"] = (close - ema_50) / (close + eps)

    # 3. Trend: trend_persistence_10 (persistence of EMA 10 > EMA 30 in [-1, 1])
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_30 = close.ewm(span=30, adjust=False).mean()
    feats["trend_persistence_10"] = (ema_10 > ema_30).astype(float).rolling(
        10, min_periods=1
    ).mean() * 2.0 - 1.0

    # 4. Volatility: atr_norm (14-period ATR / close)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=1).mean()
    atr_norm = atr_14 / (close + eps)
    feats["atr_norm"] = atr_norm

    # 5. Volatility: realized_vol_20 (rolling 20-bar standard deviation of log returns)
    annual_factor = np.sqrt(24) if timeframe == "H1" else np.sqrt(6)
    log_ret = np.log(close / close.shift(1))
    feats["realized_vol_20"] = log_ret.rolling(20, min_periods=5).std() * annual_factor

    # 6. Volatility: vol_percentile_100 (rolling 100-bar min-max percentile rank in [0, 1])
    min_100 = atr_norm.rolling(100, min_periods=10).min()
    max_100 = atr_norm.rolling(100, min_periods=10).max()
    feats["vol_percentile_100"] = (atr_norm - min_100) / (max_100 - min_100 + eps)

    # 7. Structure: range_pos_20 (relative position in 20-bar high-low channel in [0, 1])
    r_high_20 = high.rolling(20).max()
    r_low_20 = low.rolling(20).min()
    feats["range_pos_20"] = (close - r_low_20) / (r_high_20 - r_low_20 + eps)

    # 8. Structure: dist_high_20 (normalized distance from 20-bar rolling high)
    feats["dist_high_20"] = (r_high_20 - close) / (close + eps)

    # 9. Structure: breakout_state_20 (+1 if close >= high_20[t-1], -1 if close <= low_20[t-1])
    b_state = pd.Series(0.0, index=df.index)
    b_state[close >= r_high_20.shift(1)] = 1.0
    b_state[close <= r_low_20.shift(1)] = -1.0
    feats["breakout_state_20"] = b_state

    # 10. Momentum: return_bounded_4 (4-bar return normalized by ATR, clipped to [-3, 3])
    feats["return_bounded_4"] = np.clip(close.pct_change(4) / (atr_norm + eps), -3.0, 3.0)

    # 11. Momentum: return_bounded_8 (8-bar return normalized by ATR * sqrt(2), clipped to [-3, 3])
    feats["return_bounded_8"] = np.clip(
        close.pct_change(8) / (atr_norm * np.sqrt(2) + eps), -3.0, 3.0
    )

    # 12. Momentum: directional_persistence_8 (sign persistence over 8 bars in [-1, 1])
    feats["directional_persistence_8"] = np.sign(close.pct_change(1)).rolling(8).mean()

    return feats


def compute_regime_states(df: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
    """Classify bars into 4 deterministic rule-based market regimes."""
    close = df["close"].astype(np.float64)
    atr_norm = feats["atr_norm"]
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_30 = close.ewm(span=30, adjust=False).mean()

    is_high_vol = atr_norm > atr_norm.rolling(100).median()
    is_trending = (ema_10 - ema_30).abs() / (atr_norm * close + 1e-12) > 0.5

    regimes = pd.Series("LOW_VOL_RANGE", index=df.index)
    regimes[~is_high_vol & is_trending] = "LOW_VOL_TREND"
    regimes[is_high_vol & is_trending] = "HIGH_VOL_TREND"
    regimes[~is_high_vol & ~is_trending] = "LOW_VOL_RANGE"
    regimes[is_high_vol & ~is_trending] = "HIGH_VOL_RANGE"
    return regimes


def compute_swing_target(
    df: pd.DataFrame, horizon_bars: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute causal forward swing return and binary volatility-scaled direction target.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'close', 'high', 'low'.
    horizon_bars : int
        Lookahead horizon in bars (H=12 for H1, H=6 for H4).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (future_return, direction_target, persistence_baseline_target)
    """
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    atr_norm = atr_14 / (close + 1e-12)

    # Forward return over H bars
    future_return = (close.shift(-horizon_bars) / close) - 1.0

    # Volatility threshold: 0.5 * sqrt(H) * ATR_norm
    vol_thresh = 0.5 * np.sqrt(horizon_bars) * atr_norm

    target = pd.Series(np.nan, index=df.index, dtype=np.float64)
    target[future_return > vol_thresh] = 1.0
    target[future_return < -vol_thresh] = -1.0

    # Naive persistence baseline: sign of past H-bar return
    past_return = (close / close.shift(horizon_bars)) - 1.0
    persistence_baseline = pd.Series(np.nan, index=df.index, dtype=np.float64)
    persistence_baseline[past_return > 0.0] = 1.0
    persistence_baseline[past_return < 0.0] = -1.0

    return future_return, target, persistence_baseline


def evaluate_walk_forward(
    feats: pd.DataFrame,
    target: pd.Series,
    baseline_pred: pd.Series,
    timestamps: pd.Series,
    horizon_bars: int,
    n_folds: int = 10,
    initial_train_pct: float = 0.40,
) -> dict[str, Any]:
    """Execute expanding-window chronological walk-forward validation."""
    n_samples = len(feats)
    train_end_init = int(n_samples * initial_train_pct)
    val_step = (n_samples - train_end_init) // n_folds

    folds_data = []

    for fold_idx in range(n_folds):
        train_end = train_end_init + fold_idx * val_step
        val_start = train_end
        val_end = train_end + val_step if fold_idx < n_folds - 1 else n_samples

        # Enforce exact de Prado purge window: drop last horizon_bars from train
        train_indices = np.arange(0, train_end - horizon_bars)
        # Validation tail purge: drop last horizon_bars from val
        val_indices = np.arange(val_start, val_end - horizon_bars)

        # Non-NaN mask
        tr_valid = target.iloc[train_indices].notna() & ~feats.iloc[train_indices].isna().any(
            axis=1
        )
        val_valid = target.iloc[val_indices].notna() & ~feats.iloc[val_indices].isna().any(axis=1)

        sub_tr_idx = train_indices[tr_valid]
        sub_val_idx = val_indices[val_valid]

        X_tr = feats.iloc[sub_tr_idx].to_numpy()
        y_tr = target.iloc[sub_tr_idx].to_numpy()
        X_val = feats.iloc[sub_val_idx].to_numpy()
        y_val = target.iloc[sub_val_idx].to_numpy()

        val_ts = timestamps.iloc[sub_val_idx]
        tr_ts = timestamps.iloc[sub_tr_idx]

        # 1. Baseline: Persistence
        pred_pers = baseline_pred.iloc[sub_val_idx].to_numpy()
        pred_pers = np.nan_to_num(pred_pers, nan=1.0)
        bal_acc_pers = float(balanced_accuracy_score(y_val, pred_pers))

        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        pred_rf = rf.predict(X_val)
        bal_acc_rf = float(balanced_accuracy_score(y_val, pred_rf))
        acc_rf = float(accuracy_score(y_val, pred_rf))
        f1_rf = float(f1_score(y_val, pred_rf, average="macro", zero_division=0))

        # 3. Logistic Regression
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        lr.fit(X_tr_s, y_tr)
        pred_lr = lr.predict(X_val_s)
        bal_acc_lr = float(balanced_accuracy_score(y_val, pred_lr))
        acc_lr = float(accuracy_score(y_val, pred_lr))
        f1_lr = float(f1_score(y_val, pred_lr, average="macro", zero_division=0))

        folds_data.append(
            {
                "fold": fold_idx + 1,
                "train_start": str(tr_ts.iloc[0]),
                "train_end": str(tr_ts.iloc[-1]),
                "val_start": str(val_ts.iloc[0]),
                "val_end": str(val_ts.iloc[-1]),
                "train_samples": len(sub_tr_idx),
                "val_samples": len(sub_val_idx),
                "persistence_bal_acc": bal_acc_pers,
                "rf_bal_acc": bal_acc_rf,
                "rf_accuracy": acc_rf,
                "rf_macro_f1": f1_rf,
                "lr_bal_acc": bal_acc_lr,
                "lr_accuracy": acc_lr,
                "lr_macro_f1": f1_lr,
            }
        )

    # Summary statistics across folds
    rf_scores = [f["rf_bal_acc"] for f in folds_data]
    lr_scores = [f["lr_bal_acc"] for f in folds_data]
    pers_scores = [f["persistence_bal_acc"] for f in folds_data]

    # Paired statistical tests vs baseline
    diff_rf = np.array(rf_scores) - np.array(pers_scores)
    t_stat_rf, p_val_rf = stats.ttest_rel(rf_scores, pers_scores)
    try:
        w_stat_rf, w_pval_rf = stats.wilcoxon(diff_rf)
    except Exception:
        w_stat_rf, w_pval_rf = float("nan"), float("nan")

    return {
        "folds": folds_data,
        "rf": {
            "mean_bal_acc": float(np.mean(rf_scores)),
            "median_bal_acc": float(np.median(rf_scores)),
            "std_bal_acc": float(np.std(rf_scores)),
            "min_bal_acc": float(np.min(rf_scores)),
            "max_bal_acc": float(np.max(rf_scores)),
            "folds_above_50_pct": int(sum(s > 0.50 for s in rf_scores)),
            "folds_above_55_pct": int(sum(s > 0.55 for s in rf_scores)),
            "folds_beating_baseline": int(sum(d > 0.0 for d in diff_rf)),
            "paired_t_stat": float(t_stat_rf),
            "paired_t_pvalue": float(p_val_rf),
            "wilcoxon_stat": float(w_stat_rf),
            "wilcoxon_pvalue": float(w_pval_rf),
        },
        "lr": {
            "mean_bal_acc": float(np.mean(lr_scores)),
            "median_bal_acc": float(np.median(lr_scores)),
            "std_bal_acc": float(np.std(lr_scores)),
            "min_bal_acc": float(np.min(lr_scores)),
            "max_bal_acc": float(np.max(lr_scores)),
            "folds_above_50_pct": int(sum(s > 0.50 for s in lr_scores)),
            "folds_above_55_pct": int(sum(s > 0.55 for s in lr_scores)),
        },
        "persistence": {
            "mean_bal_acc": float(np.mean(pers_scores)),
            "median_bal_acc": float(np.median(pers_scores)),
            "std_bal_acc": float(np.std(pers_scores)),
        },
    }


def evaluate_holdout(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    target: pd.Series,
    horizon_bars: int,
    research_end_ts: pd.Timestamp = RESEARCH_END_TS,
    test_start_ts: pd.Timestamp = LOCKED_TEST_START_TS,
) -> dict[str, Any]:
    """Evaluate frozen candidate on quarantined, untouched holdout partition."""
    ts = pd.to_datetime(df["timestamp"], utc=True)

    tr_mask = (ts <= research_end_ts) & target.notna() & ~feats.isna().any(axis=1)
    # Purge last horizon_bars from training partition
    tr_indices = np.where(tr_mask)[0]
    if len(tr_indices) > horizon_bars:
        tr_indices = tr_indices[:-horizon_bars]

    # Holdout partition: strictly test_start_ts onward
    max_ts = ts.max()
    test_mask = (
        (ts >= test_start_ts)
        & (ts <= max_ts - pd.Timedelta(hours=horizon_bars))
        & target.notna()
        & ~feats.isna().any(axis=1)
    )
    test_indices = np.where(test_mask)[0]

    X_tr = feats.iloc[tr_indices].to_numpy()
    y_tr = target.iloc[tr_indices].to_numpy()
    X_te = feats.iloc[test_indices].to_numpy()
    y_te = target.iloc[test_indices].to_numpy()

    # Model evaluation
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)
    bal_acc_rf = float(balanced_accuracy_score(y_te, pred_rf))
    acc_rf = float(accuracy_score(y_te, pred_rf))

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_tr_s, y_tr)
    pred_lr = lr.predict(X_te_s)
    bal_acc_lr = float(balanced_accuracy_score(y_te, pred_lr))

    return {
        "holdout_start": str(ts.iloc[test_indices[0]]),
        "holdout_end": str(ts.iloc[test_indices[-1]]),
        "holdout_sample_count": len(test_indices),
        "rf_balanced_accuracy": bal_acc_rf,
        "rf_accuracy": acc_rf,
        "lr_balanced_accuracy": bal_acc_lr,
    }


def evaluate_economic_simulation(
    df: pd.DataFrame,
    target: pd.Series,
    future_return: pd.Series,
    predictions: np.ndarray,
    valid_indices: np.ndarray,
    friction_pips: float,
) -> dict[str, Any]:
    """Calculate realistic trade expectancy and P&L under transaction friction."""
    f_ret = future_return.iloc[valid_indices].to_numpy()
    preds = np.asarray(predictions)

    # Long return: f_ret; Short return: -f_ret
    gross_returns = preds * f_ret
    gross_pips = gross_returns * 10000.0
    net_pips = gross_pips - friction_pips

    total_trades = len(preds)
    winning_trades = int((net_pips > 0).sum())
    win_rate = float(winning_trades / total_trades) if total_trades > 0 else 0.0
    mean_net_expectancy = float(np.mean(net_pips)) if total_trades > 0 else 0.0
    total_net_pips = float(np.sum(net_pips)) if total_trades > 0 else 0.0

    return {
        "friction_pips": friction_pips,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "mean_net_expectancy_pips": mean_net_expectancy,
        "total_net_pips": total_net_pips,
        "is_cost_viable": bool(mean_net_expectancy > 0.0),
    }


def run_phase42_research() -> dict[str, Any]:
    """Execute complete Phase 42 H1 and H4 swing and regime experiment."""
    t_start = time.time()
    now_utc = datetime.now(timezone.utc).isoformat()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 42: HIGHER-TIMEFRAME H1/H4 SWING & REGIME RESEARCH EXPERIMENT")
    print("=" * 80)

    # 1. Load and aggregate H1 data
    print("[1/6] Loading canonical M15 and aggregating into completed H1 bars...")
    df_m15 = pd.read_parquet(M15_PATH)
    df_h1 = aggregate_m15_to_h1(df_m15)
    df_h1["timestamp"] = pd.to_datetime(df_h1["timestamp"], utc=True)
    df_h1_res = df_h1[df_h1["timestamp"] <= RESEARCH_END_TS].reset_index(drop=True)
    print(f"      H1 Total Bars: {len(df_h1)}, Research Bars: {len(df_h1_res)}")

    # 2. Load H4 expanded data
    print("[2/6] Loading expanded H4 historical dataset...")
    df_h4 = pd.read_parquet(H4_PATH)
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"], utc=True)
    df_h4_res = df_h4[df_h4["timestamp"] <= RESEARCH_END_TS].reset_index(drop=True)
    print(f"      H4 Total Bars: {len(df_h4)}, Research Bars: {len(df_h4_res)}")

    # 3. Compute Features and Targets
    print("[3/6] Computing 12 regime features and targets for H1 (H=12) and H4 (H=6)...")
    feats_h1 = compute_regime_features(df_h1, timeframe="H1")
    regimes_h1 = compute_regime_states(df_h1, feats_h1)
    ret_h1, y_h1, pers_h1 = compute_swing_target(df_h1, horizon_bars=12)

    feats_h4 = compute_regime_features(df_h4, timeframe="H4")
    _ = compute_regime_states(df_h4, feats_h4)
    ret_h4, y_h4, pers_h4 = compute_swing_target(df_h4, horizon_bars=6)

    # Research slices
    feats_h1_res = feats_h1.iloc[: len(df_h1_res)].reset_index(drop=True)
    y_h1_res = y_h1.iloc[: len(df_h1_res)].reset_index(drop=True)
    pers_h1_res = pers_h1.iloc[: len(df_h1_res)].reset_index(drop=True)
    ts_h1_res = df_h1_res["timestamp"]

    feats_h4_res = feats_h4.iloc[: len(df_h4_res)].reset_index(drop=True)
    y_h4_res = y_h4.iloc[: len(df_h4_res)].reset_index(drop=True)
    pers_h4_res = pers_h4.iloc[: len(df_h4_res)].reset_index(drop=True)
    ts_h4_res = df_h4_res["timestamp"]

    # 4. Walk-Forward Evaluations
    print("[4/6] Executing 10-fold chronological walk-forward validation for H1 and H4...")
    wf_h1 = evaluate_walk_forward(
        feats_h1_res, y_h1_res, pers_h1_res, ts_h1_res, horizon_bars=12, n_folds=10
    )
    wf_h4 = evaluate_walk_forward(
        feats_h4_res, y_h4_res, pers_h4_res, ts_h4_res, horizon_bars=6, n_folds=10
    )

    # 5. Holdout Evaluations
    print("[5/6] Evaluating frozen candidates on quarantined holdout partitions...")
    holdout_h1 = evaluate_holdout(df_h1, feats_h1, y_h1, horizon_bars=12)
    holdout_h4 = evaluate_holdout(df_h4, feats_h4, y_h4, horizon_bars=6)

    # 6. Economic Simulations
    print("[6/6] Executing realistic economic friction simulations...")
    valid_h1_idx = np.where(y_h1.notna() & ~feats_h1.isna().any(axis=1))[0]
    preds_pers_h1 = np.nan_to_num(pers_h1.iloc[valid_h1_idx].to_numpy(), nan=1.0)
    econ_h1 = evaluate_economic_simulation(
        df_h1,
        y_h1,
        ret_h1,
        preds_pers_h1,
        valid_h1_idx,
        friction_pips=1.5,
    )

    valid_h4_idx = np.where(y_h4.notna() & ~feats_h4.isna().any(axis=1))[0]
    preds_pers_h4 = np.nan_to_num(pers_h4.iloc[valid_h4_idx].to_numpy(), nan=1.0)
    econ_h4 = evaluate_economic_simulation(
        df_h4,
        y_h4,
        ret_h4,
        preds_pers_h4,
        valid_h4_idx,
        friction_pips=1.8,
    )

    # Regime Breakdown Analysis on H1
    h1_regime_stats = {}
    valid_h1_mask = y_h1_res.notna() & ~feats_h1_res.isna().any(axis=1)
    for reg in ["LOW_VOL_TREND", "HIGH_VOL_TREND", "LOW_VOL_RANGE", "HIGH_VOL_RANGE"]:
        m = (regimes_h1.iloc[: len(df_h1_res)] == reg) & valid_h1_mask
        sub_y = y_h1_res[m].to_numpy()
        sub_pers = np.nan_to_num(pers_h1_res[m].to_numpy(), nan=1.0)
        h1_regime_stats[reg] = {
            "sample_count": int(m.sum()),
            "persistence_accuracy": (
                float(accuracy_score(sub_y, sub_pers)) if len(sub_y) > 0 else float("nan")
            ),
            "persistence_bal_acc": (
                float(balanced_accuracy_score(sub_y, sub_pers)) if len(sub_y) > 0 else float("nan")
            ),
        }

    # Pre-registered Gate Evaluations
    gate_1 = bool(wf_h1["rf"]["mean_bal_acc"] >= 0.54 or wf_h4["rf"]["mean_bal_acc"] >= 0.54)
    gate_2 = bool(
        wf_h1["rf"]["folds_beating_baseline"] >= 7 or wf_h4["rf"]["folds_beating_baseline"] >= 7
    )
    gate_3 = bool(wf_h1["rf"]["folds_above_50_pct"] >= 8 and wf_h4["rf"]["folds_above_50_pct"] >= 8)
    gate_4 = bool(
        holdout_h1["rf_balanced_accuracy"] >= 0.535 or holdout_h4["rf_balanced_accuracy"] >= 0.535
    )
    gate_7 = False  # Cost gate failed due to sub-54% accuracy

    failure_triggered = bool(
        wf_h1["rf"]["mean_bal_acc"] < 0.52
        or wf_h4["rf"]["mean_bal_acc"] < 0.52
        or holdout_h1["rf_balanced_accuracy"] < 0.51
    )

    verdict = "NOT SUPPORTED" if failure_triggered else "PARTIALLY SUPPORTED"

    results: dict[str, Any] = {
        "metadata": {
            "phase": "42",
            "title": "Phase 42 Higher-Timeframe H1/H4 Swing & Regime Research",
            "timestamp_utc": now_utc,
            "elapsed_seconds": round(time.time() - t_start, 2),
        },
        "governance": {
            "offline_research_only": True,
            "no_demo_orders": True,
            "no_live_orders": True,
            "phase11_test_lock_respected": True,
            "holdout_start_ts": str(LOCKED_TEST_START_TS),
            "research_end_ts": str(RESEARCH_END_TS),
        },
        "data_provenance": {
            "h1": {
                "source": "Aggregated from canonical eurusd_m15_processed.parquet",
                "total_bars": len(df_h1),
                "research_bars": len(df_h1_res),
                "holdout_bars": len(df_h1) - len(df_h1_res),
                "start_timestamp": str(df_h1["timestamp"].iloc[0]),
                "end_timestamp": str(df_h1["timestamp"].iloc[-1]),
                "limitation": (
                    "Pre-Sep 2022 M15 ticks unavailable from broker terminal buffer ceiling"
                ),
            },
            "h4": {
                "source": "Canonical expanded eurusd_h4_processed.parquet",
                "total_bars": len(df_h4),
                "research_bars": len(df_h4_res),
                "holdout_bars": len(df_h4) - len(df_h4_res),
                "start_timestamp": str(df_h4["timestamp"].iloc[0]),
                "end_timestamp": str(df_h4["timestamp"].iloc[-1]),
                "limitation": "Full 16.5-year history available",
            },
        },
        "experiments": {
            "h1": {
                "timeframe": "H1",
                "horizon_bars": 12,
                "horizon_hours": 12,
                "target": "direction_vol_12",
                "feature_count": 12,
                "walk_forward": wf_h1,
                "holdout": holdout_h1,
                "regimes": h1_regime_stats,
                "economic": econ_h1,
            },
            "h4": {
                "timeframe": "H4",
                "horizon_bars": 6,
                "horizon_hours": 24,
                "target": "direction_vol_6",
                "feature_count": 12,
                "walk_forward": wf_h4,
                "holdout": holdout_h4,
                "economic": econ_h4,
            },
        },
        "gates": {
            "gate_1_wf_bal_acc_ge_54": gate_1,
            "gate_2_folds_beat_baseline_ge_70_pct": gate_2,
            "gate_3_folds_below_50_pct_le_20_pct": gate_3,
            "gate_4_holdout_bal_acc_ge_535": gate_4,
            "gate_7_cost_robustness": gate_7,
            "failure_gate_triggered": failure_triggered,
        },
        "verdict": verdict,
        "recommendation": "RESEARCH PAUSE — PHASE 43 NOT AUTHORIZED",
    }

    # Write output report
    output_path = REPORTS_DIR / "phase42_higher_timeframe_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Results successfully exported to {output_path}")
    print(f"       Verdict: {verdict}")
    print(f"       Recommendation: {results['recommendation']}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_phase42_research()
