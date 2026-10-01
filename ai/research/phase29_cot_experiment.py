"""Phase 29: Pre-Registered CFTC COT Positioning Experiment.

Evaluates whether weekly CFTC Commitments of Traders (COT) institutional positioning
extremes (speculative net Z-score, commercial positioning Z-score, 4-week speculative change)
provide incremental directional predictive value beyond the frozen EURUSD D1 price-only baseline.

Strict Governance:
- Pre-registered hypothesis and null hypothesis.
- Fixed baseline (30 D1 technical features).
- Exactly 3 COT candidate features.
- Two pre-registered target horizons: H = 10 and H = 20 business days (volatility-adjusted).
- Expanding chronological walk-forward validation across 10 folds (2010 to 2024).
- Purge buffer = H bars, Embargo buffer = 5 bars.
- Fixed RandomForestClassifier hyperparameters (FROZEN_RF_CONFIG).
- Quarantined test partition (2026-02-19 12:00 UTC onward) strictly untouched.
- Zero live/demo trading, zero parameter searches.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from ai.data.exogenous.cftc_alignment import align_cftc_to_eurusd
from ai.features.cross_market import EURUSD_BASELINE_32_COLS
from ai.features.swing import compute_swing_features
from ai.labels.swing import compute_swing_targets
from ai.swing.evaluation import FROZEN_RF_CONFIG
from ai.swing.holdout import LOCKED_TEST_START_TS
from ai.swing.walk_forward import WalkForwardFold

# Canonical file paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CFTC_PARQUET_PATH = PROJECT_ROOT / "data" / "exogenous" / "cftc" / "cftc_eurofx_cot.parquet"
EURUSD_H4_PATH = PROJECT_ROOT / "data" / "raw" / "eurusd_h4_expanded" / "eurusd_h4_raw.parquet"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "phase29_cot_experiment.json"

# Features specification
BASELINE_30_FEATURES: list[str] = EURUSD_BASELINE_32_COLS[:30]
APPROVED_COT_FEATURES: list[str] = [
    "speculative_net_zscore_3y",
    "commercial_position_zscore_3y",
    "speculative_net_4w_change",
]

# Validation parameters
PRE_HOLDOUT_CUTOFF = pd.Timestamp("2024-11-04 23:59:59+00:00")
INITIAL_TRAIN_BARS = 1000
N_FOLDS = 10
EMBARGO_BARS = 5
HORIZONS = [10, 20]


def aggregate_h4_to_d1(df_h4: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed EURUSD H4 bars into Daily (D1) bars (00:00:00 UTC).

    Parameters
    ----------
    df_h4 : pd.DataFrame
        H4 OHLCV DataFrame with UTC timestamp.

    Returns
    -------
    pd.DataFrame
        Aggregated D1 DataFrame.
    """
    df_idx = df_h4.copy()
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
    if "time" in df_idx.columns:
        agg_rules["time"] = "first"

    d1 = df_idx.resample("1D", origin="start_day").agg(agg_rules)
    d1 = d1.dropna(subset=["open"]).reset_index()
    return d1


def load_and_prepare_d1_data(
    h4_path: Path = EURUSD_H4_PATH,
    cftc_path: Path = CFTC_PARQUET_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, resample H4 to D1, causally align CFTC COT data, and verify point-in-time integrity.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Pre-holdout D1 research DataFrame and causal alignment audit dictionary.
    """
    if not h4_path.exists():
        raise FileNotFoundError(f"Missing H4 dataset: {h4_path}")
    if not cftc_path.exists():
        raise FileNotFoundError(f"Missing CFTC COT dataset: {cftc_path}")

    df_h4 = pd.read_parquet(h4_path)
    df_cftc = pd.read_parquet(cftc_path)

    # Standardize column aliases for the 3 approved COT features
    df_cftc = df_cftc.copy()
    df_cftc["speculative_net_zscore_3y"] = df_cftc["net_spec_zscore_3y"]
    df_cftc["commercial_position_zscore_3y"] = df_cftc["net_comm_zscore_3y"]
    df_cftc["speculative_net_4w_change"] = df_cftc["net_spec_4w_change"]

    # Resample H4 to D1
    df_d1 = aggregate_h4_to_d1(df_h4)

    # Causal point-in-time alignment
    aligned_d1, audit = align_cftc_to_eurusd(
        df_cftc=df_cftc,
        df_eurusd=df_d1,
        eurusd_timestamp_col="timestamp",
        cftc_effective_col="effective_time_utc",
    )

    if not audit.get("causally_valid", False):
        raise ValueError(f"Causal alignment failed: {audit}")

    # Enforce strict pre-holdout cutoff (2010 to 2024-11-04)
    ts = pd.to_datetime(aligned_d1["timestamp"], utc=True)
    if (ts >= LOCKED_TEST_START_TS).any():
        # Sanity check: locked test partition must never leak into pre-holdout research
        pass

    d1_pre = aligned_d1[ts <= PRE_HOLDOUT_CUTOFF].copy().reset_index(drop=True)

    return d1_pre, audit


def generate_phase29_walk_forward_folds(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_series: pd.Series,
    horizon_bars: int,
    initial_train_bars: int = INITIAL_TRAIN_BARS,
    n_folds: int = N_FOLDS,
    embargo_bars: int = EMBARGO_BARS,
) -> list[WalkForwardFold]:
    """Generate chronological expanding walk-forward folds with purge = H and embargo = 5 bars.

    Parameters
    ----------
    df : pd.DataFrame
        D1 research DataFrame.
    features_df : pd.DataFrame
        Candidate feature matrix (33 features).
    target_series : pd.Series
        Target labels (direction_vol_H).
    horizon_bars : int
        Lookahead horizon H (purge window).
    initial_train_bars : int, default 1000
        Initial warmup bars (~3.8 years).
    n_folds : int, default 10
        Number of forward chronological validation folds.
    embargo_bars : int, default 5
        Autoregressive feature memory embargo buffer.

    Returns
    -------
    list[WalkForwardFold]
        List of 10 WalkForwardFold objects.
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)
    n_raw = len(df)
    remaining_bars = n_raw - initial_train_bars
    fold_size = remaining_bars // n_folds
    purge_bars = horizon_bars

    folds: list[WalkForwardFold] = []

    for f in range(n_folds):
        val_boundary = initial_train_bars + f * fold_size
        val_end_raw = val_boundary + fold_size if f < n_folds - 1 else n_raw

        train_end = val_boundary - purge_bars
        val_start = val_boundary + embargo_bars
        val_eval_end = val_end_raw - purge_bars

        tr_raw_indices = np.arange(0, train_end)
        va_raw_indices = np.arange(val_start, val_eval_end)

        tr_valid_mask = target_series.iloc[tr_raw_indices].notna() & features_df.iloc[
            tr_raw_indices
        ].notna().all(axis=1)
        va_valid_mask = target_series.iloc[va_raw_indices].notna() & features_df.iloc[
            va_raw_indices
        ].notna().all(axis=1)

        tr_indices = tr_raw_indices[tr_valid_mask.to_numpy()]
        va_indices = va_raw_indices[va_valid_mask.to_numpy()]

        if len(tr_indices) == 0 or len(va_indices) == 0:
            raise ValueError(f"Fold {f + 1}: Empty training or validation set!")

        t_tr_start = ts.iloc[tr_indices[0]]
        t_tr_end = ts.iloc[tr_indices[-1]]
        t_va_start = ts.iloc[va_indices[0]]
        t_va_end = ts.iloc[va_indices[-1]]

        assert t_tr_end < t_va_start, f"Fold {f + 1}: Temporal overlap detected!"
        assert va_indices[0] - tr_indices[-1] >= purge_bars + embargo_bars, (
            f"Fold {f + 1}: Separation violates purge ({purge_bars}) + embargo ({embargo_bars})!"
        )

        tr_y_fold = target_series.iloc[tr_indices]
        va_y_fold = target_series.iloc[va_indices]

        tr_dist = {
            "long": int((tr_y_fold == 1.0).sum()),
            "short": int((tr_y_fold == -1.0).sum()),
        }
        va_dist = {
            "long": int((va_y_fold == 1.0).sum()),
            "short": int((va_y_fold == -1.0).sum()),
        }

        fold = WalkForwardFold(
            fold_idx=f + 1,
            train_indices=tr_indices,
            val_indices=va_indices,
            train_start_ts=t_tr_start,
            train_end_ts=t_tr_end,
            val_start_ts=t_va_start,
            val_end_ts=t_va_end,
            purge_gap_bars=purge_bars,
            train_sample_count=len(tr_indices),
            val_sample_count=len(va_indices),
            train_class_dist=tr_dist,
            val_class_dist=va_dist,
        )
        folds.append(fold)

    return folds


def evaluate_horizon_folds(
    df: pd.DataFrame,
    feats_base: pd.DataFrame,
    feats_cand: pd.DataFrame,
    target_series: pd.Series,
    horizon_bars: int,
) -> dict[str, Any]:
    """Execute expanding walk-forward evaluation across all 10 folds for a given horizon."""
    folds = generate_phase29_walk_forward_folds(
        df=df,
        features_df=feats_cand,
        target_series=target_series,
        horizon_bars=horizon_bars,
    )

    fold_records: list[dict[str, Any]] = []
    bal_base_list: list[float] = []
    bal_cand_list: list[float] = []
    f1_base_list: list[float] = []
    f1_cand_list: list[float] = []
    auc_base_list: list[float] = []
    auc_cand_list: list[float] = []

    for fold in folds:
        tr_idx = fold.train_indices
        va_idx = fold.val_indices

        y_tr = target_series.iloc[tr_idx].to_numpy()
        y_va = target_series.iloc[va_idx].to_numpy()

        # 1. Baseline Model (30 technical features)
        rf_base = RandomForestClassifier(**FROZEN_RF_CONFIG)
        rf_base.fit(feats_base.iloc[tr_idx], y_tr)
        p_base = rf_base.predict(feats_base.iloc[va_idx])
        prob_base = rf_base.predict_proba(feats_base.iloc[va_idx])
        idx_long_b = list(rf_base.classes_).index(1.0) if 1.0 in rf_base.classes_ else 1

        b_bal = float(balanced_accuracy_score(y_va, p_base))
        b_acc = float(accuracy_score(y_va, p_base))
        b_f1 = float(f1_score(y_va, p_base, average="macro", zero_division=0))
        try:
            b_auc = float(roc_auc_score((y_va == 1.0).astype(int), prob_base[:, idx_long_b]))
        except Exception:
            b_auc = 0.5

        # 2. COT Candidate Model (30 technical + 3 COT features)
        rf_cand = RandomForestClassifier(**FROZEN_RF_CONFIG)
        rf_cand.fit(feats_cand.iloc[tr_idx], y_tr)
        p_cand = rf_cand.predict(feats_cand.iloc[va_idx])
        prob_cand = rf_cand.predict_proba(feats_cand.iloc[va_idx])
        idx_long_c = list(rf_cand.classes_).index(1.0) if 1.0 in rf_cand.classes_ else 1

        c_bal = float(balanced_accuracy_score(y_va, p_cand))
        c_acc = float(accuracy_score(y_va, p_cand))
        c_f1 = float(f1_score(y_va, p_cand, average="macro", zero_division=0))
        try:
            c_auc = float(roc_auc_score((y_va == 1.0).astype(int), prob_cand[:, idx_long_c]))
        except Exception:
            c_auc = 0.5

        bal_base_list.append(b_bal)
        bal_cand_list.append(c_bal)
        f1_base_list.append(b_f1)
        f1_cand_list.append(c_f1)
        auc_base_list.append(b_auc)
        auc_cand_list.append(c_auc)

        diff_bal = c_bal - b_bal
        cm_base = confusion_matrix(y_va, p_base, labels=[-1.0, 1.0]).tolist()
        cm_cand = confusion_matrix(y_va, p_cand, labels=[-1.0, 1.0]).tolist()

        fold_records.append(
            {
                "fold": fold.fold_idx,
                "train_start": str(fold.train_start_ts),
                "train_end": str(fold.train_end_ts),
                "val_start": str(fold.val_start_ts),
                "val_end": str(fold.val_end_ts),
                "train_samples": fold.train_sample_count,
                "val_samples": fold.val_sample_count,
                "val_class_distribution": fold.val_class_dist,
                "baseline": {
                    "balanced_accuracy": round(b_bal, 6),
                    "accuracy": round(b_acc, 6),
                    "macro_f1": round(b_f1, 6),
                    "roc_auc": round(b_auc, 6),
                    "confusion_matrix": cm_base,
                },
                "cot_candidate": {
                    "balanced_accuracy": round(c_bal, 6),
                    "accuracy": round(c_acc, 6),
                    "macro_f1": round(c_f1, 6),
                    "roc_auc": round(c_auc, 6),
                    "confusion_matrix": cm_cand,
                },
                "delta_balanced_accuracy": round(diff_bal, 6),
                "delta_macro_f1": round(c_f1 - b_f1, 6),
            }
        )

    # Statistical Aggregations
    diffs = np.array(bal_cand_list) - np.array(bal_base_list)
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    # Paired t-test
    if np.allclose(diffs, 0.0):
        t_stat, p_val = 0.0, 1.0
    else:
        t_res = stats.ttest_rel(bal_cand_list, bal_base_list)
        t_stat, p_val = float(t_res.statistic), float(t_res.pvalue)

    # Wilcoxon signed-rank test
    try:
        w_res = stats.wilcoxon(bal_cand_list, bal_base_list, alternative="two-sided")
        w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, w_pval = float("nan"), 1.0

    # 95% Confidence Interval for mean difference
    dof = len(diffs) - 1
    t_crit = float(stats.t.ppf(0.975, dof)) if dof > 0 else 1.96
    se_diff = std_diff / np.sqrt(len(diffs)) if len(diffs) > 0 else 0.0
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    def _calc_stats(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        return {
            "mean": round(float(np.mean(a)), 6),
            "median": round(float(np.median(a)), 6),
            "std": round(float(np.std(a, ddof=1)), 6) if len(a) > 1 else 0.0,
            "min": round(float(np.min(a)), 6),
            "max": round(float(np.max(a)), 6),
        }

    agg_base = {
        "balanced_accuracy": _calc_stats(bal_base_list),
        "macro_f1": _calc_stats(f1_base_list),
        "roc_auc": _calc_stats(auc_base_list),
    }
    agg_cand = {
        "balanced_accuracy": _calc_stats(bal_cand_list),
        "macro_f1": _calc_stats(f1_cand_list),
        "roc_auc": _calc_stats(auc_cand_list),
    }

    # Pre-registered gate checks
    mean_cand_bal = agg_cand["balanced_accuracy"]["mean"]
    success_gate_passed = bool((mean_cand_bal >= 0.550) and (p_val < 0.05) and (mean_diff > 0.0))
    failure_gate_triggered = bool((mean_cand_bal < 0.530) or (mean_diff <= 0.0) or (p_val >= 0.05))

    return {
        "horizon_bars": horizon_bars,
        "fold_count": len(fold_records),
        "folds": fold_records,
        "baseline_summary": agg_base,
        "cot_candidate_summary": agg_cand,
        "comparison_statistics": {
            "mean_difference": round(mean_diff, 6),
            "median_difference": round(median_diff, 6),
            "std_difference": round(std_diff, 6),
            "paired_t_statistic": round(t_stat, 6),
            "paired_t_pvalue": round(p_val, 6),
            "wilcoxon_statistic": round(w_stat, 6) if not np.isnan(w_stat) else None,
            "wilcoxon_pvalue": round(w_pval, 6),
            "ci_95_lower": round(ci_lower, 6),
            "ci_95_upper": round(ci_upper, 6),
            "statistically_significant": bool(p_val < 0.05),
        },
        "success_gate_passed": success_gate_passed,
        "failure_gate_triggered": failure_gate_triggered,
    }


def run_phase29_experiment() -> dict[str, Any]:
    """Execute complete Phase 29 Pre-Registered COT Positioning Experiment."""
    start_time = time.time()
    now_utc = datetime.now(timezone.utc).isoformat()

    print("=" * 80)
    print("PHASE 29: PRE-REGISTERED CFTC COT POSITIONING EXPERIMENT")
    print("EURUSD D1 | 10 EXPANDING WALFORWARD FOLDS | FROZEN BENCHMARK RF")
    print("=" * 80)

    # 1. Load data & causal alignment
    print("\n[Step 1/5] Loading EURUSD H4 data & Causally Aligning CFTC COT...")
    df_d1, audit = load_and_prepare_d1_data()
    print(f"  Pre-Holdout D1 Research Bars : {len(df_d1):,} bars")
    print(
        f"  Date Range                   : {df_d1['timestamp'].min()} to {df_d1['timestamp'].max()}"
    )
    print(f"  Causal Alignment Audit       : {audit}")

    # 2. Features construction
    print("\n[Step 2/5] Constructing Baseline and Candidate Feature Matrices...")
    raw_feats = compute_swing_features(df_d1, timeframe="D1")
    feats_base = raw_feats[BASELINE_30_FEATURES].copy()
    feats_cand = pd.concat([feats_base, df_d1[APPROVED_COT_FEATURES]], axis=1)

    print(f"  Baseline Feature Matrix      : {feats_base.shape[1]} features (Frozen Technical)")
    print(f"  COT Candidate Matrix         : {feats_cand.shape[1]} features (30 Tech + 3 COT)")

    # 3. Targets computation
    print("\n[Step 3/5] Computing Targets for H=10 and H=20 Business Days...")
    targets = compute_swing_targets(df_d1, horizons=HORIZONS, timeframe="D1")

    # 4. Run experiments for both horizons
    horizon_results: dict[str, Any] = {}
    for h in HORIZONS:
        print(f"\n[Step 4/5] Executing 10-Fold Walk-Forward for Horizon H = {h} Business Days...")
        y_h = targets[f"direction_vol_{h}"]
        valid_bars = int(y_h.notna().sum())
        print(f"  Valid Non-Neutral Bars       : {valid_bars:,} bars")
        res_h = evaluate_horizon_folds(
            df=df_d1,
            feats_base=feats_base,
            feats_cand=feats_cand,
            target_series=y_h,
            horizon_bars=h,
        )
        horizon_results[f"H_{h}"] = res_h

        # Print Fold Table
        print(
            f"  {'Fold':<5} | {'Validation Window':<23} | {'Train N':<7} | {'Val N':<5} | "
            f"{'Base BalAcc':<11} | {'COT BalAcc':<10} | {'Diff':<8}"
        )
        print("  " + "-" * 85)
        for r in res_h["folds"]:
            f_idx = r["fold"]
            vw = f"{r['val_start'][:10]} -> {r['val_end'][:10]}"
            tr_n = r["train_samples"]
            va_n = r["val_samples"]
            b_b = r["baseline"]["balanced_accuracy"]
            c_b = r["cot_candidate"]["balanced_accuracy"]
            d_b = r["delta_balanced_accuracy"]
            print(
                f"  {f_idx:<5} | {vw:<23} | {tr_n:<7} | {va_n:<5} | "
                f"{b_b * 100:>10.2f}% | {c_b * 100:>9.2f}% | {d_b * 100:>+7.2f}%"
            )

        print("  " + "-" * 85)
        b_mean = res_h["baseline_summary"]["balanced_accuracy"]["mean"] * 100
        c_mean = res_h["cot_candidate_summary"]["balanced_accuracy"]["mean"] * 100
        diff_mean = res_h["comparison_statistics"]["mean_difference"] * 100
        p_val = res_h["comparison_statistics"]["paired_t_pvalue"]
        ci_l = res_h["comparison_statistics"]["ci_95_lower"] * 100
        ci_u = res_h["comparison_statistics"]["ci_95_upper"] * 100
        print(f"  Baseline Mean BalAcc         : {b_mean:.2f}%")
        print(f"  COT Candidate Mean BalAcc    : {c_mean:.2f}%")
        ci_str = f"[{ci_l:+.2f}%, {ci_u:+.2f}%]"
        print(f"  Mean Difference              : {diff_mean:+.2f}% (95% CI: {ci_str})")
        print(f"  Paired t-test p-value        : {p_val:.4f}")
        print(f"  Success Gate Passed          : {res_h['success_gate_passed']}")
        print(f"  Failure Gate Triggered       : {res_h['failure_gate_triggered']}")

    # 5. Scientific verdict synthesis
    print("\n[Step 5/5] Synthesizing Scientific Verdict...")
    h10 = horizon_results["H_10"]
    h20 = horizon_results["H_20"]

    both_failed = h10["failure_gate_triggered"] and h20["failure_gate_triggered"]
    both_passed = h10["success_gate_passed"] and h20["success_gate_passed"]
    one_passed = h10["success_gate_passed"] or h20["success_gate_passed"]

    if both_passed:
        scientific_verdict = "SUPPORTED"
    elif one_passed:
        scientific_verdict = "PARTIALLY SUPPORTED"
    elif (
        h10["cot_candidate_summary"]["balanced_accuracy"]["mean"] < 0.530
        and h20["cot_candidate_summary"]["balanced_accuracy"]["mean"] < 0.530
    ) or (
        h10["comparison_statistics"]["mean_difference"] <= 0.0
        and h20["comparison_statistics"]["mean_difference"] <= 0.0
    ):
        scientific_verdict = "NOT SUPPORTED"
    else:
        scientific_verdict = "NOT SUPPORTED" if both_failed else "INCONCLUSIVE"

    elapsed_seconds = round(time.time() - start_time, 2)
    print(f"  Scientific Verdict           : {scientific_verdict}")
    print(f"  Execution Time               : {elapsed_seconds}s")

    report = {
        "phase": 29,
        "title": "Phase 29 Pre-Registered CFTC COT Positioning Experiment",
        "timestamp_utc": now_utc,
        "elapsed_seconds": elapsed_seconds,
        "system_info": {
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "governance": {
            "hypothesis": (
                "Extreme weekly speculative positioning in CME Euro FX futures contains "
                "incremental information about the medium-term directional movement of EURUSD "
                "beyond information already contained in EURUSD price-derived features."
            ),
            "null_hypothesis": (
                "CFTC positioning provides no statistically meaningful incremental directional "
                "information beyond the frozen EURUSD price-only benchmark."
            ),
            "pre_registration_doc": "docs/phase-29-cot-experiment-preregistration.md",
            "model_family": "RandomForestClassifier",
            "hyperparameters": FROZEN_RF_CONFIG,
            "baseline_features": BASELINE_30_FEATURES,
            "cot_features": APPROVED_COT_FEATURES,
            "locked_test_partition_status": "QUARANTINED_AND_UNTOUCHED",
            "live_or_demo_trades": 0,
        },
        "data_alignment_audit": audit,
        "horizon_results": horizon_results,
        "scientific_verdict": scientific_verdict,
        "decision_rationale": (
            f"Horizon H=10 Mean COT BalAcc="
            f"{h10['cot_candidate_summary']['balanced_accuracy']['mean'] * 100:.2f}% "
            f"(diff={h10['comparison_statistics']['mean_difference'] * 100:+.2f}%, "
            f"p={h10['comparison_statistics']['paired_t_pvalue']:.4f}). "
            f"Horizon H=20 Mean COT BalAcc="
            f"{h20['cot_candidate_summary']['balanced_accuracy']['mean'] * 100:.2f}% "
            f"(diff={h20['comparison_statistics']['mean_difference'] * 100:+.2f}%, "
            f"p={h20['comparison_statistics']['paired_t_pvalue']:.4f})."
        ),
    }

    # Save JSON report
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {REPORT_OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run_phase29_experiment()
