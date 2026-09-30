#!/usr/bin/env python3
"""Phase 19: Market Microstructure & Data-Information Feasibility Audit Runner.

Executes a rigorous feasibility audit on MetaQuotes-Demo EURUSD tick data,
analyzing bid/ask dynamics, tick arrival intensity, signed order flow availability,
intrabar path geometry, canonical OHLC reconstruction, and session transitions.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai.microstructure.audit import (
    analyze_session_microstructure,
    compute_intrabar_path_metrics,
    compute_realized_volatility_metrics,
    compute_tick_direction_proxy,
    compute_tick_intensity_metrics,
    compute_tick_spread_metrics,
    reconstruct_ohlc_from_ticks,
    validate_tick_data_quality,
)

RAW_AUDIT_DIR = Path("data/raw/microstructure_audit")
REPORTS_DIR = Path("reports")
CANONICAL_M15_PATH = Path("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
CANONICAL_H4_PATH = Path("data/processed/eurusd_h4_expanded/eurusd_h4_processed.parquet")

# Governance Boundaries
LOCKED_PHASE11_M15_START = pd.Timestamp("2026-02-19 12:00:00+00:00")
LOCKED_PHASE15_H4_START = pd.Timestamp("2026-02-19 12:00:00+00:00")
LOCKED_PHASE18_HOLDOUT_START = pd.Timestamp("2024-11-06 00:00:00+00:00")


def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_phase19_audit() -> int:
    t_start = time.time()
    print("=" * 80)
    print("PHASE 19: MARKET MICROSTRUCTURE & DATA-INFORMATION FEASIBILITY AUDIT")
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: VERIFY ENVIRONMENT AUDIT METADATA & PARQUET SAMPLES
    # -------------------------------------------------------------------------
    print("\n[Step 1/8] Verifying MT5 Environment Audit Metadata & Tick Samples...")
    env_audit_path = REPORTS_DIR / "phase19_mt5_environment_audit.json"
    if not env_audit_path.exists():
        print("  Environment metadata not found. Running export worker...")
        from scripts.export_microstructure_ticks import main as export_main

        if export_main() != 0:
            print("[FATAL] Export worker failed.")
            return 1

    with open(env_audit_path, encoding="utf-8") as f:
        env_meta = json.load(f)

    term = env_meta["terminal"]
    print(f"  MT5 Terminal   : {term['name']} Build {term['build']} ({term['version']})")
    print(f"  Broker / Server: {term['broker']} / {term['server']}")
    print(f"  Python Bridge  : MetaTrader5 {env_meta['python_bridge']['package_version']}")
    sym_m = env_meta["symbol"]
    print(f"  EURUSD Specs   : Digits={sym_m['digits']}, Point={sym_m['point']}")
    dom_m = env_meta["depth_of_market"]
    print(
        f"  Depth of Market: Live={dom_m['live_dom_subscribed']}, "
        f"Entries={dom_m['live_dom_entries_count']}, "
        f"Historical={dom_m['historical_dom_available']}"
    )
    cal_avail = env_meta["economic_calendar"]["historical_calendar_available"]
    print(f"  Calendar API   : Historical Available={cal_avail}")

    # Verify canonical datasets exist and record pre-audit hashes
    if not CANONICAL_M15_PATH.exists():
        print(f"[FATAL] Canonical M15 dataset missing: {CANONICAL_M15_PATH}")
        return 1
    sha_m15_before = _compute_file_sha256(CANONICAL_M15_PATH)
    sha_h4_before = (
        _compute_file_sha256(CANONICAL_H4_PATH) if CANONICAL_H4_PATH.exists() else "NONE"
    )

    # -------------------------------------------------------------------------
    # STEP 2: LOAD BOUNDED TICK SAMPLES
    # -------------------------------------------------------------------------
    print("\n[Step 2/8] Loading Bounded Tick Samples (1h, 1d, 1w)...")
    samples = {}
    for s_name in ["sample_1h", "sample_1d", "sample_1w"]:
        p = RAW_AUDIT_DIR / f"{s_name}.parquet"
        if not p.exists():
            print(f"[FATAL] Missing sample parquet: {p}")
            return 1
        df_s = pd.read_parquet(p)
        df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], utc=True)

        # Governance Check: Strictly reject any ticks touching locked holdout/test partitions
        if (df_s["timestamp"] >= LOCKED_PHASE18_HOLDOUT_START).any():
            raise PermissionError(f"Sample {s_name} contains ticks inside Phase 18 holdout window!")
        if (df_s["timestamp"] >= LOCKED_PHASE11_M15_START).any():
            raise PermissionError(
                f"Sample {s_name} contains ticks inside Phase 11 locked test window!"
            )

        samples[s_name] = df_s
        print(
            f"  Loaded {s_name:<9}: {len(df_s):,d} ticks | "
            f"{df_s['timestamp'].iloc[0].isoformat()} -> {df_s['timestamp'].iloc[-1].isoformat()}"
        )

    # -------------------------------------------------------------------------
    # STEP 3: DATA QUALITY AUDIT
    # -------------------------------------------------------------------------
    print("\n[Step 3/8] Executing Tick Data Quality Audit...")
    quality_results = {}
    for s_name, df_s in samples.items():
        q = validate_tick_data_quality(df_s)
        quality_results[s_name] = q
        print(
            f"  {s_name:<9}: Monotonic={q['is_monotonic_increasing']}, "
            f"DupTimestamps={q['duplicate_timestamps']}, "
            f"NegativeSpreads={q['negative_spread_count']}, "
            f"MedianInterval={q['median_inter_tick_ms']:.1f}ms, "
            f"MaxGap={q['max_gap_seconds']:.2f}s, Pass={q['quality_pass']}"
        )
        assert q["quality_pass"], f"Quality check failed for {s_name}!"

    # -------------------------------------------------------------------------
    # STEP 4: SPREAD DYNAMICS & COMPARISON WITH M15 SPREAD
    # -------------------------------------------------------------------------
    print("\n[Step 4/8] Analyzing Bid/Ask Spread Dynamics...")
    spread_results = {}
    for s_name, df_s in samples.items():
        s = compute_tick_spread_metrics(df_s, point=1e-5)
        spread_results[s_name] = s
        pts_str = f"Mean={s['mean_spread_points']:.2f} pts ({s['mean_spread_pips']:.2f} pips)"
        print(
            f"  {s_name:<9}: {pts_str}, "
            f"Median={s['median_spread_points']:.1f} pts, "
            f"p95={s['p95_spread_points']:.1f} pts, "
            f"Min={s['min_spread_points']:.0f} pts, Max={s['max_spread_points']:.0f} pts"
        )

    # -------------------------------------------------------------------------
    # STEP 5: TICK ARRIVAL INTENSITY & DIRECTION PROXIES
    # -------------------------------------------------------------------------
    print("\n[Step 5/8] Analyzing Tick Arrival Intensity & Direction Proxies...")
    intensity_results = {}
    direction_results = {}
    for s_name, df_s in samples.items():
        intensity_results[s_name] = compute_tick_intensity_metrics(df_s)
        direction_results[s_name] = compute_tick_direction_proxy(df_s)

    dir_1d = direction_results["sample_1d"]
    inten_1d = intensity_results["sample_1d"]
    print(
        f"  Intensity (1-Day): Mean={inten_1d['ticks_per_second_mean']:.2f} ticks/sec | "
        f"Mean={inten_1d['ticks_per_minute_mean']:.1f} ticks/min | "
        f"Median={inten_1d['ticks_per_15m_median']:.0f} ticks/15m"
    )
    print(
        f"  Direction (1-Day): Mid Upticks={dir_1d['mid_uptick_share'] * 100:.1f}%, "
        f"Downticks={dir_1d['mid_downtick_share'] * 100:.1f}%, "
        f"Unchanged={dir_1d['mid_unchanged_share'] * 100:.1f}%"
    )
    has_flow = dir_1d["true_signed_order_flow_available"]
    print(f"  Signed Order Flow: True Signed Flow Available={has_flow}")
    print(f"  Classification   : {dir_1d['classification_note']}")

    # -------------------------------------------------------------------------
    # STEP 6: INTRABAR PRICE PATH & REALIZED VOLATILITY
    # -------------------------------------------------------------------------
    print("\n[Step 6/8] Computing Intrabar Price Path Geometry & Realized Volatility...")
    df_1d = samples["sample_1d"]
    path_m15 = compute_intrabar_path_metrics(df_1d, freq="15min")
    rvol_m15 = compute_realized_volatility_metrics(df_1d, freq="15min")

    path_summary = {
        "bars_analyzed": len(path_m15),
        "mean_path_length_pips": float(path_m15["path_length"].mean() / 1e-4),
        "median_realized_range_pips": float(path_m15["realized_range"].median() / 1e-4),
        "mean_path_efficiency": float(path_m15["path_efficiency"].mean()),
        "mean_direction_changes": float(path_m15["direction_changes"].mean()),
        "mean_realized_volatility": float(rvol_m15["realized_volatility"].mean()),
        "mean_tick_return_std": float(rvol_m15["tick_return_std"].mean()),
    }
    print(f"  M15 Intrabar Bars Analyzed: {path_summary['bars_analyzed']}")
    print(f"  Mean Path Length          : {path_summary['mean_path_length_pips']:.2f} pips")
    print(f"  Median Realized Range     : {path_summary['median_realized_range_pips']:.2f} pips")
    eff = path_summary["mean_path_efficiency"]
    print(f"  Mean Path Efficiency      : {eff:.4f} (displacement / path)")
    dc = path_summary["mean_direction_changes"]
    print(f"  Mean Direction Changes    : {dc:.1f} reversals per 15m")
    print(f"  Mean Realized Volatility  : {path_summary['mean_realized_volatility']:.6f}")

    # -------------------------------------------------------------------------
    # STEP 7: CANONICAL OHLC RECONSTRUCTION & COMPARISON
    # -------------------------------------------------------------------------
    print("\n[Step 7/8] Reconstructing M15 OHLC and Comparing with Canonical Dataset...")
    reconstructed_m15 = reconstruct_ohlc_from_ticks(df_1d, freq="15min", price_col="bid")

    # Load matching canonical M15 bars for 2024-10-08
    df_canon_m15 = pd.read_parquet(CANONICAL_M15_PATH)
    df_canon_m15["timestamp"] = pd.to_datetime(df_canon_m15["timestamp"], utc=True)
    mask_canon = (df_canon_m15["timestamp"] >= "2024-10-08 00:00:00+00:00") & (
        df_canon_m15["timestamp"] <= "2024-10-08 23:59:59+00:00"
    )
    canon_day = df_canon_m15[mask_canon].reset_index(drop=True)

    # Merge on timestamp
    merged_m15 = pd.merge(
        canon_day,
        reconstructed_m15,
        on="timestamp",
        suffixes=("_canon", "_recon"),
        how="inner",
    )

    diff_open = np.abs(merged_m15["open_canon"] - merged_m15["open_recon"]).max()
    diff_high = np.abs(merged_m15["high_canon"] - merged_m15["high_recon"]).max()
    diff_low = np.abs(merged_m15["low_canon"] - merged_m15["low_recon"]).max()
    diff_close = np.abs(merged_m15["close_canon"] - merged_m15["close_recon"]).max()

    reconstruction_audit = {
        "canonical_bars_count": len(canon_day),
        "reconstructed_bars_count": len(reconstructed_m15),
        "matched_bars_count": len(merged_m15),
        "max_diff_open": float(diff_open),
        "max_diff_high": float(diff_high),
        "max_diff_low": float(diff_low),
        "max_diff_close": float(diff_close),
        "mean_diff_points": float(
            (
                np.abs(merged_m15["open_canon"] - merged_m15["open_recon"])
                + np.abs(merged_m15["high_canon"] - merged_m15["high_recon"])
                + np.abs(merged_m15["low_canon"] - merged_m15["low_recon"])
                + np.abs(merged_m15["close_canon"] - merged_m15["close_recon"])
            ).mean()
            / 4.0
            / 1e-5
        ),
        "canonical_spread_constant_mode": float(canon_day["spread"].mode().iloc[0]),
        "reconstructed_spread_mean": float(reconstructed_m15["spread_mean_points"].mean()),
        "reconstructed_spread_max": float(reconstructed_m15["spread_max_points"].max()),
    }

    print(f"  Matched M15 Bars : {reconstruction_audit['matched_bars_count']} / {len(canon_day)}")
    print(f"  Max Diff (Open)  : {diff_open:.6f} ({diff_open / 1e-5:.1f} pts)")
    print(f"  Max Diff (High)  : {diff_high:.6f} ({diff_high / 1e-5:.1f} pts)")
    print(f"  Max Diff (Low)   : {diff_low:.6f} ({diff_low / 1e-5:.1f} pts)")
    print(f"  Max Diff (Close) : {diff_close:.6f} ({diff_close / 1e-5:.1f} pts)")
    mode_sp = reconstruction_audit["canonical_spread_constant_mode"]
    print(f"  Canonical Spread : Constant {mode_sp} pts snapshot")
    print(
        f"  Tick Spread Mean : {reconstruction_audit['reconstructed_spread_mean']:.1f} pts "
        f"(Max: {reconstruction_audit['reconstructed_spread_max']:.1f} pts)"
    )

    # -------------------------------------------------------------------------
    # STEP 8: SESSION MICROSTRUCTURE & 10 DETERMINISTIC CHECKS
    # -------------------------------------------------------------------------
    print("\n[Step 8/8] Analyzing Global Session Transitions & Running 10 Checks...")
    df_1w = samples["sample_1w"]
    session_metrics = analyze_session_microstructure(df_1w)
    print("\n  Weekly Global Session Microstructure Breakdown (2024-10-07 to 2024-10-11):")
    hdr = f"  {'Session':<26} | {'Share':<6} | {'Med Sprd':<9} | {'p95 Sprd':<9} | {'Ticks/Min':<9}"
    print(hdr)
    print("  " + "-" * 75)
    for sess, s_m in session_metrics.items():
        print(
            f"  {sess:<30} | {s_m['tick_share_pct']:5.1f}%  | "
            f"{s_m['median_spread_points']:5.1f} pts   | "
            f"{s_m['p95_spread_points']:5.1f} pts   | "
            f"{s_m['ticks_per_minute']:7.1f}"
        )

    # Verify canonical datasets were NOT touched
    sha_m15_after = _compute_file_sha256(CANONICAL_M15_PATH)
    sha_h4_after = _compute_file_sha256(CANONICAL_H4_PATH) if CANONICAL_H4_PATH.exists() else "NONE"
    datasets_unmodified = bool(sha_m15_before == sha_m15_after and sha_h4_before == sha_h4_after)

    dup_pct = quality_results["sample_1w"]["duplicate_quotes"] / len(df_1w) * 100.0

    # 10 Deterministic Checks
    checks = [
        {
            "check": "1. UTC timestamps on all ticks",
            "passed": bool(
                df_1w["timestamp"].dt.tz is not None and str(df_1w["timestamp"].dt.tz) == "UTC"
            ),
            "detail": "Strictly tz-aware UTC",
        },
        {
            "check": "2. Monotonicity of time_msc",
            "passed": bool(quality_results["sample_1w"]["is_monotonic_increasing"]),
            "detail": "Strictly monotonic increasing",
        },
        {
            "check": "3. Zero negative spreads",
            "passed": bool(quality_results["sample_1w"]["negative_spread_count"] == 0),
            "detail": "Zero crossed quotes",
        },
        {
            "check": "4. Zero invalid/NaN prices",
            "passed": bool(
                quality_results["sample_1w"]["invalid_bid_count"] == 0
                and quality_results["sample_1w"]["invalid_ask_count"] == 0
            ),
            "detail": "All quotes positive and finite",
        },
        {
            "check": "5. Duplicate quote rate < 1%",
            "passed": bool(dup_pct < 1.0),
            "detail": f"{dup_pct:.3f}% duplicate quotes",
        },
        {
            "check": "6. Bounded latency on queries",
            "passed": bool(
                all(x["latency_seconds"] < 120.0 for x in env_meta["tick_availability"])
            ),
            "detail": "All tick queries completed well under 120s limit",
        },
        {
            "check": "7. True signed order flow absent",
            "passed": bool(not dir_1d["true_signed_order_flow_available"]),
            "detail": "Confirmed OTC quote-driven, buyer/seller side not available",
        },
        {
            "check": "8. Historical DOM unavailable",
            "passed": bool(
                not env_meta["depth_of_market"]["historical_dom_available"]
                and env_meta["depth_of_market"]["live_dom_entries_count"] == 0
            ),
            "detail": "Zero historical DOM, empty live DOM tuple",
        },
        {
            "check": "9. Canonical datasets unmodified",
            "passed": datasets_unmodified,
            "detail": "M15 and H4 Parquet SHA256 identical before and after",
        },
        {
            "check": "10. Locked partitions unaccessed",
            "passed": bool(
                (df_1w["timestamp"] < LOCKED_PHASE18_HOLDOUT_START).all()
                and (df_1w["timestamp"] < LOCKED_PHASE11_M15_START).all()
            ),
            "detail": "All audited data in open pre-holdout research partition",
        },
    ]

    print("\n  10 Deterministic Microstructure Audit Verification Checks:")
    all_passed = True
    for c in checks:
        status_str = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            all_passed = False
        print(f"  [{status_str}] {c['check']:<35}: {c['detail']}")
    assert all_passed, "One or more deterministic audit checks failed!"

    # Final Classification per Section 22
    classification = "HISTORICAL TICKS AVAILABLE — LIMITED MICROSTRUCTURE"

    audit_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t_start, 3),
        "classification": classification,
        "environment": env_meta,
        "sample_quality": quality_results,
        "spread_metrics": spread_results,
        "intensity_metrics": intensity_results,
        "direction_proxies": direction_results,
        "intrabar_path_summary": path_summary,
        "reconstruction_audit": reconstruction_audit,
        "session_metrics": session_metrics,
        "checks": checks,
        "interpretation": (
            "Historical Bid/Ask quotes with millisecond timestamps are available from 2020 onward "
            "in MetaQuotes-Demo EURUSD. However, true signed order flow (buyer/seller initiation) "
            "is absent (last=0, volume=0, trade flags absent), Depth of Market (Level 2) is "
            "completely unavailable historically and empty live, and economic calendar data is "
            "unexposed in Python API. All derived tick metrics (intensity, realized volatility, "
            "path length) are strictly PRICE-DERIVED PROXIES rather than true microstructure flow."
        ),
    }

    out_report = REPORTS_DIR / "phase19_microstructure_audit.json"
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved complete Phase 19 audit report to {out_report}")

    print("\n" + "=" * 80)
    print(f"RESEARCH CLASSIFICATION: {classification}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_phase19_audit())
