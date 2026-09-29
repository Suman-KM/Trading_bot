"""Phase 16: Information Timing and Causality Audit Script.

Generates and displays the exact point-in-time sequence:
FEATURE TIME < DECISION TIME < OUTCOME TIME
for sample H4 decisions and corresponding M15 entry confirmation features.
"""

from __future__ import annotations

import pandas as pd

from ai.dataset.assembly import assemble_dataset
from ai.dataset.swing import aggregate_m15_to_h4
from ai.labels.swing import compute_swing_targets
from ai.swing.confirmation import align_m15_with_h4_decisions
from ai.swing.timing_audit import generate_timing_audit_report


def main() -> None:
    print("=" * 80)
    print("PHASE 16: INFORMATION-TIMING & CAUSALITY AUDIT")
    print("=" * 80)

    raw_m15 = pd.read_parquet("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
    h4_df = aggregate_m15_to_h4(raw_m15)
    ds_m15 = assemble_dataset()

    aligned_m15 = align_m15_with_h4_decisions(h4_df, ds_m15.df)
    h4_targets = compute_swing_targets(h4_df, horizons=[8], timeframe="H4")
    target_vol_8 = h4_targets["direction_vol_8"]

    # Sample representative indices spanning 2023, 2024, 2025 across multiple market hours
    sample_indices = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]

    records = generate_timing_audit_report(
        h4_df=h4_df,
        aligned_m15=aligned_m15,
        target_series=target_vol_8,
        sample_indices=sample_indices,
        horizon_bars=8,
    )

    print(f"\nAudit Sample Size: {len(records)} H4 Decision Points\n")
    print(
        f"{'Idx':<5} | {'H4 Bar Start':<20} | {'Decision Time':<20} | "
        f"{'Matched M15 Bar':<20} | {'Outcome Time (H=8)':<20} | {'Causal?':<7}"
    )
    print("-" * 105)

    for r in records:
        print(
            f"{r['h4_bar_index']:<5} | "
            f"{r['h4_bar_start'][:19]:<20} | "
            f"{r['h4_decision_time'][:19]:<20} | "
            f"{r['actual_matched_m15_timestamp'][:19]:<20} | "
            f"{r['target_outcome_time'][:19]:<20} | "
            f"{'PASS' if r['causality_verified'] else 'FAIL':<7}"
        )

    print("\nDetailed Feature and Outcome Breakdown:")
    for r in records[:3]:
        print(f"\n--- Decision Point Index {r['h4_bar_index']} ---")
        print(f"  H4 Bar Range        : {r['h4_bar_start']} -> {r['h4_decision_time']}")
        print(f"  Decision Timestamp  : {r['h4_decision_time']} (close of H4 candle)")
        print(f"  Latest Allowed M15  : {r['latest_allowed_m15_timestamp']}")
        print(f"  Matched M15 Bar     : {r['actual_matched_m15_timestamp']}")
        print(f"  Target Horizon      : 8 H4 bars (32 hours) -> {r['target_outcome_time']}")
        print(f"  M15 Direction       : {r['m15_features_used']['candle_direction']}")
        print(f"  M15 RSI-14          : {r['m15_features_used']['rsi_14']:.2f}")
        print(f"  M15 Dist EMA-20     : {r['m15_features_used']['dist_ema_20']:.6f}")
        print(f"  Actual Outcome      : {r['actual_future_outcome']}")
        print(f"  Causality Verified  : {r['causality_verified']}")


if __name__ == "__main__":
    main()
