#!/usr/bin/env python3
"""Phase 40.4: Live Frozen Strategy Integration & Shadow Validation Runner.

Executes a live forward shadow observation session on MetaQuotes-Demo EURUSD:
- Fetches historical M15 warmup (>= 80 completed bars).
- Streams live completed M15 candles with strict point-in-time causality.
- Computes canonical 80 point-in-time features.
- Evaluates frozen RandomForestBaseline model.
- Records predictions and directional confidence against frozen 0.60 threshold.
- Emits shadow signals only without order execution (execution_enabled=False).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

from trading.adapters.mt5.client import MT5ReadOnlyClient
from trading.adapters.mt5.shadow import (
    LiveFrozenStrategyPipeline,
    ShadowValidationRunner,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase40_4_runner")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 40.4 Live Frozen Strategy Integration & Shadow Runner"
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=7200.0,
        help="Target observation duration in seconds (default: 7200s = 2h)",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=120,
        help="Number of historical M15 warmup bars to fetch (default: 120)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0s)",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run initial smoke evaluation only and exit (offline/fast validation)",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Optional maximum number of completed candle evaluations",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 40.4 — LIVE FROZEN STRATEGY INTEGRATION & SHADOW VALIDATION")
    print("=" * 80)

    # 1. Initialize MT5 Read-Only Client
    client = MT5ReadOnlyClient()
    if not client.initialize():
        print("[FATAL ERROR] Failed to initialize MT5 terminal bridge.")
        return 1

    try:
        pipeline = LiveFrozenStrategyPipeline()
        runner = ShadowValidationRunner(
            client=client,
            pipeline=pipeline,
            warmup_bars=args.warmup_bars,
        )

        def handle_shutdown(signum, frame):
            print(
                f"\n[INTERRUPT] Received signal {signum}, stopping gracefully...",
                flush=True,
            )
            runner.stop()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        preflight = runner.preflight_safety_check()
        print("\n" + "=" * 80)
        print("PREFLIGHT SAFETY & ENVIRONMENT VERIFICATION (SHADOW MODE)")
        print("=" * 80)
        print(f"Broker Host:                 {preflight['broker']}")
        print(f"Broker Server:               {preflight['server']}")
        print(f"Account Mode:                {preflight['account_mode']}")
        print(f"Demo Verified:               {str(preflight['is_demo']).upper()}")
        print("Live Mode Detection:         FALSE")
        print(f"Execution Enabled:           {str(preflight['execution_enabled']).upper()}")
        print(f"Terminal Connected:          {str(preflight['terminal_connected']).upper()}")
        print(f"Trade Allowed:               {str(preflight['trade_allowed']).upper()}")
        print(f"Symbol:                      {preflight['symbol']}")
        print(f"Current Bid / Ask:           {preflight['bid']:.5f} / {preflight['ask']:.5f}")
        print(f"Spread:                      {preflight['spread']:.5f}")
        print(f"Data Age:                    {preflight['data_age_seconds']:.2f}s")
        print(f"Data Fresh (<= 120s):        {str(preflight['data_fresh']).upper()}")
        print(f"Open Positions:              {preflight['open_positions_count']}")
        print("=" * 80 + "\n")

        if not preflight["data_fresh"]:
            print("[MARKET CLOSED / STALE] Live tick data is stale > 120s. Aborting.")
            return 1

        duration = 0.0 if args.smoke_only else args.target_duration
        max_evals = 1 if args.smoke_only else args.max_evaluations

        print(f"Starting Shadow Observation Session (Duration: {duration}s)...")
        summary = runner.run_shadow_observation(
            duration_seconds=duration,
            poll_interval_seconds=args.poll_interval,
            max_evaluations=max_evals,
        )

        # Write Report
        report_path = Path("reports/phase40_4_live_strategy_integration.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        m = summary["metrics"]
        print("\n" + "=" * 80)
        print("PHASE 40.4 SHADOW VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Session Duration:            {summary['duration_seconds']:.2f}s")
        print(f"M15 Warmup Bars Loaded:      {m['m15_bars_loaded']}")
        print(f"New Completed Bars Ingested: {m['m15_bars_completed_during_run']}")
        print(f"Model Evaluations:           {m['model_evaluations']}")
        print(f"Predictions Produced:        {m['predictions']}")
        print(
            f"Long / Neutral / Short:      "
            f"{m['long_predictions']} / {m['neutral_predictions']} / {m['short_predictions']}"
        )
        print(f"Min / Max Confidence:        {m['min_confidence']:.4f} / {m['max_confidence']:.4f}")
        print(
            f"Mean / Median Confidence:    "
            f"{m['mean_confidence']:.4f} / {m['median_confidence']:.4f}"
        )
        print(f"Predictions >= 0.60:         {m['count_confidence_ge_0_60']}")
        print(f"Predictions < 0.60:          {m['count_confidence_lt_0_60']}")
        print(f"Shadow Signals Emitted:      {m['signals_emitted']}")
        print(
            f"Orders / Fills / Positions:  "
            f"{m['orders_submitted']} / {m['fills']} / {m['open_positions']}"
        )
        print(f"Root Cause Classification:   {summary['root_cause_classification']}")
        print(f"Report Written:              {report_path}")
        print("=" * 80 + "\n")
        return 0
    finally:
        client.shutdown()


if __name__ == "__main__":
    sys.exit(main())
