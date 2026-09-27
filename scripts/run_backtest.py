"""CLI runner script executing Phase 12 Research Trading Strategy & Historical Backtester.

Evaluates EURUSD M15 directional strategy on the Validation partition (14,983 rows)
with Random Forest (balanced) fitted strictly on Training data (69,937 rows).
The Phase 11 Test partition (14,988 rows) remains PERMANENTLY LOCKED.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ai.backtest.engine import BacktestEngine
from ai.backtest.metrics import calculate_performance_metrics
from ai.backtest.models import BacktestConfig
from ai.backtest.plotting import (
    plot_drawdown_curve,
    plot_equity_curve,
    plot_trade_pnl_distribution,
)
from ai.backtest.strategy import MLAssistedStrategy
from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import split_dataset
from ai.models.baselines import RandomForestBaseline
from trading.risk.limits import RiskLimits

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def run_phase12_backtest() -> None:
    """Execute complete Phase 12 backtesting study."""
    start_time = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 12 — HISTORICAL BACKTESTING SIMULATION")
    print("=" * 70)

    # 1. Dataset Assembly and Temporal Splitting
    print("\n[Step 1/6] Assembling dataset and establishing chronological partitions...")
    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits = split_dataset(ds)

    tr_start = splits.train.start_timestamp
    tr_end = splits.train.end_timestamp
    val_start = splits.val.start_timestamp
    val_end = splits.val.end_timestamp

    print(f"  Training partition:   {len(splits.train):,} rows ({tr_start} -> {tr_end})")
    print(f"  Validation partition: {len(splits.val):,} rows ({val_start} -> {val_end})")
    print(f"  Test partition:       {len(splits.test):,} rows [STRICTLY LOCKED / UNTOUCHED]")

    # Strict holdout governance verification
    assert len(splits.test) == 14988, "Test set row count altered!"

    # 2. Extract Validation Market & Feature Data
    print("\n[Step 2/6] Aligning Validation market series and point-in-time features...")
    features_full = pd.read_parquet("data/features/eurusd_m15/eurusd_m15_features.parquet")
    market_cols = ["timestamp", "open", "high", "low", "close", "spread", "atr_14"]
    val_mask = features_full["timestamp"].isin(splits.val.timestamps)
    val_market = features_full[val_mask][market_cols].reset_index(drop=True)
    val_features = splits.val.X.reset_index(drop=True)

    assert len(val_market) == len(val_features) == len(splits.val)
    print(f"  Validation data ready: {len(val_market):,} bars.")

    # 3. Fit Candidate Model Strictly on Training Partition
    print("\n[Step 3/6] Fitting Random Forest (class_weight='balanced') on Training data...")
    t_fit_start = time.time()
    rf = RandomForestBaseline(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(splits.train.X, splits.train.y)
    t_fit = time.time() - t_fit_start
    print(f"  Model fitted in {t_fit:.2f}s across {len(splits.train):,} training observations.")
    print(f"  Model classes: {rf.classes_}")

    # 4. Execute Baseline Backtest
    print("\n[Step 4/6] Running Baseline Backtest (conf=0.60, SL=1.0x ATR, TP=1.5x ATR)...")
    baseline_config = BacktestConfig(
        initial_equity=100_000.0,
        confidence_threshold=0.60,
        atr_period=14,
        stop_loss_atr_multiple=1.0,
        take_profit_atr_multiple=1.5,
        max_holding_bars=4,
        cooldown_bars=1,
        point_value=1e-5,
        default_spread_points=10.0,
        use_historical_spread=True,
        slippage_points=0.0,
        commission_per_lot=0.0,
        same_bar_sl_priority=True,
        constrain_exposure=True,
        risk_limits=RiskLimits(MIN_SIGNAL_CONFIDENCE=0.60),
    )

    baseline_strat = MLAssistedStrategy(model=rf, config=baseline_config)
    baseline_engine = BacktestEngine(strategy=baseline_strat, config=baseline_config)
    baseline_result = baseline_engine.run(df_market=val_market, features_df=val_features)
    baseline_metrics = calculate_performance_metrics(baseline_result)

    # Save Baseline Artifacts
    metrics_dict = baseline_metrics.to_dict()
    with open(REPORTS_DIR / "phase12_baseline_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    pd.DataFrame([metrics_dict]).to_csv(REPORTS_DIR / "phase12_baseline_metrics.csv", index=False)
    baseline_trades_df = baseline_result.to_trades_dataframe()
    baseline_trades_df.to_csv(REPORTS_DIR / "phase12_trade_summary.csv", index=False)

    print(f"  Baseline total trades:      {baseline_metrics.total_trades}")
    print(f"  Confidence filtered bars:   {baseline_metrics.confidence_filtered_signals:,}")
    print(f"  Ending equity:              ${baseline_metrics.ending_equity:,.2f}")
    print(f"  Net PnL:                    ${baseline_metrics.total_net_pnl:,.2f}")

    # 5. Predefined Sensitivity Analysis Scenarios
    print("\n[Step 5/6] Executing predefined sensitivity analysis scenarios...")
    scenarios = [
        ("Baseline (conf=0.60)", 0.60, 0.60, 1.0, 1.5, 0.0, 0.0),
        ("Conf=0.50 (neutral)", 0.50, 0.50, 1.0, 1.5, 0.0, 0.0),
        ("Conf=0.70 (strict)", 0.70, 0.70, 1.0, 1.5, 0.0, 0.0),
        ("Tighter SL (0.75x ATR)", 0.50, 0.50, 0.75, 1.5, 0.0, 0.0),
        ("Wider SL (1.25x ATR)", 0.50, 0.50, 1.25, 1.5, 0.0, 0.0),
        ("Tighter TP (1.00x ATR)", 0.50, 0.50, 1.0, 1.0, 0.0, 0.0),
        ("Wider TP (2.00x ATR)", 0.50, 0.50, 1.0, 2.0, 0.0, 0.0),
        ("Slippage (0.5 pip)", 0.50, 0.50, 1.0, 1.5, 5.0, 0.0),
        ("Commission ($7/lot)", 0.50, 0.50, 1.0, 1.5, 0.0, 7.0),
        ("Combined Frictions", 0.50, 0.50, 1.0, 1.5, 5.0, 7.0),
    ]

    sens_records = []
    active_scenario_result = None

    for name, conf, min_conf, sl_m, tp_m, slip, comm in scenarios:
        s_config = BacktestConfig(
            initial_equity=100_000.0,
            confidence_threshold=conf,
            atr_period=14,
            stop_loss_atr_multiple=sl_m,
            take_profit_atr_multiple=tp_m,
            max_holding_bars=4,
            cooldown_bars=1,
            point_value=1e-5,
            default_spread_points=10.0,
            use_historical_spread=True,
            slippage_points=slip,
            commission_per_lot=comm,
            same_bar_sl_priority=True,
            constrain_exposure=True,
            risk_limits=RiskLimits(MIN_SIGNAL_CONFIDENCE=min_conf),
        )
        s_strat = MLAssistedStrategy(model=rf, config=s_config)
        s_eng = BacktestEngine(strategy=s_strat, config=s_config)
        s_res = s_eng.run(df_market=val_market, features_df=val_features)
        s_met = calculate_performance_metrics(s_res)

        if name == "Conf=0.50 (neutral)":
            active_scenario_result = s_res

        sens_records.append(
            {
                "scenario": name,
                "confidence_threshold": conf,
                "sl_atr_mult": sl_m,
                "tp_atr_mult": tp_m,
                "slippage_pts": slip,
                "commission_per_lot": comm,
                "total_trades": s_met.total_trades,
                "win_rate": s_met.win_rate,
                "profit_factor": s_met.profit_factor,
                "net_pnl": s_met.total_net_pnl,
                "total_return_pct": s_met.total_return_pct,
                "max_drawdown_pct": s_met.max_drawdown_pct,
                "average_trade": s_met.average_trade_return,
                "transaction_costs": s_met.total_transaction_costs,
            }
        )
        msg = (
            f"    - {name:<26}: Trades={s_met.total_trades:>3}, "
            f"WinRate={s_met.win_rate:>5.2f}%, PF={s_met.profit_factor:>6.4f}, "
            f"NetPnL=${s_met.total_net_pnl:>8.2f}, MaxDD={s_met.max_drawdown_pct:>5.2f}%"
        )
        print(msg)

    sens_df = pd.DataFrame(sens_records)
    sens_df.to_csv(REPORTS_DIR / "phase12_sensitivity.csv", index=False)

    # 6. Generate Figures & Metadata
    print("\n[Step 6/6] Generating plots and saving study metadata...")
    # Plot baseline figures
    plot_equity_curve(
        baseline_result,
        FIGURES_DIR / "phase12_equity_curve.png",
        title="EURUSD M15 — Baseline Portfolio Equity Curve (Conf=0.60)",
    )
    plot_drawdown_curve(
        baseline_result,
        FIGURES_DIR / "phase12_drawdown.png",
        title="EURUSD M15 — Baseline Portfolio Drawdown (Conf=0.60)",
    )
    plot_trade_pnl_distribution(
        baseline_result,
        FIGURES_DIR / "phase12_trade_pnl.png",
        title="EURUSD M15 — Baseline Trade Net P&L Distribution (Conf=0.60)",
    )

    # If active scenario exists, also save active plots for comparative research
    if active_scenario_result is not None and len(active_scenario_result.trade_ledger) > 0:
        plot_equity_curve(
            active_scenario_result,
            FIGURES_DIR / "phase12_equity_curve_active_conf50.png",
            title="EURUSD M15 — Sensitivity Portfolio Equity Curve (Conf=0.50)",
        )
        plot_drawdown_curve(
            active_scenario_result,
            FIGURES_DIR / "phase12_drawdown_active_conf50.png",
            title="EURUSD M15 — Sensitivity Portfolio Drawdown (Conf=0.50)",
        )
        plot_trade_pnl_distribution(
            active_scenario_result,
            FIGURES_DIR / "phase12_trade_pnl_active_conf50.png",
            title="EURUSD M15 — Sensitivity Trade Net P&L Distribution (Conf=0.50)",
        )

    # Metadata
    metadata = {
        "phase": 12,
        "title": "EURUSD M15 Research Trading Strategy & Historical Backtest",
        "instrument": "EURUSD",
        "timeframe": "M15",
        "target": "direction_4",
        "target_threshold": 0.00050,
        "horizon_bars": 4,
        "model_architecture": (
            "RandomForestBaseline(class_weight='balanced', max_depth=10, "
            "min_samples_leaf=20, n_estimators=100)"
        ),
        "random_state": 42,
        "training_partition": {
            "rows": len(splits.train),
            "start": str(splits.train.start_timestamp),
            "end": str(splits.train.end_timestamp),
        },
        "validation_partition": {
            "rows": len(splits.val),
            "start": str(splits.val.start_timestamp),
            "end": str(splits.val.end_timestamp),
        },
        "test_partition_isolation": {
            "locked": True,
            "rows": len(splits.test),
            "start": str(splits.test.start_timestamp),
            "end": str(splits.test.end_timestamp),
            "used_for_tuning": False,
            "used_for_backtest": False,
        },
        "baseline_parameters": {
            "initial_equity": 100_000.0,
            "confidence_threshold": 0.60,
            "atr_period": 14,
            "stop_loss_atr_multiple": 1.0,
            "take_profit_atr_multiple": 1.5,
            "max_holding_bars": 4,
            "cooldown_bars": 1,
            "same_bar_sl_priority": True,
            "default_spread_points": 10.0,
            "slippage_points": 0.0,
            "commission_per_lot": 0.0,
        },
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    with open(REPORTS_DIR / "phase12_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 12 EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Elapsed time: {time.time() - start_time:.2f}s")
    print(f"Baseline trades: {baseline_metrics.total_trades}")
    print(f"Artifacts saved under {REPORTS_DIR}/")


if __name__ == "__main__":
    run_phase12_backtest()
