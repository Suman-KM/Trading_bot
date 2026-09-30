#!/usr/bin/env python3
"""Phase 20: Tick-Realistic Historical Backtest Runner & Comparison Study.

Executes and compares:
1. Phase 12 Baseline (Conf=0.60): verifies 0 trades on both candle and tick engines.
2. Phase 12 Pre-existing Research Sensitivity (Conf=0.50):
   - Candle-level execution (60 trades)
   - Tick-realistic execution (actual Bid/Ask quotes, intrabar path, floating spread)
   - Tick-realistic with slippage stress (0.5 pip / 5.0 pts)

Strict Invariants:
- Phase 11 Test Partition (14,988 rows) remains PERMANENTLY LOCKED & UNTOUCHED.
- No strategy/threshold optimization.
- Sovereign RiskEngine authority enforced.
- Batch inference utilized to satisfy Performance Requirement (Section 30).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.backtest.engine import BacktestEngine
from ai.backtest.metrics import calculate_performance_metrics
from ai.backtest.models import BacktestConfig
from ai.backtest.reconciliation import reconcile_candle_vs_tick
from ai.backtest.strategy import MLAssistedStrategy
from ai.backtest.tick_data import TickDataRepository, validate_tick_quality
from ai.backtest.tick_engine import TickBacktestEngine
from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import split_dataset
from ai.models.baselines import RandomForestBaseline
from trading.risk.limits import RiskLimits

REPORTS_DIR = Path("reports")
TICK_PARQUET = Path("data/raw/microstructure_audit/validation_trades_ticks.parquet")


class FastPrecomputedStrategy(MLAssistedStrategy):
    """Strategy wrapper utilizing precomputed batch probabilities for optimal performance."""

    def __init__(
        self,
        model: Any,
        config: BacktestConfig,
        precomputed_probs: np.ndarray,
    ) -> None:
        super().__init__(model=model, config=config)
        self.precomputed_probs = precomputed_probs

    def evaluate_bar(
        self,
        features_row: pd.Series | pd.DataFrame | np.ndarray,
        current_close: float,
        current_atr: float,
        timestamp: datetime,
    ) -> tuple[Any, dict[str, float]]:
        idx = features_row.name if hasattr(features_row, "name") else 0
        probs = self.precomputed_probs[idx]

        prob_dict = {
            "p_short": float(probs[self.short_idx]) if self.short_idx is not None else 0.0,
            "p_neutral": float(probs[self.neutral_idx]) if self.neutral_idx is not None else 0.0,
            "p_long": float(probs[self.long_idx]) if self.long_idx is not None else 0.0,
        }
        max_idx = int(probs.argmax())
        pred_class = self.classes[max_idx]
        confidence = float(probs[max_idx])
        effective_threshold = self.config.confidence_threshold

        stop_distance = float(self.config.stop_loss_atr_multiple * current_atr)
        take_profit_distance = float(self.config.take_profit_atr_multiple * current_atr)

        from trading.models.signal import Signal, SignalAction

        if pred_class == 1.0 and confidence >= effective_threshold:
            action = SignalAction.BUY
            sug_sl = current_close - stop_distance
            sug_tp = current_close + take_profit_distance
            exp_ret = 0.00050
        elif pred_class == -1.0 and confidence >= effective_threshold:
            action = SignalAction.SELL
            sug_sl = current_close + stop_distance
            sug_tp = current_close - take_profit_distance
            exp_ret = -0.00050
        else:
            action = SignalAction.HOLD
            sug_sl = None
            sug_tp = None
            exp_ret = 0.0

        signal = Signal(
            symbol=self.symbol,
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            model_version=self.model_version,
            timeframe=self.timeframe,
            expected_return=exp_ret,
            feature_version=self.feature_version,
            suggested_entry_price=current_close,
            suggested_stop_loss=sug_sl,
            suggested_take_profit=sug_tp,
        )
        return signal, prob_dict


def run_phase20_study() -> None:
    """Execute complete Phase 20 Tick-Realistic Execution comparison."""
    start_time = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 20 — TICK-REALISTIC HISTORICAL EXECUTION & COST ENGINE")
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("=" * 75)

    # 1. Dataset Assembly & Locked Holdout Governance Check
    print("\n[Step 1/7] Assembling dataset and validating holdout partitions...")
    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits = split_dataset(ds)

    print(
        f"  Training partition:   {len(splits.train):,} rows "
        f"({splits.train.start_timestamp} -> {splits.train.end_timestamp})"
    )
    print(
        f"  Validation partition: {len(splits.val):,} rows "
        f"({splits.val.start_timestamp} -> {splits.val.end_timestamp})"
    )
    print(f"  Phase 11 Test partition: {len(splits.test):,} rows [STRICTLY LOCKED]")

    # Strict governance assertion
    assert len(splits.test) == 14988, "CRITICAL: Phase 11 Test partition row count altered!"

    # 2. Extract Validation Market & Feature Data
    print("\n[Step 2/7] Aligning Validation market series and feature matrix...")
    features_full = pd.read_parquet("data/features/eurusd_m15/eurusd_m15_features.parquet")
    market_cols = ["timestamp", "open", "high", "low", "close", "spread", "atr_14"]
    val_mask = features_full["timestamp"].isin(splits.val.timestamps)
    val_market = features_full[val_mask][market_cols].reset_index(drop=True)
    val_features = splits.val.X.reset_index(drop=True)
    assert len(val_market) == len(val_features) == len(splits.val)
    print(f"  Validation dataset ready: {len(val_market):,} bars.")

    # 3. Fit Candidate ML Model Strictly on Training Partition
    print("\n[Step 3/7] Fitting Random Forest strictly on Training data...")
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
    print(f"  Random Forest fitted in {t_fit:.2f}s across {len(splits.train):,} rows.")

    # 4. Batch Model Inference across Validation Features (Section 30 compliance)
    print("\n[Step 4/7] Performing single vectorized batch inference across Validation...")
    t_inf_start = time.time()
    val_probs = rf.predict_proba(val_features)
    t_inf = time.time() - t_inf_start
    print(f"  Vectorized inference completed in {t_inf:.3f}s across {len(val_features):,} bars.")

    # 5. Load & Validate Historical Tick Feed
    print("\n[Step 5/7] Loading and verifying historical tick repository...")
    if not TICK_PARQUET.exists():
        raise FileNotFoundError(f"Required tick parquet archive missing: {TICK_PARQUET}")

    tick_df = pd.read_parquet(TICK_PARQUET)
    tick_audit = validate_tick_quality(tick_df)
    sp_met = tick_audit["spread_metrics"]
    e_ts = tick_audit["earliest_timestamp"]
    l_ts = tick_audit["latest_timestamp"]
    print(f"  Earliest: {e_ts} | Latest: {l_ts}")
    print(f"  Monotonic: {tick_audit['is_monotonic']} | Negative: {tick_audit['negative_spreads']}")
    print(
        f"  Spread mean: {sp_met['mean_points']} pts | "
        f"p95: {sp_met['p95_points']} pts | max: {sp_met['max_points']} pts"
    )
    assert tick_audit["phase11_test_lock_respected"], "CRITICAL: Tick data violates Phase 11 lock!"

    tick_repo = TickDataRepository.from_parquet(TICK_PARQUET)

    # 6. Execute Simulations
    print("\n[Step 6/7] Running comparative execution engines...")

    # A. Phase 12 Baseline (Conf=0.60)
    print("  [A] Baseline Scenario (Conf=0.60, SL=1.0x, TP=1.5x):")
    b_config = BacktestConfig(
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
    b_strat = FastPrecomputedStrategy(model=rf, config=b_config, precomputed_probs=val_probs)
    b_candle_eng = BacktestEngine(strategy=b_strat, config=b_config)
    b_candle_res = b_candle_eng.run(df_market=val_market, features_df=val_features)
    b_tick_eng = TickBacktestEngine(
        strategy=b_strat,
        tick_repo=tick_repo,
        config=b_config,
        precomputed_probs=val_probs,
    )
    b_tick_res = b_tick_eng.run(df_market=val_market, features_df=val_features)

    print(f"      Candle Baseline trades: {len(b_candle_res.trade_ledger)}")
    print(f"      Tick Baseline trades:   {len(b_tick_res.trade_ledger)}")
    assert len(b_candle_res.trade_ledger) == 0, "Baseline candle trades != 0!"
    assert len(b_tick_res.trade_ledger) == 0, "Baseline tick trades != 0!"
    print("      -> Exact Phase 12 Baseline (0 trades) successfully reproduced on both engines.")

    # B. Phase 12 Research Sensitivity Scenario (Conf=0.50, Slippage=0.0, Comm=0.0)
    print("\n  [B] Conf=0.50 Research Sensitivity Scenario (Baseline Frictions):")
    s_config = BacktestConfig(
        initial_equity=100_000.0,
        confidence_threshold=0.50,
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
        risk_limits=RiskLimits(MIN_SIGNAL_CONFIDENCE=0.50),
    )
    s_strat = FastPrecomputedStrategy(model=rf, config=s_config, precomputed_probs=val_probs)
    s_candle_eng = BacktestEngine(strategy=s_strat, config=s_config)
    s_candle_res = s_candle_eng.run(df_market=val_market, features_df=val_features)
    s_candle_met = calculate_performance_metrics(s_candle_res)

    s_tick_eng = TickBacktestEngine(
        strategy=s_strat,
        tick_repo=tick_repo,
        config=s_config,
        precomputed_probs=val_probs,
    )
    s_tick_res = s_tick_eng.run(df_market=val_market, features_df=val_features)
    s_tick_trades_df = s_tick_res.to_trades_dataframe()

    # C. Phase 12 Research Sensitivity Scenario with Slippage Stress (Slippage=5.0 pts / 0.5 pip)
    print("\n  [C] Conf=0.50 Slippage Stress Scenario (0.5 pip / 5.0 pts):")
    slip_config = BacktestConfig(
        initial_equity=100_000.0,
        confidence_threshold=0.50,
        atr_period=14,
        stop_loss_atr_multiple=1.0,
        take_profit_atr_multiple=1.5,
        max_holding_bars=4,
        cooldown_bars=1,
        point_value=1e-5,
        default_spread_points=10.0,
        use_historical_spread=True,
        slippage_points=5.0,
        commission_per_lot=0.0,
        same_bar_sl_priority=True,
        constrain_exposure=True,
        risk_limits=RiskLimits(MIN_SIGNAL_CONFIDENCE=0.50),
    )
    slip_strat = FastPrecomputedStrategy(model=rf, config=slip_config, precomputed_probs=val_probs)
    slip_tick_eng = TickBacktestEngine(
        strategy=slip_strat,
        tick_repo=tick_repo,
        config=slip_config,
        precomputed_probs=val_probs,
    )
    slip_tick_res = slip_tick_eng.run(df_market=val_market, features_df=val_features)
    slip_tick_trades_df = slip_tick_res.to_trades_dataframe()

    # 7. Trade Reconciliation and Analysis
    print("\n[Step 7/7] Reconciling trade ledgers and generating audit reports...")
    candle_trades_df = s_candle_res.to_trades_dataframe()
    rec_df, rec_summary = reconcile_candle_vs_tick(candle_trades_df, s_tick_trades_df)

    rec_df.to_csv(REPORTS_DIR / "phase20_trade_reconciliation.csv", index=False)
    s_tick_trades_df.to_csv(REPORTS_DIR / "phase20_tick_trade_summary.csv", index=False)

    # Compute key performance metrics for tick engine
    tick_wins = int((s_tick_trades_df["net_pnl"] > 0).sum()) if not s_tick_trades_df.empty else 0
    tick_losses = int((s_tick_trades_df["net_pnl"] < 0).sum()) if not s_tick_trades_df.empty else 0
    tick_trades_count = len(s_tick_trades_df)
    tick_win_rate = (tick_wins / tick_trades_count) * 100.0 if tick_trades_count > 0 else 0.0
    tick_gross_profit = (
        float(s_tick_trades_df[s_tick_trades_df["net_pnl"] > 0]["net_pnl"].sum())
        if tick_wins > 0
        else 0.0
    )
    tick_gross_loss = (
        abs(float(s_tick_trades_df[s_tick_trades_df["net_pnl"] < 0]["net_pnl"].sum()))
        if tick_losses > 0
        else 0.0
    )
    tick_pf = (tick_gross_profit / tick_gross_loss) if tick_gross_loss > 0 else 0.0
    tick_net_pnl = float(s_tick_trades_df["net_pnl"].sum()) if not s_tick_trades_df.empty else 0.0
    tick_spread_cost = (
        float(s_tick_trades_df["spread_cost"].sum()) if not s_tick_trades_df.empty else 0.0
    )

    # Max Drawdown for tick
    tick_eq_df = s_tick_res.to_equity_dataframe()
    tick_max_dd_pct = float(tick_eq_df["drawdown_pct"].max()) if not tick_eq_df.empty else 0.0

    # Slippage scenario metrics
    slip_net_pnl = (
        float(slip_tick_trades_df["net_pnl"].sum()) if not slip_tick_trades_df.empty else 0.0
    )
    slip_wins = (
        int((slip_tick_trades_df["net_pnl"] > 0).sum()) if not slip_tick_trades_df.empty else 0
    )
    slip_win_rate = (
        (slip_wins / len(slip_tick_trades_df)) * 100.0 if len(slip_tick_trades_df) > 0 else 0.0
    )

    # Intrabar collision audit
    collisions = [row for row in s_tick_res.intrabar_audit if row["collision_detected"]]

    candle_reasons = {
        str(k): int(v) for k, v in candle_trades_df["exit_reason"].value_counts().items()
    }
    tick_reasons = (
        {str(k): int(v) for k, v in s_tick_trades_df["exit_reason"].value_counts().items()}
        if not s_tick_trades_df.empty
        else {}
    )
    avg_tick_trade = round(tick_net_pnl / tick_trades_count, 2) if tick_trades_count > 0 else 0.0

    comparison_report = {
        "phase": 20,
        "title": "Phase 20 Tick-Realistic Execution vs Candle Simulation Comparison",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_reproduction": {
            "candle_trades": len(b_candle_res.trade_ledger),
            "tick_trades": len(b_tick_res.trade_ledger),
            "confirmed_reproduced": True,
        },
        "conf_50_sensitivity": {
            "candle_engine": {
                "trades": s_candle_met.total_trades,
                "win_rate": s_candle_met.win_rate,
                "profit_factor": s_candle_met.profit_factor,
                "net_pnl": s_candle_met.total_net_pnl,
                "max_drawdown_pct": s_candle_met.max_drawdown_pct,
                "average_trade": s_candle_met.average_trade_return,
                "spread_cost": s_candle_met.total_transaction_costs,
                "exit_reasons": candle_reasons,
            },
            "tick_engine_baseline": {
                "trades": tick_trades_count,
                "win_rate": round(tick_win_rate, 2),
                "profit_factor": round(tick_pf, 4),
                "net_pnl": round(tick_net_pnl, 2),
                "max_drawdown_pct": round(tick_max_dd_pct, 2),
                "average_trade": avg_tick_trade,
                "spread_cost": round(tick_spread_cost, 2),
                "slippage": 0.0,
                "exit_reasons": tick_reasons,
            },
            "tick_engine_slippage_stress": {
                "trades": len(slip_tick_trades_df),
                "win_rate": round(slip_win_rate, 2),
                "net_pnl": round(slip_net_pnl, 2),
                "slippage_points": 5.0,
            },
        },
        "reconciliation_summary": rec_summary,
        "tick_data_summary": tick_audit,
        "collisions_observed": len(collisions),
    }

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    comp_path = REPORTS_DIR / "phase20_execution_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison_report, f, indent=2, default=_json_default)

    total_time = time.time() - start_time

    # Display comparison table
    print("\n" + "=" * 75)
    print("PHASE 20 COMPARATIVE EXECUTION RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<25} {'Candle Engine':>15} {'Tick Engine':>15} {'Tick + Slip':>15}")
    print("-" * 75)
    n_slip = len(slip_tick_trades_df)
    c_tot = s_candle_met.total_trades
    t_tot = tick_trades_count
    print(f"{'Trades Total':<25} {c_tot:>15} {t_tot:>15} {n_slip:>15}")
    c_wr = f"{s_candle_met.win_rate:.2f}%"
    t_wr = f"{tick_win_rate:.2f}%"
    s_wr = f"{slip_win_rate:.2f}%"
    print(f"{'Win Rate (%)':<25} {c_wr:>15} {t_wr:>15} {s_wr:>15}")
    print(f"{'Profit Factor':<25} {s_candle_met.profit_factor:>15.4f} {tick_pf:>15.4f} {'-':>15}")
    c_pnl = f"${s_candle_met.total_net_pnl:.2f}"
    t_pnl = f"${tick_net_pnl:.2f}"
    s_pnl = f"${slip_net_pnl:.2f}"
    print(f"{'Net P&L ($)':<25} {c_pnl:>15} {t_pnl:>15} {s_pnl:>15}")
    c_dd = f"{s_candle_met.max_drawdown_pct:.2f}%"
    t_dd = f"{tick_max_dd_pct:.2f}%"
    print(f"{'Max Drawdown (%)':<25} {c_dd:>15} {t_dd:>15} {'-':>15}")
    c_sp = f"${s_candle_met.total_transaction_costs:.2f}"
    t_sp = f"${tick_spread_cost:.2f}"
    print(f"{'Spread Costs ($)':<25} {c_sp:>15} {t_sp:>15} {'-':>15}")
    print("-" * 75)
    print("Exit Reasons Distribution:")
    c_reasons = dict(candle_trades_df["exit_reason"].value_counts())
    t_reasons = dict(s_tick_trades_df["exit_reason"].value_counts())
    for r in ["STOP_LOSS", "MAX_HOLD", "TAKE_PROFIT"]:
        print(f"  - {r:<20} {c_reasons.get(r, 0):>15} {t_reasons.get(r, 0):>15}")
    print("-" * 75)
    print("Reconciliation Classification:")
    for cls_name, count in rec_summary["classifications"].items():
        print(f"  - {cls_name:<25}: {count:>3} trades")
    print(f"\nNet P&L Difference (Tick - Candle): ${rec_summary['net_pnl_difference']:+.2f}")
    print(f"Total Phase 20 study executed in {total_time:.2f}s.")
    print("=" * 75)


if __name__ == "__main__":
    run_phase20_study()
