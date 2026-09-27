"""Deterministic, leakage-safe chronological backtest simulation engine.

Executes bar-by-bar historical simulation strictly enforcing:
- Point-in-time signal generation at candle close
- Next-bar executable open price entry
- Bid/Ask spread and slippage cost modeling
- Deterministic same-bar SL/TP collision handling (SL first priority)
- Deterministic RiskEngine authority (non-bypassable)
- Max holding period exit (H=4 bars)
- Cooldown period enforcement (1 bar)
- Emergency kill switch and daily loss limits
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from ai.backtest.costs import CostModel
from ai.backtest.models import (
    ActivePosition,
    BacktestConfig,
    EquityPoint,
    SimulatedTrade,
    TradeDirection,
    TradeExitReason,
)
from ai.backtest.strategy import MLAssistedStrategy
from trading.models.portfolio import AccountInfo
from trading.models.position import Position
from trading.models.signal import SignalAction
from trading.risk.engine import RiskDecision, RiskEngine
from trading.risk.kill_switch import KillSwitch


class BacktestEngine:
    """Historical event-driven bar-by-bar backtesting engine."""

    def __init__(
        self,
        strategy: MLAssistedStrategy,
        config: BacktestConfig | None = None,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.cost_model = CostModel(self.config)

        # Initialize sovereign RiskEngine and KillSwitch
        self.kill_switch = KillSwitch()
        self.risk_engine = risk_engine or RiskEngine(
            limits=self.config.risk_limits,
            kill_switch=self.kill_switch,
        )

    def run(
        self,
        df_market: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> BacktestResult:
        """Execute chronological backtest on historical market and feature data.

        Parameters
        ----------
        df_market : pd.DataFrame
            Market dataframe containing ['timestamp', 'open', 'high', 'low',
            'close', 'spread', 'atr_14'].
        features_df : pd.DataFrame
            Point-in-time feature matrix matching rows of df_market.

        Returns
        -------
        BacktestResult
            Complete results container including trade ledger, equity curve, and metrics.
        """
        if len(df_market) != len(features_df):
            raise ValueError(
                f"Market dataframe ({len(df_market)} rows) and features dataframe "
                f"({len(features_df)} rows) must have identical length."
            )

        # Reset state
        cash_balance = float(self.config.initial_equity)
        high_water_mark = cash_balance
        daily_start_equity = cash_balance
        current_date: date | None = None

        position: ActivePosition | None = None
        pending_signal_data: dict[str, Any] | None = None
        last_exit_bar: int | None = None
        trade_id_counter = 1

        trade_ledger: list[SimulatedTrade] = []
        equity_curve: list[EquityPoint] = []
        rejections: dict[str, int] = {}
        daily_loss_events: list[tuple[datetime, float]] = []

        total_signals_evaluated = 0
        signals_generated = 0
        confidence_filtered = 0
        trades_approved = 0

        # Pre-extract arrays for maximum performance and strict deterministic indexing
        timestamps = pd.to_datetime(df_market["timestamp"]).dt.to_pydatetime()
        opens = df_market["open"].to_numpy(dtype=np.float64)
        highs = df_market["high"].to_numpy(dtype=np.float64)
        lows = df_market["low"].to_numpy(dtype=np.float64)
        closes = df_market["close"].to_numpy(dtype=np.float64)
        spreads = (
            df_market["spread"].to_numpy(dtype=np.float64)
            if "spread" in df_market.columns
            else np.full(len(df_market), self.config.default_spread_points)
        )
        atrs = (
            df_market["atr_14"].to_numpy(dtype=np.float64)
            if "atr_14" in df_market.columns
            else np.full(len(df_market), 0.0010)
        )

        n_bars = len(df_market)

        for i in range(n_bars):
            t_stamp = timestamps[i]
            open_i = opens[i]
            high_i = highs[i]
            low_i = lows[i]
            close_i = closes[i]
            spread_i = spreads[i]
            atr_i = atrs[i]

            # -------------------------------------------------------------
            # STEP 1: UTC Daily Baseline Reset
            # -------------------------------------------------------------
            bar_date = t_stamp.date()
            if current_date is None:
                current_date = bar_date
                daily_start_equity = cash_balance
            elif bar_date > current_date:
                current_date = bar_date
                # Start-of-day equity mark
                daily_start_equity = cash_balance
                if position is not None:
                    # Mark-to-market open position at open
                    if position.direction == TradeDirection.LONG:
                        daily_start_equity += (open_i - position.entry_price) * position.quantity
                    else:
                        sp_open = self.cost_model.get_spread_price(spread_i)
                        p_diff = position.entry_price - (open_i + sp_open)
                        daily_start_equity += p_diff * position.quantity

                # Reset daily loss kill switch if tripped previously
                if self.kill_switch.is_active() and self.kill_switch.reason == "DAILY_LOSS_LIMIT":
                    self.kill_switch.deactivate()

            # -------------------------------------------------------------
            # STEP 2: Execute Pending Entry from bar i-1 at bar i Open
            # -------------------------------------------------------------
            if pending_signal_data is not None and position is None:
                p_dir = pending_signal_data["direction"]
                p_qty = pending_signal_data["quantity"]
                p_stop_dist = pending_signal_data["stop_distance"]
                p_tp_dist = pending_signal_data["tp_distance"]
                p_conf = pending_signal_data["confidence"]
                p_sig_time = pending_signal_data["signal_time"]
                p_risk_amt = pending_signal_data["risk_amount"]

                # Entry execution via cost model
                fill_price, sp_entry, slip_entry, comm_entry = self.cost_model.calculate_entry_fill(
                    bar_open=open_i,
                    direction=p_dir,
                    raw_spread_points=spread_i,
                )

                # Recompute exact SL and TP from actual fill price
                if p_dir == TradeDirection.LONG:
                    sl_price = fill_price - p_stop_dist
                    tp_price = fill_price + p_tp_dist
                else:
                    sl_price = fill_price + p_stop_dist
                    tp_price = fill_price - p_tp_dist

                position = ActivePosition(
                    trade_id=trade_id_counter,
                    symbol=self.strategy.symbol,
                    direction=p_dir,
                    quantity=p_qty,
                    entry_price=fill_price,
                    entry_time=t_stamp,
                    signal_time=p_sig_time,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    confidence=p_conf,
                    stop_distance=p_stop_dist,
                    risk_amount=p_risk_amt,
                    spread_cost_entry=sp_entry,
                    commission_entry=comm_entry,
                    slippage_entry=slip_entry,
                    holding_bars=0,
                )
                trade_id_counter += 1
                pending_signal_data = None

            # -------------------------------------------------------------
            # STEP 3: Check Exits on Bar i (SL, TP, Max Hold)
            # -------------------------------------------------------------
            if position is not None:
                position.holding_bars += 1
                sp_i = self.cost_model.get_spread_price(spread_i)
                exited = False
                exit_reason = TradeExitReason.MAX_HOLD
                nominal_exit_price = close_i

                if position.direction == TradeDirection.LONG:
                    # LONG exits at Bid
                    touched_sl = low_i <= position.stop_loss
                    touched_tp = high_i >= position.take_profit

                    if touched_sl and touched_tp:
                        # Conservative ambiguity policy: Stop Loss assumed hit first
                        exited = True
                        exit_reason = TradeExitReason.STOP_LOSS
                        nominal_exit_price = min(open_i, position.stop_loss)
                    elif touched_sl:
                        exited = True
                        exit_reason = TradeExitReason.STOP_LOSS
                        nominal_exit_price = min(open_i, position.stop_loss)
                    elif touched_tp:
                        exited = True
                        exit_reason = TradeExitReason.TAKE_PROFIT
                        nominal_exit_price = max(open_i, position.take_profit)
                    elif position.holding_bars >= self.config.max_holding_bars:
                        exited = True
                        exit_reason = TradeExitReason.MAX_HOLD
                        nominal_exit_price = close_i
                else:
                    # SHORT exits at Ask (Bid + spread)
                    ask_high = high_i + sp_i
                    ask_low = low_i + sp_i

                    touched_sl = ask_high >= position.stop_loss
                    touched_tp = ask_low <= position.take_profit

                    if touched_sl and touched_tp:
                        # Conservative ambiguity policy: Stop Loss assumed hit first
                        exited = True
                        exit_reason = TradeExitReason.STOP_LOSS
                        nominal_exit_price = max(open_i, position.stop_loss - sp_i)
                    elif touched_sl:
                        exited = True
                        exit_reason = TradeExitReason.STOP_LOSS
                        nominal_exit_price = max(open_i, position.stop_loss - sp_i)
                    elif touched_tp:
                        exited = True
                        exit_reason = TradeExitReason.TAKE_PROFIT
                        nominal_exit_price = min(open_i, position.take_profit - sp_i)
                    elif position.holding_bars >= self.config.max_holding_bars:
                        exited = True
                        exit_reason = TradeExitReason.MAX_HOLD
                        nominal_exit_price = close_i

                # Process Exit
                if exited:
                    eff_exit_price, sp_exit, slip_exit, comm_exit = (
                        self.cost_model.calculate_exit_fill(
                            nominal_exit_price=nominal_exit_price,
                            direction=position.direction,
                            raw_spread_points=spread_i,
                        )
                    )

                    # PnL accounting
                    lot_ratio = position.quantity / self.config.lot_size
                    tot_comm = lot_ratio * self.config.commission_per_lot

                    if position.direction == TradeDirection.LONG:
                        # Cashflow = (Exit_Bid - Entry_Ask) * qty - total_comm
                        p_diff = eff_exit_price - position.entry_price
                        net_pnl = (p_diff * position.quantity) - tot_comm
                        spread_cost = position.quantity * position.spread_cost_entry
                        slippage_cost = position.quantity * (position.slippage_entry + slip_exit)
                        gross_pnl = net_pnl + spread_cost + slippage_cost + tot_comm
                    else:
                        # Cashflow = (Entry_Bid - Exit_Ask) * qty - total_comm
                        p_diff = position.entry_price - eff_exit_price
                        net_pnl = (p_diff * position.quantity) - tot_comm
                        spread_cost = position.quantity * sp_exit
                        slippage_cost = position.quantity * (position.slippage_entry + slip_exit)
                        gross_pnl = net_pnl + spread_cost + slippage_cost + tot_comm

                    cash_balance += net_pnl

                    completed_trade = SimulatedTrade(
                        trade_id=position.trade_id,
                        symbol=position.symbol,
                        signal_time=position.signal_time,
                        entry_time=position.entry_time,
                        exit_time=t_stamp,
                        direction=position.direction,
                        confidence=position.confidence,
                        entry_price=position.entry_price,
                        exit_price=eff_exit_price,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        quantity=position.quantity,
                        risk_amount=position.risk_amount,
                        gross_pnl=gross_pnl,
                        spread_cost=spread_cost,
                        commission=tot_comm,
                        slippage=slippage_cost,
                        net_pnl=net_pnl,
                        holding_bars=position.holding_bars,
                        exit_reason=exit_reason,
                        risk_decision="APPROVED",
                        model_version=self.strategy.model_version,
                    )
                    trade_ledger.append(completed_trade)
                    last_exit_bar = i
                    position = None

            # -------------------------------------------------------------
            # STEP 4: Mark-to-Market Portfolio & Daily Loss Limits
            # -------------------------------------------------------------
            current_equity = cash_balance
            current_exposure = 0.0

            if position is not None:
                sp_i = self.cost_model.get_spread_price(spread_i)
                if position.direction == TradeDirection.LONG:
                    unrealized = (close_i - position.entry_price) * position.quantity
                else:
                    unrealized = (position.entry_price - (close_i + sp_i)) * position.quantity
                current_equity += unrealized
                current_exposure = position.quantity * close_i

            # Check daily loss limit
            if daily_start_equity > 0:
                loss_diff = daily_start_equity - current_equity
                daily_loss_pct = (loss_diff / daily_start_equity) * 100.0
                if daily_loss_pct >= self.config.risk_limits.MAX_DAILY_LOSS_PERCENT:
                    if not self.kill_switch.is_active():
                        self.kill_switch.activate(reason="DAILY_LOSS_LIMIT")
                        daily_loss_events.append((t_stamp, daily_loss_pct))

            # Drawdown tracking
            high_water_mark = max(high_water_mark, current_equity)
            drawdown = high_water_mark - current_equity
            drawdown_pct = (drawdown / high_water_mark) * 100.0 if high_water_mark > 0 else 0.0

            equity_curve.append(
                EquityPoint(
                    timestamp=t_stamp,
                    equity=current_equity,
                    cash=cash_balance,
                    drawdown=drawdown,
                    drawdown_pct=drawdown_pct,
                    open_positions_count=1 if position is not None else 0,
                    exposure=current_exposure,
                )
            )

            # -------------------------------------------------------------
            # STEP 5: Signal Generation & Sovereign RiskEngine Evaluation
            # -------------------------------------------------------------
            # Only evaluate new signals if no position is open and no entry is pending
            if position is None and pending_signal_data is None:
                total_signals_evaluated += 1

                # Check Kill Switch
                if self.kill_switch.is_active():
                    reason = self.kill_switch.reason or "KILL_SWITCH_ACTIVE"
                    rejections[reason] = rejections.get(reason, 0) + 1
                    continue

                # Check Cooldown
                if last_exit_bar is not None and (i - last_exit_bar) <= self.config.cooldown_bars:
                    rejections["COOLDOWN_ACTIVE"] = rejections.get("COOLDOWN_ACTIVE", 0) + 1
                    continue

                # Evaluate strategy on point-in-time features of candle i
                feature_row = features_df.iloc[i]
                signal, _ = self.strategy.evaluate_bar(
                    features_row=feature_row,
                    current_close=close_i,
                    current_atr=atr_i,
                    timestamp=t_stamp,
                )

                if signal.action in (SignalAction.BUY, SignalAction.SELL):
                    signals_generated += 1

                    # Deterministic position sizing
                    risk_pct = self.config.risk_limits.MAX_POSITION_RISK_PERCENT / 100.0
                    risk_capital = current_equity * risk_pct
                    stop_dist = abs(close_i - signal.suggested_stop_loss)

                    if stop_dist > 1e-9:
                        raw_qty = risk_capital / stop_dist
                        if self.config.constrain_exposure:
                            exp_pct = self.config.risk_limits.MAX_TOTAL_EXPOSURE_PERCENT / 100.0
                            max_exposure_qty = (current_equity * exp_pct) / close_i
                            # Enforce both risk capital and gross exposure constraints
                            qty = round(min(raw_qty, max_exposure_qty) - 0.01, 2)
                        else:
                            qty = round(raw_qty, 2)
                    else:
                        qty = 0.0

                    # Construct AccountInfo snapshot for RiskEngine
                    mock_positions: dict[str, Position] = {}
                    account_snapshot = AccountInfo(
                        initial_balance=self.config.initial_equity,
                        cash_balance=cash_balance,
                        equity=current_equity,
                        positions=mock_positions,
                    )

                    # Pass through deterministic RiskEngine
                    risk_decision: RiskDecision = self.risk_engine.evaluate(
                        signal=signal,
                        account=account_snapshot,
                        daily_start_equity=daily_start_equity,
                        current_market_price=close_i,
                        override_quantity=qty if self.config.constrain_exposure else None,
                    )

                    if risk_decision.approved and risk_decision.order is not None:
                        trades_approved += 1
                        direction = (
                            TradeDirection.LONG
                            if signal.action == SignalAction.BUY
                            else TradeDirection.SHORT
                        )
                        pending_signal_data = {
                            "direction": direction,
                            "quantity": risk_decision.order.quantity,
                            "stop_distance": stop_dist,
                            "tp_distance": abs(signal.suggested_take_profit - close_i),
                            "confidence": signal.confidence,
                            "signal_time": t_stamp,
                            "risk_amount": risk_capital,
                        }
                    else:
                        rej_reason = risk_decision.reason or "UNKNOWN_REJECTION"
                        rejections[rej_reason] = rejections.get(rej_reason, 0) + 1
                else:
                    confidence_filtered += 1

        # -----------------------------------------------------------------
        # STEP 6: End of Dataset Handling
        # -----------------------------------------------------------------
        if position is not None:
            last_idx = n_bars - 1
            sp_final = self.cost_model.get_spread_price(spreads[last_idx])
            eff_exit_price, _, slip_final, _ = self.cost_model.calculate_exit_fill(
                nominal_exit_price=closes[last_idx],
                direction=position.direction,
                raw_spread_points=spreads[last_idx],
            )
            tot_comm = (position.quantity / self.config.lot_size) * self.config.commission_per_lot

            if position.direction == TradeDirection.LONG:
                net_pnl = (eff_exit_price - position.entry_price) * position.quantity - tot_comm
                spread_cost = position.quantity * position.spread_cost_entry
                slippage_cost = position.quantity * (position.slippage_entry + slip_final)
                gross_pnl = net_pnl + spread_cost + slippage_cost + tot_comm
            else:
                net_pnl = (position.entry_price - eff_exit_price) * position.quantity - tot_comm
                spread_cost = position.quantity * sp_final
                slippage_cost = position.quantity * (position.slippage_entry + slip_final)
                gross_pnl = net_pnl + spread_cost + slippage_cost + tot_comm

            cash_balance += net_pnl

            final_trade = SimulatedTrade(
                trade_id=position.trade_id,
                symbol=position.symbol,
                signal_time=position.signal_time,
                entry_time=position.entry_time,
                exit_time=timestamps[last_idx],
                direction=position.direction,
                confidence=position.confidence,
                entry_price=position.entry_price,
                exit_price=eff_exit_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                quantity=position.quantity,
                risk_amount=position.risk_amount,
                gross_pnl=gross_pnl,
                spread_cost=spread_cost,
                commission=tot_comm,
                slippage=slippage_cost,
                net_pnl=net_pnl,
                holding_bars=position.holding_bars,
                exit_reason=TradeExitReason.END_OF_DATA,
                risk_decision="APPROVED",
                model_version=self.strategy.model_version,
            )
            trade_ledger.append(final_trade)

        return BacktestResult(
            config=self.config,
            trade_ledger=trade_ledger,
            equity_curve=equity_curve,
            rejections=rejections,
            daily_loss_events=daily_loss_events,
            total_bars=n_bars,
            total_signals_evaluated=total_signals_evaluated,
            signals_generated=signals_generated,
            confidence_filtered=confidence_filtered,
            trades_approved=trades_approved,
        )


class BacktestResult:
    """Encapsulates raw results from a backtest execution."""

    def __init__(
        self,
        config: BacktestConfig,
        trade_ledger: list[SimulatedTrade],
        equity_curve: list[EquityPoint],
        rejections: dict[str, int],
        daily_loss_events: list[tuple[datetime, float]],
        total_bars: int,
        total_signals_evaluated: int,
        signals_generated: int,
        confidence_filtered: int,
        trades_approved: int,
    ) -> None:
        self.config = config
        self.trade_ledger = trade_ledger
        self.equity_curve = equity_curve
        self.rejections = rejections
        self.daily_loss_events = daily_loss_events
        self.total_bars = total_bars
        self.total_signals_evaluated = total_signals_evaluated
        self.signals_generated = signals_generated
        self.confidence_filtered = confidence_filtered
        self.trades_approved = trades_approved

    def to_trades_dataframe(self) -> pd.DataFrame:
        """Convert trade ledger into a structured pandas DataFrame."""
        if not self.trade_ledger:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trade_ledger])

    def to_equity_dataframe(self) -> pd.DataFrame:
        """Convert equity curve into a structured pandas DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "timestamp": ep.timestamp,
                    "equity": ep.equity,
                    "cash": ep.cash,
                    "drawdown": ep.drawdown,
                    "drawdown_pct": ep.drawdown_pct,
                    "open_positions": ep.open_positions_count,
                    "exposure": ep.exposure,
                }
                for ep in self.equity_curve
            ]
        )
