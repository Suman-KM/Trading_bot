"""Tick-realistic chronological historical backtesting execution engine.

Executes trade simulation using true millisecond historical tick Bid/Ask quotes:
- Point-in-time signal generation at candle close t
- Realized tick entry execution at first tick of bar t+1:
    LONG enters at actual tick Ask (Bid + spread)
    SHORT enters at actual tick Bid
- Chronological tick-by-tick stop-loss and take-profit monitoring
    LONG: SL touches when Bid <= SL; TP touches when Bid >= TP
    SHORT: SL touches when Ask >= SL; TP touches when Ask <= TP
    Chronological tick precedence strictly resolves intrabar collisions
- Max holding period exit (H=4 bars = 60 minutes) at first tick at/after bar t+4 close
- Floating dynamic spread accounting without fixed assumptions or double-counting
- Strict sovereign RiskEngine enforcement
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from ai.backtest.models import (
    BacktestConfig,
    EquityPoint,
    SimulatedTrade,
    TradeDirection,
    TradeExitReason,
)
from ai.backtest.strategy import MLAssistedStrategy
from ai.backtest.tick_data import TickDataRepository
from trading.models.portfolio import AccountInfo
from trading.risk.engine import RiskDecision, RiskEngine
from trading.risk.kill_switch import KillSwitch


class TickActivePosition:
    """Internal state tracking for an open position in the tick engine."""

    def __init__(
        self,
        trade_id: int,
        symbol: str,
        direction: TradeDirection,
        quantity: float,
        entry_price: float,
        entry_time: datetime,
        signal_time: datetime,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        stop_distance: float,
        risk_amount: float,
        entry_bid: float,
        entry_ask: float,
        entry_spread_pts: float,
        entry_bar_idx: int,
        slippage_entry: float = 0.0,
    ) -> None:
        self.trade_id = trade_id
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.signal_time = signal_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence
        self.stop_distance = stop_distance
        self.risk_amount = risk_amount
        self.entry_bid = entry_bid
        self.entry_ask = entry_ask
        self.entry_spread_pts = entry_spread_pts
        self.entry_bar_idx = entry_bar_idx
        self.slippage_entry = slippage_entry
        self.holding_bars = 0


class TickBacktestResult:
    """Encapsulates results from a tick-realistic backtest execution."""

    def __init__(
        self,
        config: BacktestConfig,
        trade_ledger: list[SimulatedTrade],
        equity_curve: list[EquityPoint],
        rejections: dict[str, int],
        total_bars: int,
        total_signals_evaluated: int,
        signals_generated: int,
        confidence_filtered: int,
        trades_approved: int,
        intrabar_audit: list[dict[str, Any]],
        unavailable_trades: list[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.trade_ledger = trade_ledger
        self.equity_curve = equity_curve
        self.rejections = rejections
        self.total_bars = total_bars
        self.total_signals_evaluated = total_signals_evaluated
        self.signals_generated = signals_generated
        self.confidence_filtered = confidence_filtered
        self.trades_approved = trades_approved
        self.intrabar_audit = intrabar_audit
        self.unavailable_trades = unavailable_trades if unavailable_trades is not None else []

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


class TickBacktestEngine:
    """Historical event-driven tick-realistic backtesting simulation engine."""

    def __init__(
        self,
        strategy: MLAssistedStrategy,
        tick_repo: TickDataRepository,
        config: BacktestConfig | None = None,
        risk_engine: RiskEngine | None = None,
        precomputed_probs: np.ndarray | None = None,
    ) -> None:
        self.strategy = strategy
        self.tick_repo = tick_repo
        self.config = config or BacktestConfig()
        self.precomputed_probs = precomputed_probs

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
    ) -> TickBacktestResult:
        """Execute chronological tick-realistic backtest across the market data.

        Parameters
        ----------
        df_market : pd.DataFrame
            Market dataframe containing:
            ['timestamp', 'open', 'high', 'low', 'close', 'spread', 'atr_14'].
        features_df : pd.DataFrame
            Point-in-time feature matrix matching rows of df_market.
        """
        if len(df_market) != len(features_df):
            raise ValueError(
                f"Market dataframe ({len(df_market)} rows) and features dataframe "
                f"({len(features_df)} rows) must have identical length."
            )

        # Precompute batch model inference if not provided to eliminate row predict_proba overhead
        if self.precomputed_probs is None:
            self.precomputed_probs = self.strategy.model.predict_proba(features_df)

        # State tracking
        cash_balance = float(self.config.initial_equity)
        high_water_mark = cash_balance
        daily_start_equity = cash_balance
        current_date: date | None = None

        position: TickActivePosition | None = None
        pending_signal: dict[str, Any] | None = None
        last_exit_bar: int | None = None
        trade_id_counter = 1

        trade_ledger: list[SimulatedTrade] = []
        equity_curve: list[EquityPoint] = []
        intrabar_audit: list[dict[str, Any]] = []
        unavailable_trades: list[dict[str, Any]] = []
        rejections: dict[str, int] = {}

        total_signals_evaluated = 0
        signals_generated = 0
        confidence_filtered = 0
        trades_approved = 0

        # Pre-extract arrays
        timestamps = pd.to_datetime(df_market["timestamp"]).dt.to_pydatetime()
        closes = df_market["close"].to_numpy(dtype=np.float64)
        atrs = (
            df_market["atr_14"].to_numpy(dtype=np.float64)
            if "atr_14" in df_market.columns
            else np.full(len(df_market), 0.0010)
        )

        n_bars = len(df_market)
        point_val = self.config.point_value
        slippage_pts = self.config.slippage_points
        slippage_price = slippage_pts * point_val

        for i in range(n_bars):
            t_stamp = timestamps[i]
            close_i = closes[i]
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
                daily_start_equity = cash_balance
                if self.kill_switch.is_active() and self.kill_switch.reason == "DAILY_LOSS_LIMIT":
                    self.kill_switch.deactivate()

            # -------------------------------------------------------------
            # STEP 2: Execute Pending Entry from bar i-1 at bar i Start
            # -------------------------------------------------------------
            if pending_signal is not None and position is None:
                p_dir = pending_signal["direction"]
                p_qty = pending_signal["quantity"]
                p_stop_dist = pending_signal["stop_distance"]
                p_tp_dist = pending_signal["tp_distance"]
                p_conf = pending_signal["confidence"]
                p_sig_time = pending_signal["signal_time"]
                p_risk_amt = pending_signal["risk_amount"]

                # Find the first available tick at or after bar i start within continuous coverage
                entry_tick = self.tick_repo.get_first_tick_at_or_after(t_stamp)
                if entry_tick is None:
                    # STRICT DATA INTEGRITY: Do not synthesize fake fills or jump gaps!
                    unavailable_record = {
                        "trade_id": trade_id_counter,
                        "signal_time": p_sig_time.isoformat(),
                        "bar_time": t_stamp.isoformat(),
                        "direction": p_dir.value,
                        "confidence": p_conf,
                        "reason": "ENTRY_TICK_UNAVAILABLE",
                        "status": "DATA_UNAVAILABLE",
                    }
                    unavailable_trades.append(unavailable_record)
                    rejections["TICK_DATA_UNAVAILABLE"] = (
                        rejections.get("TICK_DATA_UNAVAILABLE", 0) + 1
                    )
                    trade_record = SimulatedTrade(
                        trade_id=trade_id_counter,
                        symbol=self.strategy.symbol,
                        signal_time=p_sig_time,
                        entry_time=t_stamp,
                        exit_time=t_stamp,
                        direction=p_dir,
                        confidence=p_conf,
                        entry_price=0.0,
                        exit_price=0.0,
                        stop_loss=0.0,
                        take_profit=0.0,
                        quantity=p_qty,
                        risk_amount=p_risk_amt,
                        gross_pnl=0.0,
                        spread_cost=0.0,
                        commission=0.0,
                        slippage=0.0,
                        net_pnl=0.0,
                        holding_bars=0,
                        exit_reason=TradeExitReason.DATA_UNAVAILABLE,
                        risk_decision="APPROVED",
                        model_version=self.strategy.model_version,
                    )
                    trade_ledger.append(trade_record)
                    trade_id_counter += 1
                    pending_signal = None
                else:
                    entry_bid = entry_tick.bid
                    entry_ask = entry_tick.ask
                    entry_spread_pts = entry_tick.spread_points
                    entry_time = entry_tick.timestamp

                    if p_dir == TradeDirection.LONG:
                        # LONG enters at Ask + slippage
                        fill_price = entry_ask + slippage_price
                        sl_price = fill_price - p_stop_dist
                        tp_price = fill_price + p_tp_dist
                    else:
                        # SHORT enters at Bid - slippage
                        fill_price = entry_bid - slippage_price
                        sl_price = fill_price + p_stop_dist
                        tp_price = fill_price - p_tp_dist

                    position = TickActivePosition(
                        trade_id=trade_id_counter,
                        symbol=self.strategy.symbol,
                        direction=p_dir,
                        quantity=p_qty,
                        entry_price=fill_price,
                        entry_time=entry_time,
                        signal_time=p_sig_time,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        confidence=p_conf,
                        stop_distance=p_stop_dist,
                        risk_amount=p_risk_amt,
                        entry_bid=entry_bid,
                        entry_ask=entry_ask,
                        entry_spread_pts=entry_spread_pts,
                        entry_bar_idx=i,
                        slippage_entry=slippage_price,
                    )
                    trade_id_counter += 1
                    pending_signal = None

            # -------------------------------------------------------------
            # STEP 3: Check Exits on Bar i using Historical Ticks
            # -------------------------------------------------------------
            if position is not None:
                position.holding_bars += 1
                bar_end_time = t_stamp + timedelta(minutes=15)
                is_max_hold_bar = position.holding_bars >= self.config.max_holding_bars

                # Query ticks from position entry (or bar start) to bar end
                tick_start_search = max(position.entry_time, t_stamp)
                t_slice_ms, b_slice, a_slice = self.tick_repo.get_raw_slice(
                    tick_start_search, bar_end_time
                )

                exited = False
                exit_reason = TradeExitReason.MAX_HOLD
                exit_tick_time = bar_end_time
                exit_bid = close_i
                exit_ask = close_i + (self.config.default_spread_points * point_val)
                exit_spread_pts = self.config.default_spread_points
                collision_detected = False
                first_event = None

                n_ticks_in_bar = len(t_slice_ms)
                if n_ticks_in_bar > 0:
                    # Chronological tick-by-tick path scanning
                    for k in range(n_ticks_in_bar):
                        cur_t_ms = int(t_slice_ms[k])
                        cur_dt = datetime.fromtimestamp(cur_t_ms / 1000.0, tz=timezone.utc)
                        cur_bid = float(b_slice[k])
                        cur_ask = float(a_slice[k])

                        if position.direction == TradeDirection.LONG:
                            # LONG: SL when Bid <= stop_loss; TP when Bid >= take_profit
                            hit_sl = cur_bid <= position.stop_loss
                            hit_tp = cur_bid >= position.take_profit

                            if hit_sl and hit_tp:
                                # Within same tick (e.g. quote gap), conservative policy triggers SL
                                exited = True
                                exit_reason = TradeExitReason.STOP_LOSS
                                exit_tick_time = cur_dt
                                exit_bid = cur_bid
                                exit_ask = cur_ask
                                exit_spread_pts = (cur_ask - cur_bid) / point_val
                                collision_detected = True
                                first_event = "STOP_LOSS (simultaneous)"
                                break
                            elif hit_sl:
                                exited = True
                                exit_reason = TradeExitReason.STOP_LOSS
                                exit_tick_time = cur_dt
                                exit_bid = cur_bid
                                exit_ask = cur_ask
                                exit_spread_pts = (cur_ask - cur_bid) / point_val
                                first_event = "STOP_LOSS"
                                break
                            elif hit_tp:
                                exited = True
                                exit_reason = TradeExitReason.TAKE_PROFIT
                                exit_tick_time = cur_dt
                                exit_bid = cur_bid
                                exit_ask = cur_ask
                                exit_spread_pts = (cur_ask - cur_bid) / point_val
                                first_event = "TAKE_PROFIT"
                                break

                        else:  # SHORT
                            # SHORT: SL when Ask >= stop_loss; TP when Ask <= take_profit
                            hit_sl = cur_ask >= position.stop_loss
                            hit_tp = cur_ask <= position.take_profit

                            if hit_sl and hit_tp:
                                exited = True
                                exit_reason = TradeExitReason.STOP_LOSS
                                exit_tick_time = cur_dt
                                exit_bid = cur_bid
                                exit_ask = cur_ask
                                exit_spread_pts = (cur_ask - cur_bid) / point_val
                                collision_detected = True
                                first_event = "STOP_LOSS (simultaneous)"
                                break
                            elif hit_sl:
                                exited = True
                                exit_reason = TradeExitReason.STOP_LOSS
                                exit_tick_time = cur_dt
                                exit_bid = cur_bid
                                exit_ask = cur_ask
                                exit_spread_pts = (cur_ask - cur_bid) / point_val
                                first_event = "STOP_LOSS"
                                break
                            elif hit_tp:
                                exited = True
                                exit_reason = TradeExitReason.TAKE_PROFIT
                                exit_tick_time = cur_dt
                                exit_bid = cur_bid
                                exit_ask = cur_ask
                                exit_spread_pts = (cur_ask - cur_bid) / point_val
                                first_event = "TAKE_PROFIT"
                                break

                    # If not exited by SL/TP and at max-hold bar, exit at final tick
                    if not exited and is_max_hold_bar:
                        exited = True
                        exit_reason = TradeExitReason.MAX_HOLD
                        last_k = n_ticks_in_bar - 1
                        last_ms = int(t_slice_ms[last_k])
                        exit_tick_time = datetime.fromtimestamp(last_ms / 1000.0, tz=timezone.utc)
                        exit_bid = float(b_slice[last_k])
                        exit_ask = float(a_slice[last_k])
                        exit_spread_pts = (exit_ask - exit_bid) / point_val
                        first_event = "MAX_HOLD"

                else:
                    # STRICT DATA INTEGRITY: No tick data in bar. DO NOT use candle extremes!
                    exited = True
                    exit_reason = TradeExitReason.DATA_UNAVAILABLE
                    exit_tick_time = t_stamp
                    exit_bid = position.entry_bid
                    exit_ask = position.entry_ask
                    exit_spread_pts = position.entry_spread_pts
                    first_event = "DATA_UNAVAILABLE"
                    unavailable_record = {
                        "trade_id": position.trade_id,
                        "signal_time": position.signal_time.isoformat(),
                        "entry_time": position.entry_time.isoformat(),
                        "bar_time": t_stamp.isoformat(),
                        "direction": position.direction.value,
                        "reason": "HOLDING_TICKS_UNAVAILABLE",
                        "status": "DATA_UNAVAILABLE",
                    }
                    unavailable_trades.append(unavailable_record)
                    rejections["TICK_DATA_UNAVAILABLE"] = (
                        rejections.get("TICK_DATA_UNAVAILABLE", 0) + 1
                    )

                # Process Completed Exit
                if exited:
                    if exit_reason == TradeExitReason.DATA_UNAVAILABLE:
                        eff_exit_price = position.entry_price
                        p_diff = 0.0
                        spread_cost = 0.0
                        tot_comm = 0.0
                        slippage_cost = 0.0
                        net_pnl = 0.0
                        gross_pnl = 0.0
                    else:
                        if position.direction == TradeDirection.LONG:
                            # LONG sells at Bid - slippage
                            eff_exit_price = exit_bid - slippage_price
                            p_diff = eff_exit_price - position.entry_price
                            spread_cost = position.quantity * (
                                position.entry_ask - position.entry_bid
                            )
                        else:
                            # SHORT buys at Ask + slippage
                            eff_exit_price = exit_ask + slippage_price
                            p_diff = position.entry_price - eff_exit_price
                            spread_cost = position.quantity * (exit_ask - exit_bid)

                        lot_ratio = position.quantity / self.config.lot_size
                        tot_comm = lot_ratio * self.config.commission_per_lot
                        slippage_cost = position.quantity * (
                            position.slippage_entry + slippage_price
                        )
                        net_pnl = (p_diff * position.quantity) - tot_comm
                        gross_pnl = net_pnl + spread_cost + slippage_cost + tot_comm

                        cash_balance += net_pnl

                    completed_trade = SimulatedTrade(
                        trade_id=position.trade_id,
                        symbol=position.symbol,
                        signal_time=position.signal_time,
                        entry_time=position.entry_time,
                        exit_time=exit_tick_time,
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

                    intrabar_audit.append(
                        {
                            "trade_id": position.trade_id,
                            "signal_time": position.signal_time.isoformat(),
                            "entry_time": position.entry_time.isoformat(),
                            "exit_time": exit_tick_time.isoformat(),
                            "direction": position.direction.value,
                            "entry_price": round(position.entry_price, 5),
                            "exit_price": round(eff_exit_price, 5),
                            "entry_spread_pts": round(position.entry_spread_pts, 1),
                            "exit_spread_pts": round(exit_spread_pts, 1),
                            "ticks_processed": n_ticks_in_bar,
                            "first_event": first_event,
                            "collision_detected": collision_detected,
                            "net_pnl": round(net_pnl, 2),
                            "exit_reason": exit_reason.value,
                            "holding_bars": position.holding_bars,
                        }
                    )

                    last_exit_bar = i
                    position = None

            # -------------------------------------------------------------
            # STEP 4: Mark-to-Market Portfolio
            # -------------------------------------------------------------
            current_equity = cash_balance
            current_exposure = 0.0

            if position is not None:
                if position.direction == TradeDirection.LONG:
                    unrealized = (close_i - position.entry_price) * position.quantity
                else:
                    unrealized = (position.entry_price - close_i) * position.quantity
                current_equity += unrealized
                current_exposure = position.quantity * close_i

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
            if position is None and pending_signal is None:
                total_signals_evaluated += 1

                if self.kill_switch.is_active():
                    reason = self.kill_switch.reason or "KILL_SWITCH_ACTIVE"
                    rejections[reason] = rejections.get(reason, 0) + 1
                    continue

                if last_exit_bar is not None and (i - last_exit_bar) <= self.config.cooldown_bars:
                    rejections["COOLDOWN_ACTIVE"] = rejections.get("COOLDOWN_ACTIVE", 0) + 1
                    continue

                # Precomputed inference lookup for bar i
                probs = self.precomputed_probs[i]
                max_idx = int(probs.argmax())
                pred_class = self.strategy.classes[max_idx]
                confidence = float(probs[max_idx])
                eff_threshold = self.config.confidence_threshold

                stop_dist = float(self.config.stop_loss_atr_multiple * atr_i)
                tp_dist = float(self.config.take_profit_atr_multiple * atr_i)

                from trading.models.signal import Signal, SignalAction

                if pred_class == 1.0 and confidence >= eff_threshold:
                    action = SignalAction.BUY
                    sug_sl = close_i - stop_dist
                    sug_tp = close_i + tp_dist
                    exp_ret = 0.00050
                elif pred_class == -1.0 and confidence >= eff_threshold:
                    action = SignalAction.SELL
                    sug_sl = close_i + stop_dist
                    sug_tp = close_i - tp_dist
                    exp_ret = -0.00050
                else:
                    action = SignalAction.HOLD
                    sug_sl = None
                    sug_tp = None
                    exp_ret = 0.0

                signal = Signal(
                    symbol=self.strategy.symbol,
                    action=action,
                    confidence=confidence,
                    timestamp=t_stamp,
                    model_version=self.strategy.model_version,
                    timeframe=self.strategy.timeframe,
                    expected_return=exp_ret,
                    feature_version=self.strategy.feature_version,
                    suggested_entry_price=close_i,
                    suggested_stop_loss=sug_sl,
                    suggested_take_profit=sug_tp,
                )

                if signal.action in (SignalAction.BUY, SignalAction.SELL):
                    signals_generated += 1

                    risk_pct = self.config.risk_limits.MAX_POSITION_RISK_PERCENT / 100.0
                    risk_capital = current_equity * risk_pct
                    if stop_dist > 1e-9:
                        raw_qty = risk_capital / stop_dist
                        if self.config.constrain_exposure:
                            exp_pct = self.config.risk_limits.MAX_TOTAL_EXPOSURE_PERCENT / 100.0
                            max_exp_qty = (current_equity * exp_pct) / close_i
                            qty = round(min(raw_qty, max_exp_qty) - 0.01, 2)
                        else:
                            qty = round(raw_qty, 2)
                    else:
                        qty = 0.0

                    account_snapshot = AccountInfo(
                        initial_balance=self.config.initial_equity,
                        cash_balance=cash_balance,
                        equity=current_equity,
                        positions={},
                    )

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
                        pending_signal = {
                            "direction": direction,
                            "quantity": risk_decision.order.quantity,
                            "stop_distance": stop_dist,
                            "tp_distance": tp_dist,
                            "confidence": signal.confidence,
                            "signal_time": t_stamp,
                            "risk_amount": risk_capital,
                        }
                    else:
                        rej = risk_decision.reason or "UNKNOWN_REJECTION"
                        rejections[rej] = rejections.get(rej, 0) + 1
                else:
                    confidence_filtered += 1

        return TickBacktestResult(
            config=self.config,
            trade_ledger=trade_ledger,
            equity_curve=equity_curve,
            rejections=rejections,
            total_bars=n_bars,
            total_signals_evaluated=total_signals_evaluated,
            signals_generated=signals_generated,
            confidence_filtered=confidence_filtered,
            trades_approved=trades_approved,
            intrabar_audit=intrabar_audit,
            unavailable_trades=unavailable_trades,
        )
