"""Deterministic market-data replay engine for paper trading verification."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading.execution.paper_broker import PaperBroker
from trading.execution.service import TradingExecutionService
from trading.models.order import OrderSide
from trading.models.signal import Signal
from trading.replay.models import MarketEvent, ReplayCheckpoint, ReplayMode

logger = logging.getLogger(__name__)


class ReplayEngine:
    """Deterministic replay engine streaming historical market events into PaperBroker.

    Invariants:
    - Zero predictive signal generation; executes only deterministic pre-scheduled orders.
    - Deterministic bracket evaluation: When both SL and TP are touched in the same bar,
      Stop-Loss is prioritized first (capital-preservation collision rule).
    - Checkpoint/resume capability producing byte-identical final states.
    """

    def __init__(
        self,
        broker: Optional[PaperBroker] = None,
        events: Optional[List[MarketEvent]] = None,
        execution_service: Optional[TradingExecutionService] = None,
        mode: ReplayMode = ReplayMode.FAST_REPLAY,
        speedup: float = 100.0,
        paper_broker: Optional[PaperBroker] = None,
    ) -> None:
        self.broker = broker or paper_broker
        if self.broker is None:
            raise ValueError("Either broker or paper_broker must be provided to ReplayEngine.")
        self.events = events or []
        self.execution_service = execution_service
        self.mode = mode
        self.speedup = max(1.0, float(speedup))

        self._cursor: int = 0
        self._scheduled_orders: List[Dict[str, Any]] = []
        self._executed_scheduled_orders: List[str] = []
        self._last_event_time: Optional[datetime] = None
        self._executed_brackets_count: int = 0

    def schedule_order(
        self,
        trigger_time: datetime,
        signal: Signal,
        schedule_id: Optional[str] = None,
    ) -> str:
        """Schedule a deterministic test signal to be evaluated at or after trigger_time."""
        sid = schedule_id or f"sched-{uuid.uuid4().hex[:8]}"
        t_time = trigger_time if trigger_time.tzinfo else trigger_time.replace(tzinfo=timezone.utc)
        self._scheduled_orders.append(
            {
                "id": sid,
                "trigger_time": t_time,
                "signal": signal,
                "executed": False,
            }
        )
        return sid

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def is_finished(self) -> bool:
        return self._cursor >= len(self.events)

    def step(self) -> Optional[MarketEvent]:
        """Process exactly one market event in chronological sequence."""
        if self.is_finished:
            return None

        event = self.events[self._cursor]

        # Handle real-time simulated pacing if configured
        if self.mode == ReplayMode.REALTIME_SIMULATED_REPLAY and self._last_event_time is not None:
            time_delta = (event.timestamp - self._last_event_time).total_seconds()
            if time_delta > 0:
                sleep_duration = min(time_delta / self.speedup, 1.0)
                time.sleep(sleep_duration)

        # 1. Check for scheduled test orders triggered at this timestamp
        current_price = event.close or event.mid or event.last or 1.0
        if self.execution_service is not None:
            for item in self._scheduled_orders:
                if not item["executed"] and event.timestamp >= item["trigger_time"]:
                    item["executed"] = True
                    self._executed_scheduled_orders.append(item["id"])
                    self.execution_service.process_signal(
                        signal_input=item["signal"],
                        current_market_price=current_price,
                    )

        # 2. Evaluate bracket orders and update mark-to-market valuations
        self._process_market_event(event)

        self._last_event_time = event.timestamp
        self._cursor += 1
        return event

    def _process_market_event(self, event: MarketEvent) -> None:
        """Update market price and evaluate bracket collision rules deterministically."""
        pos = self.broker.get_position(event.symbol)
        if pos is None:
            return

        effective_price = event.close or event.mid or event.last or pos.current_price

        # Check bracket triggers if candle extremes are available
        hi = event.high if event.high is not None else effective_price
        lo = event.low if event.low is not None else effective_price

        sl_hit = False
        tp_hit = False
        sl_price = pos.stop_loss or effective_price
        tp_price = pos.take_profit or effective_price

        if pos.side == OrderSide.BUY:
            if pos.stop_loss is not None and lo <= pos.stop_loss:
                sl_hit = True
                sl_price = pos.stop_loss
            if pos.take_profit is not None and hi >= pos.take_profit:
                tp_hit = True
                tp_price = pos.take_profit
        else:  # SELL
            if pos.stop_loss is not None and hi >= pos.stop_loss:
                sl_hit = True
                sl_price = pos.stop_loss
            if pos.take_profit is not None and lo <= pos.take_profit:
                tp_hit = True
                tp_price = pos.take_profit

        # Deterministic collision rule: In the event both bounds are touched,
        # Stop-Loss is executed first (capital preservation invariant)
        if sl_hit and tp_hit:
            logger.info("Replay bracket collision on %s: triggering STOP-LOSS first", event.symbol)
            self.broker.close_position(event.symbol, exit_price=sl_price)
            self._executed_brackets_count += 1
        elif sl_hit:
            self.broker.close_position(event.symbol, exit_price=sl_price)
            self._executed_brackets_count += 1
        elif tp_hit:
            self.broker.close_position(event.symbol, exit_price=tp_price)
            self._executed_brackets_count += 1
        else:
            # Simple mark-to-market price update
            self.broker.update_market_price(event.symbol, price=effective_price)

    def run(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Execute replay up to max_steps or end of event stream."""
        steps_executed = 0
        while not self.is_finished:
            if max_steps is not None and steps_executed >= max_steps:
                break
            self.step()
            steps_executed += 1
        return self.get_summary()

    def create_checkpoint(self, checkpoint_id: Optional[str] = None) -> ReplayCheckpoint:
        """Create a serializable checkpoint of current replay state."""
        cid = checkpoint_id or f"chkpt-{self._cursor}-{uuid.uuid4().hex[:8]}"
        account = self.broker.get_account()
        active_pos = {k: p.model_dump(mode="json") for k, p in self.broker.get_positions().items()}
        current_ts = (
            self.events[self._cursor - 1].timestamp
            if self._cursor > 0
            else (self.events[0].timestamp if self.events else datetime.now(timezone.utc))
        )
        return ReplayCheckpoint(
            checkpoint_id=cid,
            timestamp=current_ts,
            cursor_index=self._cursor,
            account_state=account.model_dump(mode="json"),
            active_positions=active_pos,
            closed_positions_count=len(self.broker.get_closed_positions()),
            orders_count=len(self.broker.get_all_orders()),
            executions_count=len(self.broker.get_execution_reports()),
            kill_switch_active=False,
            metadata={
                "executed_brackets_count": self._executed_brackets_count,
                "scheduled_executed": list(self._executed_scheduled_orders),
            },
        )

    def resume_from_checkpoint(self, checkpoint: ReplayCheckpoint) -> None:
        """Resume replay state from a previously saved checkpoint."""
        self._cursor = checkpoint.cursor_index
        self._executed_brackets_count = int(checkpoint.metadata.get("executed_brackets_count", 0))
        sched_exec = set(checkpoint.metadata.get("scheduled_executed", []))
        for item in self._scheduled_orders:
            if item["id"] in sched_exec:
                item["executed"] = True

    def get_summary(self) -> Dict[str, Any]:
        """Return operational summary of the replay session."""
        account = self.broker.get_account()
        metrics = self.broker.get_metrics()
        return {
            "mode": self.mode.value,
            "events_replayed": self._cursor,
            "total_events": len(self.events),
            "is_finished": self.is_finished,
            "final_equity": account.equity,
            "cash_balance": account.cash_balance,
            "realized_pnl": account.realized_pnl,
            "unrealized_pnl": account.unrealized_pnl,
            "open_positions": len(self.broker.get_positions()),
            "closed_positions": len(self.broker.get_closed_positions()),
            "orders_count": len(self.broker.get_all_orders()),
            "brackets_triggered": self._executed_brackets_count,
            "metrics": metrics,
        }
