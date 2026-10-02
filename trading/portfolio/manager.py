import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from trading.execution.paper_broker import PaperBroker
from trading.models.portfolio import AccountInfo


class PortfolioManager:
    """Manages simulated portfolio state, daily loss calculations, and market exposure."""

    def __init__(
        self,
        broker: PaperBroker,
        initial_daily_equity: Optional[float] = None,
        repository: Optional[Any] = None,
        today_utc: Optional[Any] = None,
    ) -> None:
        self.broker = broker
        self.repository = repository
        current_account = self.broker.get_account()
        start_equity = initial_daily_equity or current_account.equity
        if not math.isfinite(start_equity) or start_equity <= 0:
            raise ValueError(
                f"Daily start equity must be a positive finite number, got {start_equity}"
            )
        self._daily_start_equity: float = float(start_equity)
        if today_utc is not None:
            self._daily_date: str = (
                today_utc.strftime("%Y-%m-%d") if hasattr(today_utc, "strftime") else str(today_utc)
            )
        else:
            self._daily_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.repository:
            self.recover_from_repository(today_utc=self._daily_date)

    def set_repository(self, repository: Any) -> None:
        """Attach persistent repository and recover state."""
        self.repository = repository
        if self.repository:
            self.recover_from_repository(today_utc=self._daily_date)

    def recover_from_repository(self, today_utc: Optional[Any] = None) -> None:
        """Recover daily risk baseline using UTC calendar date."""
        if not self.repository:
            return
        state = self.repository.get_risk_state()
        if today_utc is not None:
            check_date = (
                today_utc.strftime("%Y-%m-%d") if hasattr(today_utc, "strftime") else str(today_utc)
            )
        else:
            check_date = self._daily_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_date = check_date
        if state:
            persisted_date = state.get("date_utc")
            if persisted_date == check_date:
                # Same UTC calendar day: preserve baseline equity
                self._daily_start_equity = float(
                    state.get("baseline_equity", self._daily_start_equity)
                )
            else:
                # Next calendar day: roll forward daily baseline
                current_account = self.broker.get_account()
                self._daily_start_equity = current_account.equity
                self._sync_to_repository()
        else:
            self._sync_to_repository()

    def _sync_to_repository(self) -> None:
        """Sync daily risk state to persistent repository."""
        if not self.repository:
            return
        date_str = self._daily_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state = self.repository.get_risk_state() or {}
        ks_active = bool(state.get("kill_switch_active", 0))
        ks_reason = state.get("kill_switch_reason")
        ks_ts = state.get("kill_switch_timestamp")
        daily_loss_pct = self.get_daily_loss_percent()
        self.repository.save_risk_state(
            date_utc=date_str,
            daily_loss=daily_loss_pct,
            baseline_equity=self._daily_start_equity,
            kill_switch_active=ks_active,
            kill_switch_reason=ks_reason,
            kill_switch_timestamp=ks_ts,
        )

    @property
    def daily_start_equity(self) -> float:
        """Return the baseline equity recorded at the start of the trading day."""
        return self._daily_start_equity

    def reset_daily_baseline(self, new_baseline: Optional[float] = None) -> None:
        """Reset the baseline equity for a new trading day."""
        account = self.broker.get_account()
        baseline = new_baseline or account.equity
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError(f"Baseline must be a positive finite number, got {baseline}")
        self._daily_start_equity = float(baseline)
        self._sync_to_repository()

    def get_daily_loss_percent(self) -> float:
        """Calculate percentage loss relative to daily baseline equity."""
        current_equity = self.broker.get_account().equity
        if self._daily_start_equity <= 0:
            return 0.0
        drawdown = ((self._daily_start_equity - current_equity) / self._daily_start_equity) * 100.0
        return max(0.0, round(drawdown, 4))

    def get_total_exposure_percent(self) -> float:
        """Calculate gross portfolio exposure as a percentage of current equity."""
        account = self.broker.get_account()
        if account.equity <= 0:
            return 100.0
        gross_exposure = sum(pos.quantity * pos.current_price for pos in account.positions.values())
        return round((gross_exposure / account.equity) * 100.0, 4)

    def get_open_position_count(self) -> int:
        """Return number of currently open positions."""
        return len(self.broker.get_positions())

    def get_account_info(self) -> AccountInfo:
        """Return current account snapshot from broker."""
        return self.broker.get_account()

    def get_summary(self) -> Dict[str, Any]:
        """Return comprehensive portfolio and risk summary."""
        account = self.get_account_info()
        return {
            "daily_start_equity": self._daily_start_equity,
            "equity": account.equity,
            "cash_balance": account.cash_balance,
            "realized_pnl": account.realized_pnl,
            "unrealized_pnl": account.unrealized_pnl,
            "daily_loss_percent": self.get_daily_loss_percent(),
            "total_exposure_percent": self.get_total_exposure_percent(),
            "open_positions_count": len(account.positions),
            "positions": {k: p.model_dump() for k, p in account.positions.items()},
        }
