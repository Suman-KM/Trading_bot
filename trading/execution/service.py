"""Execution service strictly coordinating the safe signal-to-order pipeline."""

import logging
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, ValidationError

from trading.execution.paper_broker import PaperBroker
from trading.logging_config import log_risk_audit, setup_safety_logger
from trading.models.order import Order
from trading.models.signal import Signal
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskDecision, RiskEngine, RiskReason


class ExecutionResult(BaseModel):
    """Result returned by the trading execution pipeline."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    reason: Optional[str] = None
    order: Optional[Order] = None
    signal_symbol: str
    metadata: Dict[str, Any] = {}


class TradingExecutionService:
    """Coordinates signal validation, risk engine approval, and paper execution."""

    def __init__(
        self,
        risk_engine: RiskEngine,
        paper_broker: PaperBroker,
        portfolio_manager: PortfolioManager,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.risk_engine = risk_engine
        self.broker = paper_broker
        self.portfolio_manager = portfolio_manager
        self.logger = logger or setup_safety_logger()

    def process_signal(
        self,
        signal_input: Union[Signal, Dict[str, Any]],
        current_market_price: Optional[float] = None,
    ) -> ExecutionResult:
        """Process incoming signal through strict safety verification.

        INVARIANT: PaperBroker is NEVER invoked if the RiskEngine rejects or fails.
        """
        # Step 1: Strict Signal Schema Validation
        if isinstance(signal_input, Signal):
            signal = signal_input
        else:
            try:
                signal = Signal.model_validate(signal_input)
            except ValidationError as err:
                error_msg = f"SCHEMA_VALIDATION_ERROR: {err.errors()[0]['msg']}"
                raw_symbol = (
                    str(signal_input.get("symbol", "UNKNOWN"))
                    if isinstance(signal_input, dict)
                    else "UNKNOWN"
                )
                log_risk_audit(
                    logger=self.logger,
                    symbol=raw_symbol,
                    action="UNKNOWN",
                    approved=False,
                    reason=RiskReason.INVALID_SIGNAL.value,
                    extra_msg=error_msg,
                )
                return ExecutionResult(
                    approved=False,
                    reason=RiskReason.INVALID_SIGNAL.value,
                    order=None,
                    signal_symbol=raw_symbol,
                    metadata={"error": error_msg},
                )

        # Step 2: Query current simulated account and daily baseline
        account = self.broker.get_account()
        daily_baseline = self.portfolio_manager.daily_start_equity

        # Step 3: Sovereign Risk Engine Evaluation & Position Sizing
        decision: RiskDecision = self.risk_engine.evaluate(
            signal=signal,
            account=account,
            daily_start_equity=daily_baseline,
            current_market_price=current_market_price,
        )

        # Step 4: Audit Logging
        order_id = decision.order.order_id if decision.order else None
        log_risk_audit(
            logger=self.logger,
            symbol=signal.symbol,
            action=signal.action.value,
            approved=decision.approved,
            reason=decision.reason,
            order_id=order_id,
            confidence=signal.confidence,
        )

        # Step 5: Enforce Non-Bypassable Boundary
        if not decision.approved or decision.order is None:
            # Rejection: Broker MUST NOT be called
            return ExecutionResult(
                approved=False,
                reason=decision.reason,
                order=None,
                signal_symbol=signal.symbol,
                metadata=decision.metadata,
            )

        # Step 6: Execute exclusively on Paper Broker Simulator
        executed_order = self.broker.submit_order(
            decision.order, current_market_price=current_market_price
        )

        return ExecutionResult(
            approved=True,
            reason=None,
            order=executed_order,
            signal_symbol=signal.symbol,
            metadata=decision.metadata,
        )
