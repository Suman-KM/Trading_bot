"""Non-executing MT5 broker adapter boundary stub for architecture and safety validation."""

from __future__ import annotations

from typing import Dict, List, Optional

from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.adapter import BrokerAdapter
from trading.execution.capabilities import (
    DEFAULT_MT5_CAPABILITIES,
    BrokerCapabilities,
)
from trading.execution.translation import (
    MT5TradeRequest,
    translate_order_to_mt5_request,
)
from trading.execution.validation import (
    BrokerSymbolSpecification,
    BrokerValidationError,
    validate_order_for_broker,
)
from trading.models.execution import ExecutionReport
from trading.models.order import Order
from trading.models.portfolio import AccountInfo
from trading.models.position import Position


class BrokerExecutionDisabledError(RuntimeError):
    """Raised when an order submission or modification is attempted on a non-executing adapter."""

    pass


class MT5ConnectionError(RuntimeError):
    """Raised when an MT5 RPC/socket connection error or timeout occurs."""

    pass


class MT5ResponseError(RuntimeError):
    """Raised when MT5 returns a malformed or unrecognizable response."""

    pass


class MT5BrokerAdapter(BrokerAdapter):
    """Safe, non-executing MetaTrader 5 broker adapter for architecture validation.

    Absolute Safety Invariants:
    1. Default: is_live is permanently False.
    2. Default: execution_enabled is False.
    3. Any call to submit_order() raises BrokerExecutionDisabledError.
    4. Unsupported broker operations fail closed via UnsupportedBrokerOperationError.
    5. Zero credentials stored or transmitted; zero network sockets opened.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        server: Optional[str] = None,
        execution_enabled: bool = False,
        audit_trail: Optional[AuditTrail] = None,
        capabilities: Optional[BrokerCapabilities] = None,
        symbol_specs: Optional[Dict[str, BrokerSymbolSpecification]] = None,
    ) -> None:
        self._account_id = account_id
        self._server = server
        self._execution_enabled = execution_enabled
        self._audit_trail = audit_trail
        self._capabilities = capabilities or DEFAULT_MT5_CAPABILITIES
        self._symbol_specs = symbol_specs or {
            "EURUSD": BrokerSymbolSpecification(
                symbol="EURUSD",
                min_volume=0.01,
                max_volume=100.0,
                volume_step=0.01,
                contract_size=100_000.0,
                price_digits=5,
                point=0.00001,
                tick_size=0.00001,
            )
        }
        self._simulated_orders: Dict[str, Order] = {}
        self._simulated_executions: List[ExecutionReport] = []

    @property
    def is_live(self) -> bool:
        """Always False. Live execution is forbidden in this phase."""
        return False

    @property
    def execution_enabled(self) -> bool:
        """Declare whether external order submission is enabled (always False by default)."""
        return self._execution_enabled

    @property
    def capabilities(self) -> BrokerCapabilities:
        """Declared broker capabilities."""
        return self._capabilities

    def get_symbol_info(self, symbol: str) -> Optional[BrokerSymbolSpecification]:
        """Fetch broker symbol specifications."""
        return self._symbol_specs.get(symbol.strip().upper())

    def get_account(self) -> AccountInfo:
        """Fetch current simulated/stub account info."""
        return AccountInfo(
            initial_balance=100_000.0,
            cash_balance=100_000.0,
            margin_used=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            equity=100_000.0,
            positions={},
        )

    def get_positions(self) -> Dict[str, Position]:
        """Fetch active open positions."""
        return {}

    def get_position(self, symbol: str) -> Optional[Position]:
        """Fetch an active position by symbol."""
        return None

    def get_order(self, order_id: str) -> Optional[Order]:
        """Fetch order by ID."""
        return self._simulated_orders.get(order_id)

    def get_all_orders(self) -> List[Order]:
        """Fetch all simulated orders."""
        return list(self._simulated_orders.values())

    def get_execution_reports(self) -> List[ExecutionReport]:
        """Fetch execution reports history."""
        return list(self._simulated_executions)

    def update_market_price(self, symbol: str, price: float) -> None:
        """Update market price."""
        pass

    def validate_order(
        self, order: Order, current_market_price: Optional[float] = None
    ) -> BrokerSymbolSpecification:
        """Validate order against broker symbol specification; fail closed if invalid."""
        spec = self.get_symbol_info(order.symbol)
        if spec is None:
            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.ORDER_REJECTED,
                    symbol=order.symbol,
                    order_id=order.order_id,
                    details={"reason": f"Unknown broker symbol: {order.symbol}"},
                )
            raise BrokerValidationError(f"Unknown broker symbol: {order.symbol}")

        try:
            validate_order_for_broker(order, spec, current_market_price=current_market_price)
        except BrokerValidationError as err:
            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.ORDER_REJECTED,
                    symbol=order.symbol,
                    order_id=order.order_id,
                    details={"reason": str(err)},
                )
            raise
        return spec

    def translate_order(
        self, order: Order, current_market_price: Optional[float] = None
    ) -> MT5TradeRequest:
        """Translate order to MT5 trade request."""
        spec = self.validate_order(order, current_market_price=current_market_price)
        return translate_order_to_mt5_request(
            order, spec, current_market_price=current_market_price
        )

    def submit_order(self, order: Order, current_market_price: Optional[float] = None) -> Order:
        """Attempt order submission to broker.

        INVARIANT: If execution_enabled is False (default), unconditionally raise
        BrokerExecutionDisabledError. Zero orders reach an MT5 terminal.
        """
        if not self._execution_enabled:
            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.EXECUTION_ERROR,
                    symbol=order.symbol,
                    order_id=order.order_id,
                    details={
                        "reason": "BROKER_EXECUTION_DISABLED",
                        "adapter": "MT5BrokerAdapter",
                    },
                )
            raise BrokerExecutionDisabledError(
                "MT5_EXECUTION_DISABLED: MT5 broker execution is disabled. "
                "No live or demo orders can be submitted."
            )

        # Fails closed on any unsupported capability
        self._capabilities.assert_supported("supports_market_orders")
        raise NotImplementedError("Live MT5 submission is not implemented in this phase.")

    def cancel_order(self, order_id: str) -> Order:
        """Cancel order."""
        if not self._execution_enabled:
            raise BrokerExecutionDisabledError(
                "MT5_EXECUTION_DISABLED: Cannot cancel orders while execution is disabled."
            )
        self._capabilities.assert_supported("supports_cancel")
        raise NotImplementedError("Live MT5 cancellation is not implemented in this phase.")

    def close_position(self, symbol: str, exit_price: float) -> Order:
        """Close position."""
        if not self._execution_enabled:
            raise BrokerExecutionDisabledError(
                "MT5_EXECUTION_DISABLED: Cannot close positions while execution is disabled."
            )
        raise NotImplementedError("Live MT5 close is not implemented in this phase.")

    def simulate_dry_run_submission(
        self, order: Order, current_market_price: Optional[float] = None
    ) -> MT5TradeRequest:
        """Perform full validation and translation dry-run without submitting order.

        Emits audit events and returns translated trade request for architectural verification.
        """
        trade_req = self.translate_order(order, current_market_price=current_market_price)
        if self._audit_trail:
            self._audit_trail.record(
                event_type=AuditEventType.ORDER_VALIDATED,
                symbol=order.symbol,
                order_id=order.order_id,
                details={
                    "adapter": "MT5BrokerAdapter",
                    "action": "DRY_RUN_VALIDATION_SUCCESS",
                    "volume_lots": trade_req.volume,
                    "price": trade_req.price,
                },
            )
        return trade_req
