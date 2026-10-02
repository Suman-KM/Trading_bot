"""MetaTrader 5 broker adapter with controlled demo execution and fail-closed safety."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from trading.adapters.mt5.client import MT5ReadOnlyClient
    from trading.adapters.mt5.reconciliation import ReconciliationReport

from trading.adapters.mt5.safety import (
    assert_demo_execution_authorized,
)
from trading.adapters.mt5.transport import (
    MT5DemoExecutionTransport,
)
from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.adapter import BrokerAdapter
from trading.execution.capabilities import (
    DEFAULT_MT5_CAPABILITIES,
    BrokerCapabilities,
)
from trading.execution.exceptions import (
    BrokerExecutionDisabledError,
    MT5ConnectionError,
    MT5ResponseError,
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
from trading.models.order import Order, OrderSide, OrderStatus, OrderType
from trading.models.portfolio import AccountInfo
from trading.models.position import Position


class MT5BrokerAdapter(BrokerAdapter):
    """MetaTrader 5 broker adapter supporting controlled DEMO execution with fail-closed safety.

    Absolute Safety Invariants:
    1. Default: is_live is permanently False.
    2. Default: execution_enabled is False.
    3. Live accounts (trade_mode == 2 or REAL) are strictly forbidden; any detected live account
       immediately raises LiveAccountForbiddenError.
    4. Multi-condition demo authorization required before order submission:
       - DEMO_ONLY == True
       - is_live == False
       - execution_enabled == True
       - demo_execution_enabled == True
       - Account verified as DEMO (trade_mode == 0)
    5. Atomic in-memory state rollback on persistence failure via capture_snapshot/restore_snapshot.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        server: Optional[str] = None,
        execution_enabled: bool = False,
        demo_execution_enabled: bool = False,
        transport: Optional[Any] = None,
        audit_trail: Optional[AuditTrail] = None,
        capabilities: Optional[BrokerCapabilities] = None,
        symbol_specs: Optional[Dict[str, BrokerSymbolSpecification]] = None,
        client: Optional[MT5ReadOnlyClient] = None,
        repository: Optional[Any] = None,
        initial_balance: float = 100_000.0,
    ) -> None:
        self._account_id = account_id
        self._server = server
        self._execution_enabled = execution_enabled
        self._demo_execution_enabled = demo_execution_enabled
        self._audit_trail = audit_trail
        self._capabilities = capabilities or DEFAULT_MT5_CAPABILITIES
        self._client = client
        self._repository = repository
        self._initial_balance = float(initial_balance)
        self._cash_balance = float(initial_balance)
        self._realized_pnl = 0.0
        self._total_costs = 0.0

        self.transport = transport
        self._positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        self._orders: Dict[str, Order] = {}
        self._executions: List[ExecutionReport] = []
        self._seen_execution_ids: set[str] = set()

        if self._repository:
            try:
                pos = self._repository.get_active_positions()
                if isinstance(pos, dict):
                    self._positions = pos
                closed = self._repository.get_closed_positions()
                if isinstance(closed, list):
                    self._closed_positions = closed
                orders = self._repository.get_all_orders()
                if isinstance(orders, list):
                    self._orders = {o.order_id: o for o in orders}
                execs = self._repository.get_all_executions()
                if isinstance(execs, list):
                    self._executions = execs
            except Exception:
                pass

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

    @property
    def is_live(self) -> bool:
        """Always False. Live execution is strictly forbidden in this platform."""
        return False

    @property
    def execution_enabled(self) -> bool:
        """Declare whether external order submission is enabled."""
        return self._execution_enabled

    @property
    def demo_execution_enabled(self) -> bool:
        """Declare whether controlled DEMO execution is authorized."""
        return self._demo_execution_enabled

    @property
    def capabilities(self) -> BrokerCapabilities:
        """Declared broker capabilities."""
        return self._capabilities

    @property
    def readonly_client(self) -> Optional[MT5ReadOnlyClient]:
        """Associated read-only client if configured."""
        return self._client

    @property
    def is_readonly_connected(self) -> bool:
        """Return True if read-only client session is connected."""
        return bool(self._client and self._client.is_connected)

    @property
    def market_data_available(self) -> bool:
        """Return True if read-only market data is retrievable."""
        return self.is_readonly_connected

    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture in-memory state for atomic rollback."""
        return {
            "initial_balance": self._initial_balance,
            "cash_balance": self._cash_balance,
            "realized_pnl": self._realized_pnl,
            "total_costs": self._total_costs,
            "positions": {k: v.model_copy(deep=True) for k, v in self._positions.items()},
            "closed_positions": [p.model_copy(deep=True) for p in self._closed_positions],
            "orders": {k: v.model_copy(deep=True) for k, v in self._orders.items()},
            "executions": [r.model_copy(deep=True) for r in self._executions],
            "seen_execution_ids": set(self._seen_execution_ids),
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore in-memory state from snapshot."""
        self._initial_balance = snapshot["initial_balance"]
        self._cash_balance = snapshot["cash_balance"]
        self._realized_pnl = snapshot["realized_pnl"]
        self._total_costs = snapshot["total_costs"]
        self._positions = {k: v.model_copy(deep=True) for k, v in snapshot["positions"].items()}
        self._closed_positions = [p.model_copy(deep=True) for p in snapshot["closed_positions"]]
        self._orders = {k: v.model_copy(deep=True) for k, v in snapshot["orders"].items()}
        self._executions = [r.model_copy(deep=True) for r in snapshot["executions"]]
        self._seen_execution_ids = set(snapshot["seen_execution_ids"])

    def get_symbol_info(self, symbol: str) -> Optional[BrokerSymbolSpecification]:
        """Fetch broker symbol specifications."""
        sym = symbol.strip().upper()
        if self._client and self._client.is_connected:
            try:
                live_spec = self._client.get_symbol_specification(sym)
                self._symbol_specs[sym] = live_spec
                return live_spec
            except Exception:
                pass
        return self._symbol_specs.get(sym)

    def get_account(self) -> AccountInfo:
        """Fetch current account info."""
        if self._client and self._client.is_connected:
            try:
                acc_meta = self._client.get_account_metadata()
                unrealized = round(acc_meta.equity - acc_meta.balance, 2)
                return AccountInfo(
                    initial_balance=acc_meta.balance,
                    cash_balance=acc_meta.balance,
                    margin_used=acc_meta.margin,
                    realized_pnl=self._realized_pnl,
                    unrealized_pnl=unrealized,
                    equity=acc_meta.equity,
                    positions={k: p.model_copy() for k, p in self._positions.items()},
                )
            except Exception:
                pass

        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        equity = self._initial_balance + self._realized_pnl + unrealized
        margin_used = sum(p.quantity * p.current_price for p in self._positions.values())
        return AccountInfo(
            initial_balance=self._initial_balance,
            cash_balance=round(self._cash_balance, 4),
            margin_used=round(margin_used, 4),
            realized_pnl=round(self._realized_pnl, 4),
            unrealized_pnl=round(unrealized, 4),
            equity=round(equity, 4),
            positions={k: p.model_copy() for k, p in self._positions.items()},
        )

    def get_positions(self) -> Dict[str, Position]:
        """Fetch active open positions."""
        return {k: p.model_copy() for k, p in self._positions.items()}

    def get_position(self, symbol: str) -> Optional[Position]:
        """Fetch an active position by symbol."""
        pos = self._positions.get(symbol.strip().upper())
        return pos.model_copy() if pos else None

    def get_order(self, order_id: str) -> Optional[Order]:
        """Fetch order by ID."""
        return self._orders.get(order_id)

    def get_all_orders(self) -> List[Order]:
        """Fetch all orders."""
        return list(self._orders.values())

    def get_execution_reports(self) -> List[ExecutionReport]:
        """Fetch execution reports history."""
        return list(self._executions)

    def update_market_price(self, symbol: str, price: float) -> None:
        """Update market price for open position mark-to-market."""
        sym = symbol.strip().upper()
        if sym in self._positions:
            self._positions[sym].update_price(price)

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
        self,
        order: Order,
        current_market_price: Optional[float] = None,
        position: int = 0,
    ) -> MT5TradeRequest:
        """Translate order to MT5 trade request."""
        spec = self.validate_order(order, current_market_price=current_market_price)
        return translate_order_to_mt5_request(
            order,
            spec,
            current_market_price=current_market_price,
            position=position,
        )

    def submit_order(
        self,
        order: Order,
        current_market_price: Optional[float] = None,
        position_ticket: int = 0,
    ) -> Order:
        """Submit order to MT5 demo broker after multi-condition authorization check.

        Invariants:
        - If execution is disabled or demo is not authorized, fails closed.
        - If connected account is LIVE (2), raises LiveAccountForbiddenError.
        - Enforces atomic state rollback via capture_snapshot/restore_snapshot.
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

        # Fetch account metadata for verification if client available
        acc_meta = None
        if self._client and self._client.is_connected:
            acc_meta = self._client.get_account_metadata()

        # Multi-condition gate evaluation
        assert_demo_execution_authorized(
            is_live=self.is_live,
            execution_enabled=self.execution_enabled,
            demo_execution_enabled=self.demo_execution_enabled,
            account_meta=acc_meta,
        )

        self._capabilities.assert_supported("supports_market_orders")

        snapshot = self.capture_snapshot()
        try:
            return self._submit_order_internal(
                order,
                current_market_price=current_market_price,
                position_ticket=position_ticket,
            )
        except Exception:
            self.restore_snapshot(snapshot)
            raise

    def _submit_order_internal(
        self,
        order: Order,
        current_market_price: Optional[float] = None,
        position_ticket: int = 0,
    ) -> Order:
        if (
            position_ticket == 0
            and order.source == "close_position"
            and self._client
            and self._client.is_connected
        ):
            try:
                open_pos = self._client.get_open_positions(order.symbol)
                if open_pos:
                    position_ticket = int(open_pos[0].get("ticket", 0))
            except Exception:
                pass

        spec = self.validate_order(order, current_market_price=current_market_price)
        trade_req = self.translate_order(
            order,
            current_market_price=current_market_price,
            position=position_ticket,
        )

        # Dispatch via transport
        transport = self.transport or MT5DemoExecutionTransport(client=self._client)
        resp: Any = transport.send_trade_request(trade_req)

        # 1. Handle Rejection
        status_val = resp.status.value if hasattr(resp.status, "value") else str(resp.status)
        if status_val == "REJECTED":
            order.transition_to(OrderStatus.REJECTED)
            order.rejection_reason = resp.rejection_reason or "Broker rejected order."
            self._orders[order.order_id] = order
            if self._repository:
                self._repository.save_order(order)
            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.ORDER_REJECTED,
                    symbol=order.symbol,
                    order_id=order.order_id,
                    details={"reason": resp.rejection_reason},
                )
            return order

        # 2. Handle Duplicate Acknowledgement
        if resp.broker_execution_id in self._seen_execution_ids:
            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.ORDER_FILLED,
                    symbol=order.symbol,
                    order_id=order.order_id,
                    details={
                        "action": "IDEMPOTENT_DUPLICATE_ACK",
                        "broker_execution_id": resp.broker_execution_id,
                    },
                )
            return self._orders.get(order.order_id, order)

        self._seen_execution_ids.add(resp.broker_execution_id)

        # 3. Handle Fill
        fill_qty = resp.filled_volume * spec.contract_size
        exec_price = resp.executed_price or current_market_price or order.price or 1.0

        if status_val == "PARTIALLY_FILLED":
            if order.status != OrderStatus.PARTIALLY_FILLED:
                order.transition_to(OrderStatus.PARTIALLY_FILLED)
        else:
            if order.status != OrderStatus.FILLED:
                order.transition_to(OrderStatus.FILLED)

        order.fill_price = exec_price
        order.fill_timestamp = resp.timestamp
        self._orders[order.order_id] = order

        # 4. Update Position
        symbol_key = order.symbol.strip().upper()
        existing_pos = self._positions.get(symbol_key)
        closed_position_to_save: Optional[Position] = None

        if existing_pos is None:
            new_pos = Position(
                position_id=f"pos-{order.order_id[:8]}",
                symbol=symbol_key,
                side=order.side,
                quantity=fill_qty,
                entry_price=exec_price,
                current_price=exec_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                entry_timestamp=resp.timestamp,
            )
            self._positions[symbol_key] = new_pos
        else:
            if existing_pos.side == order.side:
                # Accumulate position
                total_qty = existing_pos.quantity + fill_qty
                existing_pos.entry_price = (
                    (existing_pos.entry_price * existing_pos.quantity) + (exec_price * fill_qty)
                ) / total_qty
                existing_pos.quantity = total_qty
                existing_pos.current_price = exec_price
            else:
                # Opposite side close / net
                closed_qty = min(existing_pos.quantity, fill_qty)
                pnl = (
                    (exec_price - existing_pos.entry_price) * closed_qty
                    if existing_pos.side == OrderSide.BUY
                    else (existing_pos.entry_price - exec_price) * closed_qty
                )
                self._realized_pnl += pnl
                self._cash_balance += pnl
                remaining_qty = existing_pos.quantity - closed_qty
                if remaining_qty <= 1e-9:
                    existing_pos.close(exit_price=exec_price, exit_timestamp=resp.timestamp)
                    closed_position_to_save = existing_pos.model_copy()
                    self._closed_positions.append(existing_pos.model_copy())
                    del self._positions[symbol_key]
                else:
                    existing_pos.quantity = remaining_qty
                    existing_pos.current_price = exec_price

        # 5. Record Execution Report
        exec_report = ExecutionReport(
            execution_id=resp.broker_execution_id,
            order_id=order.order_id,
            client_request_id=order.client_request_id,
            symbol=order.symbol,
            side=order.side,
            requested_price=order.price or exec_price,
            executed_price=exec_price,
            quantity=fill_qty,
            spread=0.0,
            slippage=0.0,
            commission=0.0,
            execution_status=order.status,
            timestamp=resp.timestamp,
        )
        self._executions.append(exec_report)

        # 6. Persistence
        if self._repository:
            self._repository.save_order(order)
            self._repository.save_execution(exec_report)
            if symbol_key in self._positions:
                self._repository.save_position(self._positions[symbol_key])
            elif closed_position_to_save:
                self._repository.save_position(closed_position_to_save)
            self._sync_snapshot()

        # 7. Audit Trail
        if self._audit_trail:
            self._audit_trail.record(
                event_type=AuditEventType.ORDER_FILLED,
                symbol=order.symbol,
                order_id=order.order_id,
                details={
                    "fill_price": exec_price,
                    "volume_lots": resp.filled_volume,
                    "quantity": fill_qty,
                    "broker_order_id": resp.broker_order_id,
                    "broker_execution_id": resp.broker_execution_id,
                },
            )

        return order

    def _sync_snapshot(self) -> None:
        """Persist account snapshot to repository."""
        if self._repository:
            acc = self.get_account()
            self._repository.save_account_snapshot(
                account=acc,
                total_costs=self._total_costs,
            )

    def close_position(self, symbol: str, exit_price: float) -> Order:
        """Close an active position on EURUSD."""
        if not self._execution_enabled:
            raise BrokerExecutionDisabledError(
                "MT5_EXECUTION_DISABLED: Cannot close positions while execution is disabled."
            )

        acc_meta = None
        if self._client and self._client.is_connected:
            acc_meta = self._client.get_account_metadata()

        assert_demo_execution_authorized(
            is_live=self.is_live,
            execution_enabled=self.execution_enabled,
            demo_execution_enabled=self.demo_execution_enabled,
            account_meta=acc_meta,
        )

        cleaned = symbol.strip().upper()
        pos = self._positions.get(cleaned)
        if pos is None:
            raise KeyError(f"No active position for {cleaned}")

        close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
        close_order = Order(
            symbol=cleaned,
            side=close_side,
            quantity=pos.quantity,
            order_type=OrderType.MARKET,
            price=exit_price,
            source="close_position",
            status=OrderStatus.VALIDATED,
        )

        # Submit closing order
        broker_ticket = 0
        if self._client and self._client.is_connected:
            try:
                open_pos = self._client.get_open_positions(cleaned)
                if open_pos:
                    broker_ticket = int(open_pos[0].get("ticket", 0))
            except Exception:
                pass

        return self.submit_order(
            close_order,
            current_market_price=exit_price,
            position_ticket=broker_ticket,
        )

    def cancel_order(self, order_id: str) -> Order:
        """Cancel order."""
        if not self._execution_enabled:
            raise BrokerExecutionDisabledError(
                "MT5_EXECUTION_DISABLED: Cannot cancel orders while execution is disabled."
            )
        self._capabilities.assert_supported("supports_cancel")
        raise NotImplementedError("Live MT5 cancellation is not implemented in this phase.")

    def simulate_dry_run_submission(
        self, order: Order, current_market_price: Optional[float] = None
    ) -> MT5TradeRequest:
        """Perform full validation and translation dry-run without submitting order."""
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

    def reconcile(self, strict_fail_closed: bool = False) -> ReconciliationReport:
        """Perform reconciliation between local position state and MT5 terminal."""
        from trading.adapters.mt5.reconciliation import reconcile_positions

        broker_positions: List[Dict[str, Any]] = []
        broker_acc = None
        if self._client and self._client.is_connected:
            broker_positions = self._client.get_open_positions()
            acc = self._client.get_account_metadata()
            if acc:
                broker_acc = {"balance": acc.balance, "equity": acc.equity}

        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        internal_equity = self._initial_balance + self._realized_pnl + unrealized
        margin_used = sum(p.quantity * p.current_price for p in self._positions.values())
        internal_acc = AccountInfo(
            initial_balance=self._initial_balance,
            cash_balance=round(self._cash_balance, 4),
            margin_used=round(margin_used, 4),
            realized_pnl=round(self._realized_pnl, 4),
            unrealized_pnl=round(unrealized, 4),
            equity=round(internal_equity, 4),
            positions={k: p.model_copy() for k, p in self._positions.items()},
        )

        spec = self.get_symbol_info("EURUSD")
        contract_size = spec.contract_size if spec else 100_000.0

        return reconcile_positions(
            internal_positions=self._positions,
            broker_positions=broker_positions,
            internal_account=internal_acc,
            broker_account=broker_acc,
            contract_size=contract_size,
            audit_trail=self._audit_trail,
            strict_fail_closed=strict_fail_closed,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics."""
        account = self.get_account()
        orders = list(self._orders.values())
        return {
            "orders_received": len(orders),
            "orders_accepted": sum(1 for o in orders if o.status == OrderStatus.FILLED),
            "orders_rejected": sum(1 for o in orders if o.status == OrderStatus.REJECTED),
            "orders_filled": sum(1 for o in orders if o.status == OrderStatus.FILLED),
            "orders_cancelled": sum(1 for o in orders if o.status == OrderStatus.CANCELLED),
            "execution_errors": sum(1 for o in orders if o.status == OrderStatus.REJECTED),
            "open_positions": len(self._positions),
            "closed_positions": len(self._closed_positions),
            "gross_pnl": 0.0,
            "total_costs": round(self._total_costs, 4),
            "net_pnl": round(self._realized_pnl, 4),
            "daily_loss": 0.0,
            "current_exposure": sum(p.quantity * p.current_price for p in self._positions.values()),
            "equity": account.equity,
            "cash_balance": account.cash_balance,
        }


__all__ = [
    "BrokerExecutionDisabledError",
    "MT5BrokerAdapter",
    "MT5ConnectionError",
    "MT5ResponseError",
]
