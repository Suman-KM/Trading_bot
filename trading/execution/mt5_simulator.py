"""Deterministic, simulated MT5 broker transport and adapter for end-to-end execution testing."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.adapter import BrokerAdapter
from trading.execution.capabilities import (
    DEFAULT_MT5_CAPABILITIES,
    BrokerCapabilities,
)
from trading.execution.costs import TransactionCostConfig
from trading.execution.exceptions import (
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


class SimulatedResponseStatus(str, Enum):
    """Simulated MT5 trade response status types."""

    ACCEPTED = "ACCEPTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    DUPLICATE_ACK = "DUPLICATE_ACK"
    CONNECTION_ERROR = "CONNECTION_ERROR"


class SimulatedBrokerResponse(BaseModel):
    """Deterministic simulated broker response structure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    client_request_id: Optional[str] = Field(None, description="Idempotent client request ID")
    internal_order_id: str = Field(..., description="System internal order UUID")
    broker_order_id: str = Field(..., description="Simulated broker ticket / order ID")
    broker_execution_id: str = Field(..., description="Simulated broker deal / fill ID")
    status: SimulatedResponseStatus = Field(..., description="Simulated return status")
    symbol: str = Field(..., description="Trading instrument symbol")
    filled_volume: float = Field(..., ge=0.0, description="Volume filled in lots")
    remaining_volume: float = Field(..., ge=0.0, description="Remaining volume in lots")
    executed_price: Optional[float] = Field(None, description="Executed fill price")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason if rejected")
    timestamp: datetime = Field(..., description="Response UTC timestamp")


class SimulatedMT5Transport:
    """Safe, deterministic in-memory simulated transport.

    ABSOLUTE SAFETY INVARIANTS:
    - Zero network sockets opened.
    - Zero calls to MT5 or order_send().
    - Deterministic responses according to configured simulation mode.
    """

    def __init__(self, mode: str = "FILLED") -> None:
        self.mode = mode
        self.submissions: List[MT5TradeRequest] = []
        self.responses: List[SimulatedBrokerResponse] = []
        self.partial_fill_fraction: float = 0.4
        self._partial_fill_state: Dict[str, int] = {}
        self._execution_counter: int = 1000

    @property
    def submission_count(self) -> int:
        """Count of trade requests submitted to this transport."""
        return len(self.submissions)

    def reset(self) -> None:
        """Reset submission history and state."""
        self.submissions.clear()
        self.responses.clear()
        self._partial_fill_state.clear()
        self._execution_counter = 1000

    def send_trade_request(self, req: MT5TradeRequest) -> SimulatedBrokerResponse:
        """Simulate sending trade request to broker; returns deterministic response."""
        self.submissions.append(req)
        now_utc = datetime.now(timezone.utc)
        self._execution_counter += 1
        broker_ticket = f"mt5-ord-{self._execution_counter}"
        deal_id = f"mt5-deal-{self._execution_counter}"

        # 1. Connection Error Simulation
        if self.mode == "CONNECTION_ERROR":
            raise MT5ConnectionError(
                "SIMULATED_CONNECTION_DROPPED: Connection to broker was dropped."
            )

        # 2. Timeout Simulation
        if self.mode == "TIMEOUT":
            raise MT5ConnectionError(
                "SIMULATED_BROKER_TIMEOUT: Broker did not respond within timeout window."
            )

        # 3. Malformed Response Simulation
        if self.mode == "MALFORMED_RESPONSE":
            raise MT5ResponseError(
                "SIMULATED_MALFORMED_RESPONSE: "
                "Broker returned non-finite price or missing execution ID."
            )

        # 4. Rejection Simulation
        if self.mode == "REJECTED":
            resp = SimulatedBrokerResponse(
                client_request_id=req.client_request_id,
                internal_order_id=req.internal_order_id,
                broker_order_id=broker_ticket,
                broker_execution_id=deal_id,
                status=SimulatedResponseStatus.REJECTED,
                symbol=req.symbol,
                filled_volume=0.0,
                remaining_volume=req.volume,
                executed_price=None,
                rejection_reason="SIMULATED_BROKER_REJECTION: Insufficient broker liquidity.",
                timestamp=now_utc,
            )
            self.responses.append(resp)
            return resp

        # 5. Partial Fill Simulation
        if self.mode == "PARTIALLY_FILLED":
            step = self._partial_fill_state.get(req.internal_order_id, 0)
            if step == 0:
                first_fill_vol = round(req.volume * self.partial_fill_fraction, 4)
                remaining = round(req.volume - first_fill_vol, 4)
                self._partial_fill_state[req.internal_order_id] = 1
                resp = SimulatedBrokerResponse(
                    client_request_id=req.client_request_id,
                    internal_order_id=req.internal_order_id,
                    broker_order_id=broker_ticket,
                    broker_execution_id=deal_id,
                    status=SimulatedResponseStatus.PARTIALLY_FILLED,
                    symbol=req.symbol,
                    filled_volume=first_fill_vol,
                    remaining_volume=remaining,
                    executed_price=req.price,
                    rejection_reason=None,
                    timestamp=now_utc,
                )
            else:
                remaining = round(req.volume * (1.0 - self.partial_fill_fraction), 4)
                resp = SimulatedBrokerResponse(
                    client_request_id=req.client_request_id,
                    internal_order_id=req.internal_order_id,
                    broker_order_id=broker_ticket,
                    broker_execution_id=deal_id,
                    status=SimulatedResponseStatus.FILLED,
                    symbol=req.symbol,
                    filled_volume=remaining,
                    remaining_volume=0.0,
                    executed_price=req.price,
                    rejection_reason=None,
                    timestamp=now_utc,
                )
            self.responses.append(resp)
            return resp

        # 6. Default: Full Fill Simulation
        resp = SimulatedBrokerResponse(
            client_request_id=req.client_request_id,
            internal_order_id=req.internal_order_id,
            broker_order_id=broker_ticket,
            broker_execution_id=deal_id,
            status=SimulatedResponseStatus.FILLED,
            symbol=req.symbol,
            filled_volume=req.volume,
            remaining_volume=0.0,
            executed_price=req.price,
            rejection_reason=None,
            timestamp=now_utc,
        )
        self.responses.append(resp)
        return resp


class SimulatedMT5BrokerAdapter(BrokerAdapter):
    """Adapter that routes validated orders through SimulatedMT5Transport for pipeline verification.

    Maintains simulated in-memory and persistent accounting, position tracking,
    and audit trail logging.
    """

    def __init__(
        self,
        transport: Optional[SimulatedMT5Transport] = None,
        initial_balance: float = 100_000.0,
        symbol_specs: Optional[Dict[str, BrokerSymbolSpecification]] = None,
        cost_config: Optional[TransactionCostConfig] = None,
        audit_trail: Optional[AuditTrail] = None,
        repository: Optional[Any] = None,
    ) -> None:
        self.transport = transport or SimulatedMT5Transport()
        self._initial_balance = float(initial_balance)
        self._cash_balance = float(initial_balance)
        self._realized_pnl = 0.0
        self._total_costs = 0.0
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._executions: List[ExecutionReport] = []
        self._seen_execution_ids: set[str] = set()
        self._cost_config = cost_config or TransactionCostConfig()
        self._audit_trail = audit_trail
        self._repository = repository
        self._symbol_specs = symbol_specs or {
            "EURUSD": BrokerSymbolSpecification(
                symbol="EURUSD",
                min_volume=0.01,
                max_volume=500.0,
                volume_step=0.01,
                contract_size=100_000.0,
                price_digits=5,
                point=0.00001,
                tick_size=0.00001,
            )
        }

    @property
    def is_live(self) -> bool:
        """Always False. Simulated broker only."""
        return False

    @property
    def is_simulated(self) -> bool:
        """Explicit declaration that this is a simulated broker backend."""
        return True

    @property
    def execution_enabled(self) -> bool:
        """True for simulation harness."""
        return True

    @property
    def capabilities(self) -> BrokerCapabilities:
        """Declared simulation capabilities."""
        return DEFAULT_MT5_CAPABILITIES

    def get_symbol_info(self, symbol: str) -> Optional[BrokerSymbolSpecification]:
        """Fetch specification details for symbol."""
        return self._symbol_specs.get(symbol.strip().upper())

    def get_account(self) -> AccountInfo:
        """Return simulated account equity and cash."""
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
        """Return copy of active open positions."""
        return {k: p.model_copy() for k, p in self._positions.items()}

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return active open position for symbol."""
        pos = self._positions.get(symbol.strip().upper())
        return pos.model_copy() if pos else None

    def get_all_orders(self) -> List[Order]:
        """Return list of all orders."""
        return list(self._orders.values())

    def get_order(self, order_id: str) -> Optional[Order]:
        """Fetch order by ID."""
        return self._orders.get(order_id)

    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture in-memory state for atomic rollback."""
        return {
            "initial_balance": self._initial_balance,
            "cash_balance": self._cash_balance,
            "realized_pnl": self._realized_pnl,
            "total_costs": self._total_costs,
            "positions": {k: v.model_copy(deep=True) for k, v in self._positions.items()},
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
        self._orders = {k: v.model_copy(deep=True) for k, v in snapshot["orders"].items()}
        self._executions = [r.model_copy(deep=True) for r in snapshot["executions"]]
        self._seen_execution_ids = set(snapshot["seen_execution_ids"])

    def submit_order(self, order: Order, current_market_price: Optional[float] = None) -> Order:
        """Validate, translate, simulate broker execution, and update local state."""
        snapshot = self.capture_snapshot()
        try:
            return self._submit_order_internal(order, current_market_price=current_market_price)
        except Exception:
            self.restore_snapshot(snapshot)
            raise

    def _submit_order_internal(
        self, order: Order, current_market_price: Optional[float] = None
    ) -> Order:
        spec = self.get_symbol_info(order.symbol)
        if spec is None:
            raise BrokerValidationError(f"Unknown symbol: {order.symbol}")

        # 1. Broker Compatibility Validation
        validate_order_for_broker(order, spec, current_market_price=current_market_price)

        # 2. MT5 Request Translation
        trade_req = translate_order_to_mt5_request(
            order, spec, current_market_price=current_market_price
        )

        # 3. Dispatch to Simulated Transport
        sim_resp = self.transport.send_trade_request(trade_req)

        # 4. Handle Simulated Rejection
        if sim_resp.status == SimulatedResponseStatus.REJECTED:
            order.transition_to(OrderStatus.REJECTED)
            order.rejection_reason = sim_resp.rejection_reason
            self._orders[order.order_id] = order
            if self._repository:
                self._repository.save_order(order)
            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.ORDER_REJECTED,
                    symbol=order.symbol,
                    order_id=order.order_id,
                    details={"reason": sim_resp.rejection_reason},
                )
            return order

        # 5. Handle Duplicate Acknowledgement
        if sim_resp.broker_execution_id in self._seen_execution_ids:
            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.ORDER_FILLED,
                    symbol=order.symbol,
                    order_id=order.order_id,
                    details={
                        "action": "IDEMPOTENT_DUPLICATE_ACK",
                        "broker_execution_id": sim_resp.broker_execution_id,
                    },
                )
            return self._orders.get(order.order_id, order)

        self._seen_execution_ids.add(sim_resp.broker_execution_id)

        # 6. Apply Execution Fill (Full or Partial)
        fill_qty = sim_resp.filled_volume * spec.contract_size
        exec_price = sim_resp.executed_price or current_market_price or order.price or 1.0

        if sim_resp.status == SimulatedResponseStatus.PARTIALLY_FILLED:
            if order.status != OrderStatus.PARTIALLY_FILLED:
                order.transition_to(OrderStatus.PARTIALLY_FILLED)
            order.fill_price = exec_price
            order.fill_timestamp = sim_resp.timestamp
        else:
            if order.status != OrderStatus.FILLED:
                order.transition_to(OrderStatus.FILLED)
            order.fill_price = exec_price
            order.fill_timestamp = sim_resp.timestamp

        self._orders[order.order_id] = order

        # 7. Update Position
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
                entry_timestamp=sim_resp.timestamp,
            )
            self._positions[symbol_key] = new_pos
        else:
            if existing_pos.side == order.side:
                # Add to existing position
                total_qty = existing_pos.quantity + fill_qty
                existing_pos.entry_price = (
                    (existing_pos.entry_price * existing_pos.quantity) + (exec_price * fill_qty)
                ) / total_qty
                existing_pos.quantity = total_qty
                existing_pos.current_price = exec_price
            else:
                # Opposite side: netting / closing
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
                    existing_pos.close(exit_price=exec_price, exit_timestamp=sim_resp.timestamp)
                    closed_position_to_save = existing_pos.model_copy()
                    del self._positions[symbol_key]
                else:
                    existing_pos.quantity = remaining_qty
                    existing_pos.current_price = exec_price

        # 8. Create and Store ExecutionReport
        exec_report = ExecutionReport(
            execution_id=sim_resp.broker_execution_id,
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
            swap=0.0,
            gross_pnl=0.0,
            net_pnl=0.0,
            execution_status=order.status,
            timestamp=sim_resp.timestamp,
        )
        self._executions.append(exec_report)

        # 9. Persistence
        if self._repository:
            self._repository.save_order(order)
            self._repository.save_execution(exec_report)
            if symbol_key in self._positions:
                self._repository.save_position(self._positions[symbol_key])
            elif closed_position_to_save is not None:
                self._repository.save_position(closed_position_to_save)
            self._repository.save_account_snapshot(
                account=self.get_account(),
                total_costs=self._total_costs,
                daily_pnl=self._realized_pnl,
            )

        # 10. Audit Trail
        if self._audit_trail:
            self._audit_trail.record(
                event_type=AuditEventType.ORDER_FILLED,
                symbol=order.symbol,
                order_id=order.order_id,
                details={
                    "fill_quantity": fill_qty,
                    "fill_price": exec_price,
                    "broker_execution_id": sim_resp.broker_execution_id,
                    "remaining_volume_lots": sim_resp.remaining_volume,
                    "partial": sim_resp.status == SimulatedResponseStatus.PARTIALLY_FILLED,
                },
            )

        return order

    def cancel_order(self, order_id: str) -> Order:
        """Cancel an open/pending order."""
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not found")
        order = self._orders[order_id]
        order.transition_to(OrderStatus.CANCELLED)
        if self._repository:
            self._repository.save_order(order)
        if self._audit_trail:
            self._audit_trail.record(
                event_type=AuditEventType.ORDER_CANCELLED,
                symbol=order.symbol,
                order_id=order.order_id,
            )
        return order.model_copy()

    def close_position(self, symbol: str, exit_price: float) -> Order:
        """Close an active position at the specified exit price."""
        cleaned = symbol.strip().upper()
        pos = self._positions.get(cleaned)
        if pos is None:
            raise KeyError(f"No active position for {cleaned}")

        close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
        close_order = Order(
            order_id=f"ord-close-{pos.position_id}",
            symbol=cleaned,
            side=close_side,
            quantity=pos.quantity,
            order_type=OrderType.MARKET,
            price=exit_price,
            source="close_position",
            status=OrderStatus.PENDING,
        )
        return self.submit_order(close_order, current_market_price=exit_price)

    def update_market_price(self, symbol: str, price: float) -> None:
        """Update market price for mark-to-market valuations."""
        cleaned = symbol.strip().upper()
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Market price must be positive and finite, got {price}")
        if cleaned in self._positions:
            pos = self._positions[cleaned]
            pos.update_price(price)
            if self._repository:
                self._repository.save_position(pos)
                self._repository.save_account_snapshot(
                    account=self.get_account(),
                    total_costs=self._total_costs,
                    daily_pnl=self._realized_pnl,
                )
            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.POSITION_UPDATED,
                    symbol=cleaned,
                    details={"current_price": price, "unrealized_pnl": pos.unrealized_pnl},
                )

    def get_execution_reports(self) -> List[ExecutionReport]:
        """Fetch history of all execution reports."""
        return [r.model_copy() for r in self._executions]
