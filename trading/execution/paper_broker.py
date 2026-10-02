"""Deterministic, in-memory Paper Broker Simulator for risk and execution testing."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from trading.audit.trail import AuditEventType, AuditTrail
from trading.execution.costs import CostBreakdown, TransactionCostConfig
from trading.models.execution import ExecutionReport
from trading.models.order import Order, OrderSide, OrderStatus, OrderType
from trading.models.portfolio import AccountInfo
from trading.models.position import Position

if TYPE_CHECKING:
    pass


class InconsistentStateError(ValueError):
    """Raised when persisted paper-trading state violates internal consistency."""

    pass


class PaperBroker:
    """Deterministic, persistent-capable broker simulator with zero external connectivity.

    Guarantees:
    - Pure in-memory state with zero network connectivity.
    - Explicit transaction-cost accounting (spread, slippage, commission, swap).
    - Structured execution reports and audit trails.
    - Deterministic order state transitions.
    - Continuous position lifecycle and MTM valuation.
    - Crash/restart recovery from persistent storage with consistency validation.
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        cost_config: Optional[TransactionCostConfig] = None,
        audit_trail: Optional[AuditTrail] = None,
        repository: Optional[Any] = None,
    ) -> None:
        if not math.isfinite(initial_balance) or initial_balance <= 0:
            raise ValueError(
                f"Initial balance must be a positive finite number, got {initial_balance}"
            )
        self._initial_balance: float = float(initial_balance)
        self._cash_balance: float = float(initial_balance)
        self._gross_pnl: float = 0.0
        self._total_costs: float = 0.0
        self._realized_pnl: float = 0.0
        self._positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        self._orders: Dict[str, Order] = {}
        self._execution_reports: List[ExecutionReport] = []
        self._cost_config = cost_config or TransactionCostConfig()
        self._audit_trail = audit_trail
        self._repository = repository
        if self._repository:
            self.recover_from_repository()

    @property
    def cost_config(self) -> TransactionCostConfig:
        return self._cost_config

    @property
    def audit_trail(self) -> Optional[AuditTrail]:
        return self._audit_trail

    def set_audit_trail(self, audit_trail: AuditTrail) -> None:
        self._audit_trail = audit_trail

    @property
    def repository(self) -> Optional[Any]:
        return self._repository

    def set_repository(self, repository: Any) -> None:
        self._repository = repository
        if self._repository:
            self.recover_from_repository()

    def recover_from_repository(self) -> None:
        """Reconstruct in-memory state from persistent repository and validate consistency."""
        if not self._repository:
            return

        # 1. Load orders
        orders = self._repository.get_all_orders()
        self._orders = {o.order_id: o for o in orders}

        # 2. Load execution reports
        self._execution_reports = self._repository.get_all_executions()

        # 3. Load active positions
        self._positions = self._repository.get_active_positions()

        # 4. Load closed positions
        self._closed_positions = self._repository.get_closed_positions()

        # 5. Load latest account snapshot
        snapshot = self._repository.get_latest_account_snapshot()
        if snapshot:
            self._initial_balance = float(snapshot["initial_balance"])
            self._cash_balance = float(snapshot["cash_balance"])
            self._realized_pnl = float(snapshot["realized_pnl"])
            self._total_costs = float(snapshot["total_costs"])
            self._gross_pnl = round(self._realized_pnl + self._total_costs, 4)
        else:
            self._total_costs = sum(p.costs for p in self._closed_positions)
            self._realized_pnl = sum(p.realized_pnl for p in self._closed_positions)
            self._gross_pnl = sum(p.gross_pnl for p in self._closed_positions)

        # 6. Validate consistency
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        reconstructed_equity = round(self._initial_balance + self._realized_pnl + unrealized, 4)

        if snapshot:
            persisted_equity = round(float(snapshot["equity"]), 4)
            if abs(reconstructed_equity - persisted_equity) > 0.05:
                raise InconsistentStateError(
                    f"INCONSISTENT_STATE: Reconstructed equity ({reconstructed_equity:.2f}) "
                    f"does not match persisted equity ({persisted_equity:.2f})"
                )

    def _sync_snapshot(self) -> None:
        """Helper to persist account snapshot after state mutations."""
        if not self._repository:
            return
        account = self.get_account()
        exposure = sum(p.quantity * p.current_price for p in self._positions.values())
        self._repository.save_account_snapshot(
            account=account,
            total_costs=self._total_costs,
            daily_pnl=self._realized_pnl,
            current_exposure=exposure,
        )

    def get_account(self) -> AccountInfo:
        """Return current snapshot of simulated account equity, cash, and positions."""
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
        """Return copy of active positions."""
        return {k: p.model_copy() for k, p in self._positions.items()}

    def get_closed_positions(self) -> List[Position]:
        """Return copy of historical closed positions."""
        return [p.model_copy() for p in self._closed_positions]

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return position for a given symbol if active."""
        cleaned = symbol.strip().upper()
        pos = self._positions.get(cleaned)
        return pos.model_copy() if pos else None

    def get_order(self, order_id: str) -> Optional[Order]:
        """Return an order by ID."""
        order = self._orders.get(order_id)
        return order.model_copy() if order else None

    def get_all_orders(self) -> List[Order]:
        """Return list of all registered orders."""
        return [o.model_copy() for o in self._orders.values()]

    def get_execution_reports(self) -> List[ExecutionReport]:
        """Return list of all execution reports."""
        return [r.model_copy() for r in self._execution_reports]

    def update_market_price(self, symbol: str, price: float) -> Optional[Order]:
        """Update market price for a symbol, mark-to-market, and check SL/TP boundaries."""
        cleaned = symbol.strip().upper()
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Market price must be positive and finite, got {price}")

        if cleaned not in self._positions:
            return None

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
                details={
                    "current_price": price,
                    "unrealized_pnl": pos.unrealized_pnl,
                },
            )

        # Check automated SL/TP triggers
        if pos.is_stop_loss_hit(price):
            return self.close_position(cleaned, exit_price=pos.stop_loss or price)
        elif pos.is_take_profit_hit(price):
            return self.close_position(cleaned, exit_price=pos.take_profit or price)

        return None

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending/created order."""
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

    def submit_order(self, order: Order, current_market_price: Optional[float] = None) -> Order:
        """Execute market order with explicit validation, cost accounting, and state transition."""
        # Idempotency check: if order was already submitted and final, return it
        if order.order_id in self._orders:
            existing = self._orders[order.order_id]
            if existing.status in {OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED}:
                return existing.model_copy()

        now = datetime.now(timezone.utc)
        fill_price = current_market_price if current_market_price is not None else order.price

        # Validate fill price
        if fill_price is None or not math.isfinite(fill_price) or fill_price <= 0:
            order.transition_to(OrderStatus.REJECTED, reason="INVALID_FILL_PRICE")
            self._orders[order.order_id] = order
            self._record_execution(
                order=order,
                executed_price=None,
                costs=CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0),
                rejection_reason="INVALID_FILL_PRICE",
            )
            return order.model_copy()

        # Validate quantity
        if order.quantity <= 0 or not math.isfinite(order.quantity):
            order.transition_to(OrderStatus.REJECTED, reason="INVALID_QUANTITY")
            self._orders[order.order_id] = order
            self._record_execution(
                order=order,
                executed_price=None,
                costs=CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0),
                rejection_reason="INVALID_QUANTITY",
            )
            return order.model_copy()

        symbol = order.symbol.strip().upper()
        existing_pos = self._positions.get(symbol)

        # Branch A: Opening new position
        if existing_pos is None:
            entry_costs = self._cost_config.calculate_entry_costs(order.quantity, fill_price)
            capital_required = (fill_price * order.quantity) + entry_costs.total_cost

            if order.side == OrderSide.BUY:
                if self._cash_balance < capital_required:
                    order.transition_to(OrderStatus.REJECTED, reason="INSUFFICIENT_CASH")
                    self._orders[order.order_id] = order
                    self._record_execution(
                        order=order,
                        executed_price=None,
                        costs=entry_costs,
                        rejection_reason="INSUFFICIENT_CASH",
                    )
                    return order.model_copy()
                self._cash_balance -= capital_required
            else:
                # For short paper position, reserve entry costs from cash
                if self._cash_balance < entry_costs.total_cost:
                    order.transition_to(OrderStatus.REJECTED, reason="INSUFFICIENT_CASH")
                    self._orders[order.order_id] = order
                    self._record_execution(
                        order=order,
                        executed_price=None,
                        costs=entry_costs,
                        rejection_reason="INSUFFICIENT_CASH",
                    )
                    return order.model_copy()
                self._cash_balance -= entry_costs.total_cost

            self._total_costs += entry_costs.total_cost
            self._realized_pnl -= entry_costs.total_cost

            new_pos = Position(
                symbol=symbol,
                side=order.side,
                quantity=order.quantity,
                entry_price=fill_price,
                current_price=fill_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                entry_timestamp=now,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                gross_pnl=0.0,
                costs=entry_costs.total_cost,
                net_pnl=-entry_costs.total_cost,
                status="OPEN",
            )
            self._positions[symbol] = new_pos

            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.POSITION_OPENED,
                    symbol=symbol,
                    order_id=order.order_id,
                    details={
                        "side": order.side.value,
                        "quantity": order.quantity,
                        "entry_price": fill_price,
                        "entry_costs": entry_costs.total_cost,
                    },
                )

            # Fill order
            order.transition_to(OrderStatus.FILLED)
            order.fill_price = fill_price
            order.fill_timestamp = now
            self._orders[order.order_id] = order

            self._record_execution(
                order=order,
                executed_price=fill_price,
                costs=entry_costs,
                net_pnl=-entry_costs.total_cost,
            )
            return order.model_copy()

        # Branch B: Adding to existing position (same side)
        elif existing_pos.side == order.side:
            entry_costs = self._cost_config.calculate_entry_costs(order.quantity, fill_price)
            capital_required = (fill_price * order.quantity) + entry_costs.total_cost

            if order.side == OrderSide.BUY:
                if self._cash_balance < capital_required:
                    order.transition_to(OrderStatus.REJECTED, reason="INSUFFICIENT_CASH")
                    self._orders[order.order_id] = order
                    self._record_execution(
                        order=order,
                        executed_price=None,
                        costs=entry_costs,
                        rejection_reason="INSUFFICIENT_CASH",
                    )
                    return order.model_copy()
                self._cash_balance -= capital_required
            else:
                self._cash_balance -= entry_costs.total_cost

            self._total_costs += entry_costs.total_cost
            self._realized_pnl -= entry_costs.total_cost

            total_qty = existing_pos.quantity + order.quantity
            weighted_entry = (
                (existing_pos.entry_price * existing_pos.quantity) + (fill_price * order.quantity)
            ) / total_qty
            existing_pos.quantity = total_qty
            existing_pos.entry_price = weighted_entry
            existing_pos.costs += entry_costs.total_cost
            existing_pos.update_price(fill_price)

            if self._audit_trail:
                self._audit_trail.record(
                    event_type=AuditEventType.POSITION_UPDATED,
                    symbol=symbol,
                    order_id=order.order_id,
                    details={"action": "SCALE_IN", "new_quantity": total_qty},
                )

            order.transition_to(OrderStatus.FILLED)
            order.fill_price = fill_price
            order.fill_timestamp = now
            self._orders[order.order_id] = order

            self._record_execution(
                order=order,
                executed_price=fill_price,
                costs=entry_costs,
                net_pnl=-entry_costs.total_cost,
            )
            return order.model_copy()

        # Branch C: Opposite side order (closing/flattening position)
        else:
            closed_qty = min(existing_pos.quantity, order.quantity)
            exit_costs = self._cost_config.calculate_exit_costs(closed_qty, fill_price)

            if existing_pos.side == OrderSide.BUY:
                trade_gross_pnl = (fill_price - existing_pos.entry_price) * closed_qty
                self._cash_balance += (
                    (existing_pos.entry_price * closed_qty)
                    + trade_gross_pnl
                    - exit_costs.total_cost
                )
            else:
                trade_gross_pnl = (existing_pos.entry_price - fill_price) * closed_qty
                self._cash_balance += trade_gross_pnl - exit_costs.total_cost

            trade_net_pnl = trade_gross_pnl - exit_costs.total_cost

            self._gross_pnl += trade_gross_pnl
            self._total_costs += exit_costs.total_cost
            self._realized_pnl += trade_net_pnl

            remaining_qty = existing_pos.quantity - closed_qty
            if remaining_qty <= 1e-9:
                existing_pos.close(
                    exit_price=fill_price,
                    exit_timestamp=now,
                    costs=existing_pos.costs + exit_costs.total_cost,
                )
                self._closed_positions.append(existing_pos.model_copy())
                del self._positions[symbol]

                if self._audit_trail:
                    self._audit_trail.record(
                        event_type=AuditEventType.POSITION_CLOSED,
                        symbol=symbol,
                        order_id=order.order_id,
                        details={
                            "exit_price": fill_price,
                            "gross_pnl": existing_pos.gross_pnl,
                            "net_pnl": existing_pos.net_pnl,
                        },
                    )
            else:
                existing_pos.quantity = remaining_qty
                existing_pos.costs += exit_costs.total_cost
                existing_pos.update_price(fill_price)

            order.transition_to(OrderStatus.FILLED)
            order.fill_price = fill_price
            order.fill_timestamp = now
            self._orders[order.order_id] = order

            self._record_execution(
                order=order,
                executed_price=fill_price,
                costs=exit_costs,
                gross_pnl=trade_gross_pnl,
                net_pnl=trade_net_pnl,
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
            symbol=cleaned,
            side=close_side,
            quantity=pos.quantity,
            order_type=OrderType.MARKET,
            price=exit_price,
            source="close_position",
            status=OrderStatus.PENDING,
        )
        return self.submit_order(close_order, current_market_price=exit_price)

    def _record_execution(
        self,
        order: Order,
        executed_price: Optional[float],
        costs: CostBreakdown,
        gross_pnl: float = 0.0,
        net_pnl: float = 0.0,
        rejection_reason: Optional[str] = None,
    ) -> None:
        """Record an immutable execution report and audit log event."""
        report = ExecutionReport(
            order_id=order.order_id,
            client_request_id=order.client_request_id,
            symbol=order.symbol,
            side=order.side,
            requested_price=order.price or (executed_price or 0.0),
            executed_price=executed_price,
            quantity=order.quantity,
            spread=costs.spread_cost,
            slippage=costs.slippage_cost,
            commission=costs.commission,
            swap=costs.swap_cost if hasattr(costs, "swap_cost") else 0.0,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            execution_status=order.status,
            rejection_reason=rejection_reason or order.rejection_reason,
        )
        self._execution_reports.append(report)

        if self._repository:
            self._repository.save_order(order)
            self._repository.save_execution(report)
            if order.symbol in self._positions:
                self._repository.save_position(self._positions[order.symbol])
            elif self._closed_positions and self._closed_positions[-1].symbol == order.symbol:
                self._repository.save_position(self._closed_positions[-1])
            self._sync_snapshot()

        if self._audit_trail:
            event_type = (
                AuditEventType.ORDER_FILLED
                if order.status == OrderStatus.FILLED
                else AuditEventType.ORDER_REJECTED
            )
            self._audit_trail.record(
                event_type=event_type,
                symbol=order.symbol,
                order_id=order.order_id,
                details={
                    "status": order.status.value,
                    "fill_price": executed_price,
                    "quantity": order.quantity,
                    "costs": costs.total_cost,
                    "rejection_reason": rejection_reason,
                },
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational engineering metrics."""
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
            "gross_pnl": round(self._gross_pnl, 4),
            "total_costs": round(self._total_costs, 4),
            "net_pnl": round(self._realized_pnl, 4),
            "daily_loss": 0.0,
            "current_exposure": sum(p.quantity * p.current_price for p in self._positions.values()),
            "equity": account.equity,
            "cash_balance": account.cash_balance,
        }
