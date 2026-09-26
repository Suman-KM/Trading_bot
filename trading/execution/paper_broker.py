"""In-memory Paper Broker Simulator for risk and execution testing."""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

from trading.models.order import Order, OrderSide, OrderStatus, OrderType
from trading.models.portfolio import AccountInfo
from trading.models.position import Position


class PaperBroker:
    """Deterministic, 100% in-memory broker simulator with zero external connectivity."""

    def __init__(self, initial_balance: float = 100_000.0) -> None:
        if not math.isfinite(initial_balance) or initial_balance <= 0:
            raise ValueError(
                f"Initial balance must be a positive finite number, got {initial_balance}"
            )
        self._initial_balance: float = float(initial_balance)
        self._cash_balance: float = float(initial_balance)
        self._realized_pnl: float = 0.0
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}

    def get_account(self) -> AccountInfo:
        """Return current snapshot of simulated account equity, cash, and positions."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        equity = self._initial_balance + self._realized_pnl + unrealized
        return AccountInfo(
            initial_balance=self._initial_balance,
            cash_balance=round(self._cash_balance, 4),
            realized_pnl=round(self._realized_pnl, 4),
            unrealized_pnl=round(unrealized, 4),
            equity=round(equity, 4),
            positions={k: p.model_copy() for k, p in self._positions.items()},
        )

    def get_positions(self) -> Dict[str, Position]:
        """Return copy of active positions."""
        return {k: p.model_copy() for k, p in self._positions.items()}

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

    def update_market_price(self, symbol: str, price: float) -> None:
        """Update market price for a symbol and mark-to-market open position."""
        cleaned = symbol.strip().upper()
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Market price must be positive and finite, got {price}")
        if cleaned in self._positions:
            self._positions[cleaned].update_price(price)

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending order."""
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not found")
        order = self._orders[order_id]
        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot cancel order in state {order.status.value}")
        order.status = OrderStatus.CANCELLED
        return order.model_copy()

    def submit_order(self, order: Order, current_market_price: Optional[float] = None) -> Order:
        """Simulate market order validation and execution."""
        fill_price = current_market_price if current_market_price is not None else order.price
        if fill_price is None or not math.isfinite(fill_price) or fill_price <= 0:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "INVALID_FILL_PRICE"
            self._orders[order.order_id] = order
            return order.model_copy()

        if order.quantity <= 0 or not math.isfinite(order.quantity):
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "INVALID_QUANTITY"
            self._orders[order.order_id] = order
            return order.model_copy()

        symbol = order.symbol.strip().upper()
        now = datetime.now(timezone.utc)

        existing_pos = self._positions.get(symbol)

        if existing_pos is None:
            # Opening new position
            cost = fill_price * order.quantity
            if order.side == OrderSide.BUY:
                if self._cash_balance < cost:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "INSUFFICIENT_CASH"
                    self._orders[order.order_id] = order
                    return order.model_copy()
                self._cash_balance -= cost

            self._positions[symbol] = Position(
                symbol=symbol,
                side=order.side,
                quantity=order.quantity,
                entry_price=fill_price,
                current_price=fill_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )
        elif existing_pos.side == order.side:
            # Adding to existing position (weighted average entry price)
            cost = fill_price * order.quantity
            if order.side == OrderSide.BUY:
                if self._cash_balance < cost:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "INSUFFICIENT_CASH"
                    self._orders[order.order_id] = order
                    return order.model_copy()
                self._cash_balance -= cost

            total_qty = existing_pos.quantity + order.quantity
            weighted_entry = (
                (existing_pos.entry_price * existing_pos.quantity) + (fill_price * order.quantity)
            ) / total_qty
            existing_pos.quantity = total_qty
            existing_pos.entry_price = weighted_entry
            existing_pos.update_price(fill_price)
        else:
            # Opposite side order: closing / flattening position
            closed_qty = min(existing_pos.quantity, order.quantity)
            if existing_pos.side == OrderSide.BUY:
                trade_pnl = (fill_price - existing_pos.entry_price) * closed_qty
                self._cash_balance += (existing_pos.entry_price * closed_qty) + trade_pnl
            else:
                trade_pnl = (existing_pos.entry_price - fill_price) * closed_qty
                self._cash_balance += trade_pnl

            self._realized_pnl += trade_pnl

            remaining_qty = existing_pos.quantity - closed_qty
            if remaining_qty <= 1e-9:
                del self._positions[symbol]
            else:
                existing_pos.quantity = remaining_qty
                existing_pos.update_price(fill_price)

        # Mark order filled
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_timestamp = now
        self._orders[order.order_id] = order
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
