"""Broker adapter boundary interface and Paper Broker adapter implementation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from trading.models.execution import ExecutionReport
from trading.models.order import Order
from trading.models.portfolio import AccountInfo
from trading.models.position import Position

if TYPE_CHECKING:
    from trading.execution.paper_broker import PaperBroker


class BrokerAdapter(ABC):
    """Abstract interface defining boundary between ExecutionService and Broker."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Fetch current simulated account equity, cash, and balances."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Fetch active open positions."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Fetch an active position by symbol."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Fetch an order by its unique ID."""
        pass

    @abstractmethod
    def get_all_orders(self) -> List[Order]:
        """Fetch all registered orders."""
        pass

    @abstractmethod
    def submit_order(self, order: Order, current_market_price: Optional[float] = None) -> Order:
        """Submit an order for execution."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> Order:
        """Cancel an open/pending order."""
        pass

    @abstractmethod
    def close_position(self, symbol: str, exit_price: float) -> Order:
        """Close an active position."""
        pass

    @abstractmethod
    def update_market_price(self, symbol: str, price: float) -> None:
        """Update market price for mark-to-market valuations."""
        pass

    @abstractmethod
    def get_execution_reports(self) -> List[ExecutionReport]:
        """Fetch history of all execution reports."""
        pass

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """Declare whether adapter connects to real/live capital. Always False for paper."""
        pass


class PaperBrokerAdapter(BrokerAdapter):
    """Concrete broker adapter wrapping the in-memory PaperBroker.

    Guarantees 100% in-memory execution with zero network connectivity.
    """

    def __init__(self, broker: "PaperBroker") -> None:
        self._broker = broker

    @property
    def is_live(self) -> bool:
        return False

    def get_account(self) -> AccountInfo:
        return self._broker.get_account()

    def get_positions(self) -> Dict[str, Position]:
        return self._broker.get_positions()

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._broker.get_position(symbol)

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._broker.get_order(order_id)

    def get_all_orders(self) -> List[Order]:
        return self._broker.get_all_orders()

    def submit_order(self, order: Order, current_market_price: Optional[float] = None) -> Order:
        return self._broker.submit_order(order, current_market_price=current_market_price)

    def cancel_order(self, order_id: str) -> Order:
        return self._broker.cancel_order(order_id)

    def close_position(self, symbol: str, exit_price: float) -> Order:
        return self._broker.close_position(symbol, exit_price=exit_price)

    def update_market_price(self, symbol: str, price: float) -> None:
        self._broker.update_market_price(symbol, price)

    def get_execution_reports(self) -> List[ExecutionReport]:
        return self._broker.get_execution_reports()
