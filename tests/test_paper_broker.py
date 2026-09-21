"""Unit tests for the in-memory Paper Broker Simulator."""

from trading.execution.paper_broker import PaperBroker
from trading.models.order import Order, OrderSide, OrderStatus, OrderType


def make_order(
    symbol: str = "EURUSD",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 100.0,
    price: float = 1.1000,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        price=price,
        source="unit_test",
        status=OrderStatus.PENDING,
    )


def test_paper_broker_initialization():
    """Verify initial paper broker state."""
    broker = PaperBroker(initial_balance=100_000.0)
    account = broker.get_account()
    assert account.initial_balance == 100_000.0
    assert account.cash_balance == 100_000.0
    assert account.equity == 100_000.0
    assert account.realized_pnl == 0.0
    assert account.unrealized_pnl == 0.0
    assert len(account.positions) == 0


def test_paper_order_fill_buy():
    """Test 12: Order fill for BUY position."""
    broker = PaperBroker(initial_balance=100_000.0)
    order = make_order(symbol="EURUSD", side=OrderSide.BUY, quantity=10_000.0, price=1.1000)

    filled = broker.submit_order(order)
    assert filled.status == OrderStatus.FILLED
    assert filled.fill_price == 1.1000
    assert filled.fill_timestamp is not None

    account = broker.get_account()
    # Cost = 10,000 * 1.1000 = 11,000
    assert account.cash_balance == 100_000.0 - 11_000.0
    assert "EURUSD" in account.positions
    pos = account.positions["EURUSD"]
    assert pos.quantity == 10_000.0
    assert pos.entry_price == 1.1000
    assert pos.current_price == 1.1000
    assert pos.unrealized_pnl == 0.0
    assert account.equity == 100_000.0


def test_pnl_calculation_and_position_close():
    """Test 14: Unrealized mark-to-market and realized P&L on position closure."""
    broker = PaperBroker(initial_balance=100_000.0)
    # Buy 1,000 shares @ $50.0 = $50,000
    order = make_order(symbol="XYZ", side=OrderSide.BUY, quantity=1_000.0, price=50.0)
    broker.submit_order(order)

    # Price rises to $55.0 -> unrealized gain = (55 - 50) * 1,000 = +$5,000
    broker.update_market_price("XYZ", 55.0)
    account = broker.get_account()
    assert account.positions["XYZ"].unrealized_pnl == 5_000.0
    assert account.unrealized_pnl == 5_000.0
    assert account.equity == 105_000.0

    # Close position at $55.0
    close_order = broker.close_position("XYZ", exit_price=55.0)
    assert close_order.status == OrderStatus.FILLED
    assert close_order.fill_price == 55.0

    account_after = broker.get_account()
    assert "XYZ" not in account_after.positions
    assert account_after.unrealized_pnl == 0.0
    assert account_after.realized_pnl == 5_000.0
    assert account_after.cash_balance == 105_000.0
    assert account_after.equity == 105_000.0


def test_rejected_order_insufficient_cash():
    """Test 13a: Order rejected due to insufficient cash."""
    broker = PaperBroker(initial_balance=1_000.0)
    # Attempt to buy $50,000 worth with only $1,000 cash
    order = make_order(symbol="AAPL", side=OrderSide.BUY, quantity=500.0, price=100.0)
    result = broker.submit_order(order)

    assert result.status == OrderStatus.REJECTED
    assert result.rejection_reason == "INSUFFICIENT_CASH"
    assert "AAPL" not in broker.get_positions()
    assert broker.get_account().cash_balance == 1_000.0


def test_rejected_order_invalid_parameters():
    """Test 13b: Order rejected due to invalid price or quantity in submission."""
    broker = PaperBroker(initial_balance=100_000.0)

    # Invalid fill price passed to submit_order
    order1 = make_order(price=10.0)
    res1 = broker.submit_order(order1, current_market_price=-10.0)
    assert res1.status == OrderStatus.REJECTED
    assert res1.rejection_reason == "INVALID_FILL_PRICE"

    # Invalid quantity on order
    order2 = make_order(quantity=100.0)
    object.__setattr__(order2, "quantity", -5.0)
    res2 = broker.submit_order(order2)
    assert res2.status == OrderStatus.REJECTED
    assert res2.rejection_reason == "INVALID_QUANTITY"


def test_cancel_order():
    """Test cancellation of a pending order."""
    broker = PaperBroker()
    order = make_order()
    broker._orders[order.order_id] = order
    cancelled = broker.cancel_order(order.order_id)
    assert cancelled.status == OrderStatus.CANCELLED
