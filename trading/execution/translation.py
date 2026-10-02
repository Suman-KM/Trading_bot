"""Order translation layer mapping internal domain orders to MT5 trade requests."""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from trading.execution.validation import (
    BrokerSymbolSpecification,
    BrokerValidationError,
    validate_order_for_broker,
)
from trading.models.order import Order, OrderSide


class OrderTranslationError(ValueError):
    """Raised when an internal order cannot be translated safely to an MT5 trade request."""

    pass


# Standard MT5 Constants
MT5_TRADE_ACTION_DEAL = 1
MT5_ORDER_TYPE_BUY = 0
MT5_ORDER_TYPE_SELL = 1
MT5_ORDER_FILLING_FOK = 0
MT5_ORDER_FILLING_IOC = 1
MT5_ORDER_TIME_GTC = 0


class MT5TradeRequest(BaseModel):
    """Direct representation of an MqlTradeRequest structure for MetaTrader 5."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: int = Field(MT5_TRADE_ACTION_DEAL, description="Trade action type (e.g. DEAL)")
    magic: int = Field(100001, description="EA/System identification magic number")
    order: int = Field(0, description="Order ticket (0 for new orders)")
    symbol: str = Field(..., description="Symbol name")
    volume: float = Field(..., description="Order volume in standard lots")
    price: float = Field(..., description="Requested execution price")
    stop_loss: float = Field(0.0, description="Stop Loss price (0.0 if not set)")
    take_profit: float = Field(0.0, description="Take Profit price (0.0 if not set)")
    deviation: int = Field(20, description="Max allowed price deviation in points")
    order_type: int = Field(..., description="0 = BUY, 1 = SELL")
    type_filling: int = Field(MT5_ORDER_FILLING_FOK, description="Execution policy")
    type_time: int = Field(MT5_ORDER_TIME_GTC, description="Order expiration type")
    comment: Optional[str] = Field(None, description="Trade comment / client ID tag")
    position: int = Field(0, description="Position ticket for closing/modifying (0 for new orders)")
    client_request_id: Optional[str] = Field(
        None, description="Idempotent client request identifier"
    )
    internal_order_id: str = Field(..., description="Internal system order UUID")


def translate_order_to_mt5_request(
    order: Order,
    spec: BrokerSymbolSpecification,
    magic: int = 100001,
    deviation: int = 20,
    current_market_price: Optional[float] = None,
    position: int = 0,
) -> MT5TradeRequest:
    """Translate an internal Order to an MT5TradeRequest without altering intent or risk.

    Invariants:
    - Never modifies requested risk, SL/TP levels, direction, or quantity.
    - Fails closed by raising OrderTranslationError if translation cannot be performed safely.
    """
    try:
        validate_order_for_broker(order, spec, current_market_price=current_market_price)
    except BrokerValidationError as err:
        raise OrderTranslationError(f"TRANSLATION_FAILED: {str(err)}") from err

    # 1. Map Side
    if order.side == OrderSide.BUY:
        mt5_type = MT5_ORDER_TYPE_BUY
    elif order.side == OrderSide.SELL:
        mt5_type = MT5_ORDER_TYPE_SELL
    else:
        raise OrderTranslationError(f"Unsupported order side: {order.side}")

    # 2. Map Volume
    volume_lots = round(order.quantity / spec.contract_size, 4)
    # Ensure exact consistency: volume in lots * contract_size must equal quantity within 0.01 units
    reconstructed_qty = volume_lots * spec.contract_size
    if not math.isclose(reconstructed_qty, order.quantity, rel_tol=1e-5):
        raise OrderTranslationError(
            f"TRANSLATION_FAILED: Quantity {order.quantity} cannot be mapped exactly to "
            f"broker lot precision without truncation. Reconstructed: {reconstructed_qty}."
        )

    # 3. Map Price
    target_price = current_market_price or order.price or order.requested_price
    if target_price is None or not math.isfinite(target_price) or target_price <= 0:
        raise OrderTranslationError("TRANSLATION_FAILED: Missing or invalid execution price.")

    # 4. Map SL / TP
    sl_val = round(order.stop_loss, spec.price_digits) if order.stop_loss is not None else 0.0
    tp_val = round(order.take_profit, spec.price_digits) if order.take_profit is not None else 0.0

    # 5. Comment tag
    comment_tag = order.client_request_id or f"ord_{order.order_id[:8]}"
    if len(comment_tag) > 31:
        comment_tag = comment_tag[:31]

    return MT5TradeRequest(
        action=MT5_TRADE_ACTION_DEAL,
        magic=magic,
        order=0,
        symbol=order.symbol.strip().upper(),
        volume=volume_lots,
        price=round(target_price, spec.price_digits),
        stop_loss=sl_val,
        take_profit=tp_val,
        deviation=deviation,
        order_type=mt5_type,
        type_filling=MT5_ORDER_FILLING_FOK,
        type_time=MT5_ORDER_TIME_GTC,
        comment=comment_tag,
        position=position,
        client_request_id=order.client_request_id,
        internal_order_id=order.order_id,
    )
