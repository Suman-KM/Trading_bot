"""Broker-specific symbol specification and order parameter validation."""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from trading.models.order import Order, OrderSide


class BrokerValidationError(ValueError):
    """Raised when an order violates broker-specific symbol, volume, or price constraints."""

    pass


class BrokerSymbolSpecification(BaseModel):
    """Specification of broker trading conditions and volume constraints for a symbol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(..., description="Instrument symbol (e.g. 'EURUSD')")
    min_volume: float = Field(0.01, description="Minimum allowed order volume in lots")
    max_volume: float = Field(100.0, description="Maximum allowed order volume in lots")
    volume_step: float = Field(0.01, description="Volume increment step in lots")
    contract_size: float = Field(100_000.0, description="Contract units per 1.0 standard lot")
    price_digits: int = Field(5, description="Number of decimal places for quotes")
    point: float = Field(0.00001, description="Point value")
    tick_size: float = Field(0.00001, description="Minimum price movement increment")


def validate_order_for_broker(
    order: Order,
    spec: BrokerSymbolSpecification,
    current_market_price: Optional[float] = None,
) -> None:
    """Validate that an order conforms strictly to broker specifications before submission.

    IMPORTANT: This validation is NOT a replacement for the Sovereign Risk Engine.
    RiskEngine remains the authoritative gate for portfolio-level exposure, daily loss,
    position risk, and emergency halts. This function validates broker execution compatibility.

    Raises:
        BrokerValidationError: If any broker-specific rule is violated.
    """
    # 1. Symbol Validation
    if order.symbol.strip().upper() != spec.symbol.strip().upper():
        raise BrokerValidationError(
            f"BROKER_VALIDATION_ERROR: Order symbol '{order.symbol}' does not match "
            f"broker specification symbol '{spec.symbol}'."
        )

    # 2. Side Validation
    if order.side not in (OrderSide.BUY, OrderSide.SELL):
        raise BrokerValidationError(f"BROKER_VALIDATION_ERROR: Invalid order side '{order.side}'.")

    # 3. Quantity & Volume Validation
    if not math.isfinite(order.quantity) or order.quantity <= 0:
        raise BrokerValidationError(
            "BROKER_VALIDATION_ERROR: Order quantity must be positive finite, "
            f"got {order.quantity}."
        )

    lots = round(order.quantity / spec.contract_size, 6)
    if lots < spec.min_volume:
        raise BrokerValidationError(
            f"BROKER_VALIDATION_ERROR: Order volume {lots:.4f} lots ({order.quantity:.1f} units) "
            f"is below minimum allowed volume of {spec.min_volume} lots."
        )
    if lots > spec.max_volume:
        raise BrokerValidationError(
            f"BROKER_VALIDATION_ERROR: Order volume {lots:.4f} lots ({order.quantity:.1f} units) "
            f"exceeds maximum allowed volume of {spec.max_volume} lots."
        )

    # Check volume step
    steps = round((lots - spec.min_volume) / spec.volume_step, 6)
    if abs(steps - round(steps)) > 1e-4:
        raise BrokerValidationError(
            f"BROKER_VALIDATION_ERROR: Order volume {lots:.4f} lots does not align with "
            f"broker volume step of {spec.volume_step} lots."
        )

    # 4. Price Validation
    eval_price = current_market_price if current_market_price is not None else order.price
    if eval_price is None or not math.isfinite(eval_price) or eval_price <= 0:
        raise BrokerValidationError(
            f"BROKER_VALIDATION_ERROR: Price must be a positive finite number, got {eval_price}."
        )

    # 5. Stop-Loss Validation
    if order.stop_loss is not None:
        if not math.isfinite(order.stop_loss) or order.stop_loss <= 0:
            raise BrokerValidationError(
                "BROKER_VALIDATION_ERROR: Stop-loss must be positive finite, "
                f"got {order.stop_loss}."
            )
        if order.side == OrderSide.BUY and order.stop_loss >= eval_price:
            raise BrokerValidationError(
                f"BROKER_VALIDATION_ERROR: For BUY orders, stop-loss ({order.stop_loss}) "
                f"must be strictly below entry price ({eval_price})."
            )
        if order.side == OrderSide.SELL and order.stop_loss <= eval_price:
            raise BrokerValidationError(
                f"BROKER_VALIDATION_ERROR: For SELL orders, stop-loss ({order.stop_loss}) "
                f"must be strictly above entry price ({eval_price})."
            )

    # 6. Take-Profit Validation
    if order.take_profit is not None:
        if not math.isfinite(order.take_profit) or order.take_profit <= 0:
            raise BrokerValidationError(
                "BROKER_VALIDATION_ERROR: Take-profit must be positive finite, "
                f"got {order.take_profit}."
            )
        if order.side == OrderSide.BUY and order.take_profit <= eval_price:
            raise BrokerValidationError(
                f"BROKER_VALIDATION_ERROR: For BUY orders, take-profit ({order.take_profit}) "
                f"must be strictly above entry price ({eval_price})."
            )
        if order.side == OrderSide.SELL and order.take_profit >= eval_price:
            raise BrokerValidationError(
                f"BROKER_VALIDATION_ERROR: For SELL orders, take-profit ({order.take_profit}) "
                f"must be strictly below entry price ({eval_price})."
            )
