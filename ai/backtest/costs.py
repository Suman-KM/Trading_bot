"""Cost modeling and price conversion for historical trade simulation.

Converts MT5 spread points to price units and calculates realistic transaction frictions
including bid/ask spread, round-turn broker commissions, and fixed slippage.
"""

from __future__ import annotations

import math

from ai.backtest.models import BacktestConfig, TradeDirection


class CostModel:
    """Deterministic cost calculator for spread, commission, and slippage."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def get_spread_price(self, raw_spread_points: float | None) -> float:
        """Resolve spread in price units from historical points or fallback default.

        Formula:
            spread_points = max(raw_spread_points or default_spread_points, min_spread_points)
            spread_price = spread_points * point_value
        """
        if raw_spread_points is None or math.isnan(raw_spread_points) or raw_spread_points <= 0:
            points = self.config.default_spread_points
        else:
            points = (
                raw_spread_points
                if self.config.use_historical_spread
                else self.config.default_spread_points
            )

        points = max(points, self.config.min_spread_points)
        return float(points * self.config.point_value)

    def get_slippage_price(self) -> float:
        """Return fixed per-execution slippage in price units."""
        return float(self.config.slippage_points * self.config.point_value)

    def calculate_entry_fill(
        self,
        bar_open: float,
        direction: TradeDirection,
        raw_spread_points: float | None,
    ) -> tuple[float, float, float, float]:
        """Compute entry fill price, spread cost, slippage cost, and entry commission.

        Parameters
        ----------
        bar_open : float
            Next-bar opening price (Bid price in FX convention).
        direction : TradeDirection
            LONG or SHORT.
        raw_spread_points : float | None
            Historical spread points recorded on the entry bar.

        Returns
        -------
        tuple[float, float, float, float]
            (fill_price, spread_price, slippage_price, entry_commission_per_lot)
        """
        spread_price = self.get_spread_price(raw_spread_points)
        slippage_price = self.get_slippage_price()

        if direction == TradeDirection.LONG:
            # LONG enters at Ask: Bid_open + spread + slippage
            fill_price = bar_open + spread_price + slippage_price
        else:
            # SHORT enters at Bid: Bid_open - slippage
            fill_price = bar_open - slippage_price

        return fill_price, spread_price, slippage_price, self.config.commission_per_lot / 2.0

    def calculate_exit_fill(
        self,
        nominal_exit_price: float,
        direction: TradeDirection,
        raw_spread_points: float | None,
    ) -> tuple[float, float, float, float]:
        """Compute exit fill price, spread price, slippage price, and exit commission.

        Parameters
        ----------
        nominal_exit_price : float
            Trigger or nominal exit price (Bid price in FX convention).
        direction : TradeDirection
            Direction of the original position being closed.
        raw_spread_points : float | None
            Historical spread points recorded on the exit bar.

        Returns
        -------
        tuple[float, float, float, float]
            (effective_exit_price, spread_price, slippage_price, exit_commission_per_lot)
        """
        spread_price = self.get_spread_price(raw_spread_points)
        slippage_price = self.get_slippage_price()

        if direction == TradeDirection.LONG:
            # Closing LONG sells at Bid: nominal_exit_price - slippage
            effective_exit_price = nominal_exit_price - slippage_price
        else:
            # Closing SHORT buys at Ask: nominal_exit_price + spread + slippage
            effective_exit_price = nominal_exit_price + spread_price + slippage_price

        return (
            effective_exit_price,
            spread_price,
            slippage_price,
            self.config.commission_per_lot / 2.0,
        )

    def calculate_trade_costs(
        self,
        quantity: float,
        spread_price_entry: float,
        spread_price_exit: float,
        slippage_price_entry: float,
        slippage_price_exit: float,
    ) -> tuple[float, float, float, float]:
        """Calculate total dollar costs for a completed round-turn trade.

        Returns
        -------
        tuple[float, float, float, float]
            (spread_cost, commission, slippage_cost, total_transaction_costs)
        """
        # Spread cost paid across the round-turn (LONG at entry, SHORT at exit)
        # LONG: enters at Ask (crosses spread). Spread cost = qty * spread_entry.
        # SHORT: exits at Ask (crosses spread). Spread cost = qty * spread_exit.
        spread_cost = quantity * max(spread_price_entry, spread_price_exit)

        # Slippage cost incurred on both entry fill and exit fill
        slippage_cost = quantity * (slippage_price_entry + slippage_price_exit)

        # Round-turn commission based on standard lot volume
        lots = quantity / self.config.lot_size
        commission = lots * self.config.commission_per_lot

        total_costs = spread_cost + commission + slippage_cost
        return spread_cost, commission, slippage_cost, total_costs
