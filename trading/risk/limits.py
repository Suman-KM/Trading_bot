"""Risk limits configuration and deterministic position sizing logic."""

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLimits(BaseModel):
    """Configurable risk thresholds for the trading system."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    MAX_DAILY_LOSS_PERCENT: float = Field(
        default=1.0, description="Maximum allowable daily account drawdown in percent."
    )
    MAX_POSITION_RISK_PERCENT: float = Field(
        default=0.5, description="Maximum allowable capital risk per single trade in percent."
    )
    MAX_OPEN_POSITIONS: int = Field(
        default=3, description="Maximum number of simultaneous open positions allowed."
    )
    MAX_TOTAL_EXPOSURE_PERCENT: float = Field(
        default=20.0, description="Maximum total portfolio market exposure in percent of equity."
    )
    MIN_SIGNAL_CONFIDENCE: float = Field(
        default=0.60, description="Minimum AI signal confidence score required for approval."
    )

    @field_validator(
        "MAX_DAILY_LOSS_PERCENT",
        "MAX_POSITION_RISK_PERCENT",
        "MAX_TOTAL_EXPOSURE_PERCENT",
        "MIN_SIGNAL_CONFIDENCE",
    )
    @classmethod
    def validate_positive_percentages(cls, v: float, info) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{info.field_name} must be a positive finite number, got {v}")
        if "CONFIDENCE" in info.field_name and (v > 1.0):
            raise ValueError(f"{info.field_name} must be <= 1.0, got {v}")
        if "PERCENT" in info.field_name and (v > 100.0):
            raise ValueError(f"{info.field_name} cannot exceed 100.0%, got {v}")
        return float(v)

    @field_validator("MAX_OPEN_POSITIONS")
    @classmethod
    def validate_max_positions(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"MAX_OPEN_POSITIONS must be at least 1, got {v}")
        return int(v)


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_loss_price: float,
    risk_percent: float,
    max_risk_percent: float = 0.5,
) -> float:
    """Calculate deterministic position quantity based on risk capital and stop-loss distance.

    Formula:
        risk_capital = account_equity * (risk_percent / 100.0)
        stop_distance = |entry_price - stop_loss_price|
        quantity = risk_capital / stop_distance

    No leverage is assumed. Explicit errors are raised for invalid boundaries.
    """
    if not math.isfinite(account_equity) or account_equity <= 0:
        raise ValueError(f"Account equity must be a positive finite number, got {account_equity}")

    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError(f"Entry price must be a positive finite number, got {entry_price}")

    if not math.isfinite(stop_loss_price) or stop_loss_price <= 0:
        raise ValueError(f"Stop loss price must be a positive finite number, got {stop_loss_price}")

    stop_distance = abs(entry_price - stop_loss_price)
    if stop_distance <= 1e-9:
        raise ValueError(
            f"Zero stop distance: entry ({entry_price}) and stop ({stop_loss_price}) are identical"
        )

    if not math.isfinite(risk_percent) or risk_percent <= 0:
        raise ValueError(f"Risk percent must be a positive finite number, got {risk_percent}")

    if risk_percent > max_risk_percent:
        raise ValueError(
            f"Risk percent ({risk_percent}%) exceeds max position risk ({max_risk_percent}%)"
        )

    risk_capital = account_equity * (risk_percent / 100.0)
    quantity = risk_capital / stop_distance

    if not math.isfinite(quantity) or quantity <= 0:
        raise ValueError(f"Calculated quantity is invalid: {quantity}")

    return round(quantity, 4)
