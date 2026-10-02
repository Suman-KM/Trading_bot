"""Signal data model for Member 2 AI signal ingestion."""

import math
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class SignalAction(str, Enum):
    """Permitted trading actions emitted by AI signals."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Signal(BaseModel):
    """Strictly validated trading signal model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    action: SignalAction
    confidence: float
    timestamp: datetime
    model_version: str
    timeframe: str
    expected_return: float
    feature_version: str
    client_request_id: Optional[str] = None
    suggested_entry_price: Optional[float] = None
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty")
        return cleaned

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Confidence must be a finite number")
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return float(v)

    @field_validator("expected_return")
    @classmethod
    def validate_expected_return(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Expected return must be a finite number (no NaN or Inf)")
        return float(v)

    @field_validator("model_version", "timeframe", "feature_version")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError(f"{info.field_name} cannot be empty")
        return cleaned

    @field_validator("suggested_entry_price", "suggested_stop_loss", "suggested_take_profit")
    @classmethod
    def validate_optional_prices(cls, v: Optional[float], info) -> Optional[float]:
        if v is not None:
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{info.field_name} must be a positive finite number, got {v}")
            return float(v)
        return None
