"""Models and schemas for deterministic market-data replay."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ReplayMode(str, Enum):
    """Execution modes for market-data replay."""

    FAST_REPLAY = "FAST_REPLAY"
    REALTIME_SIMULATED_REPLAY = "REALTIME_SIMULATED_REPLAY"


class MarketEvent(BaseModel):
    """Single discrete market price observation for replay."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    symbol: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[float] = None


class ReplayCheckpoint(BaseModel):
    """Serializable snapshot of replay progress and engine state."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    timestamp: datetime
    cursor_index: int
    account_state: Dict[str, Any]
    active_positions: Dict[str, Any]
    closed_positions_count: int
    orders_count: int
    executions_count: int
    kill_switch_active: bool
    kill_switch_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
