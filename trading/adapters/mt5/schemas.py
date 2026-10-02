"""Schemas for MetaTrader 5 read-only data structures."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class MT5Timeframe(str, Enum):
    """Supported standard MT5 bar timeframes."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class MT5TerminalMetadata(BaseModel):
    """Read-only metadata describing the MetaTrader 5 terminal environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., description="Terminal application name (e.g. MetaTrader 5)")
    company: str = Field(..., description="Terminal distributor company")
    build: int = Field(..., description="Terminal build number (e.g. 6230)")
    version: str = Field(..., description="Formatted terminal version string")
    connected: bool = Field(..., description="Whether terminal is connected to broker trade server")
    trade_allowed: bool = Field(
        ...,
        description="Internal terminal flag for EA trading (NOTE: project gate remains disabled)",
    )
    ping_last: int = Field(..., description="Last ping to broker server in microseconds")
    path: str = Field(..., description="Installation path of the terminal executable")


class MT5AccountMetadata(BaseModel):
    """Sanitized, read-only metadata describing the connected MT5 account."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    login_masked: str = Field(..., description="Sanitized account login identifier (e.g. ***1234)")
    company: str = Field(..., description="Broker company name (e.g. MetaQuotes Ltd.)")
    server: str = Field(..., description="Broker trade server name (e.g. MetaQuotes-Demo)")
    currency: str = Field(..., description="Account base currency (e.g. USD)")
    leverage: int = Field(..., description="Account leverage ratio (e.g. 100 for 1:100)")
    balance: float = Field(..., description="Current cash balance")
    equity: float = Field(..., description="Current account equity")
    margin: float = Field(..., description="Currently used margin")
    margin_free: float = Field(..., description="Available free margin")
    trade_mode: int = Field(..., description="0 = DEMO, 1 = CONTEST, 2 = REAL")
    is_demo: bool = Field(..., description="True if account is confirmed as DEMO")


class MT5TickData(BaseModel):
    """Validated, normalized point-in-time tick record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(..., description="Trading instrument symbol")
    timestamp_utc: datetime = Field(..., description="Timezone-aware UTC timestamp")
    bid: float = Field(..., gt=0, description="Bid price")
    ask: float = Field(..., gt=0, description="Ask price")
    last: float = Field(default=0.0, ge=0, description="Last traded price if available")
    volume: float = Field(default=0.0, ge=0, description="Tick volume")
    flags: int = Field(default=0, description="MT5 tick flags bitfield")
    spread: float = Field(..., ge=0, description="Calculated spread in price units (ask - bid)")


class MT5BarData(BaseModel):
    """Validated, normalized OHLC candlestick bar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(..., description="Trading instrument symbol")
    timeframe: MT5Timeframe = Field(..., description="Bar timeframe")
    timestamp_utc: datetime = Field(..., description="Timezone-aware UTC candle open timestamp")
    open: float = Field(..., gt=0, description="Bar open price")
    high: float = Field(..., gt=0, description="Bar high price")
    low: float = Field(..., gt=0, description="Bar low price")
    close: float = Field(..., gt=0, description="Bar close price")
    tick_volume: int = Field(..., ge=0, description="Bar tick volume")
    spread: int = Field(default=0, ge=0, description="Spread in points")
    real_volume: float = Field(default=0.0, ge=0, description="Bar real volume")
