"""MT5 read-only adapter module."""

from trading.adapters.mt5.client import MT5_TIMEFRAME_MAP, MT5ReadOnlyClient
from trading.adapters.mt5.normalizer import (
    MT5DataNormalizationError,
    mask_login,
    normalize_account_info,
    normalize_bar,
    normalize_symbol_info,
    normalize_terminal_info,
    normalize_tick,
    normalize_timestamp,
)
from trading.adapters.mt5.schemas import (
    MT5AccountMetadata,
    MT5BarData,
    MT5TerminalMetadata,
    MT5TickData,
    MT5Timeframe,
)

__all__ = [
    "MT5AccountMetadata",
    "MT5BarData",
    "MT5DataNormalizationError",
    "MT5ReadOnlyClient",
    "MT5TerminalMetadata",
    "MT5TickData",
    "MT5Timeframe",
    "MT5_TIMEFRAME_MAP",
    "mask_login",
    "normalize_account_info",
    "normalize_bar",
    "normalize_symbol_info",
    "normalize_terminal_info",
    "normalize_tick",
    "normalize_timestamp",
]
