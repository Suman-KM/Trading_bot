"""Deterministic, fail-closed normalization layer for MetaTrader 5 broker data."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Union

from trading.adapters.mt5.schemas import (
    MT5AccountMetadata,
    MT5BarData,
    MT5TerminalMetadata,
    MT5TickData,
    MT5Timeframe,
)
from trading.execution.validation import BrokerSymbolSpecification


class MT5DataNormalizationError(ValueError):
    """Raised when MT5 data fails schema or integrity validation (fail-closed)."""

    pass


def mask_login(login: Any) -> str:
    """Mask account login for secure logging and reporting."""
    if login is None:
        return "REDACTED"
    s = str(login).strip()
    if len(s) <= 4:
        return "***"
    return f"***{s[-4:]}"


def _get_field(data: Any, key: str, default: Any = None) -> Any:
    """Safely extract field from dict, namedtuple, or object."""
    if isinstance(data, Mapping):
        return data.get(key, default)
    if hasattr(data, key):
        return getattr(data, key)
    return default


def normalize_terminal_info(
    raw_info: Any, version_info: Optional[tuple] = None
) -> MT5TerminalMetadata:
    """Normalize raw MT5 terminal info into MT5TerminalMetadata."""
    if raw_info is None:
        raise MT5DataNormalizationError("Terminal info cannot be None.")

    name = str(_get_field(raw_info, "name", "MetaTrader 5"))
    company = str(_get_field(raw_info, "company", "Unknown"))
    path = str(_get_field(raw_info, "path", ""))
    connected = bool(_get_field(raw_info, "connected", False))
    trade_allowed = bool(_get_field(raw_info, "trade_allowed", False))
    ping_last = int(_get_field(raw_info, "ping_last", 0))

    if version_info and len(version_info) >= 3:
        build = int(version_info[1])
        version = f"{version_info[0]}.{version_info[1]} ({version_info[2]})"
    else:
        build = int(_get_field(raw_info, "build", 0))
        version = f"Build {build}"

    return MT5TerminalMetadata(
        name=name,
        company=company,
        build=build,
        version=version,
        connected=connected,
        trade_allowed=trade_allowed,
        ping_last=ping_last,
        path=path,
    )


def normalize_account_info(raw_info: Any) -> MT5AccountMetadata:
    """Normalize raw MT5 account info into MT5AccountMetadata with masked login."""
    if raw_info is None:
        raise MT5DataNormalizationError("Account info cannot be None.")

    login_val = _get_field(raw_info, "login")
    login_masked = mask_login(login_val)
    company = str(_get_field(raw_info, "company", "Unknown"))
    server = str(_get_field(raw_info, "server", "Unknown"))
    currency = str(_get_field(raw_info, "currency", "USD"))
    leverage = int(_get_field(raw_info, "leverage", 1))

    balance = float(_get_field(raw_info, "balance", 0.0))
    equity = float(_get_field(raw_info, "equity", 0.0))
    margin = float(_get_field(raw_info, "margin", 0.0))
    margin_free = float(_get_field(raw_info, "margin_free", 0.0))
    trade_mode = int(_get_field(raw_info, "trade_mode", 0))

    for val_name, val in [
        ("balance", balance),
        ("equity", equity),
        ("margin", margin),
        ("margin_free", margin_free),
    ]:
        if not math.isfinite(val):
            raise MT5DataNormalizationError(f"Account {val_name} must be a finite number: {val}")

    # trade_mode: 0 = DEMO, 1 = CONTEST, 2 = REAL
    is_demo = (trade_mode == 0) or ("demo" in server.lower())

    return MT5AccountMetadata(
        login_masked=login_masked,
        company=company,
        server=server,
        currency=currency,
        leverage=leverage,
        balance=balance,
        equity=equity,
        margin=margin,
        margin_free=margin_free,
        trade_mode=trade_mode,
        is_demo=is_demo,
    )


def normalize_symbol_info(raw_info: Any) -> BrokerSymbolSpecification:
    """Normalize raw MT5 symbol info into domain BrokerSymbolSpecification."""
    if raw_info is None:
        raise MT5DataNormalizationError("Symbol info cannot be None.")

    symbol = str(_get_field(raw_info, "name", "")).strip().upper()
    if not symbol:
        raise MT5DataNormalizationError("Symbol name cannot be empty.")

    min_vol = float(_get_field(raw_info, "volume_min", 0.01))
    max_vol = float(_get_field(raw_info, "volume_max", 100.0))
    vol_step = float(_get_field(raw_info, "volume_step", 0.01))
    contract_size = float(
        _get_field(
            raw_info,
            "trade_contract_size",
            _get_field(raw_info, "contract_size", 100_000.0),
        )
    )
    digits = int(_get_field(raw_info, "digits", 5))
    point = float(_get_field(raw_info, "point", 10**-digits))
    tick_size = float(_get_field(raw_info, "trade_tick_size", point))

    # Assert positive finite values
    for name, v in [
        ("min_volume", min_vol),
        ("max_volume", max_vol),
        ("volume_step", vol_step),
        ("contract_size", contract_size),
        ("point", point),
        ("tick_size", tick_size),
    ]:
        if not math.isfinite(v) or v <= 0:
            raise MT5DataNormalizationError(f"Symbol {name} must be positive finite: {v}")

    if min_vol > max_vol:
        raise MT5DataNormalizationError(
            f"min_volume ({min_vol}) cannot exceed max_volume ({max_vol})"
        )

    return BrokerSymbolSpecification(
        symbol=symbol,
        min_volume=min_vol,
        max_volume=max_vol,
        volume_step=vol_step,
        contract_size=contract_size,
        price_digits=digits,
        point=point,
        tick_size=tick_size,
    )


def normalize_timestamp(
    ts_val: Union[int, float, str, datetime],
    broker_utc_offset_hours: float = 3.0,
) -> datetime:
    """Convert raw MT5 timestamp (broker server time epoch or datetime) to timezone-aware UTC."""
    if isinstance(ts_val, datetime):
        if ts_val.tzinfo is None:
            # Assume naive datetime is in broker server time
            broker_dt = ts_val.replace(tzinfo=timezone.utc)
            return broker_dt - timedelta(hours=broker_utc_offset_hours)
        return ts_val.astimezone(timezone.utc)

    if isinstance(ts_val, str):
        ts_str = ts_val.strip()
        try:
            ts_val = float(ts_str)
        except ValueError:
            try:
                dt = datetime.fromisoformat(ts_str)
                return normalize_timestamp(dt, broker_utc_offset_hours=broker_utc_offset_hours)
            except Exception:
                raise MT5DataNormalizationError(f"Invalid timestamp string: {ts_val}")

    if not isinstance(ts_val, (int, float)) or not math.isfinite(ts_val) or ts_val <= 0:
        raise MT5DataNormalizationError(f"Invalid timestamp value: {ts_val}")

    # Raw integer epoch seconds or milliseconds
    # MT5 time_msc is in milliseconds if > 1e11
    if ts_val > 1e11:
        epoch_sec = ts_val / 1000.0
    else:
        epoch_sec = float(ts_val)

    # In MT5, epoch_sec represents Unix epoch interpreted in broker server time
    dt_server = datetime.fromtimestamp(epoch_sec, tz=timezone.utc)
    return dt_server - timedelta(hours=broker_utc_offset_hours)


def normalize_tick(
    raw_tick: Any,
    symbol: str,
    broker_utc_offset_hours: float = 3.0,
) -> MT5TickData:
    """Normalize raw MT5 tick into schema-validated, timezone-aware MT5TickData."""
    if raw_tick is None:
        raise MT5DataNormalizationError("Tick data cannot be None.")

    bid = float(_get_field(raw_tick, "bid", 0.0))
    ask = float(_get_field(raw_tick, "ask", 0.0))
    last = float(_get_field(raw_tick, "last", 0.0))
    vol = float(_get_field(raw_tick, "volume_real", _get_field(raw_tick, "volume", 0.0)))
    flags = int(_get_field(raw_tick, "flags", 0))

    if not math.isfinite(bid) or bid <= 0:
        raise MT5DataNormalizationError(f"Invalid bid price: {bid}")
    if not math.isfinite(ask) or ask <= 0:
        raise MT5DataNormalizationError(f"Invalid ask price: {ask}")
    if ask < bid:
        raise MT5DataNormalizationError(f"Inverted spread: ask {ask} < bid {bid}")

    # Timestamp extraction (prefer time_msc for microsecond precision)
    raw_time = _get_field(raw_tick, "time_msc", _get_field(raw_tick, "time"))
    if raw_time is None:
        raise MT5DataNormalizationError("Tick has no timestamp.")

    ts_utc = normalize_timestamp(raw_time, broker_utc_offset_hours=broker_utc_offset_hours)
    spread = round(ask - bid, 6)

    return MT5TickData(
        symbol=symbol.strip().upper(),
        timestamp_utc=ts_utc,
        bid=bid,
        ask=ask,
        last=last if math.isfinite(last) and last >= 0 else 0.0,
        volume=vol if math.isfinite(vol) and vol >= 0 else 0.0,
        flags=flags,
        spread=spread,
    )


def normalize_bar(
    raw_bar: Any,
    symbol: str,
    timeframe: MT5Timeframe,
    broker_utc_offset_hours: float = 3.0,
) -> MT5BarData:
    """Normalize raw MT5 OHLC record into schema-validated, timezone-aware MT5BarData."""
    if raw_bar is None:
        raise MT5DataNormalizationError("Bar data cannot be None.")

    open_val = float(_get_field(raw_bar, "open", 0.0))
    high_val = float(_get_field(raw_bar, "high", 0.0))
    low_val = float(_get_field(raw_bar, "low", 0.0))
    close_val = float(_get_field(raw_bar, "close", 0.0))
    tick_vol = int(_get_field(raw_bar, "tick_volume", 0))
    spread = int(_get_field(raw_bar, "spread", 0))
    real_vol = float(_get_field(raw_bar, "real_volume", 0.0))

    for name, v in [
        ("open", open_val),
        ("high", high_val),
        ("low", low_val),
        ("close", close_val),
    ]:
        if not math.isfinite(v) or v <= 0:
            raise MT5DataNormalizationError(f"Invalid candle {name} price: {v}")

    if high_val < low_val:
        raise MT5DataNormalizationError(
            f"Candle High ({high_val}) cannot be less than Low ({low_val})"
        )
    if high_val < max(open_val, close_val):
        raise MT5DataNormalizationError(
            f"Candle High ({high_val}) cannot be less than max(Open={open_val}, Close={close_val})"
        )
    if low_val > min(open_val, close_val):
        raise MT5DataNormalizationError(
            f"Candle Low ({low_val}) cannot be greater than min(Open={open_val}, Close={close_val})"
        )
    if tick_vol < 0:
        raise MT5DataNormalizationError(f"Candle tick volume cannot be negative: {tick_vol}")

    raw_time = _get_field(raw_bar, "time")
    if raw_time is None:
        raise MT5DataNormalizationError("Bar has no timestamp.")

    ts_utc = normalize_timestamp(raw_time, broker_utc_offset_hours=broker_utc_offset_hours)

    return MT5BarData(
        symbol=symbol.strip().upper(),
        timeframe=timeframe,
        timestamp_utc=ts_utc,
        open=open_val,
        high=high_val,
        low=low_val,
        close=close_val,
        tick_volume=tick_vol,
        spread=max(0, spread),
        real_volume=max(0.0, real_vol),
    )
