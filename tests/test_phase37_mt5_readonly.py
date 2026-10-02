"""Phase 37: MT5 Read-Only Connectivity, Account & Market-Data Validation Test Suite.

Verifies MT5 read-only client, normalization layer, timezone conversion, schema integrity,
failure handling, API readiness, and the invariant that MT5 execution remains disabled.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from trading.adapters.mt5.client import MT5_TIMEFRAME_MAP, MT5ReadOnlyClient
from trading.adapters.mt5.normalizer import (
    MT5DataNormalizationError,
    mask_login,
    normalize_account_info,
    normalize_bar,
    normalize_symbol_info,
    normalize_tick,
    normalize_timestamp,
)
from trading.adapters.mt5.schemas import MT5Timeframe
from trading.api.app import create_app
from trading.api.dependencies import (
    TradingContext,
    reset_trading_context,
    set_trading_context,
)
from trading.execution.capabilities import UnsupportedBrokerOperationError
from trading.execution.mt5_adapter import (
    BrokerExecutionDisabledError,
    MT5BrokerAdapter,
    MT5ConnectionError,
)
from trading.execution.validation import BrokerSymbolSpecification
from trading.models.order import Order, OrderSide, OrderStatus, OrderType

# -----------------------------------------------------------------------------
# Fixtures & Mock Helpers
# -----------------------------------------------------------------------------


class MockMT5Backend:
    """Mock backend providing simulated read-only MT5 primitives for deterministic testing."""

    def __init__(self, connected: bool = True, init_success: bool = True) -> None:
        self.connected = connected
        self.init_success = init_success

    def initialize(self) -> bool:
        return self.init_success

    def shutdown(self) -> None:
        pass

    def terminal_info(self) -> dict:
        return {
            "name": "MetaTrader 5",
            "company": "MetaQuotes Ltd.",
            "build": 6230,
            "connected": self.connected,
            "trade_allowed": False,
            "ping_last": 150000,
            "path": "C:\\Program Files\\MetaTrader 5",
        }

    def version(self) -> tuple:
        return (500, 6230, "25 Sep 2026")

    def account_info(self) -> dict:
        return {
            "login": 98765432,
            "server": "MetaQuotes-Demo",
            "company": "MetaQuotes Ltd.",
            "currency": "USD",
            "leverage": 100,
            "balance": 100000.0,
            "equity": 100000.0,
            "margin": 0.0,
            "margin_free": 100000.0,
            "trade_mode": 0,
        }

    def symbol_info(self, symbol: str) -> dict:
        if symbol.upper() != "EURUSD":
            return None
        return {
            "name": "EURUSD",
            "digits": 5,
            "point": 0.00001,
            "trade_contract_size": 100000.0,
            "volume_min": 0.01,
            "volume_max": 500.0,
            "volume_step": 0.01,
            "trade_tick_size": 0.00001,
        }

    def symbol_info_tick(self, symbol: str) -> dict:
        if symbol.upper() != "EURUSD":
            return None
        return {
            "time": 1790971200,
            "time_msc": 1790971200500,
            "bid": 1.12550,
            "ask": 1.12560,
            "last": 0.0,
            "volume": 0,
            "flags": 1026,
        }

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start: int, count: int) -> list:
        if symbol.upper() != "EURUSD":
            return None
        base_time = 1790971200
        return [
            {
                "time": base_time - (i * 900),
                "open": 1.12500,
                "high": 1.12600,
                "low": 1.12450,
                "close": 1.12550,
                "tick_volume": 1500,
                "spread": 1,
                "real_volume": 0.0,
            }
            for i in range(count)
        ]


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------


def test_01_environment_discovery():
    """Environment discovery verifies client instantiation and Wine prefix handling."""
    client = MT5ReadOnlyClient(wine_prefix="/custom/wine/prefix")
    assert client._wine_prefix == "/custom/wine/prefix"
    assert client.execution_enabled is False
    assert client.is_connected is False


def test_02_adapter_execution_permanently_disabled():
    """MT5BrokerAdapter remains execution-disabled even when paired with read-only client."""
    mock_b = MockMT5Backend(connected=True)
    client = MT5ReadOnlyClient(mock_backend=mock_b)
    client.initialize()

    adapter = MT5BrokerAdapter(client=client)
    assert adapter.is_live is False
    assert adapter.execution_enabled is False
    assert adapter.is_readonly_connected is True
    assert adapter.market_data_available is True

    # Order submissions must be rejected
    order = Order(
        order_id="ord_test",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10_000.0,
        price=1.1250,
        status=OrderStatus.VALIDATED,
    )
    try:
        adapter.submit_order(order)
        assert False, "submit_order should have raised BrokerExecutionDisabledError"
    except BrokerExecutionDisabledError as exc:
        assert "MT5_EXECUTION_DISABLED" in str(exc)

    try:
        adapter.cancel_order("ord_test")
        assert False, "cancel_order should have raised BrokerExecutionDisabledError"
    except BrokerExecutionDisabledError:
        pass


def test_03_mt5_connectivity_state():
    """MT5 client manages connection lifecycle and reports connected state."""
    mock_b = MockMT5Backend(connected=True, init_success=True)
    client = MT5ReadOnlyClient(mock_backend=mock_b)

    assert not client.is_connected
    ok = client.initialize()
    assert ok is True
    assert client.is_connected is True

    term_meta = client.get_terminal_metadata()
    assert term_meta.connected is True
    assert term_meta.build == 6230

    client.shutdown()
    assert client.is_connected is False


def test_04_account_info_normalization():
    """Account metadata normalization masks login and extracts balances safely."""
    raw_acc = {
        "login": 12345678,
        "company": "MetaQuotes Ltd.",
        "server": "MetaQuotes-Demo",
        "currency": "USD",
        "leverage": 100,
        "balance": 50000.0,
        "equity": 50250.0,
        "margin": 1000.0,
        "margin_free": 49250.0,
        "trade_mode": 0,
    }
    meta = normalize_account_info(raw_acc)
    assert meta.login_masked == "***5678"
    assert meta.balance == 50000.0
    assert meta.equity == 50250.0
    assert meta.is_demo is True
    assert meta.currency == "USD"
    assert mask_login(None) == "REDACTED"
    assert mask_login(123) == "***"


def test_05_symbol_info_normalization():
    """Symbol specification normalization converts raw metadata to BrokerSymbolSpecification."""
    raw_sym = {
        "name": "EURUSD",
        "digits": 5,
        "point": 0.00001,
        "trade_contract_size": 100000.0,
        "volume_min": 0.01,
        "volume_max": 500.0,
        "volume_step": 0.01,
        "trade_tick_size": 0.00001,
    }
    spec = normalize_symbol_info(raw_sym)
    assert isinstance(spec, BrokerSymbolSpecification)
    assert spec.symbol == "EURUSD"
    assert spec.price_digits == 5
    assert spec.point == 0.00001
    assert spec.min_volume == 0.01
    assert spec.max_volume == 500.0
    assert spec.volume_step == 0.01
    assert spec.contract_size == 100000.0


def test_06_eurusd_specification_validation():
    """EURUSD specifications retrieved via client match standard broker contract."""
    mock_b = MockMT5Backend()
    client = MT5ReadOnlyClient(mock_backend=mock_b)
    client.initialize()

    spec = client.get_symbol_specification("EURUSD")
    assert spec.symbol == "EURUSD"
    assert spec.contract_size == 100_000.0
    assert spec.min_volume <= spec.max_volume
    assert spec.volume_step > 0


def test_07_tick_normalization():
    """Tick normalization validates prices, spread, and timezone conversion."""
    raw_tick = {
        "time": 1790971200,
        "time_msc": 1790971200500,
        "bid": 1.12550,
        "ask": 1.12565,
        "last": 0.0,
        "volume": 10.0,
        "flags": 1026,
    }
    tick = normalize_tick(raw_tick, symbol="EURUSD", broker_utc_offset_hours=3.0)
    assert tick.symbol == "EURUSD"
    assert tick.bid == 1.12550
    assert tick.ask == 1.12565
    assert math.isclose(tick.spread, 0.00015, rel_tol=1e-5)
    assert tick.timestamp_utc.tzinfo == timezone.utc


def test_08_ohlc_normalization():
    """OHLC candlestick normalization enforces price boundaries and volume sanity."""
    raw_bar = {
        "time": 1790971200,
        "open": 1.12500,
        "high": 1.12650,
        "low": 1.12450,
        "close": 1.12600,
        "tick_volume": 1200,
        "spread": 1,
        "real_volume": 0.0,
    }
    bar = normalize_bar(
        raw_bar,
        symbol="EURUSD",
        timeframe=MT5Timeframe.M15,
        broker_utc_offset_hours=3.0,
    )
    assert bar.symbol == "EURUSD"
    assert bar.timeframe == MT5Timeframe.M15
    assert bar.open == 1.12500
    assert bar.high == 1.12650
    assert bar.low == 1.12450
    assert bar.close == 1.12600
    assert bar.tick_volume == 1200
    assert bar.timestamp_utc.tzinfo == timezone.utc


def test_09_utc_timezone_conversion():
    """Explicitly verifies broker timestamp (UTC+3) converts to UTC by 3-hour difference."""
    # 1790971200 is 2026-10-02 20:00:00 when treated as UTC
    # With broker_utc_offset_hours=3.0, the true UTC time is 17:00:00
    ts_utc = normalize_timestamp(1790971200, broker_utc_offset_hours=3.0)
    expected_utc = datetime(2026, 10, 2, 17, 0, 0, tzinfo=timezone.utc)
    assert ts_utc == expected_utc

    # Datetime input conversion
    naive_dt = datetime(2026, 10, 2, 20, 0, 0)
    ts_from_dt = normalize_timestamp(naive_dt, broker_utc_offset_hours=3.0)
    assert ts_from_dt == expected_utc


def test_10_malformed_market_data_rejection():
    """Malformed market data triggers fail-closed MT5DataNormalizationError."""
    # Inverted spread (ask < bid)
    bad_tick = {"time": 1790971200, "bid": 1.1260, "ask": 1.1250}
    try:
        normalize_tick(bad_tick, "EURUSD")
        assert False, "Should raise MT5DataNormalizationError on inverted spread"
    except MT5DataNormalizationError as exc:
        assert "Inverted spread" in str(exc)

    # Negative price
    bad_bid = {"time": 1790971200, "bid": -1.1250, "ask": 1.1260}
    try:
        normalize_tick(bad_bid, "EURUSD")
        assert False
    except MT5DataNormalizationError:
        pass

    # High < Low candle
    bad_ohlc_1 = {
        "time": 1790971200,
        "open": 1.12,
        "high": 1.10,
        "low": 1.15,
        "close": 1.12,
        "tick_volume": 10,
    }
    try:
        normalize_bar(bad_ohlc_1, "EURUSD", MT5Timeframe.M15)
        assert False
    except MT5DataNormalizationError:
        pass

    # High < Open candle
    bad_ohlc_2 = {
        "time": 1790971200,
        "open": 1.15,
        "high": 1.14,
        "low": 1.10,
        "close": 1.12,
        "tick_volume": 10,
    }
    try:
        normalize_bar(bad_ohlc_2, "EURUSD", MT5Timeframe.M15)
        assert False
    except MT5DataNormalizationError:
        pass

    # Negative volume
    bad_vol = {
        "time": 1790971200,
        "open": 1.12,
        "high": 1.13,
        "low": 1.11,
        "close": 1.12,
        "tick_volume": -5,
    }
    try:
        normalize_bar(bad_vol, "EURUSD", MT5Timeframe.M15)
        assert False
    except MT5DataNormalizationError:
        pass


def test_11_connection_failure_fail_closed():
    """Uninitialized or failed MT5 connection fails closed without inventing data."""
    mock_b = MockMT5Backend(init_success=False)
    client = MT5ReadOnlyClient(mock_backend=mock_b)

    ok = client.initialize()
    assert ok is False
    assert client.is_connected is False

    try:
        client.get_terminal_metadata()
        assert False
    except MT5ConnectionError:
        pass

    try:
        client.get_latest_tick("EURUSD")
        assert False
    except MT5ConnectionError:
        pass


def test_12_symbol_failure_fail_closed():
    """Unknown or invalid symbol queries fail closed."""
    mock_b = MockMT5Backend()
    client = MT5ReadOnlyClient(mock_backend=mock_b)
    client.initialize()

    try:
        client.get_symbol_specification("UNKNOWN_COIN")
        assert False
    except MT5DataNormalizationError:
        pass


def test_13_api_readiness_exposure():
    """API /readiness endpoint exposes MT5 read-only status without blocking paper trading."""
    reset_trading_context()
    mock_b = MockMT5Backend(connected=True)
    client = MT5ReadOnlyClient(mock_backend=mock_b)
    client.initialize()

    adapter = MT5BrokerAdapter(client=client)
    ctx = TradingContext(mt5_adapter=adapter)
    set_trading_context(ctx)

    app = create_app()
    client_http = TestClient(app)
    resp = client_http.get("/readiness")
    assert resp.status_code == 200
    data = resp.json()

    assert data["ready"] is True
    assert data["paper_broker_initialized"] is True
    assert data["mt5_adapter_available"] is True
    assert data["mt5_execution_enabled"] is False
    assert data["mt5_readonly_connected"] is True
    assert data["mt5_market_data_available"] is True
    reset_trading_context()


def test_14_execution_remains_disabled_when_connected():
    """Connected MT5 read-only client does NOT enable execution or modify project safety."""
    mock_b = MockMT5Backend(connected=True)
    client = MT5ReadOnlyClient(mock_backend=mock_b)
    client.initialize()
    assert client.is_connected is True

    adapter = MT5BrokerAdapter(client=client)
    assert adapter.execution_enabled is False

    # Symbol info and account info can be retrieved in read-only mode
    spec = adapter.get_symbol_info("EURUSD")
    assert spec is not None
    assert spec.symbol == "EURUSD"

    acc = adapter.get_account()
    assert acc.equity == 100000.0


def test_15_no_order_submission_path():
    """MT5ReadOnlyClient contains zero order execution paths."""
    client = MT5ReadOnlyClient()
    try:
        client.order_send()
        assert False
    except BrokerExecutionDisabledError as exc:
        assert "FORBIDDEN" in str(exc)

    try:
        client.order_check()
        assert False
    except BrokerExecutionDisabledError as exc:
        assert "FORBIDDEN" in str(exc)

    try:
        client.cancel_order()
        assert False
    except UnsupportedBrokerOperationError as exc:
        assert "FORBIDDEN" in str(exc)


def test_16_timeframe_constants():
    """Verifies all standard timeframes map to valid MT5 API constants."""
    assert MT5_TIMEFRAME_MAP[MT5Timeframe.M1] == 1
    assert MT5_TIMEFRAME_MAP[MT5Timeframe.M5] == 5
    assert MT5_TIMEFRAME_MAP[MT5Timeframe.M15] == 15
    assert MT5_TIMEFRAME_MAP[MT5Timeframe.M30] == 30
    assert MT5_TIMEFRAME_MAP[MT5Timeframe.H1] == 16385
    assert MT5_TIMEFRAME_MAP[MT5Timeframe.H4] == 16388
    assert MT5_TIMEFRAME_MAP[MT5Timeframe.D1] == 16408


def test_17_live_environment_inspection_report_validation():
    """Validates the live inspection artifact generated by scripts/inspect_mt5_readonly.py."""
    import json
    from pathlib import Path

    import pytest

    report_p = Path("reports/mt5_readonly_inspection.json")
    if not report_p.exists():
        pytest.skip("reports/mt5_readonly_inspection.json does not exist.")

    with open(report_p, encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "INITIALIZED"
    assert data["terminal"]["name"] == "MetaTrader 5"
    assert data["terminal"]["connected"] is True

    # Account metadata validation
    acc = data["account"]
    assert acc["login_masked"].startswith("***")
    assert acc["server"] == "MetaQuotes-Demo"
    assert acc["trade_mode"] == 0
    assert acc["balance"] > 0
    assert acc["equity"] > 0

    # Symbol metadata validation
    sym = data["symbol"]
    assert sym["name"] == "EURUSD"
    assert sym["digits"] == 5
    assert sym["contract_size"] == 100000.0
    assert sym["volume_min"] == 0.01

    # Latest tick validation
    tick = data["latest_tick"]
    assert tick["bid"] > 0
    assert tick["ask"] >= tick["bid"]

    # OHLC sample validation
    ohlc = data["ohlc_sample"]
    assert len(ohlc) == 5
    for bar in ohlc:
        assert bar["high"] >= bar["low"]
        assert bar["high"] >= max(bar["open"], bar["close"])
        assert bar["low"] <= min(bar["open"], bar["close"])
