"""Strictly read-only MetaTrader 5 client for market-data and environment diagnostics."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading.adapters.mt5.normalizer import (
    normalize_account_info,
    normalize_bar,
    normalize_symbol_info,
    normalize_terminal_info,
    normalize_tick,
)
from trading.adapters.mt5.schemas import (
    MT5AccountMetadata,
    MT5BarData,
    MT5TerminalMetadata,
    MT5TickData,
    MT5Timeframe,
)
from trading.execution.capabilities import UnsupportedBrokerOperationError
from trading.execution.mt5_adapter import BrokerExecutionDisabledError, MT5ConnectionError
from trading.execution.validation import BrokerSymbolSpecification

logger = logging.getLogger(__name__)

# Standard MT5 Timeframe Constants
MT5_TIMEFRAME_MAP = {
    MT5Timeframe.M1: 1,
    MT5Timeframe.M5: 5,
    MT5Timeframe.M15: 15,
    MT5Timeframe.M30: 30,
    MT5Timeframe.H1: 16385,
    MT5Timeframe.H4: 16388,
    MT5Timeframe.D1: 16408,
}


class MT5ReadOnlyClient:
    """Strictly read-only client for querying MT5 terminal, account, and market data.

    ABSOLUTE SAFETY INVARIANTS:
    1. Zero execution methods: no order_send(), order_check(), or position modification.
    2. Any attempt to access or call order execution triggers BrokerExecutionDisabledError.
    3. execution_enabled is hardcoded to False.
    4. Fails closed if MT5 is unavailable or returns an error.
    """

    def __init__(
        self,
        wine_prefix: Optional[str] = None,
        terminal_path: Optional[str] = None,
        broker_utc_offset_hours: float = 3.0,
        mock_backend: Optional[Any] = None,
    ) -> None:
        self._wine_prefix = (
            wine_prefix or os.environ.get("WINEPREFIX") or str(Path.home() / ".wine-mt5-demo")
        )
        self._terminal_path = terminal_path
        self._broker_utc_offset_hours = broker_utc_offset_hours
        self._mock_backend = mock_backend
        self._is_initialized = False

    @property
    def execution_enabled(self) -> bool:
        """Always False. Order execution is permanently disabled in this client."""
        return False

    @property
    def is_connected(self) -> bool:
        """Return True if client has established an active read-only session."""
        return self._is_initialized

    def order_send(self, *args, **kwargs) -> Any:
        """Blocked execution entrypoint. Always raises BrokerExecutionDisabledError."""
        raise BrokerExecutionDisabledError(
            "MT5_EXECUTION_FORBIDDEN: MT5ReadOnlyClient does not support order_send()."
        )

    def order_check(self, *args, **kwargs) -> Any:
        """Blocked execution entrypoint. Always raises BrokerExecutionDisabledError."""
        raise BrokerExecutionDisabledError(
            "MT5_EXECUTION_FORBIDDEN: MT5ReadOnlyClient does not support order_check()."
        )

    def cancel_order(self, *args, **kwargs) -> Any:
        """Blocked execution entrypoint."""
        raise UnsupportedBrokerOperationError(
            "MT5_EXECUTION_FORBIDDEN: MT5ReadOnlyClient cannot cancel orders."
        )

    def close_position(self, *args, **kwargs) -> Any:
        """Blocked execution entrypoint."""
        raise UnsupportedBrokerOperationError(
            "MT5_EXECUTION_FORBIDDEN: MT5ReadOnlyClient cannot close positions."
        )

    def initialize(self) -> bool:
        """Initialize the MT5 terminal session in read-only mode."""
        if self._mock_backend is not None:
            self._is_initialized = bool(self._mock_backend.initialize())
            return self._is_initialized

        try:
            import MetaTrader5 as mt5

            if self._terminal_path:
                init_ok = mt5.initialize(path=self._terminal_path)
            else:
                init_ok = mt5.initialize()
            if not init_ok:
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                self._is_initialized = False
                return False
            self._is_initialized = True
            return True
        except ImportError:
            # Fall back to testing Wine subprocess execution if running on Linux host
            res = self._execute_wine_command("import MetaTrader5 as mt5; print(mt5.initialize())")
            if res.get("status") == "ok" and "True" in res.get("stdout", ""):
                self._is_initialized = True
                return True
            self._is_initialized = False
            return False

    def shutdown(self) -> None:
        """Cleanly terminate the MT5 session."""
        if self._mock_backend is not None:
            self._mock_backend.shutdown()
            self._is_initialized = False
            return

        try:
            import MetaTrader5 as mt5

            mt5.shutdown()
        except ImportError:
            pass
        self._is_initialized = False

    def get_terminal_metadata(self) -> MT5TerminalMetadata:
        """Retrieve and normalize terminal metadata."""
        if not self._is_initialized:
            raise MT5ConnectionError("MT5 client is not initialized.")

        if self._mock_backend is not None:
            raw_term = self._mock_backend.terminal_info()
            version_info = getattr(self._mock_backend, "version", lambda: (500, 6230, "Mock"))()
            return normalize_terminal_info(raw_term, version_info)

        try:
            import MetaTrader5 as mt5

            raw_term = mt5.terminal_info()
            v_info = mt5.version()
            return normalize_terminal_info(raw_term, v_info)
        except ImportError:
            res = self._run_wine_diagnostic_query("terminal")
            return normalize_terminal_info(res.get("terminal"), res.get("version"))

    def get_account_metadata(self) -> MT5AccountMetadata:
        """Retrieve and normalize connected account metadata."""
        if not self._is_initialized:
            raise MT5ConnectionError("MT5 client is not initialized.")

        if self._mock_backend is not None:
            raw_acc = self._mock_backend.account_info()
            return normalize_account_info(raw_acc)

        try:
            import MetaTrader5 as mt5

            raw_acc = mt5.account_info()
            return normalize_account_info(raw_acc)
        except ImportError:
            res = self._run_wine_diagnostic_query("account")
            return normalize_account_info(res.get("account"))

    def get_symbol_specification(self, symbol: str) -> BrokerSymbolSpecification:
        """Retrieve and normalize broker symbol specifications."""
        if not self._is_initialized:
            raise MT5ConnectionError("MT5 client is not initialized.")

        sym = symbol.strip().upper()
        if self._mock_backend is not None:
            raw_sym = self._mock_backend.symbol_info(sym)
            return normalize_symbol_info(raw_sym)

        try:
            import MetaTrader5 as mt5

            mt5.symbol_select(sym, True)
            raw_sym = mt5.symbol_info(sym)
            return normalize_symbol_info(raw_sym)
        except ImportError:
            res = self._run_wine_diagnostic_query(f"symbol_{sym}")
            return normalize_symbol_info(res.get("symbol"))

    def get_latest_tick(self, symbol: str) -> MT5TickData:
        """Retrieve and normalize latest market tick for symbol."""
        if not self._is_initialized:
            raise MT5ConnectionError("MT5 client is not initialized.")

        sym = symbol.strip().upper()
        if self._mock_backend is not None:
            raw_tick = self._mock_backend.symbol_info_tick(sym)
            return normalize_tick(
                raw_tick, symbol=sym, broker_utc_offset_hours=self._broker_utc_offset_hours
            )

        try:
            import MetaTrader5 as mt5

            mt5.symbol_select(sym, True)
            raw_tick = mt5.symbol_info_tick(sym)
            return normalize_tick(
                raw_tick, symbol=sym, broker_utc_offset_hours=self._broker_utc_offset_hours
            )
        except ImportError:
            res = self._run_wine_diagnostic_query(f"tick_{sym}")
            return normalize_tick(
                res.get("tick"), symbol=sym, broker_utc_offset_hours=self._broker_utc_offset_hours
            )

    def get_historical_bars(
        self, symbol: str, timeframe: MT5Timeframe, count: int = 5
    ) -> List[MT5BarData]:
        """Retrieve a small historical OHLC sample."""
        if not self._is_initialized:
            raise MT5ConnectionError("MT5 client is not initialized.")
        if count <= 0 or count > 500:
            raise ValueError(f"Count must be between 1 and 500: {count}")

        sym = symbol.strip().upper()
        if self._mock_backend is not None:
            raw_rates = self._mock_backend.copy_rates_from_pos(
                sym, MT5_TIMEFRAME_MAP[timeframe], 0, count
            )
            return [
                normalize_bar(
                    r,
                    sym,
                    timeframe,
                    broker_utc_offset_hours=self._broker_utc_offset_hours,
                )
                for r in (raw_rates or [])
            ]

        try:
            import MetaTrader5 as mt5

            mt5.symbol_select(sym, True)
            raw_rates = mt5.copy_rates_from_pos(sym, MT5_TIMEFRAME_MAP[timeframe], 0, count)
            if raw_rates is None:
                return []
            return [
                normalize_bar(
                    r,
                    sym,
                    timeframe,
                    broker_utc_offset_hours=self._broker_utc_offset_hours,
                )
                for r in raw_rates
            ]
        except ImportError:
            res = self._run_wine_diagnostic_query(f"bars_{sym}_{timeframe.value}_{count}")
            raw_bars = res.get("bars", [])
            return [
                normalize_bar(
                    r,
                    sym,
                    timeframe,
                    broker_utc_offset_hours=self._broker_utc_offset_hours,
                )
                for r in raw_bars
            ]

    def _execute_wine_command(self, python_code: str) -> Dict[str, Any]:
        """Execute a Python snippet inside the Wine Python environment."""
        wine_prefix = self._wine_prefix
        cmd = [
            "wine",
            "C:\\Python\\python.exe",
            "-c",
            python_code,
        ]
        env = os.environ.copy()
        env["WINEPREFIX"] = wine_prefix
        env["WINEDEBUG"] = "-all"
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def _run_wine_diagnostic_query(self, query_type: str) -> Dict[str, Any]:
        """Run a structured JSON-producing diagnostic query inside Wine."""
        code = """
import json, sys
import MetaTrader5 as mt5
from datetime import datetime, timezone

if not mt5.initialize():
    print(json.dumps({'status': 'error', 'last_error': mt5.last_error()}))
    sys.exit(0)

res = {'status': 'ok'}
q = '__QUERY_TYPE__'
if q == 'terminal':
    t = mt5.terminal_info()
    res['terminal'] = t._asdict() if t else None
    res['version'] = list(mt5.version()) if mt5.version() else None
elif q == 'account':
    a = mt5.account_info()
    res['account'] = a._asdict() if a else None
elif q.startswith('symbol_'):
    s_name = q.split('_', 1)[1]
    mt5.symbol_select(s_name, True)
    s = mt5.symbol_info(s_name)
    sym_d = None
    if s:
        sym_d = {
            k: getattr(s, k)
            for k in dir(s)
            if not k.startswith('_') and not callable(getattr(s, k))
        }
    res['symbol'] = sym_d
elif q.startswith('tick_'):
    s_name = q.split('_', 1)[1]
    mt5.symbol_select(s_name, True)
    t = mt5.symbol_info_tick(s_name)
    res['tick'] = t._asdict() if t else None
elif q.startswith('bars_'):
    _, s_name, tf_name, cnt = q.split('_')
    cnt = int(cnt)
    tf = getattr(mt5, f'TIMEFRAME_{tf_name}', mt5.TIMEFRAME_M15)
    mt5.symbol_select(s_name, True)
    rates = mt5.copy_rates_from_pos(s_name, tf, 0, cnt)
    bar_list = []
    if rates is not None:
        for r in rates:
            bar_list.append({
                k: float(r[k]) if isinstance(r[k], (int, float)) else str(r[k])
                for k in r.dtype.names
            })
    res['bars'] = bar_list

mt5.shutdown()
print(json.dumps(res))
""".replace("__QUERY_TYPE__", query_type)
        exec_res = self._execute_wine_command(code)
        if exec_res.get("status") != "ok":
            raise MT5ConnectionError(f"Wine diagnostic query failed: {exec_res.get('stderr')}")

        lines = [
            line for line in exec_res.get("stdout", "").splitlines() if line.strip().startswith("{")
        ]
        if not lines:
            raise MT5ConnectionError("No valid JSON payload returned from MT5 Wine bridge.")

        data = json.loads(lines[-1])
        if data.get("status") != "ok":
            raise MT5ConnectionError(f"MT5 returned error: {data.get('last_error')}")
        return data
