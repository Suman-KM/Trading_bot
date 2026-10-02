"""Safe, deterministic MT5 demo execution transport with fail-closed safeguards."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from trading.adapters.mt5.safety import (
    DEMO_ONLY,
    LiveAccountForbiddenError,
)
from trading.execution.exceptions import (
    BrokerExecutionDisabledError,
    MT5ConnectionError,
    MT5ResponseError,
)
from trading.execution.translation import MT5TradeRequest

if TYPE_CHECKING:
    from trading.adapters.mt5.client import MT5ReadOnlyClient

logger = logging.getLogger("trading.adapters.mt5.transport")


class MT5ExecutionStatus(str, Enum):
    """MT5 trade execution response status."""

    FILLED = "FILLED"
    PLACED = "PLACED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class MT5ExecutionResponse(BaseModel):
    """Unified response model representing the result of an MT5 trade request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    client_request_id: Optional[str] = Field(None, description="Idempotent client request ID")
    internal_order_id: str = Field(..., description="System internal order ID")
    broker_order_id: str = Field(..., description="MT5 order ticket ID")
    broker_execution_id: str = Field(..., description="MT5 deal ticket / execution ID")
    status: MT5ExecutionStatus = Field(..., description="Execution status")
    symbol: str = Field(..., description="Trading instrument symbol")
    filled_volume: float = Field(..., ge=0.0, description="Filled volume in lots")
    remaining_volume: float = Field(0.0, ge=0.0, description="Remaining unfilled volume in lots")
    executed_price: Optional[float] = Field(None, description="Executed fill price")
    rejection_reason: Optional[str] = Field(None, description="Reason if rejected")
    timestamp: datetime = Field(..., description="UTC timestamp of the execution response")
    retcode: int = Field(0, description="MT5 return code")
    comment: Optional[str] = Field(None, description="MT5 response comment")


class MT5DemoExecutionTransport:
    """Safe, non-bypassable transport for submitting orders exclusively to an MT5 DEMO account.

    Invariants:
    1. DEMO_ONLY must be True.
    2. Account trade mode must be verified as DEMO (0) prior to order_send().
    3. If a REAL/LIVE account (2) is detected anywhere in the transport, execution immediately
       fails closed by raising LiveAccountForbiddenError.
    4. Runs `order_check()` before `order_send()`; if check fails, order_send is aborted.
    """

    def __init__(
        self,
        client: Optional[MT5ReadOnlyClient] = None,
        mock_backend: Optional[Any] = None,
        wine_prefix: Optional[str] = None,
        wine_cmd: str = "wine",
        wine_python: str = "C:\\Python\\python.exe",
    ) -> None:
        self._client = client
        self._mock_backend = mock_backend
        self._wine_prefix = wine_prefix or os.path.expanduser("~/.wine-mt5-demo")
        self._wine_cmd = wine_cmd
        self._wine_python = wine_python
        self.submissions: List[MT5TradeRequest] = []
        self.responses: List[MT5ExecutionResponse] = []

    @property
    def submission_count(self) -> int:
        """Count of trade requests submitted to this transport."""
        return len(self.submissions)

    def send_trade_request(self, req: MT5TradeRequest) -> MT5ExecutionResponse:
        """Submit a trade request to the MT5 demo terminal after strict safety verification."""
        if not DEMO_ONLY:
            raise BrokerExecutionDisabledError(
                "SAFETY_INVARIANT_VIOLATION: DEMO_ONLY global flag is False."
            )

        self.submissions.append(req)

        # 1. Mock Backend Dispatch (for fast deterministic unit testing)
        if self._mock_backend is not None:
            resp = self._send_mock_trade(req)
            self.responses.append(resp)
            return resp

        # 2. Native MetaTrader5 Library Dispatch (if running natively on Windows)
        try:
            import MetaTrader5 as mt5  # noqa: F401

            resp = self._send_native_trade(req)
            self.responses.append(resp)
            return resp
        except ImportError:
            # 3. Wine Subprocess Bridge Dispatch (Linux environment)
            resp = self._send_wine_trade(req)
            self.responses.append(resp)
            return resp

    def close_position(
        self,
        symbol: str,
        ticket: Optional[int] = None,
        volume: Optional[float] = None,
        price: Optional[float] = None,
        deviation: int = 20,
        magic: int = 100001,
        comment: Optional[str] = None,
    ) -> MT5ExecutionResponse:
        """Close an active position on the MT5 demo terminal."""
        if not DEMO_ONLY:
            raise BrokerExecutionDisabledError("SAFETY_INVARIANT_VIOLATION: DEMO_ONLY is False.")

        sym = symbol.strip().upper()

        # If ticket or volume is not specified, query MT5 for active position
        if ticket is None or volume is None or price is None:
            positions = []
            if self._client and self._client.is_connected:
                positions = self._client.get_open_positions(sym)

            if not positions:
                raise MT5ResponseError(f"No active MT5 position found for symbol {sym} to close.")

            target_pos = positions[0]
            ticket = int(target_pos.get("ticket", 0))
            volume = float(target_pos.get("volume", 0.0))
            pos_type = int(target_pos.get("type", 0))

            # Opposite direction to close: 0 (BUY) -> 1 (SELL), 1 (SELL) -> 0 (BUY)
            close_type = 1 if pos_type == 0 else 0

            # Fetch fresh tick for close price
            if price is None and self._client and self._client.is_connected:
                tick = self._client.get_symbol_tick(sym)
                price = tick.bid if close_type == 1 else tick.ask
            elif price is None:
                price = float(target_pos.get("price_current", target_pos.get("price_open", 1.0)))
        else:
            close_type = 1  # Default to SELL close unless determined

        trade_req = MT5TradeRequest(
            action=1,  # TRADE_ACTION_DEAL
            magic=magic,
            order=0,
            symbol=sym,
            volume=round(volume, 4),
            price=price,
            stop_loss=0.0,
            take_profit=0.0,
            deviation=deviation,
            order_type=close_type,
            type_filling=0,  # ORDER_FILLING_FOK
            type_time=0,  # ORDER_TIME_GTC
            comment=comment or f"Close #{ticket}",
            position=ticket,
            client_request_id=f"close-{ticket}",
            internal_order_id=f"close-ord-{ticket}",
        )
        return self.send_trade_request(trade_req)

    def _send_mock_trade(self, req: MT5TradeRequest) -> MT5ExecutionResponse:
        """Execute request against a mock backend object."""
        now_utc = datetime.now(timezone.utc)
        fn_send = getattr(self._mock_backend, "order_send", None)
        fn_check = getattr(self._mock_backend, "order_check", None)

        # Safety Check on mock
        fn_acc = getattr(self._mock_backend, "account_info", None)
        if fn_acc:
            acc = fn_acc()
            if hasattr(acc, "trade_mode") and acc.trade_mode == 2:
                raise LiveAccountForbiddenError("LIVE_ACCOUNT_DETECTED: Fatal safety abort.")

        req_dict = {
            "action": req.action,
            "magic": req.magic,
            "order": req.order,
            "symbol": req.symbol,
            "volume": req.volume,
            "price": req.price,
            "sl": req.stop_loss,
            "tp": req.take_profit,
            "deviation": req.deviation,
            "type": req.order_type,
            "type_filling": req.type_filling,
            "type_time": req.type_time,
            "comment": req.comment,
        }
        if req.position > 0:
            req_dict["position"] = req.position

        if fn_check:
            chk = fn_check(req_dict)
            if hasattr(chk, "retcode") and chk.retcode != 0:
                chk_cm = getattr(chk, "comment", "Check failed")
                return MT5ExecutionResponse(
                    client_request_id=req.client_request_id,
                    internal_order_id=req.internal_order_id,
                    broker_order_id="0",
                    broker_execution_id="0",
                    status=MT5ExecutionStatus.REJECTED,
                    symbol=req.symbol,
                    filled_volume=0.0,
                    remaining_volume=req.volume,
                    executed_price=None,
                    rejection_reason=f"MOCK_CHECK_REJECTED: {chk_cm}",
                    timestamp=now_utc,
                    retcode=getattr(chk, "retcode", -1),
                    comment=getattr(chk, "comment", None),
                )

        if fn_send:
            res = fn_send(req_dict)
            retcode = getattr(res, "retcode", 10009)
            if retcode in (10009, 10010):  # DONE or DONE_PARTIAL
                return MT5ExecutionResponse(
                    client_request_id=req.client_request_id,
                    internal_order_id=req.internal_order_id,
                    broker_order_id=str(getattr(res, "order", 5001)),
                    broker_execution_id=str(getattr(res, "deal", getattr(res, "order", 5001))),
                    status=MT5ExecutionStatus.FILLED,
                    symbol=req.symbol,
                    filled_volume=getattr(res, "volume", req.volume),
                    remaining_volume=0.0,
                    executed_price=getattr(res, "price", req.price),
                    timestamp=now_utc,
                    retcode=retcode,
                    comment=getattr(res, "comment", "Done"),
                )
            else:
                ord_cm = getattr(res, "comment", "Rejected")
                return MT5ExecutionResponse(
                    client_request_id=req.client_request_id,
                    internal_order_id=req.internal_order_id,
                    broker_order_id=str(getattr(res, "order", 0)),
                    broker_execution_id=str(getattr(res, "deal", 0)),
                    status=MT5ExecutionStatus.REJECTED,
                    symbol=req.symbol,
                    filled_volume=0.0,
                    remaining_volume=req.volume,
                    executed_price=None,
                    rejection_reason=f"MOCK_ORDER_REJECTED: retcode={retcode} comment={ord_cm}",
                    timestamp=now_utc,
                    retcode=retcode,
                    comment=getattr(res, "comment", None),
                )

        # Default fallback mock response
        return MT5ExecutionResponse(
            client_request_id=req.client_request_id,
            internal_order_id=req.internal_order_id,
            broker_order_id="mock-5001",
            broker_execution_id="mock-deal-5001",
            status=MT5ExecutionStatus.FILLED,
            symbol=req.symbol,
            filled_volume=req.volume,
            remaining_volume=0.0,
            executed_price=req.price,
            timestamp=now_utc,
            retcode=10009,
            comment="Done",
        )

    def _send_native_trade(self, req: MT5TradeRequest) -> MT5ExecutionResponse:
        """Execute request using native MetaTrader5 Python package."""
        import MetaTrader5 as mt5

        now_utc = datetime.now(timezone.utc)
        if not mt5.initialize():
            err = mt5.last_error()
            raise MT5ConnectionError(f"MT5 initialization failed: {err}")

        try:
            acc = mt5.account_info()
            if acc is None:
                raise MT5ConnectionError(f"Failed to query account info: {mt5.last_error()}")

            if acc.trade_mode == 2:
                raise LiveAccountForbiddenError("LIVE_ACCOUNT_DETECTED: Fatal safety abort.")

            if acc.trade_mode != 0:
                raise BrokerExecutionDisabledError(
                    f"NON_DEMO_ACCOUNT: trade_mode is {acc.trade_mode}, expected DEMO (0)."
                )

            req_dict = {
                "action": req.action,
                "magic": req.magic,
                "order": req.order,
                "symbol": req.symbol,
                "volume": req.volume,
                "price": req.price,
                "sl": req.stop_loss,
                "tp": req.take_profit,
                "deviation": req.deviation,
                "type": req.order_type,
                "type_filling": req.type_filling,
                "type_time": req.type_time,
                "comment": req.comment,
            }
            if req.position > 0:
                req_dict["position"] = req.position

            chk = mt5.order_check(req_dict)
            if chk is None or chk.retcode != 0:
                retcode = chk.retcode if chk else -1
                comment = chk.comment if chk else str(mt5.last_error())
                return MT5ExecutionResponse(
                    client_request_id=req.client_request_id,
                    internal_order_id=req.internal_order_id,
                    broker_order_id="0",
                    broker_execution_id="0",
                    status=MT5ExecutionStatus.REJECTED,
                    symbol=req.symbol,
                    filled_volume=0.0,
                    remaining_volume=req.volume,
                    executed_price=None,
                    rejection_reason=f"ORDER_CHECK_FAILED: retcode={retcode} comment={comment}",
                    timestamp=now_utc,
                    retcode=retcode,
                    comment=comment,
                )

            res = mt5.order_send(req_dict)
            if res is None:
                err = mt5.last_error()
                raise MT5ResponseError(f"order_send returned None: {err}")

            if res.retcode in (10009, 10010):
                return MT5ExecutionResponse(
                    client_request_id=req.client_request_id,
                    internal_order_id=req.internal_order_id,
                    broker_order_id=str(res.order),
                    broker_execution_id=str(res.deal if res.deal else res.order),
                    status=MT5ExecutionStatus.FILLED,
                    symbol=req.symbol,
                    filled_volume=res.volume,
                    remaining_volume=0.0,
                    executed_price=res.price,
                    timestamp=now_utc,
                    retcode=res.retcode,
                    comment=res.comment,
                )
            else:
                return MT5ExecutionResponse(
                    client_request_id=req.client_request_id,
                    internal_order_id=req.internal_order_id,
                    broker_order_id=str(res.order),
                    broker_execution_id=str(res.deal),
                    status=MT5ExecutionStatus.REJECTED,
                    symbol=req.symbol,
                    filled_volume=0.0,
                    remaining_volume=req.volume,
                    executed_price=None,
                    rejection_reason=(
                        f"ORDER_SEND_REJECTED: retcode={res.retcode} comment={res.comment}"
                    ),
                    timestamp=now_utc,
                    retcode=res.retcode,
                    comment=res.comment,
                )
        finally:
            mt5.shutdown()

    def _send_wine_trade(self, req: MT5TradeRequest) -> MT5ExecutionResponse:
        """Execute request by dispatching a verified Python script into the Wine MT5 environment."""
        now_utc = datetime.now(timezone.utc)
        payload = {
            "action": req.action,
            "magic": req.magic,
            "order": req.order,
            "symbol": req.symbol,
            "volume": req.volume,
            "price": req.price,
            "sl": req.stop_loss,
            "tp": req.take_profit,
            "deviation": req.deviation,
            "type": req.order_type,
            "type_filling": req.type_filling,
            "type_time": req.type_time,
            "comment": req.comment or "",
        }
        if req.position > 0:
            payload["position"] = req.position

        json_payload_str = json.dumps(payload)

        script_code = f"""
import json, sys
import MetaTrader5 as mt5

out = {{"status": "error", "error": "unknown"}}
try:
    if not mt5.initialize():
        out = {{"status": "error", "error": f"init_failed: {{mt5.last_error()}}"}}
        print(json.dumps(out))
        sys.exit(0)

    acc = mt5.account_info()
    if acc is None:
        out = {{"status": "error", "error": f"account_info_failed: {{mt5.last_error()}}"}}
        print(json.dumps(out))
        sys.exit(0)

    # Invariant 1: Reject Live Account
    if acc.trade_mode == 2:
        out = {{"status": "fatal_live", "error": "LIVE_ACCOUNT: Real capital detected."}}
        print(json.dumps(out))
        sys.exit(0)

    # Invariant 2: Require Demo Account
    if acc.trade_mode != 0:
        out = {{"status": "error", "error": f"NON_DEMO_ACCOUNT: trade_mode={{acc.trade_mode}}"}}
        print(json.dumps(out))
        sys.exit(0)

    req_data = json.loads({json.dumps(json_payload_str)})

    # Ensure symbol is selected
    mt5.symbol_select(req_data["symbol"], True)

    # 1. order_check
    chk = mt5.order_check(req_data)
    if chk is None or chk.retcode != 0:
        retcode = chk.retcode if chk else -1
        comment = chk.comment if chk else str(mt5.last_error())
        out = {{"status": "check_failed", "retcode": retcode, "comment": comment}}
        print(json.dumps(out))
        sys.exit(0)

    # 2. order_send
    res = mt5.order_send(req_data)
    if res is None:
        err = mt5.last_error()
        out = {{"status": "error", "error": f"order_send_failed: {{err}}"}}
        print(json.dumps(out))
        sys.exit(0)

    out = {{
        "status": "ok",
        "retcode": res.retcode,
        "deal": res.deal,
        "order": res.order,
        "volume": res.volume,
        "price": res.price,
        "comment": res.comment,
    }}
    print(json.dumps(out))
finally:
    mt5.shutdown()
"""
        env = os.environ.copy()
        env["WINEPREFIX"] = self._wine_prefix
        env["WINEDEBUG"] = "-all"

        cmd = [self._wine_cmd, self._wine_python, "-c", script_code]
        try:
            with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=out_f,
                    stderr=err_f,
                    timeout=30,
                )
                out_f.seek(0)
                err_f.seek(0)
                stdout = out_f.read().decode("utf-8", errors="replace")
                stderr = err_f.read().decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            raise MT5ConnectionError("MT5_TIMEOUT: Wine order execution subprocess timed out.")
        except Exception as exc:
            raise MT5ConnectionError(f"MT5_SUBPROCESS_ERROR: {str(exc)}")

        if proc.returncode != 0 and not stdout:
            raise MT5ConnectionError(f"Wine execution failed with code {proc.returncode}: {stderr}")

        # Extract JSON line
        json_lines = [line for line in stdout.splitlines() if line.strip().startswith("{")]
        if not json_lines:
            raise MT5ResponseError(
                f"No valid JSON returned from Wine. stdout: {stdout[:80]}, stderr: {stderr[:80]}"
            )

        try:
            result_data = json.loads(json_lines[-1])
        except Exception as exc:
            raise MT5ResponseError(f"Failed to parse JSON response: {exc}")

        # Safety Check
        if result_data.get("status") == "fatal_live":
            raise LiveAccountForbiddenError(
                "FATAL_SAFETY_VIOLATION: Wine execution bridge detected LIVE account."
            )

        if result_data.get("status") == "error":
            raise MT5ConnectionError(f"MT5 trade bridge error: {result_data.get('error')}")

        if result_data.get("status") == "check_failed":
            retcode = result_data.get("retcode", -1)
            comment = result_data.get("comment", "Check failed")
            return MT5ExecutionResponse(
                client_request_id=req.client_request_id,
                internal_order_id=req.internal_order_id,
                broker_order_id="0",
                broker_execution_id="0",
                status=MT5ExecutionStatus.REJECTED,
                symbol=req.symbol,
                filled_volume=0.0,
                remaining_volume=req.volume,
                executed_price=None,
                rejection_reason=f"ORDER_CHECK_FAILED: retcode={retcode} comment={comment}",
                timestamp=now_utc,
                retcode=retcode,
                comment=comment,
            )

        # Standard ok / filled response
        retcode = result_data.get("retcode", 0)
        deal = str(result_data.get("deal", 0))
        order_ticket = str(result_data.get("order", 0))
        exec_price = float(result_data.get("price", 0.0))
        vol = float(result_data.get("volume", req.volume))
        comment = result_data.get("comment", "")

        if retcode in (10009, 10010):  # DONE or DONE_PARTIAL
            return MT5ExecutionResponse(
                client_request_id=req.client_request_id,
                internal_order_id=req.internal_order_id,
                broker_order_id=order_ticket,
                broker_execution_id=deal if deal != "0" else order_ticket,
                status=MT5ExecutionStatus.FILLED,
                symbol=req.symbol,
                filled_volume=vol,
                remaining_volume=0.0,
                executed_price=exec_price if exec_price > 0 else req.price,
                timestamp=now_utc,
                retcode=retcode,
                comment=comment,
            )
        else:
            return MT5ExecutionResponse(
                client_request_id=req.client_request_id,
                internal_order_id=req.internal_order_id,
                broker_order_id=order_ticket,
                broker_execution_id=deal,
                status=MT5ExecutionStatus.REJECTED,
                symbol=req.symbol,
                filled_volume=0.0,
                remaining_volume=req.volume,
                executed_price=None,
                rejection_reason=f"MT5_TRADE_REJECTED: retcode={retcode} comment={comment}",
                timestamp=now_utc,
                retcode=retcode,
                comment=comment,
            )
