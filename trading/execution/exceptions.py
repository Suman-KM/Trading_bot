"""Exceptions for broker execution boundaries and MT5 connectivity."""

from __future__ import annotations


class BrokerExecutionDisabledError(RuntimeError):
    """Raised when an order submission or modification is attempted on a non-executing adapter."""

    pass


class MT5ConnectionError(RuntimeError):
    """Raised when an MT5 RPC/socket connection error or timeout occurs."""

    pass


class MT5ResponseError(RuntimeError):
    """Raised when MT5 returns a malformed or unrecognizable response."""

    pass
