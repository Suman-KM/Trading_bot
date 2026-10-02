"""Broker capability model declaring supported and unsupported broker features."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class UnsupportedBrokerOperationError(RuntimeError):
    """Raised when an operation is attempted that is not supported by the broker."""

    pass


class BrokerCapabilities(BaseModel):
    """Explicit declaration of broker capabilities and limitations.

    Ensures unsupported operations fail closed before submission.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    supports_market_orders: bool = True
    supports_limit_orders: bool = False
    supports_stop_orders: bool = False
    supports_position_query: bool = True
    supports_account_query: bool = True
    supports_cancel: bool = True
    supports_partial_fills: bool = False
    supports_hedging: bool = False
    supports_netting: bool = True

    def assert_supported(self, capability_name: str) -> None:
        """Verify that a specific capability is supported; fail closed if not."""
        if not getattr(self, capability_name, False):
            raise UnsupportedBrokerOperationError(
                f"UNSUPPORTED_BROKER_OPERATION: Capability '{capability_name}' is not supported."
            )


DEFAULT_PAPER_CAPABILITIES = BrokerCapabilities(
    supports_market_orders=True,
    supports_limit_orders=False,
    supports_stop_orders=False,
    supports_position_query=True,
    supports_account_query=True,
    supports_cancel=True,
    supports_partial_fills=False,
    supports_hedging=False,
    supports_netting=True,
)

DEFAULT_MT5_CAPABILITIES = BrokerCapabilities(
    supports_market_orders=True,
    supports_limit_orders=True,
    supports_stop_orders=True,
    supports_position_query=True,
    supports_account_query=True,
    supports_cancel=True,
    supports_partial_fills=True,
    supports_hedging=True,
    supports_netting=True,
)
