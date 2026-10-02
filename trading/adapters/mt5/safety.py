"""Deterministic, fail-closed safety gate and verification for MT5 demo account execution.

ABSOLUTE GOVERNANCE INVARIANTS:
1. Live accounts (trade_mode == 2 or REAL) are strictly FORBIDDEN.
2. Orders can only be submitted if account is positively verified as DEMO (trade_mode == 0).
3. Requires explicit multi-condition authorization:
   - DEMO_ONLY == True
   - is_live == False
   - DEMO_EXECUTION_ENABLED == True
   - ACCOUNT_TRADE_MODE == DEMO
   - execution_enabled == True
4. Any ambiguity (UNKNOWN, UNAVAILABLE, UNVERIFIED) MUST fail closed.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict

from trading.adapters.mt5.schemas import MT5AccountMetadata, MT5TerminalMetadata
from trading.execution.exceptions import BrokerExecutionDisabledError
from trading.execution.validation import BrokerSymbolSpecification

# Strict Global Flag
DEMO_ONLY: bool = True

# Standard MT5 Trade Mode Constants
ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2


class DemoAccountVerificationError(RuntimeError):
    """Raised when an MT5 account fails positive verification as a DEMO account."""

    pass


class LiveAccountForbiddenError(BrokerExecutionDisabledError):
    """Raised when a LIVE/REAL trading account is detected. Fatal safety violation."""

    pass


class DemoExecutionNotAuthorizedError(BrokerExecutionDisabledError):
    """Raised when demo execution is attempted without explicit multi-condition authorization."""

    pass


class AccountTradeMode(str, Enum):
    """Standardized account trade mode classifications."""

    DEMO = "DEMO"
    CONTEST = "CONTEST"
    REAL = "REAL"
    UNKNOWN = "UNKNOWN"


class DemoVerificationReport(BaseModel):
    """Detailed verification audit result confirming account is DEMO."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verified: bool
    account_login_masked: str
    company: str
    server: str
    currency: str
    leverage: int
    balance: float
    equity: float
    trade_mode_code: int
    trade_mode_str: str
    is_demo: bool
    is_live_detected: bool
    terminal_connected: bool
    symbol: Optional[str] = None
    min_volume: Optional[float] = None
    max_volume: Optional[float] = None
    volume_step: Optional[float] = None
    contract_size: Optional[float] = None
    verification_notes: str


class DemoAccountVerifier:
    """Authority for positively verifying MT5 account classification before order submission."""

    @staticmethod
    def classify_trade_mode(trade_mode: int) -> AccountTradeMode:
        """Classify numerical MT5 trade_mode into enum."""
        if trade_mode == ACCOUNT_TRADE_MODE_DEMO:
            return AccountTradeMode.DEMO
        if trade_mode == ACCOUNT_TRADE_MODE_CONTEST:
            return AccountTradeMode.CONTEST
        if trade_mode == ACCOUNT_TRADE_MODE_REAL:
            return AccountTradeMode.REAL
        return AccountTradeMode.UNKNOWN

    @classmethod
    def verify(
        cls,
        account_meta: Optional[MT5AccountMetadata],
        terminal_meta: Optional[MT5TerminalMetadata] = None,
        symbol_spec: Optional[BrokerSymbolSpecification] = None,
    ) -> DemoVerificationReport:
        """Verify that account is confirmed DEMO; fail closed on live or unverified state."""
        if account_meta is None:
            raise DemoAccountVerificationError(
                "DEMO_VERIFICATION_FAILED: Account metadata is None or unavailable."
            )

        # 1. Fatal Real Money Check
        if account_meta.trade_mode == ACCOUNT_TRADE_MODE_REAL:
            raise LiveAccountForbiddenError(
                "FATAL_SAFETY_VIOLATION: Connected MT5 account is LIVE/REAL. "
                "Order submission is permanently forbidden in this environment."
            )

        # 2. Strict Demo Mode Verification
        mode_enum = cls.classify_trade_mode(account_meta.trade_mode)
        if mode_enum != AccountTradeMode.DEMO:
            raise DemoAccountVerificationError(
                f"DEMO_VERIFICATION_FAILED: Account trade mode is {mode_enum.value} "
                f"(code {account_meta.trade_mode}), expected DEMO (code 0)."
            )

        if not account_meta.is_demo:
            raise DemoAccountVerificationError(
                "DEMO_VERIFICATION_FAILED: Account is_demo flag is False."
            )

        # 3. Numeric Integrity Checks
        if not math.isfinite(account_meta.balance) or account_meta.balance <= 0:
            raise DemoAccountVerificationError(
                f"DEMO_VERIFICATION_FAILED: Invalid account balance: {account_meta.balance}"
            )
        if not math.isfinite(account_meta.equity) or account_meta.equity <= 0:
            raise DemoAccountVerificationError(
                f"DEMO_VERIFICATION_FAILED: Invalid account equity: {account_meta.equity}"
            )

        # 4. Terminal Metadata Checks
        term_connected = True
        if terminal_meta is not None:
            if not terminal_meta.connected:
                raise DemoAccountVerificationError(
                    "DEMO_VERIFICATION_FAILED: MT5 terminal is disconnected from broker server."
                )
            term_connected = terminal_meta.connected

        # 5. Symbol Specification Verification
        sym_name = None
        min_vol = None
        max_vol = None
        vol_step = None
        contract_size = None
        if symbol_spec is not None:
            sym_name = symbol_spec.symbol.upper()
            if sym_name != "EURUSD":
                raise DemoAccountVerificationError(
                    f"DEMO_VERIFICATION_FAILED: Only EURUSD authorized, got {sym_name}."
                )
            min_vol = symbol_spec.min_volume
            max_vol = symbol_spec.max_volume
            vol_step = symbol_spec.volume_step
            contract_size = symbol_spec.contract_size

            if min_vol <= 0 or not math.isfinite(min_vol):
                raise DemoAccountVerificationError(
                    f"DEMO_VERIFICATION_FAILED: Invalid min_volume: {min_vol}"
                )

        return DemoVerificationReport(
            verified=True,
            account_login_masked=account_meta.login_masked,
            company=account_meta.company,
            server=account_meta.server,
            currency=account_meta.currency,
            leverage=account_meta.leverage,
            balance=account_meta.balance,
            equity=account_meta.equity,
            trade_mode_code=account_meta.trade_mode,
            trade_mode_str=mode_enum.value,
            is_demo=True,
            is_live_detected=False,
            terminal_connected=term_connected,
            symbol=sym_name,
            min_volume=min_vol,
            max_volume=max_vol,
            volume_step=vol_step,
            contract_size=contract_size,
            verification_notes="Account successfully and positively verified as MetaTrader 5 DEMO.",
        )


def assert_demo_execution_authorized(
    is_live: bool,
    execution_enabled: bool,
    demo_execution_enabled: bool,
    account_meta: Optional[MT5AccountMetadata],
) -> DemoVerificationReport:
    """Enforce explicit multi-condition demo execution authorization before any order submission.

    Rules:
    - is_live MUST be False
    - execution_enabled MUST be True
    - demo_execution_enabled MUST be True
    - DEMO_ONLY MUST be True
    - Account MUST be verified DEMO (trade_mode == 0)
    - If account is LIVE (trade_mode == 2), raise LiveAccountForbiddenError immediately.
    """
    # Invariant 1: No live execution
    if is_live:
        raise LiveAccountForbiddenError(
            "LIVE_ACCOUNT_FORBIDDEN: is_live is True. Live trading is strictly forbidden."
        )

    # Invariant 2: Execution enabled gate
    if not execution_enabled:
        raise BrokerExecutionDisabledError(
            "MT5_EXECUTION_DISABLED: BrokerAdapter.execution_enabled is False."
        )

    # Invariant 3: Explicit demo authorization gate
    if not demo_execution_enabled:
        raise DemoExecutionNotAuthorizedError(
            "DEMO_EXECUTION_DISABLED: Demo execution is not authorized "
            "(demo_execution_enabled is False)."
        )

    # Invariant 4: Global safety flag
    if not DEMO_ONLY:
        raise BrokerExecutionDisabledError(
            "SAFETY_INVARIANT_VIOLATION: DEMO_ONLY safety flag is False."
        )

    # Invariant 5: Positive Account Classification
    report = DemoAccountVerifier.verify(account_meta)
    return report
