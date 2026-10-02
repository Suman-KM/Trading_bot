"""Dependency injection container and providers for the FastAPI application."""

from typing import Optional

from trading.audit.trail import AuditTrail
from trading.execution.paper_broker import PaperBroker
from trading.execution.service import TradingExecutionService
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskEngine
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits


class TradingContext:
    """Encapsulates the runtime state of the trading safety core."""

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        limits: Optional[RiskLimits] = None,
        kill_switch: Optional[KillSwitch] = None,
        audit_trail: Optional[AuditTrail] = None,
    ) -> None:
        self.audit_trail: AuditTrail = audit_trail or AuditTrail()
        self.limits: RiskLimits = limits or RiskLimits()
        self.kill_switch: KillSwitch = kill_switch or KillSwitch(audit_trail=self.audit_trail)
        self.kill_switch.set_audit_trail(self.audit_trail)
        self.risk_engine: RiskEngine = RiskEngine(limits=self.limits, kill_switch=self.kill_switch)
        self.paper_broker: PaperBroker = PaperBroker(
            initial_balance=initial_balance,
            audit_trail=self.audit_trail,
        )
        self.portfolio_manager: PortfolioManager = PortfolioManager(broker=self.paper_broker)
        self.execution_service: TradingExecutionService = TradingExecutionService(
            risk_engine=self.risk_engine,
            paper_broker=self.paper_broker,
            portfolio_manager=self.portfolio_manager,
            audit_trail=self.audit_trail,
        )


# Global default container instance
_default_context: Optional[TradingContext] = None


def get_trading_context() -> TradingContext:
    """Retrieve the application trading context."""
    global _default_context
    if _default_context is None:
        _default_context = TradingContext()
    return _default_context


def set_trading_context(context: TradingContext) -> None:
    """Set the application trading context (useful for testing)."""
    global _default_context
    _default_context = context


def reset_trading_context(initial_balance: float = 100_000.0) -> TradingContext:
    """Reset the default trading context to a fresh state."""
    global _default_context
    _default_context = TradingContext(initial_balance=initial_balance)
    return _default_context


def get_execution_service() -> TradingExecutionService:
    """Dependency provider for TradingExecutionService."""
    return get_trading_context().execution_service


def get_portfolio_manager() -> PortfolioManager:
    """Dependency provider for PortfolioManager."""
    return get_trading_context().portfolio_manager


def get_paper_broker() -> PaperBroker:
    """Dependency provider for PaperBroker."""
    return get_trading_context().paper_broker


def get_risk_engine() -> RiskEngine:
    """Dependency provider for RiskEngine."""
    return get_trading_context().risk_engine


def get_kill_switch() -> KillSwitch:
    """Dependency provider for KillSwitch."""
    return get_trading_context().kill_switch


def get_audit_trail() -> AuditTrail:
    """Dependency provider for AuditTrail."""
    return get_trading_context().audit_trail
