"""Unit tests verifying the end-to-end safe execution flow and non-bypassable boundary."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from trading.execution.paper_broker import PaperBroker
from trading.execution.service import TradingExecutionService
from trading.models.order import OrderStatus
from trading.models.signal import Signal, SignalAction
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskEngine, RiskReason
from trading.risk.kill_switch import KillSwitch
from trading.risk.limits import RiskLimits


def make_valid_signal() -> Signal:
    """Create valid signal with 10% exposure (well within 20% limit)."""
    return Signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.90,
        timestamp=datetime.now(timezone.utc),
        model_version="ml_v3",
        timeframe="15m",
        expected_return=0.02,
        feature_version="feat_v3",
        suggested_entry_price=100.0,
        suggested_stop_loss=95.0,  # 5.0 distance -> 100 qty -> $10,000 exposure = 10%
    )


def test_approved_signal_executes_on_paper_broker():
    """Approved signal flows through entire pipeline and is filled by PaperBroker."""
    broker = PaperBroker(initial_balance=100_000.0)
    risk_engine = RiskEngine()
    portfolio_mgr = PortfolioManager(broker=broker)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
    )

    signal = make_valid_signal()
    result = service.process_signal(signal)

    assert result.approved is True
    assert result.reason is None
    assert result.order is not None
    assert result.order.status == OrderStatus.FILLED
    assert result.order.symbol == "EURUSD"
    # Position must be created in PaperBroker
    assert "EURUSD" in broker.get_positions()


def test_broker_never_called_when_risk_rejects_confidence():
    """INVARIANT: PaperBroker.submit_order is NEVER invoked when confidence is low."""
    broker = PaperBroker(initial_balance=100_000.0)
    broker.submit_order = MagicMock(wraps=broker.submit_order)

    risk_engine = RiskEngine(limits=RiskLimits(MIN_SIGNAL_CONFIDENCE=0.60))
    portfolio_mgr = PortfolioManager(broker=broker)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
    )

    # Low confidence signal (0.45 < 0.60)
    signal = Signal(
        symbol="EURUSD",
        action=SignalAction.BUY,
        confidence=0.45,
        timestamp=datetime.now(timezone.utc),
        model_version="ml_v3",
        timeframe="15m",
        expected_return=0.02,
        feature_version="feat_v3",
        suggested_entry_price=100.0,
        suggested_stop_loss=95.0,
    )

    result = service.process_signal(signal)

    assert result.approved is False
    assert result.reason == RiskReason.CONFIDENCE_TOO_LOW.value
    assert result.order is None
    # Crucial assertion: PaperBroker was NEVER touched
    broker.submit_order.assert_not_called()
    assert len(broker.get_positions()) == 0


def test_broker_never_called_when_kill_switch_active():
    """INVARIANT: PaperBroker.submit_order is NEVER invoked when kill switch is engaged."""
    broker = PaperBroker(initial_balance=100_000.0)
    broker.submit_order = MagicMock(wraps=broker.submit_order)

    kill_switch = KillSwitch()
    kill_switch.activate("CIRCUIT_BREAKER_TRIPPED")

    risk_engine = RiskEngine(kill_switch=kill_switch)
    portfolio_mgr = PortfolioManager(broker=broker)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
    )

    signal = make_valid_signal()
    result = service.process_signal(signal)

    assert result.approved is False
    assert result.reason == RiskReason.KILL_SWITCH_ACTIVE.value
    assert result.order is None
    # Crucial assertion: PaperBroker was NEVER touched
    broker.submit_order.assert_not_called()
    assert len(broker.get_positions()) == 0


def test_broker_never_called_on_invalid_signal_schema():
    """INVARIANT: Malformed input is rejected at perimeter before RiskEngine or PaperBroker."""
    broker = PaperBroker(initial_balance=100_000.0)
    broker.submit_order = MagicMock(wraps=broker.submit_order)

    risk_engine = RiskEngine()
    portfolio_mgr = PortfolioManager(broker=broker)
    service = TradingExecutionService(
        risk_engine=risk_engine,
        paper_broker=broker,
        portfolio_manager=portfolio_mgr,
    )

    bad_signal_payload = {"symbol": "EURUSD", "action": "INVALID_ACTION"}
    result = service.process_signal(bad_signal_payload)

    assert result.approved is False
    assert result.reason == RiskReason.INVALID_SIGNAL.value
    assert result.order is None
    broker.submit_order.assert_not_called()
