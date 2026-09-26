"""Unit and integration tests for FastAPI Paper Trading endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from trading.api.app import create_app
from trading.api.dependencies import (
    get_trading_context,
    reset_trading_context,
)
from trading.models.order import OrderSide
from trading.models.position import Position


@pytest.fixture
def client():
    """Create fresh application and test client for each test."""
    reset_trading_context(initial_balance=100_000.0)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def make_valid_signal_payload(
    symbol: str = "BNBUSDT",
    action: str = "BUY",
    confidence: float = 0.85,
    entry: float = 100.0,
    stop_loss: float = 95.0,
) -> dict:
    return {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": "xgb_v1",
        "timeframe": "15m",
        "expected_return": 0.015,
        "feature_version": "features_v2",
        "suggested_entry_price": entry,
        "suggested_stop_loss": stop_loss,
    }


def test_get_health(client: TestClient):
    """Test 1: GET /health returns DEMO and PAPER environment status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["environment"] == "DEMO"
    assert data["trading_backend"] == "PAPER"


def test_get_account(client: TestClient):
    """Test 2: GET /account returns initial paper balance and equity."""
    response = client.get("/account")
    assert response.status_code == 200
    data = response.json()
    assert data["balance"] == 100_000.0
    assert data["equity"] == 100_000.0
    assert data["available_cash"] == 100_000.0
    assert data["realized_pnl"] == 0.0
    assert data["unrealized_pnl"] == 0.0


def test_get_positions_empty_initially(client: TestClient):
    """Test 3: GET /positions returns empty list when no positions are open."""
    response = client.get("/positions")
    assert response.status_code == 200
    assert response.json() == []


def test_post_valid_signal(client: TestClient):
    """Test 4: POST /signals with valid signal returns HTTP 200 and filled simulated order."""
    payload = make_valid_signal_payload()
    response = client.post("/signals", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["approved"] is True
    assert data["reason"] is None
    assert data["signal_symbol"] == "BNBUSDT"
    assert data["order"] is not None
    assert data["order"]["status"] == "FILLED"
    assert data["order"]["fill_price"] == 100.0

    # Verify position is reflected in GET /positions
    pos_res = client.get("/positions")
    assert pos_res.status_code == 200
    positions = pos_res.json()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "BNBUSDT"


def test_post_invalid_signal_schema(client: TestClient):
    """Test 5: POST /signals with malformed schema returns HTTP 422."""
    bad_payload = {
        "symbol": "",  # Empty symbol invalid
        "action": "INVALID_ACTION",
        "confidence": 1.5,  # > 1.0 invalid
    }
    response = client.post("/signals", json=bad_payload)
    assert response.status_code == 422


def test_post_low_confidence_signal(client: TestClient):
    """Test 6: POST /signals with confidence < 0.60 returns HTTP 409 rejection."""
    payload = make_valid_signal_payload(confidence=0.45)
    response = client.post("/signals", json=payload)
    assert response.status_code == 409
    data = response.json()
    assert data["approved"] is False
    assert data["reason"] == "CONFIDENCE_TOO_LOW"
    assert data["order"] is None


def test_post_hold_signal(client: TestClient):
    """Test 7: POST /signals with action HOLD returns HTTP 409 rejection."""
    payload = make_valid_signal_payload(action="HOLD")
    response = client.post("/signals", json=payload)
    assert response.status_code == 409
    data = response.json()
    assert data["approved"] is False
    assert data["reason"] == "SIGNAL_ACTION_HOLD"
    assert data["order"] is None


def test_post_excessive_exposure_signal(client: TestClient):
    """Test 8: POST /signals causing gross exposure > 20% returns HTTP 409."""
    # Extremely tight stop generates huge quantity: 50,000 units * $100 = $5,000,000
    payload = make_valid_signal_payload(entry=100.0, stop_loss=99.99)
    response = client.post("/signals", json=payload)
    assert response.status_code == 409
    data = response.json()
    assert data["approved"] is False
    assert data["reason"] == "EXCESSIVE_EXPOSURE"


def test_post_max_positions_limit(client: TestClient):
    """Test 9: POST /signals exceeding max open positions (3) returns HTTP 409."""
    # Open 3 positions
    ctx = get_trading_context()
    for sym in ("SYM1", "SYM2", "SYM3"):
        ctx.paper_broker._positions[sym] = Position(
            symbol=sym,
            side=OrderSide.BUY,
            quantity=10.0,
            entry_price=10.0,
            current_price=10.0,
        )

    # 4th position must be rejected
    payload = make_valid_signal_payload(symbol="SYM4")
    response = client.post("/signals", json=payload)
    assert response.status_code == 409
    data = response.json()
    assert data["approved"] is False
    assert data["reason"] == "MAX_OPEN_POSITIONS_REACHED"


def test_post_daily_loss_limit(client: TestClient):
    """Test 10: POST /signals when daily loss >= 1.0% returns HTTP 409."""
    ctx = get_trading_context()
    # Baseline 100,000; simulate realized loss of $1,200 (1.2% loss)
    ctx.paper_broker._realized_pnl = -1_200.0

    payload = make_valid_signal_payload()
    response = client.post("/signals", json=payload)
    assert response.status_code == 409
    data = response.json()
    assert data["approved"] is False
    assert data["reason"] == "DAILY_LOSS_LIMIT_EXCEEDED"
    assert ctx.kill_switch.is_active() is True


def test_post_signal_when_kill_switch_active(client: TestClient):
    """Test 11: POST /signals returns HTTP 409 when kill switch is engaged."""
    ctx = get_trading_context()
    ctx.kill_switch.activate("CIRCUIT_BREAKER_MANUAL")

    payload = make_valid_signal_payload()
    response = client.post("/signals", json=payload)
    assert response.status_code == 409
    data = response.json()
    assert data["approved"] is False
    assert data["reason"] == "KILL_SWITCH_ACTIVE"


def test_activate_kill_switch_endpoint(client: TestClient):
    """Test 12: POST /risk/kill-switch/activate activates kill switch with reason."""
    response = client.post(
        "/risk/kill-switch/activate",
        json={"reason": "MANUAL_TEST_HALT"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "activated"
    assert data["is_active"] is True
    assert data["reason"] == "MANUAL_TEST_HALT"

    # Verify status endpoint reflects activation
    status_res = client.get("/risk/status")
    assert status_res.status_code == 200
    assert status_res.json()["kill_switch_active"] is True
    assert status_res.json()["kill_switch_reason"] == "MANUAL_TEST_HALT"


def test_deactivate_kill_switch_endpoint(client: TestClient):
    """Test 13: POST /risk/kill-switch/deactivate restores system operations."""
    # Activate first
    client.post("/risk/kill-switch/activate", json={"reason": "PRE_TEST"})
    # Now deactivate
    response = client.post("/risk/kill-switch/deactivate")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deactivated"
    assert data["is_active"] is False

    status_res = client.get("/risk/status")
    assert status_res.json()["kill_switch_active"] is False


def test_get_orders_history(client: TestClient):
    """Test 14: GET /orders returns registered simulated orders."""
    # Submit 1 order
    client.post("/signals", json=make_valid_signal_payload(symbol="SOLUSDT"))

    response = client.get("/orders")
    assert response.status_code == 200
    orders = response.json()
    assert len(orders) >= 1
    assert orders[0]["symbol"] == "SOLUSDT"
    assert orders[0]["status"] == "FILLED"


def test_rejected_signal_never_reaches_paper_broker(client: TestClient):
    """Test 15: INVARIANT assertion: PaperBroker is NEVER invoked on rejected signals."""
    ctx = get_trading_context()
    spy_submit = MagicMock(wraps=ctx.paper_broker.submit_order)
    ctx.paper_broker.submit_order = spy_submit

    # 1. Low confidence rejection
    client.post("/signals", json=make_valid_signal_payload(confidence=0.30))
    spy_submit.assert_not_called()

    # 2. HOLD action rejection
    client.post("/signals", json=make_valid_signal_payload(action="HOLD"))
    spy_submit.assert_not_called()

    # 3. Kill switch active rejection
    ctx.kill_switch.activate("HALT")
    client.post("/signals", json=make_valid_signal_payload())
    spy_submit.assert_not_called()
