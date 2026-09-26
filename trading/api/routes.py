"""API route handlers implementing the Paper Trading endpoints."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from trading.api.dependencies import (
    get_execution_service,
    get_kill_switch,
    get_paper_broker,
    get_portfolio_manager,
    get_risk_engine,
)
from trading.api.schemas import (
    AccountResponse,
    HealthResponse,
    KillSwitchActivateRequest,
    KillSwitchResponse,
    PositionResponse,
    RiskStatusResponse,
    SignalResponse,
)
from trading.execution.paper_broker import PaperBroker
from trading.execution.service import TradingExecutionService
from trading.models.signal import Signal
from trading.portfolio.manager import PortfolioManager
from trading.risk.engine import RiskEngine
from trading.risk.kill_switch import KillSwitch

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="System Health Status")
def health_check() -> HealthResponse:
    """Return system operational status declaring DEMO and PAPER trading backend."""
    return HealthResponse(
        status="ok",
        environment="DEMO",
        trading_backend="PAPER",
    )


@router.get("/account", response_model=AccountResponse, summary="Simulated Account Info")
def get_account(
    portfolio_mgr: PortfolioManager = Depends(get_portfolio_manager),
) -> AccountResponse:
    """Return simulated paper account equity, cash balance, and P&L."""
    account = portfolio_mgr.get_account_info()
    return AccountResponse(
        balance=account.initial_balance,
        equity=account.equity,
        realized_pnl=account.realized_pnl,
        unrealized_pnl=account.unrealized_pnl,
        available_cash=account.cash_balance,
    )


@router.get("/positions", response_model=List[PositionResponse], summary="Active Open Positions")
def get_positions(
    broker: PaperBroker = Depends(get_paper_broker),
) -> List[PositionResponse]:
    """Return all active simulated positions."""
    positions = broker.get_positions()
    return [
        PositionResponse(
            symbol=pos.symbol,
            side=pos.side.value,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            unrealized_pnl=pos.unrealized_pnl,
            realized_pnl=pos.realized_pnl,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
        )
        for pos in positions.values()
    ]


@router.post(
    "/signals",
    response_model=SignalResponse,
    responses={
        200: {"description": "Signal approved and simulated order filled"},
        409: {"description": "Signal rejected by Risk Engine or Kill Switch"},
        422: {"description": "Signal schema validation error"},
    },
    summary="Submit AI Trading Signal",
)
def submit_signal(
    signal: Signal,
    service: TradingExecutionService = Depends(get_execution_service),
) -> Any:
    """Process an AI signal through the non-bypassable Risk Engine pipeline."""
    result = service.process_signal(signal)

    if not result.approved:
        # Return HTTP 409 Conflict with structured rejection reason
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=result.model_dump(),
        )

    return SignalResponse(
        approved=result.approved,
        reason=result.reason,
        order=result.order.model_dump() if result.order else None,
        signal_symbol=result.signal_symbol,
        metadata=result.metadata,
    )


@router.get("/orders", response_model=List[Dict[str, Any]], summary="Simulated Orders History")
def get_orders(
    broker: PaperBroker = Depends(get_paper_broker),
) -> List[Dict[str, Any]]:
    """Return list of all registered simulated orders."""
    orders = broker.get_all_orders()
    return [o.model_dump() for o in orders]


@router.get("/risk/status", response_model=RiskStatusResponse, summary="Risk Engine & Safety State")
def get_risk_status(
    risk_engine: RiskEngine = Depends(get_risk_engine),
    kill_switch: KillSwitch = Depends(get_kill_switch),
    portfolio_mgr: PortfolioManager = Depends(get_portfolio_manager),
) -> RiskStatusResponse:
    """Return current risk thresholds, exposure, daily loss, and kill switch state."""
    return RiskStatusResponse(
        kill_switch_active=kill_switch.is_active(),
        kill_switch_reason=kill_switch.reason,
        daily_loss_percent=portfolio_mgr.get_daily_loss_percent(),
        open_positions=portfolio_mgr.get_open_position_count(),
        gross_exposure_percent=portfolio_mgr.get_total_exposure_percent(),
        configured_limits=risk_engine.limits.model_dump(),
        trading_mode="DEMO/PAPER",
    )


@router.post(
    "/risk/kill-switch/activate",
    response_model=KillSwitchResponse,
    summary="Emergency Kill Switch Activation",
)
def activate_kill_switch(
    payload: KillSwitchActivateRequest,
    kill_switch: KillSwitch = Depends(get_kill_switch),
) -> KillSwitchResponse:
    """Activate the kill switch, rejecting all subsequent orders."""
    kill_switch.activate(reason=payload.reason)
    return KillSwitchResponse(
        status="activated",
        is_active=True,
        reason=kill_switch.reason,
    )


@router.post(
    "/risk/kill-switch/deactivate",
    response_model=KillSwitchResponse,
    summary="Deactivate Emergency Kill Switch",
)
def deactivate_kill_switch(
    kill_switch: KillSwitch = Depends(get_kill_switch),
) -> KillSwitchResponse:
    """Deactivate the emergency kill switch and restore normal operations."""
    kill_switch.deactivate()
    return KillSwitchResponse(
        status="deactivated",
        is_active=False,
        reason=None,
    )
