"""API route handlers implementing the Paper Trading endpoints."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from trading.api.dependencies import (
    TradingContext,
    get_audit_trail,
    get_execution_service,
    get_kill_switch,
    get_paper_broker,
    get_portfolio_manager,
    get_risk_engine,
    get_trading_context,
)
from trading.api.schemas import (
    AccountResponse,
    AuditEventResponse,
    ExecutionReportResponse,
    HealthResponse,
    KillSwitchActivateRequest,
    KillSwitchResponse,
    MetricsResponse,
    PositionResponse,
    ReadinessResponse,
    RiskStatusResponse,
    SignalResponse,
)
from trading.audit.trail import AuditTrail
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


@router.get("/readiness", response_model=ReadinessResponse, summary="Engine Readiness Verification")
def get_readiness(
    broker: PaperBroker = Depends(get_paper_broker),
    risk_engine: RiskEngine = Depends(get_risk_engine),
    kill_switch: KillSwitch = Depends(get_kill_switch),
    portfolio_mgr: PortfolioManager = Depends(get_portfolio_manager),
    context: TradingContext = Depends(get_trading_context),
) -> ReadinessResponse:
    """Verify internal operational readiness of the paper-trading platform."""
    account = broker.get_account()
    db_conn = context.database_manager.is_connected() if context.database_manager else False
    pers_healthy = context.persistence_healthy
    daily_loss_pct = portfolio_mgr.get_daily_loss_percent()
    daily_loss_ok = daily_loss_pct < risk_engine.limits.MAX_DAILY_LOSS_PERCENT
    portfolio_valid = account.equity > 0

    is_ready = (
        broker is not None
        and risk_engine is not None
        and not kill_switch.is_active()
        and portfolio_valid
        and db_conn
        and pers_healthy
        and daily_loss_ok
    )
    resp = ReadinessResponse(
        ready=is_ready,
        paper_broker_initialized=True,
        risk_engine_available=True,
        kill_switch_active=kill_switch.is_active(),
        portfolio_state_valid=portfolio_valid,
        unrecovered_execution_errors=0,
        database_connected=db_conn,
        persistence_healthy=pers_healthy,
        recovery_error=context.recovery_error,
        environment="DEMO",
        trading_backend="PAPER",
    )
    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=resp.model_dump(),
        )
    return resp


@router.get(
    "/executions",
    response_model=List[ExecutionReportResponse],
    summary="Auditable Execution Reports History",
)
def get_executions(
    broker: PaperBroker = Depends(get_paper_broker),
) -> List[ExecutionReportResponse]:
    """Return all auditable paper trade execution reports with explicit cost breakdown."""
    reports = broker.get_execution_reports()
    return [
        ExecutionReportResponse(
            execution_id=r.execution_id,
            order_id=r.order_id,
            client_request_id=r.client_request_id,
            timestamp=r.timestamp.isoformat(),
            symbol=r.symbol,
            side=r.side.value,
            requested_price=r.requested_price,
            executed_price=r.executed_price,
            quantity=r.quantity,
            spread=r.spread,
            slippage=r.slippage,
            commission=r.commission,
            gross_pnl=r.gross_pnl,
            net_pnl=r.net_pnl,
            execution_status=r.execution_status.value,
            rejection_reason=r.rejection_reason,
        )
        for r in reports
    ]


@router.get(
    "/audit/events",
    response_model=List[AuditEventResponse],
    summary="Immutable Structured Audit Trail",
)
def get_audit_events(
    audit_trail: AuditTrail = Depends(get_audit_trail),
    limit: int = 100,
) -> List[AuditEventResponse]:
    """Return immutable audit trail events."""
    events = audit_trail.get_events(limit=limit)
    return [
        AuditEventResponse(
            event_id=e.event_id,
            timestamp=e.timestamp.isoformat(),
            event_type=e.event_type.value,
            symbol=e.symbol,
            order_id=e.order_id,
            position_id=e.position_id,
            source=e.source,
            details=e.details,
        )
        for e in events
    ]


@router.get("/metrics", response_model=MetricsResponse, summary="Engineering Operational Metrics")
def get_metrics(
    broker: PaperBroker = Depends(get_paper_broker),
    portfolio_mgr: PortfolioManager = Depends(get_portfolio_manager),
    context: TradingContext = Depends(get_trading_context),
) -> MetricsResponse:
    """Return operational engineering metrics (orders received, filled, costs, exposure)."""
    metrics = broker.get_metrics()
    repo_stats = context.repository.get_persistence_stats() if context.repository else {}
    db_conn = context.database_manager.is_connected() if context.database_manager else False
    return MetricsResponse(
        orders_received=metrics["orders_received"],
        orders_accepted=metrics["orders_accepted"],
        orders_rejected=metrics["orders_rejected"],
        orders_filled=metrics["orders_filled"],
        orders_cancelled=metrics["orders_cancelled"],
        execution_errors=metrics["execution_errors"],
        open_positions=metrics["open_positions"],
        closed_positions=metrics["closed_positions"],
        gross_pnl=metrics["gross_pnl"],
        total_costs=metrics["total_costs"],
        net_pnl=metrics["net_pnl"],
        daily_loss_percent=portfolio_mgr.get_daily_loss_percent(),
        current_exposure_percent=portfolio_mgr.get_total_exposure_percent(),
        database_connected=db_conn,
        persisted_orders_count=repo_stats.get("orders_count", 0),
        persisted_positions_count=repo_stats.get("positions_count", 0),
        persisted_executions_count=repo_stats.get("executions_count", 0),
        persisted_audit_events_count=repo_stats.get("audit_events_count", 0),
        last_persistence_timestamp=repo_stats.get("last_persistence_timestamp"),
    )
