"""Pydantic schemas for the FastAPI endpoints."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response declaring DEMO and PAPER mode."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    environment: str = "DEMO"
    trading_backend: str = "PAPER"


class AccountResponse(BaseModel):
    """Simulated paper account snapshot."""

    model_config = ConfigDict(extra="forbid")

    balance: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    available_cash: float


class PositionResponse(BaseModel):
    """Simulated active position model."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class RiskStatusResponse(BaseModel):
    """Non-secret risk engine and kill switch status."""

    model_config = ConfigDict(extra="forbid")

    kill_switch_active: bool
    kill_switch_reason: Optional[str] = None
    daily_loss_percent: float
    open_positions: int
    gross_exposure_percent: float
    configured_limits: Dict[str, Any]
    trading_mode: str = "DEMO/PAPER"


class KillSwitchActivateRequest(BaseModel):
    """Request payload to activate the emergency kill switch."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="MANUAL_HALT", min_length=1)


class KillSwitchResponse(BaseModel):
    """Response returned upon kill switch state change."""

    model_config = ConfigDict(extra="forbid")

    status: str
    is_active: bool
    reason: Optional[str] = None


class SignalResponse(BaseModel):
    """Response payload returned when a signal is submitted."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    reason: Optional[str] = None
    order: Optional[Dict[str, Any]] = None
    signal_symbol: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    """Readiness status declaring whether the paper trading engine is internally ready."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    paper_broker_initialized: bool
    risk_engine_available: bool
    kill_switch_active: bool
    portfolio_state_valid: bool
    unrecovered_execution_errors: int
    database_connected: bool = True
    persistence_healthy: bool = True
    database_integrity_valid: bool = True
    audit_integrity_valid: bool = True
    quarantine_enforced: bool = True
    execution_service_ready: bool = True
    mt5_adapter_available: bool = False
    mt5_execution_enabled: bool = False
    mt5_readonly_connected: bool = False
    mt5_market_data_available: bool = False
    mt5_demo_account_verified: bool = False
    mt5_live_account_detected: bool = False
    mt5_trade_permissions_verified: bool = False
    mt5_symbol_verified: bool = False
    reconciliation_healthy: bool = True
    recovery_error: Optional[str] = None
    environment: str = "DEMO"
    trading_backend: str = "PAPER"


class ExecutionReportResponse(BaseModel):
    """Execution report response for auditability and TCA."""

    model_config = ConfigDict(extra="forbid")

    execution_id: str
    order_id: str
    client_request_id: Optional[str] = None
    timestamp: str
    symbol: str
    side: str
    requested_price: float
    executed_price: Optional[float] = None
    quantity: float
    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    execution_status: str
    rejection_reason: Optional[str] = None


class AuditEventResponse(BaseModel):
    """Structured audit trail event response."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    timestamp: str
    event_type: str
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    source: str
    details: Dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Operational engineering metrics."""

    model_config = ConfigDict(extra="forbid")

    orders_received: int
    orders_accepted: int
    orders_rejected: int
    orders_filled: int
    orders_cancelled: int
    execution_errors: int
    open_positions: int
    closed_positions: int
    gross_pnl: float
    total_costs: float
    net_pnl: float
    daily_loss_percent: float
    current_exposure_percent: float
    database_connected: bool = True
    persisted_orders_count: int = 0
    persisted_positions_count: int = 0
    persisted_executions_count: int = 0
    persisted_audit_events_count: int = 0
    last_persistence_timestamp: Optional[str] = None
