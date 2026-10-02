"""Deterministic schema migration runner for the Paper Trading database."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading.persistence.database import PaperDatabaseManager

logger = logging.getLogger(__name__)

MIGRATION_V1_SQL = """
-- 1. Orders table
CREATE TABLE IF NOT EXISTS paper_orders (
    order_id TEXT PRIMARY KEY,
    client_request_id TEXT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    requested_price REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    fill_price REAL,
    fill_timestamp TEXT,
    rejection_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON paper_orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_client_req ON paper_orders(client_request_id);

-- 2. Executions table
CREATE TABLE IF NOT EXISTS paper_executions (
    execution_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    client_request_id TEXT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    requested_price REAL NOT NULL,
    executed_price REAL,
    quantity REAL NOT NULL,
    spread REAL NOT NULL DEFAULT 0.0,
    slippage REAL NOT NULL DEFAULT 0.0,
    commission REAL NOT NULL DEFAULT 0.0,
    swap REAL NOT NULL DEFAULT 0.0,
    gross_pnl REAL NOT NULL DEFAULT 0.0,
    net_pnl REAL NOT NULL DEFAULT 0.0,
    execution_status TEXT NOT NULL,
    rejection_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_executions_order_id ON paper_executions(order_id);
CREATE INDEX IF NOT EXISTS idx_executions_timestamp ON paper_executions(timestamp);

-- 3. Positions table
CREATE TABLE IF NOT EXISTS paper_positions (
    position_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    entry_timestamp TEXT NOT NULL,
    exit_timestamp TEXT,
    gross_pnl REAL NOT NULL DEFAULT 0.0,
    costs REAL NOT NULL DEFAULT 0.0,
    net_pnl REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON paper_positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON paper_positions(symbol);

-- 4. Audit events table
CREATE TABLE IF NOT EXISTS paper_audit_events (
    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    order_id TEXT,
    position_id TEXT,
    symbol TEXT,
    details_json TEXT NOT NULL,
    prev_hash TEXT,
    event_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON paper_audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON paper_audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_events_seq ON paper_audit_events(sequence_id);

-- 5. Account snapshots table
CREATE TABLE IF NOT EXISTS paper_account_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    initial_balance REAL NOT NULL,
    cash_balance REAL NOT NULL,
    equity REAL NOT NULL,
    margin_used REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    total_costs REAL NOT NULL,
    daily_pnl REAL NOT NULL,
    current_exposure REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_account_snapshots_ts ON paper_account_snapshots(timestamp);

-- 6. Risk state table (singleton row)
CREATE TABLE IF NOT EXISTS paper_risk_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    date_utc TEXT NOT NULL,
    daily_loss REAL NOT NULL DEFAULT 0.0,
    baseline_equity REAL NOT NULL,
    kill_switch_active INTEGER NOT NULL DEFAULT 0,
    kill_switch_reason TEXT,
    kill_switch_timestamp TEXT,
    updated_at TEXT NOT NULL
);

-- 7. Idempotency records table
CREATE TABLE IF NOT EXISTS paper_idempotency_records (
    client_request_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    status TEXT NOT NULL,
    execution_report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class PaperSchemaMigrator:
    """Manages versioned, deterministic schema migrations for Paper Trading."""

    MIGRATIONS = [
        (1, "001_initial_paper_trading_schema", MIGRATION_V1_SQL),
    ]

    def __init__(self, db: Optional[PaperDatabaseManager] = None) -> None:
        self.db = db

    def apply_all(self) -> int:
        if self.db is None:
            raise ValueError("No PaperDatabaseManager configured for migrator.")
        return self.apply_migrations(self.db)

    def get_current_version(self) -> int:
        if self.db is None:
            return 0
        applied = self.get_applied_migrations(self.db)
        return max([m["version"] for m in applied], default=0)

    @classmethod
    def apply_migrations(cls, db: PaperDatabaseManager) -> int:
        """Apply all pending migrations inside an atomic transaction."""
        applied_count = 0
        with db.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at_utc TEXT NOT NULL
                );
                """
            )
            rows = conn.execute("SELECT version FROM paper_schema_migrations;").fetchall()
            applied_versions = {row[0] for row in rows}

            for version, name, sql in cls.MIGRATIONS:
                if version not in applied_versions:
                    logger.info("Applying paper schema migration %d: %s", version, name)
                    conn.executescript(sql)
                    now_utc = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """
                        INSERT INTO paper_schema_migrations (version, name, applied_at_utc)
                        VALUES (?, ?, ?);
                        """,
                        (version, name, now_utc),
                    )
                    applied_count += 1
        return applied_count

    @classmethod
    def get_applied_migrations(cls, db: PaperDatabaseManager) -> List[Dict[str, Any]]:
        """Return list of applied migrations."""
        with db.connection() as conn:
            try:
                sql = (
                    "SELECT version, name, applied_at_utc "
                    "FROM paper_schema_migrations ORDER BY version;"
                )
                rows = conn.execute(sql).fetchall()
                return [
                    {"version": row[0], "name": row[1], "applied_at_utc": row[2]} for row in rows
                ]
            except Exception:
                return []
