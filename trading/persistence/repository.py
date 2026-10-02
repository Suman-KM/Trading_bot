"""Repository providing CRUD and query operations for persistent paper trading state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from trading.audit.trail import AuditEvent, AuditEventType
from trading.models.execution import ExecutionReport
from trading.models.order import Order, OrderSide, OrderStatus, OrderType
from trading.models.portfolio import AccountInfo
from trading.models.position import Position, PositionStatus
from trading.persistence.database import PaperDatabaseManager


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _str_to_dt(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    return datetime.fromisoformat(val)


def _enum_val(v: Any) -> Optional[str]:
    if v is None:
        return None
    if hasattr(v, "value"):
        return str(v.value)
    return str(v)


class PaperTradingRepository:
    """Encapsulates all SQL queries and data mapping for paper trading state."""

    def __init__(self, db: PaperDatabaseManager) -> None:
        self.db = db

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def save_order(self, order: Order, conn: Optional[sqlite3.Connection] = None) -> None:
        """Insert or replace an order record."""
        sql = """
        INSERT OR REPLACE INTO paper_orders (
            order_id, client_request_id, symbol, direction, order_type,
            quantity, requested_price, stop_loss, take_profit, status,
            created_at, updated_at, fill_price, fill_timestamp, rejection_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        req_price = (
            order.requested_price
            if order.requested_price is not None
            else (order.price if order.price is not None else 0.0)
        )
        c_at = order.created_at or order.timestamp
        u_at = order.updated_at or order.timestamp
        params = (
            order.order_id,
            order.client_request_id,
            order.symbol,
            _enum_val(order.side),
            _enum_val(order.order_type),
            order.quantity,
            req_price,
            order.stop_loss,
            order.take_profit,
            _enum_val(order.status),
            _dt_to_str(c_at),
            _dt_to_str(u_at),
            order.fill_price,
            _dt_to_str(order.fill_timestamp),
            order.rejection_reason,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self.db.transaction() as c:
                c.execute(sql, params)

    def get_order(
        self, order_id: str, conn: Optional[sqlite3.Connection] = None
    ) -> Optional[Order]:
        """Fetch an order by order_id."""
        sql = "SELECT * FROM paper_orders WHERE order_id = ?;"
        if conn is not None:
            row = conn.execute(sql, (order_id,)).fetchone()
        else:
            with self.db.connection() as c:
                row = c.execute(sql, (order_id,)).fetchone()
        if not row:
            return None
        return self._row_to_order(row)

    def get_all_orders(self, conn: Optional[sqlite3.Connection] = None) -> List[Order]:
        """Fetch all orders ordered by creation time."""
        sql = "SELECT * FROM paper_orders ORDER BY created_at ASC;"
        if conn is not None:
            rows = conn.execute(sql).fetchall()
        else:
            with self.db.connection() as c:
                rows = c.execute(sql).fetchall()
        return [self._row_to_order(r) for r in rows]

    def _row_to_order(self, row: sqlite3.Row) -> Order:
        c_at = _str_to_dt(row["created_at"]) or datetime.now(timezone.utc)
        u_at = _str_to_dt(row["updated_at"]) or datetime.now(timezone.utc)
        return Order(
            order_id=row["order_id"],
            client_request_id=row["client_request_id"],
            symbol=row["symbol"],
            side=OrderSide(row["direction"]),
            order_type=OrderType(row["order_type"]),
            quantity=row["quantity"],
            price=row["requested_price"],
            requested_price=row["requested_price"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            status=OrderStatus(row["status"]),
            timestamp=c_at,
            created_at=c_at,
            updated_at=u_at,
            fill_price=row["fill_price"],
            fill_timestamp=_str_to_dt(row["fill_timestamp"]),
            rejection_reason=row["rejection_reason"],
        )

    # -------------------------------------------------------------------------
    # Executions
    # -------------------------------------------------------------------------

    def save_execution(
        self, report: ExecutionReport, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        """Insert an execution report."""
        sql = """
        INSERT OR REPLACE INTO paper_executions (
            execution_id, order_id, client_request_id, timestamp, symbol,
            side, requested_price, executed_price, quantity, spread,
            slippage, commission, swap, gross_pnl, net_pnl, execution_status,
            rejection_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            report.execution_id,
            report.order_id,
            report.client_request_id,
            _dt_to_str(report.timestamp),
            report.symbol,
            _enum_val(report.side),
            report.requested_price,
            report.executed_price,
            report.quantity,
            report.spread,
            report.slippage,
            report.commission,
            report.swap,
            report.gross_pnl,
            report.net_pnl,
            _enum_val(report.execution_status),
            report.rejection_reason,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self.db.transaction() as c:
                c.execute(sql, params)

    def get_all_executions(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> List[ExecutionReport]:
        """Fetch all execution reports ordered chronologically."""
        sql = "SELECT * FROM paper_executions ORDER BY timestamp ASC;"
        if conn is not None:
            rows = conn.execute(sql).fetchall()
        else:
            with self.db.connection() as c:
                rows = c.execute(sql).fetchall()
        return [self._row_to_execution(r) for r in rows]

    def _row_to_execution(self, row: sqlite3.Row) -> ExecutionReport:
        return ExecutionReport(
            execution_id=row["execution_id"],
            order_id=row["order_id"],
            client_request_id=row["client_request_id"],
            timestamp=_str_to_dt(row["timestamp"]) or datetime.now(timezone.utc),
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            requested_price=row["requested_price"],
            executed_price=row["executed_price"],
            quantity=row["quantity"],
            spread=row["spread"],
            slippage=row["slippage"],
            commission=row["commission"],
            swap=row["swap"],
            gross_pnl=row["gross_pnl"],
            net_pnl=row["net_pnl"],
            execution_status=OrderStatus(row["execution_status"]),
            rejection_reason=row["rejection_reason"],
        )

    # -------------------------------------------------------------------------
    # Positions
    # -------------------------------------------------------------------------

    def save_position(self, position: Position, conn: Optional[sqlite3.Connection] = None) -> None:
        """Insert or update a position."""
        sql = """
        INSERT OR REPLACE INTO paper_positions (
            position_id, symbol, direction, quantity, entry_price, current_price,
            exit_price, stop_loss, take_profit, entry_timestamp, exit_timestamp,
            gross_pnl, costs, net_pnl, unrealized_pnl, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            position.position_id,
            position.symbol,
            _enum_val(position.side),
            position.quantity,
            position.entry_price,
            position.current_price,
            position.exit_price,
            position.stop_loss,
            position.take_profit,
            _dt_to_str(position.entry_timestamp),
            _dt_to_str(position.exit_timestamp),
            position.gross_pnl,
            position.costs,
            position.net_pnl,
            position.unrealized_pnl,
            _enum_val(position.status),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self.db.transaction() as c:
                c.execute(sql, params)

    def get_active_positions(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> Dict[str, Position]:
        """Fetch all active (OPEN or ACTIVE) positions mapped by symbol."""
        sql = "SELECT * FROM paper_positions WHERE status IN ('OPEN', 'ACTIVE');"
        if conn is not None:
            rows = conn.execute(sql).fetchall()
        else:
            with self.db.connection() as c:
                rows = c.execute(sql).fetchall()
        positions: Dict[str, Position] = {}
        for r in rows:
            pos = self._row_to_position(r)
            positions[pos.symbol] = pos
        return positions

    def get_closed_positions(self, conn: Optional[sqlite3.Connection] = None) -> List[Position]:
        """Fetch all historical closed positions."""
        sql = "SELECT * FROM paper_positions WHERE status = 'CLOSED' ORDER BY exit_timestamp ASC;"
        if conn is not None:
            rows = conn.execute(sql).fetchall()
        else:
            with self.db.connection() as c:
                rows = c.execute(sql).fetchall()
        return [self._row_to_position(r) for r in rows]

    def _row_to_position(self, row: sqlite3.Row) -> Position:
        pos = Position(
            position_id=row["position_id"],
            symbol=row["symbol"],
            side=OrderSide(row["direction"]),
            quantity=row["quantity"],
            entry_price=row["entry_price"],
            current_price=row["current_price"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            entry_timestamp=_str_to_dt(row["entry_timestamp"]),
            exit_price=row["exit_price"],
            exit_timestamp=_str_to_dt(row["exit_timestamp"]),
            gross_pnl=row["gross_pnl"],
            costs=row["costs"],
            net_pnl=row["net_pnl"],
            status=PositionStatus(row["status"]),
        )
        pos.unrealized_pnl = row["unrealized_pnl"]
        return pos

    # -------------------------------------------------------------------------
    # Audit Events
    # -------------------------------------------------------------------------

    def save_audit_event(
        self, event: AuditEvent, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        """Append an immutable audit event to persistent storage with hash chaining."""

        def _execute(c: sqlite3.Connection) -> None:
            last_row = c.execute(
                "SELECT event_hash FROM paper_audit_events ORDER BY sequence_id DESC LIMIT 1;"
            ).fetchone()
            prev_hash = last_row["event_hash"] if last_row and last_row["event_hash"] else "0" * 64

            details_str = json.dumps(event.details, sort_keys=True)
            ts_str = _dt_to_str(event.timestamp) or datetime.now(timezone.utc).isoformat()
            etype_str = _enum_val(event.event_type) or ""

            hash_payload = f"{prev_hash}:{event.event_id}:{ts_str}:{etype_str}:{details_str}"
            event_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()

            sql = """
            INSERT INTO paper_audit_events (
                event_id, timestamp, event_type, source, order_id, position_id,
                symbol, details_json, prev_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            params = (
                event.event_id,
                ts_str,
                etype_str,
                event.source,
                event.order_id,
                event.position_id,
                event.symbol,
                details_str,
                prev_hash,
                event_hash,
            )
            c.execute(sql, params)

        if conn is not None:
            _execute(conn)
        else:
            with self.db.transaction() as c:
                _execute(c)

    def get_audit_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[AuditEvent]:
        """Fetch audit events, optionally filtered by event_type."""
        if event_type:
            sql = """
            SELECT * FROM paper_audit_events
            WHERE event_type = ?
            ORDER BY sequence_id ASC
            LIMIT ?;
            """
            params: tuple[Any, ...] = (event_type, limit)
        else:
            sql = "SELECT * FROM paper_audit_events ORDER BY sequence_id ASC LIMIT ?;"
            params = (limit,)

        if conn is not None:
            rows = conn.execute(sql, params).fetchall()
        else:
            with self.db.connection() as c:
                rows = c.execute(sql, params).fetchall()

        events: List[AuditEvent] = []
        for r in rows:
            details = json.loads(r["details_json"]) if r["details_json"] else {}
            events.append(
                AuditEvent(
                    event_id=r["event_id"],
                    sequence_id=r["sequence_id"] if "sequence_id" in r.keys() else None,
                    timestamp=_str_to_dt(r["timestamp"]) or datetime.now(timezone.utc),
                    event_type=AuditEventType(r["event_type"]),
                    source=r["source"],
                    order_id=r["order_id"],
                    position_id=r["position_id"],
                    symbol=r["symbol"],
                    details=details,
                    prev_hash=r["prev_hash"] if "prev_hash" in r.keys() else None,
                    event_hash=r["event_hash"] if "event_hash" in r.keys() else None,
                )
            )
        return events

    def verify_audit_trail_integrity(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> Tuple[bool, Optional[str]]:
        """Verify sequence ordering and cryptographic hash chaining of all audit events."""
        sql = (
            "SELECT sequence_id, event_id, timestamp, event_type, details_json, "
            "prev_hash, event_hash FROM paper_audit_events ORDER BY sequence_id ASC;"
        )
        if conn is not None:
            rows = conn.execute(sql).fetchall()
        else:
            with self.db.connection() as c:
                rows = c.execute(sql).fetchall()

        expected_prev = "0" * 64
        for r in rows:
            seq = r["sequence_id"]
            if r["prev_hash"] != expected_prev:
                return False, f"Tamper detected at sequence {seq}: prev_hash mismatch."
            payload = (
                f"{r['prev_hash']}:{r['event_id']}:{r['timestamp']}:"
                f"{r['event_type']}:{r['details_json']}"
            )
            computed_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if computed_hash != r["event_hash"]:
                return False, f"Tamper detected at sequence {seq}: payload hash mismatch."
            expected_prev = r["event_hash"]

        return True, None

    # -------------------------------------------------------------------------
    # Account Snapshots
    # -------------------------------------------------------------------------

    def save_account_snapshot(
        self,
        account: AccountInfo,
        total_costs: float = 0.0,
        daily_pnl: float = 0.0,
        current_exposure: float = 0.0,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Persist an account snapshot."""
        sql = """
        INSERT OR REPLACE INTO paper_account_snapshots (
            snapshot_id, timestamp, initial_balance, cash_balance, equity,
            margin_used, realized_pnl, unrealized_pnl, total_costs,
            daily_pnl, current_exposure
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        snapshot_id = f"snap-{uuid.uuid4().hex[:12]}"
        now_utc = datetime.now(timezone.utc).isoformat()
        params = (
            snapshot_id,
            now_utc,
            account.initial_balance,
            account.cash_balance,
            account.equity,
            getattr(account, "margin_used", 0.0),
            account.realized_pnl,
            account.unrealized_pnl,
            total_costs,
            daily_pnl,
            current_exposure,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self.db.transaction() as c:
                c.execute(sql, params)

    def get_latest_account_snapshot(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch the most recent account snapshot."""
        sql = "SELECT * FROM paper_account_snapshots ORDER BY timestamp DESC LIMIT 1;"
        if conn is not None:
            row = conn.execute(sql).fetchone()
        else:
            with self.db.connection() as c:
                row = c.execute(sql).fetchone()
        if not row:
            return None
        return dict(row)

    # -------------------------------------------------------------------------
    # Risk State & Kill Switch
    # -------------------------------------------------------------------------

    def save_risk_state(
        self,
        date_utc: str,
        daily_loss: float,
        baseline_equity: float,
        kill_switch_active: bool,
        kill_switch_reason: Optional[str] = None,
        kill_switch_timestamp: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Persist daily risk metrics and kill switch state in singleton row (id=1)."""
        sql = """
        INSERT INTO paper_risk_state (
            id, date_utc, daily_loss, baseline_equity, kill_switch_active,
            kill_switch_reason, kill_switch_timestamp, updated_at
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            date_utc = excluded.date_utc,
            daily_loss = excluded.daily_loss,
            baseline_equity = excluded.baseline_equity,
            kill_switch_active = excluded.kill_switch_active,
            kill_switch_reason = excluded.kill_switch_reason,
            kill_switch_timestamp = excluded.kill_switch_timestamp,
            updated_at = excluded.updated_at;
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        params = (
            date_utc,
            daily_loss,
            baseline_equity,
            1 if kill_switch_active else 0,
            kill_switch_reason,
            kill_switch_timestamp,
            now_utc,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self.db.transaction() as c:
                c.execute(sql, params)

    def get_risk_state(self, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
        """Fetch the current persisted risk state."""
        sql = "SELECT * FROM paper_risk_state WHERE id = 1;"
        if conn is not None:
            row = conn.execute(sql).fetchone()
        else:
            with self.db.connection() as c:
                row = c.execute(sql).fetchone()
        if not row:
            return None
        return dict(row)

    # -------------------------------------------------------------------------
    # Idempotency Records
    # -------------------------------------------------------------------------

    def save_idempotency_record(
        self,
        client_request_id: str,
        order_id: str,
        execution_id: str,
        status: str,
        execution_report_json: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Save a processed client request record for idempotency deduplication."""
        sql = """
        INSERT OR REPLACE INTO paper_idempotency_records (
            client_request_id, order_id, execution_id, status,
            execution_report_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?);
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        params = (
            client_request_id,
            order_id,
            execution_id,
            status,
            execution_report_json,
            now_utc,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self.db.transaction() as c:
                c.execute(sql, params)

    def get_idempotency_record(
        self, client_request_id: str, conn: Optional[sqlite3.Connection] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch an idempotency record by client_request_id."""
        sql = "SELECT * FROM paper_idempotency_records WHERE client_request_id = ?;"
        if conn is not None:
            row = conn.execute(sql, (client_request_id,)).fetchone()
        else:
            with self.db.connection() as c:
                row = c.execute(sql, (client_request_id,)).fetchone()
        if not row:
            return None
        return dict(row)

    # -------------------------------------------------------------------------
    # Diagnostics & Stats
    # -------------------------------------------------------------------------

    def get_persistence_stats(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """Return operational persistence statistics."""

        def _query(c: sqlite3.Connection) -> Dict[str, Any]:
            orders_cnt = c.execute("SELECT COUNT(*) FROM paper_orders;").fetchone()[0]
            execs_cnt = c.execute("SELECT COUNT(*) FROM paper_executions;").fetchone()[0]
            pos_cnt = c.execute("SELECT COUNT(*) FROM paper_positions;").fetchone()[0]
            audit_cnt = c.execute("SELECT COUNT(*) FROM paper_audit_events;").fetchone()[0]
            idemp_cnt = c.execute("SELECT COUNT(*) FROM paper_idempotency_records;").fetchone()[0]
            last_ts_row = c.execute(
                "SELECT timestamp FROM paper_audit_events ORDER BY timestamp DESC LIMIT 1;"
            ).fetchone()
            last_ts = last_ts_row[0] if last_ts_row else None
            return {
                "orders_count": orders_cnt,
                "executions_count": execs_cnt,
                "positions_count": pos_cnt,
                "audit_events_count": audit_cnt,
                "idempotency_count": idemp_cnt,
                "last_persistence_timestamp": last_ts,
            }

        if conn is not None:
            return _query(conn)
        with self.db.connection() as c:
            return _query(c)

    def verify_integrity(self, conn: Optional[sqlite3.Connection] = None) -> bool:
        """Run SQLite PRAGMA integrity_check."""

        def _check(c: sqlite3.Connection) -> bool:
            row = c.execute("PRAGMA integrity_check;").fetchone()
            return bool(row and row[0] == "ok")

        if conn is not None:
            return _check(conn)
        with self.db.connection() as c:
            return _check(c)
