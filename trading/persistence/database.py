"""Isolated, thread-safe database connection manager for Paper Trading.

Ensures strict separation between simulated paper-trading storage and any future
external broker database. Employs SQLite with WAL mode, foreign key enforcement,
and explicit ACID transaction boundaries.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional


class PaperDatabaseManager:
    """Thread-safe SQLite database manager dedicated to paper-trading persistence.

    Invariants:
    - Dedicated strictly to simulated paper trading (namespace: paper_trading).
    - Foreign keys enforced on every connection.
    - WAL journal mode for concurrent readers and atomic writers.
    - Parameterized queries exclusively; no dynamic SQL concatenation.
    """

    DEFAULT_DB_PATH = Path("data/paper_trading/paper_trading.db")

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        if db_path is None:
            env_path = os.getenv("PAPER_TRADING_DB_PATH")
            self.db_path = Path(env_path) if env_path else self.DEFAULT_DB_PATH
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
        else:
            self.db_path = Path(db_path)

        self._is_memory = str(self.db_path) == ":memory:"
        if not self._is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_connection()

    def _init_connection(self) -> None:
        """Initialize the shared SQLite connection with strict safety pragmas."""
        target = ":memory:" if self._is_memory else str(self.db_path)
        # check_same_thread=False is safe when all access is guarded by self._lock
        conn = sqlite3.connect(
            target,
            check_same_thread=False,
            isolation_level=None,  # Explicit transaction management
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            if not self._is_memory:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
        self._conn = conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an atomic ACID transaction.

        Serializes write transactions and rolls back completely on any unhandled exception.
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("PaperDatabaseManager is closed.")
            conn = self._conn
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE;")
            try:
                yield conn
                if conn.in_transaction:
                    conn.execute("COMMIT;")
            except Exception:
                if conn.in_transaction:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                raise

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing thread-safe read access to the database."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("PaperDatabaseManager is closed.")
            yield self._conn

    def is_connected(self) -> bool:
        """Check if the database connection is open and responsive."""
        with self._lock:
            if self._conn is None:
                return False
            try:
                cur = self._conn.execute("SELECT 1;")
                return cur.fetchone()[0] == 1
            except Exception:
                return False

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
