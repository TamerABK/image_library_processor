from __future__ import annotations

import sqlite3
import time
from contextlib import nullcontext
from pathlib import Path

from app_paths import app_data_path
from automatic_scan.perf import InstrumentedSqliteConnection, get_active_profiler
from automatic_scan.profiling import profile_count, profile_scope
from image_cache_batch import active_cache_session


UNIFIED_CACHE_DB_FILENAME = "image_analysis_cache.sqlite3"


class _CacheConnection(InstrumentedSqliteConnection):
    def __exit__(self, *args):
        had_transaction = self.in_transaction
        started = time.perf_counter()
        succeeded = False
        try:
            with profile_scope("cache.sqlite_transaction_exit", detail="deep"):
                result = super().__exit__(*args)
            succeeded = True
            return result
        finally:
            if had_transaction:
                if not succeeded:
                    profile_count("cache.transaction_exit_failures")
                elif args and args[0] is not None:
                    profile_count("cache.transaction_rollbacks")
                else:
                    profile_count("cache.transaction_commits")
                    profiler = get_active_profiler()
                    if profiler is not None:
                        profiler.record_commit(time.perf_counter() - started)
            self.close()


class SharedImageCacheDatabase:
    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path is not None else app_data_path(
            UNIFIED_CACHE_DB_FILENAME
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS image_entries (
                    path TEXT PRIMARY KEY,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    is_raw INTEGER,
                    extension TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_image_entries_mtime
                ON image_entries (mtime_ns);
                """
            )

    def connect(self) -> sqlite3.Connection:
        session = active_cache_session(self._db_path)
        if session is not None:
            return session.connection()
        return self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        started = time.perf_counter()
        with profile_scope("cache.sqlite_connect", detail="deep"):
            connection = sqlite3.connect(
                self._db_path,
                timeout=30,
                factory=_CacheConnection,
            )
        profiler = get_active_profiler()
        if profiler is not None:
            profiler.record_database_connect(time.perf_counter() - started)
        connection.row_factory = sqlite3.Row
        with profile_scope("cache.sqlite_pragmas", detail="deep"):
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def using_connection(self, connection: sqlite3.Connection | None = None):
        """Borrow a scan's read connection, or own a short-lived connection."""
        return self.connect() if connection is None else nullcontext(connection)

    def upsert_image_entry(
        self,
        connection: sqlite3.Connection,
        path: Path,
        file_size: int,
        mtime_ns: int,
        *,
        width: int | None = None,
        height: int | None = None,
        is_raw: bool | None = None,
        extension: str | None = None,
    ) -> None:
        normalized_extension = None
        if extension is not None:
            normalized_extension = extension.strip().lower() or None

        # All stage rows refer to this fingerprint. Updating it in place would
        # make another stage's old result look valid. Foreign-key cascades remove
        # those results before the new file revision is registered.
        connection.execute(
            "DELETE FROM image_entries WHERE path = ? AND (file_size != ? OR mtime_ns != ?)",
            (str(path), file_size, mtime_ns),
        )

        connection.execute(
            """
            INSERT INTO image_entries (
                path,
                file_size,
                mtime_ns,
                width,
                height,
                is_raw,
                extension
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_size = excluded.file_size,
                mtime_ns = excluded.mtime_ns,
                width = COALESCE(excluded.width, image_entries.width),
                height = COALESCE(excluded.height, image_entries.height),
                is_raw = COALESCE(excluded.is_raw, image_entries.is_raw),
                extension = COALESCE(excluded.extension, image_entries.extension),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                str(path),
                file_size,
                mtime_ns,
                width,
                height,
                None if is_raw is None else int(is_raw),
                normalized_extension,
            ),
        )
