"""Thread-affine, bounded deferred writes for the unified analysis cache.

Only the project processing worker enables this session. SQL is buffered between
image operations; the SQLite transaction itself exists only during ``flush``.
Other application/database users retain their ordinary immediate-commit API.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from automatic_scan.profiling import profile_count, profile_scope


_ACTIVE = threading.local()
SUPPORTED_BATCH_SIZES = (1, 32, 64, 128, 256, 512)


class CacheBatchWriteError(RuntimeError):
    """A cache batch rolled back and this session cannot safely continue."""


def active_cache_session(path: Path):
    session = getattr(_ACTIVE, "session", None)
    return session if session is not None and session.path == path.resolve() else None


class CacheBatchSession:
    """A processing-thread-owned connection and a bounded queue of SQL groups.

    A group is one successful ``with storage.connect()`` block. Groups preserve
    order and are committed together; a failed group never enters the queue.
    No transaction stays open while image analysis or inference executes.
    """

    def __init__(self, path: Path, connection_factory, *, batch_size: int = 256,
                 max_delay_seconds: float = 1.0):
        if batch_size not in SUPPORTED_BATCH_SIZES:
            raise ValueError(f"cache batch size must be one of {SUPPORTED_BATCH_SIZES}")
        if max_delay_seconds <= 0:
            raise ValueError("cache maximum batching delay must be positive")
        self.path = path.resolve()
        self.batch_size = batch_size
        self.max_delay_seconds = max_delay_seconds
        self._owner = threading.get_ident()
        self._connection = connection_factory()
        self._groups: list[tuple[float, list[tuple[str, str, object]]]] = []
        self._first_pending_at: float | None = None
        self._failed: Exception | None = None
        self._closed = False
        self.commits = 0
        self.rollbacks = 0
        self.max_pending = 0
        self.records_committed = 0
        self.reused_connections = 0
        self.total_buffer_wait_seconds = 0.0
        self.max_buffer_wait_seconds = 0.0
        self.wal_peak_bytes = 0

    def _assert_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("cache connection belongs to another thread")
        if self._closed:
            raise RuntimeError("cache batch session is closed")
        if self._failed is not None:
            raise CacheBatchWriteError("cache batch failed; reopen processing") from self._failed

    def __enter__(self):
        self._assert_owner()
        if getattr(_ACTIVE, "session", None) is not None:
            raise RuntimeError("nested cache batch sessions are unsupported")
        _ACTIVE.session = self
        return self

    def __exit__(self, *_):
        try:
            self.close()
        finally:
            if getattr(_ACTIVE, "session", None) is self:
                del _ACTIVE.session

    def connection(self):
        self._assert_owner()
        self.reused_connections += 1
        profile_count("cache.connection_reuses")
        return _BufferedConnection(self)

    def add(self, statements: list[tuple[str, str, object]]):
        self._assert_owner()
        if not statements:
            return
        if not self._groups:
            self._first_pending_at = time.monotonic()
        self._groups.append((time.monotonic(), statements))
        self.max_pending = max(self.max_pending, len(self._groups))
        profile_count("cache.write_groups_queued")
        if len(self._groups) >= self.batch_size or (
            self._first_pending_at is not None and
            time.monotonic() - self._first_pending_at >= self.max_delay_seconds
        ):
            self.flush()

    def flush(self):
        self._assert_owner()
        if not self._groups:
            return
        groups = self._groups
        self._groups = []
        self._first_pending_at = None
        now = time.monotonic()
        for queued_at, _ in groups:
            elapsed = now - queued_at
            self.total_buffer_wait_seconds += elapsed
            self.max_buffer_wait_seconds = max(self.max_buffer_wait_seconds, elapsed)
        with profile_scope("cache.batch.flush", detail="deep", units=len(groups), batch_size=len(groups)):
            try:
                self._connection.execute("BEGIN")
                for _, group in groups:
                    for kind, sql, parameters in group:
                        if kind == "many":
                            self._connection.executemany(sql, parameters)
                        else:
                            self._connection.execute(sql, parameters)
                with profile_scope("cache.batch.commit", detail="deep", units=len(groups),
                                   batch_size=len(groups)):
                    self._connection.commit()
            except Exception as exc:
                self._connection.rollback()
                self.rollbacks += 1
                profile_count("cache.batch.rollbacks")
                self._failed = exc
                raise CacheBatchWriteError("cache batch rolled back") from exc
        self.commits += 1
        self.records_committed += len(groups)
        try:
            self.wal_peak_bytes = max(self.wal_peak_bytes,
                self.path.with_name(self.path.name + "-wal").stat().st_size)
        except OSError:
            pass
        profile_count("cache.batch.commits")
        profile_count("cache.batch.records_committed", len(groups))

    def read(self, sql, parameters=()):
        self.flush()
        return self._connection.execute(sql, parameters)

    def script(self, sql):
        self.flush()
        return self._connection.executescript(sql)

    def close(self):
        if self._closed:
            return
        try:
            if self._failed is None:
                self.flush()
        finally:
            profile_count("cache.batch.max_pending", self.max_pending)
            profile_count("cache.batch.buffer_wait_ns", int(self.total_buffer_wait_seconds * 1e9))
            profile_count("cache.batch.max_buffer_wait_ns", int(self.max_buffer_wait_seconds * 1e9))
            profile_count("cache.batch.wal_peak_bytes", self.wal_peak_bytes)
            self._connection.close()
            self._closed = True


class _BufferedConnection:
    """A short context representing one atomic write group, not a live transaction."""

    def __init__(self, session: CacheBatchSession):
        self._session = session
        self._statements: list[tuple[str, str, object]] = []

    def __enter__(self):
        self._session._assert_owner()
        return self

    def __exit__(self, exception_type, *_):
        if exception_type is None:
            self._submit()
        else:
            self._statements.clear()
        return False

    def _submit(self):
        if self._statements:
            statements, self._statements = self._statements, []
            self._session.add(statements)

    def execute(self, sql, parameters=()):
        self._session._assert_owner()
        keyword = sql.lstrip().split(None, 1)[0].upper()
        if keyword in {"SELECT", "WITH", "PRAGMA", "EXPLAIN"}:
            self._submit()
            return self._session.read(sql, parameters)
        self._statements.append(("one", sql, tuple(parameters)))
        return None

    def executemany(self, sql, parameters):
        self._session._assert_owner()
        self._statements.append(("many", sql, tuple(tuple(row) for row in parameters)))
        return None

    def executescript(self, sql):
        self._submit()
        return self._session.script(sql)

    def commit(self):
        self._submit()
        self._session.flush()

    def rollback(self):
        self._statements.clear()


def batch_size_from_environment() -> int | None:
    """``legacy`` selects the previous connection-per-operation baseline."""
    value = os.environ.get("ROOM36_CACHE_WRITE_BATCH_SIZE", "256").strip().lower()
    if value == "legacy":
        return None
    size = int(value)
    if size not in SUPPORTED_BATCH_SIZES:
        raise ValueError(f"ROOM36_CACHE_WRITE_BATCH_SIZE must be legacy or {SUPPORTED_BATCH_SIZES}")
    return size


def max_delay_from_environment() -> float:
    milliseconds = int(os.environ.get("ROOM36_CACHE_BATCH_MAX_DELAY_MS", "1000"))
    if milliseconds < 1:
        raise ValueError("ROOM36_CACHE_BATCH_MAX_DELAY_MS must be positive")
    return milliseconds / 1000
