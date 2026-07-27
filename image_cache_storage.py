from __future__ import annotations

import sqlite3
from pathlib import Path

from app_paths import app_data_path


UNIFIED_CACHE_DB_FILENAME = "image_analysis_cache.sqlite3"


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
        connection = sqlite3.connect(self._db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

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
