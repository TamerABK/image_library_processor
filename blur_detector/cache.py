"""
Persistent cache for blur scan results.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from image_cache_storage import SharedImageCacheDatabase

if TYPE_CHECKING:
    from .blur_detector import BlurResult


class BlurScanCache:
    CACHE_VERSION = 2

    def __init__(self, db_path: str | Path | None = None):
        self._storage = SharedImageCacheDatabase(db_path)
        self._db_path = self._storage.db_path
        self._storage.initialize()
        self._initialize()

    def get(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
    ) -> "BlurResult" | None:
        from .blur_detector import BlurResult

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    blur_scan_cache.laplacian,
                    blur_scan_cache.sobel,
                    blur_scan_cache.local_contrast,
                    blur_scan_cache.lap_norm,
                    blur_scan_cache.sobel_norm,
                    blur_scan_cache.contrast_norm,
                    blur_scan_cache.final_score,
                    blur_scan_cache.status
                FROM blur_scan_cache
                INNER JOIN image_entries
                    ON image_entries.path = blur_scan_cache.path
                WHERE image_entries.path = ?
                  AND image_entries.file_size = ?
                  AND image_entries.mtime_ns = ?
                  AND blur_scan_cache.cache_version = ?
                """,
                (str(path), file_size, mtime_ns, self.CACHE_VERSION),
            ).fetchone()

        if row is None:
            return None

        return BlurResult(
            laplacian=row["laplacian"],
            sobel=row["sobel"],
            local_contrast=row["local_contrast"],
            lap_norm=row["lap_norm"],
            sobel_norm=row["sobel_norm"],
            contrast_norm=row["contrast_norm"],
            final_score=row["final_score"],
            status=row["status"],
        )

    def put(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
        result: "BlurResult",
        *,
        width: int | None = None,
        height: int | None = None,
        is_raw: bool | None = None,
    ) -> None:
        with self._connect() as connection:
            self._storage.upsert_image_entry(
                connection,
                path,
                file_size,
                mtime_ns,
                width=width,
                height=height,
                is_raw=is_raw,
                extension=path.suffix,
            )
            connection.execute(
                """
                INSERT INTO blur_scan_cache (
                    path,
                    cache_version,
                    laplacian,
                    sobel,
                    local_contrast,
                    lap_norm,
                    sobel_norm,
                    contrast_norm,
                    final_score,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    cache_version = excluded.cache_version,
                    laplacian = excluded.laplacian,
                    sobel = excluded.sobel,
                    local_contrast = excluded.local_contrast,
                    lap_norm = excluded.lap_norm,
                    sobel_norm = excluded.sobel_norm,
                    contrast_norm = excluded.contrast_norm,
                    final_score = excluded.final_score,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    str(path),
                    self.CACHE_VERSION,
                    result.laplacian,
                    result.sobel,
                    result.local_contrast,
                    result.lap_norm,
                    result.sobel_norm,
                    result.contrast_norm,
                    result.final_score,
                    result.status,
                ),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS blur_scan_cache (
                    path TEXT PRIMARY KEY,
                    cache_version INTEGER NOT NULL DEFAULT 1,
                    laplacian REAL NOT NULL,
                    sobel REAL NOT NULL,
                    local_contrast REAL NOT NULL,
                    lap_norm REAL NOT NULL,
                    sobel_norm REAL NOT NULL,
                    contrast_norm REAL NOT NULL,
                    final_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (path) REFERENCES image_entries(path) ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return self._storage.connect()
