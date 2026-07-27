"""
Persistent cache for duplicate detection image metadata and hashes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from image_cache_storage import SharedImageCacheDatabase

from .models import PhotoInfo


class DuplicatePhotoCache:
    """
    SQLite-backed cache keyed by photo path and invalidated by file metadata.
    """

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
    ) -> PhotoInfo | None:
        normalized_path = str(path)

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    image_entries.width,
                    image_entries.height,
                    duplicate_photo_cache.phash,
                    duplicate_photo_cache.dhash
                FROM duplicate_photo_cache
                INNER JOIN image_entries
                    ON image_entries.path = duplicate_photo_cache.path
                WHERE image_entries.path = ?
                  AND image_entries.file_size = ?
                  AND image_entries.mtime_ns = ?
                  AND duplicate_photo_cache.cache_version = ?
                """,
                (normalized_path, file_size, mtime_ns, self.CACHE_VERSION),
            ).fetchone()

        if row is None:
            return None

        return PhotoInfo(
            path=path,
            width=row["width"],
            height=row["height"],
            file_size=file_size,
            phash=self._deserialize_hash(row["phash"]),
            dhash=self._deserialize_hash(row["dhash"]),
        )

    def put(
        self,
        photo: PhotoInfo,
        mtime_ns: int,
        *,
        is_raw: bool | None = None,
    ) -> None:
        with self._connect() as connection:
            self._storage.upsert_image_entry(
                connection,
                photo.path,
                photo.file_size,
                mtime_ns,
                width=photo.width,
                height=photo.height,
                is_raw=is_raw,
                extension=photo.path.suffix,
            )
            connection.execute(
                """
                INSERT INTO duplicate_photo_cache (
                    path,
                    cache_version,
                    phash,
                    dhash
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    cache_version = excluded.cache_version,
                    phash = excluded.phash,
                    dhash = excluded.dhash,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    str(photo.path),
                    self.CACHE_VERSION,
                    self._serialize_hash(photo.phash),
                    self._serialize_hash(photo.dhash),
                ),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS duplicate_photo_cache (
                    path TEXT PRIMARY KEY,
                    cache_version INTEGER NOT NULL DEFAULT 1,
                    phash INTEGER NOT NULL,
                    dhash INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (path) REFERENCES image_entries(path) ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return self._storage.connect()

    @staticmethod
    def _serialize_hash(value: int) -> str:
        return format(value, "x")

    @staticmethod
    def _deserialize_hash(value: int | str) -> int:
        if isinstance(value, int):
            return value

        return int(value, 16)
