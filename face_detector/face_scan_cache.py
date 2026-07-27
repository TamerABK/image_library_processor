"""
Persistent cache for per-image face scan results.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from image_cache_storage import SharedImageCacheDatabase

from .models import EmbeddedFace


class FaceScanCache:
    COVERAGE_ALL_FACES = "all_faces"
    COVERAGE_RECOGNIZED_ONLY = "recognized_only"

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
        include_unknown_faces: bool = True,
        database_signature: str | None = None,
    ) -> list[EmbeddedFace] | None:
        with self._connect() as connection:
            entry = connection.execute(
                """
                SELECT
                    face_scan_cache_entries.path,
                    face_scan_cache_entries.coverage,
                    face_scan_cache_entries.database_signature
                FROM face_scan_cache_entries
                INNER JOIN image_entries
                    ON image_entries.path = face_scan_cache_entries.path
                WHERE image_entries.path = ?
                  AND image_entries.file_size = ?
                  AND image_entries.mtime_ns = ?
                """,
                (str(path), file_size, mtime_ns),
            ).fetchone()

            if entry is None:
                return None
            if (
                include_unknown_faces
                and entry["coverage"] != self.COVERAGE_ALL_FACES
            ):
                return None
            if (
                not include_unknown_faces
                and entry["coverage"] == self.COVERAGE_RECOGNIZED_ONLY
                and entry["database_signature"] != database_signature
            ):
                return None

            rows = connection.execute(
                """
                SELECT
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                    confidence,
                    landmarks,
                    landmarks_rows,
                    landmarks_cols,
                    embedding
                FROM face_scan_cache_faces
                WHERE entry_path = ?
                ORDER BY face_index
                """,
                (str(path),),
            ).fetchall()

        return [
            EmbeddedFace(
                path=path,
                bbox=(
                    row["bbox_x"],
                    row["bbox_y"],
                    row["bbox_w"],
                    row["bbox_h"],
                ),
                confidence=row["confidence"],
                landmarks=self._deserialize_array(
                    row["landmarks"],
                    row["landmarks_rows"],
                    row["landmarks_cols"],
                ),
                embedding=self._deserialize_vector(row["embedding"]),
            )
            for row in rows
        ]

    def put(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
        faces: list[EmbeddedFace],
        coverage: str = COVERAGE_ALL_FACES,
        database_signature: str | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        is_raw: bool | None = None,
    ) -> None:
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT
                    face_scan_cache_entries.coverage,
                    image_entries.file_size,
                    image_entries.mtime_ns
                FROM face_scan_cache_entries
                INNER JOIN image_entries
                    ON image_entries.path = face_scan_cache_entries.path
                WHERE face_scan_cache_entries.path = ?
                """,
                (str(path),),
            ).fetchone()

            if (
                existing is not None
                and existing["coverage"] == self.COVERAGE_ALL_FACES
                and coverage == self.COVERAGE_RECOGNIZED_ONLY
                and existing["file_size"] == file_size
                and existing["mtime_ns"] == mtime_ns
            ):
                return

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
                INSERT INTO face_scan_cache_entries (
                    path,
                    face_count,
                    coverage,
                    database_signature
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    face_count = excluded.face_count,
                    coverage = excluded.coverage,
                    database_signature = excluded.database_signature,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    str(path),
                    len(faces),
                    coverage,
                    database_signature,
                ),
            )
            connection.execute(
                """
                DELETE FROM face_scan_cache_faces
                WHERE entry_path = ?
                """,
                (str(path),),
            )

            if faces:
                connection.executemany(
                    """
                    INSERT INTO face_scan_cache_faces (
                        entry_path,
                        face_index,
                        bbox_x,
                        bbox_y,
                        bbox_w,
                        bbox_h,
                        confidence,
                        landmarks,
                        landmarks_rows,
                        landmarks_cols,
                        embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            str(path),
                            index,
                            face.bbox[0],
                            face.bbox[1],
                            face.bbox[2],
                            face.bbox[3],
                            face.confidence,
                            self._serialize_array(face.landmarks),
                            int(face.landmarks.shape[0]),
                            int(face.landmarks.shape[1]),
                            self._serialize_vector(face.embedding),
                        )
                        for index, face in enumerate(faces)
                    ],
                )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS face_scan_cache_entries (
                    path TEXT PRIMARY KEY,
                    face_count INTEGER NOT NULL,
                    coverage TEXT NOT NULL DEFAULT 'all_faces',
                    database_signature TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (path) REFERENCES image_entries(path) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS face_scan_cache_faces (
                    entry_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_w INTEGER NOT NULL,
                    bbox_h INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    landmarks BLOB NOT NULL,
                    landmarks_rows INTEGER NOT NULL,
                    landmarks_cols INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (entry_path, face_index),
                    FOREIGN KEY (entry_path) REFERENCES face_scan_cache_entries(path)
                        ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return self._storage.connect()

    @staticmethod
    def _serialize_array(value: np.ndarray) -> bytes:
        return np.asarray(value, dtype=np.float32).tobytes()

    @staticmethod
    def _deserialize_array(payload: bytes, rows: int, cols: int) -> np.ndarray:
        return np.frombuffer(payload, dtype=np.float32).reshape(rows, cols).copy()

    @staticmethod
    def _serialize_vector(value: np.ndarray) -> bytes:
        return np.asarray(value, dtype=np.float32).tobytes()

    @staticmethod
    def _deserialize_vector(payload: bytes) -> np.ndarray:
        return np.frombuffer(payload, dtype=np.float32).copy()
