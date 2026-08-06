from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from image_cache_storage import SharedImageCacheDatabase

from grouping.models import ScanError, VibeDuplicateSubgroup, VibeGroup, VibeGroupingResult, VibeImageFeatures


def compute_model_fingerprint(model_path: Path) -> str:
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = model_path.stat()
    payload = {
        "filename": model_path.name,
        "file_size": stat.st_size,
        "sha256": digest.hexdigest(),
    }
    return json.dumps(payload, sort_keys=True)


class VibeFeatureCache:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._storage = SharedImageCacheDatabase(db_path)
        self._storage.initialize()
        self._initialize()

    def get(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
        *,
        model_fingerprint: str,
        preprocessing_version: int,
        feature_version: int,
        people_signature: str | None,
        subject_scene_preprocessing_version: int | None = None,
    ) -> VibeImageFeatures | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    image_entries.path,
                    vibe_feature_cache.timestamp_value,
                    vibe_feature_cache.timestamp_source,
                    vibe_feature_cache.semantic_embedding,
                    vibe_feature_cache.semantic_dimension,
                    vibe_feature_cache.recognized_person_ids_json,
                    vibe_feature_cache.dominant_people_names_json,
                    vibe_feature_cache.color_features,
                    vibe_feature_cache.color_dimension,
                    vibe_feature_cache.composition_features,
                    vibe_feature_cache.composition_dimension,
                    vibe_feature_cache.face_layout,
                    vibe_feature_cache.face_layout_dimension,
                    vibe_feature_cache.face_scale_summary,
                    vibe_feature_cache.face_scale_dimension,
                    vibe_feature_cache.background_embedding,
                    vibe_feature_cache.background_dimension,
                    vibe_feature_cache.quality_score,
                    vibe_feature_cache.brightness,
                    vibe_feature_cache.face_count,
                    vibe_feature_cache.face_area_ratio,
                    vibe_feature_cache.metadata_json,
                    image_entries.width,
                    image_entries.height
                FROM vibe_feature_cache
                INNER JOIN image_entries
                    ON image_entries.path = vibe_feature_cache.path
                WHERE image_entries.path = ?
                  AND image_entries.file_size = ?
                  AND image_entries.mtime_ns = ?
                  AND vibe_feature_cache.model_fingerprint = ?
                  AND vibe_feature_cache.preprocessing_version = ?
                  AND vibe_feature_cache.feature_version = ?
                  AND COALESCE(vibe_feature_cache.people_signature, '') = COALESCE(?, '')
                """,
                (
                    str(path),
                    file_size,
                    mtime_ns,
                    model_fingerprint,
                    preprocessing_version,
                    feature_version,
                    people_signature,
                ),
            ).fetchone()

        if row is None:
            return None

        color_features = None
        if row["color_features"] is not None and row["color_dimension"] is not None:
            color_features = self._deserialize_vector(row["color_features"], row["color_dimension"])

        composition_features = None
        if row["composition_features"] is not None and row["composition_dimension"] is not None:
            composition_features = self._deserialize_vector(
                row["composition_features"],
                row["composition_dimension"],
            )

        face_layout = None
        if row["face_layout"] is not None and row["face_layout_dimension"] is not None:
            face_layout = self._deserialize_vector(row["face_layout"], row["face_layout_dimension"])

        face_scale_summary = None
        if row["face_scale_summary"] is not None and row["face_scale_dimension"] is not None:
            face_scale_summary = self._deserialize_vector(
                row["face_scale_summary"],
                row["face_scale_dimension"],
            )

        background_embedding = None
        if row["background_embedding"] is not None and row["background_dimension"] is not None:
            background_embedding = self._deserialize_vector(
                row["background_embedding"],
                row["background_dimension"],
            )

        subject_scene_embedding = None
        if subject_scene_preprocessing_version is not None:
            subject_scene_embedding = self._get_subject_scene_embedding(
                path,
                file_size,
                mtime_ns,
                model_fingerprint=model_fingerprint,
                subject_scene_preprocessing_version=subject_scene_preprocessing_version,
            )

        return VibeImageFeatures(
            image_path=str(path),
            semantic_embedding=self._deserialize_vector(
                row["semantic_embedding"],
                row["semantic_dimension"],
            ),
            capture_timestamp=(
                None if row["timestamp_value"] is None else float(row["timestamp_value"])
            ),
            timestamp_source=str(row["timestamp_source"]),
            recognized_person_ids=tuple(json.loads(row["recognized_person_ids_json"])),
            color_features=color_features,
            composition_features=composition_features,
            face_layout=face_layout,
            face_scale_summary=face_scale_summary,
            subject_scene_embedding=subject_scene_embedding,
            background_embedding=background_embedding,
            action_scores=None,
            scene_scores=None,
            shot_type_scores=None,
            width=row["width"],
            height=row["height"],
            file_mtime_ns=mtime_ns,
            file_size=file_size,
            quality_score=(
                None if row["quality_score"] is None else float(row["quality_score"])
            ),
            brightness=(
                None if row["brightness"] is None else float(row["brightness"])
            ),
            face_count=int(row["face_count"]),
            face_area_ratio=float(row["face_area_ratio"]),
            dominant_people_names=tuple(json.loads(row["dominant_people_names_json"])),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def put(
        self,
        features: VibeImageFeatures,
        *,
        model_fingerprint: str,
        preprocessing_version: int,
        feature_version: int,
        people_signature: str | None,
        subject_scene_preprocessing_version: int | None = None,
    ) -> None:
        path = Path(features.image_path)
        with self._connect() as connection:
            self._storage.upsert_image_entry(
                connection,
                path,
                features.file_size,
                features.file_mtime_ns,
                width=features.width,
                height=features.height,
                extension=path.suffix,
            )
            connection.execute(
                """
                INSERT INTO vibe_feature_cache (
                    path,
                    model_fingerprint,
                    preprocessing_version,
                    feature_version,
                    people_signature,
                    timestamp_value,
                    timestamp_source,
                    semantic_embedding,
                    semantic_dimension,
                    recognized_person_ids_json,
                    dominant_people_names_json,
                    color_features,
                    color_dimension,
                    composition_features,
                    composition_dimension,
                    face_layout,
                    face_layout_dimension,
                    face_scale_summary,
                    face_scale_dimension,
                    background_embedding,
                    background_dimension,
                    quality_score,
                    brightness,
                    face_count,
                    face_area_ratio,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    model_fingerprint = excluded.model_fingerprint,
                    preprocessing_version = excluded.preprocessing_version,
                    feature_version = excluded.feature_version,
                    people_signature = excluded.people_signature,
                    timestamp_value = excluded.timestamp_value,
                    timestamp_source = excluded.timestamp_source,
                    semantic_embedding = excluded.semantic_embedding,
                    semantic_dimension = excluded.semantic_dimension,
                    recognized_person_ids_json = excluded.recognized_person_ids_json,
                    dominant_people_names_json = excluded.dominant_people_names_json,
                    color_features = excluded.color_features,
                    color_dimension = excluded.color_dimension,
                    composition_features = excluded.composition_features,
                    composition_dimension = excluded.composition_dimension,
                    face_layout = excluded.face_layout,
                    face_layout_dimension = excluded.face_layout_dimension,
                    face_scale_summary = excluded.face_scale_summary,
                    face_scale_dimension = excluded.face_scale_dimension,
                    background_embedding = excluded.background_embedding,
                    background_dimension = excluded.background_dimension,
                    quality_score = excluded.quality_score,
                    brightness = excluded.brightness,
                    face_count = excluded.face_count,
                    face_area_ratio = excluded.face_area_ratio,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    str(path),
                    model_fingerprint,
                    preprocessing_version,
                    feature_version,
                    people_signature,
                    features.capture_timestamp,
                    features.timestamp_source,
                    self._serialize_vector(features.semantic_embedding),
                    int(features.semantic_embedding.shape[0]),
                    json.dumps(list(features.recognized_person_ids), sort_keys=True),
                    json.dumps(list(features.dominant_people_names), ensure_ascii=True),
                    None if features.color_features is None else self._serialize_vector(features.color_features),
                    None if features.color_features is None else int(features.color_features.shape[0]),
                    None
                    if features.composition_features is None
                    else self._serialize_vector(features.composition_features),
                    None
                    if features.composition_features is None
                    else int(features.composition_features.shape[0]),
                    None if features.face_layout is None else self._serialize_vector(features.face_layout),
                    None if features.face_layout is None else int(features.face_layout.shape[0]),
                    None
                    if features.face_scale_summary is None
                    else self._serialize_vector(features.face_scale_summary),
                    None
                    if features.face_scale_summary is None
                    else int(features.face_scale_summary.shape[0]),
                    None
                    if features.background_embedding is None
                    else self._serialize_vector(features.background_embedding),
                    None
                    if features.background_embedding is None
                    else int(features.background_embedding.shape[0]),
                    features.quality_score,
                    features.brightness,
                    features.face_count,
                    features.face_area_ratio,
                    json.dumps(features.metadata, sort_keys=True),
                ),
            )
            if (
                subject_scene_preprocessing_version is not None
                and features.subject_scene_embedding is not None
            ):
                connection.execute(
                    """
                    INSERT INTO vibe_subject_scene_cache (
                        path,
                        model_fingerprint,
                        subject_scene_preprocessing_version,
                        subject_scene_embedding,
                        subject_scene_dimension
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        model_fingerprint = excluded.model_fingerprint,
                        subject_scene_preprocessing_version = excluded.subject_scene_preprocessing_version,
                        subject_scene_embedding = excluded.subject_scene_embedding,
                        subject_scene_dimension = excluded.subject_scene_dimension,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        str(path),
                        model_fingerprint,
                        subject_scene_preprocessing_version,
                        self._serialize_vector(features.subject_scene_embedding),
                        int(features.subject_scene_embedding.shape[0]),
                    ),
                )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vibe_feature_cache (
                    path TEXT PRIMARY KEY,
                    model_fingerprint TEXT NOT NULL,
                    preprocessing_version INTEGER NOT NULL,
                    feature_version INTEGER NOT NULL,
                    people_signature TEXT,
                    timestamp_value REAL,
                    timestamp_source TEXT NOT NULL,
                    semantic_embedding BLOB NOT NULL,
                    semantic_dimension INTEGER NOT NULL,
                    recognized_person_ids_json TEXT NOT NULL,
                    dominant_people_names_json TEXT NOT NULL DEFAULT '[]',
                    color_features BLOB,
                    color_dimension INTEGER,
                    composition_features BLOB,
                    composition_dimension INTEGER,
                    face_layout BLOB,
                    face_layout_dimension INTEGER,
                    face_scale_summary BLOB,
                    face_scale_dimension INTEGER,
                    background_embedding BLOB,
                    background_dimension INTEGER,
                    quality_score REAL,
                    brightness REAL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    face_area_ratio REAL NOT NULL DEFAULT 0.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (path) REFERENCES image_entries(path) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS vibe_subject_scene_cache (
                    path TEXT PRIMARY KEY,
                    model_fingerprint TEXT NOT NULL,
                    subject_scene_preprocessing_version INTEGER NOT NULL,
                    subject_scene_embedding BLOB NOT NULL,
                    subject_scene_dimension INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (path) REFERENCES image_entries(path) ON DELETE CASCADE
                );
                """
            )
            self._ensure_columns(
                connection,
                {
                    "face_layout": "BLOB",
                    "face_layout_dimension": "INTEGER",
                    "face_scale_summary": "BLOB",
                    "face_scale_dimension": "INTEGER",
                    "background_embedding": "BLOB",
                    "background_dimension": "INTEGER",
                },
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vibe_feature_cache_versions
                ON vibe_feature_cache (
                    model_fingerprint,
                    preprocessing_version,
                    feature_version,
                    people_signature
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vibe_subject_scene_cache_versions
                ON vibe_subject_scene_cache (
                    model_fingerprint,
                    subject_scene_preprocessing_version
                )
                """
            )

    def _get_subject_scene_embedding(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
        *,
        model_fingerprint: str,
        subject_scene_preprocessing_version: int,
    ) -> np.ndarray | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    vibe_subject_scene_cache.subject_scene_embedding,
                    vibe_subject_scene_cache.subject_scene_dimension
                FROM vibe_subject_scene_cache
                INNER JOIN image_entries
                    ON image_entries.path = vibe_subject_scene_cache.path
                WHERE image_entries.path = ?
                  AND image_entries.file_size = ?
                  AND image_entries.mtime_ns = ?
                  AND vibe_subject_scene_cache.model_fingerprint = ?
                  AND vibe_subject_scene_cache.subject_scene_preprocessing_version = ?
                """,
                (
                    str(path),
                    file_size,
                    mtime_ns,
                    model_fingerprint,
                    subject_scene_preprocessing_version,
                ),
            ).fetchone()
        if row is None:
            return None
        return self._deserialize_vector(
            row["subject_scene_embedding"],
            row["subject_scene_dimension"],
        )

    def _connect(self) -> sqlite3.Connection:
        return self._storage.connect()

    @staticmethod
    def _ensure_columns(
        connection: sqlite3.Connection,
        columns: dict[str, str],
    ) -> None:
        existing = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(vibe_feature_cache)")
        }
        for column_name, column_type in columns.items():
            if column_name in existing:
                continue
            connection.execute(
                f"ALTER TABLE vibe_feature_cache ADD COLUMN {column_name} {column_type}"
            )

    @staticmethod
    def _serialize_vector(value: np.ndarray) -> bytes:
        return np.asarray(value, dtype=np.float32).tobytes()

    @staticmethod
    def _deserialize_vector(payload: bytes, dimension: int) -> np.ndarray:
        return np.frombuffer(payload, dtype=np.float32, count=dimension).copy()


class VibeGroupingResultCache:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._storage = SharedImageCacheDatabase(db_path)
        self._storage.initialize()
        self._initialize()

    def get(self, folder_path: Path, cache_key: str) -> VibeGroupingResult | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT result_json
                FROM vibe_grouping_result_cache
                WHERE folder_path = ?
                  AND cache_key = ?
                """,
                (str(folder_path), cache_key),
            ).fetchone()

        if row is None:
            return None
        return self._deserialize_result(json.loads(row["result_json"]))

    def put(
        self,
        folder_path: Path,
        cache_key: str,
        result: VibeGroupingResult,
    ) -> None:
        payload = json.dumps(self._serialize_result(result), sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO vibe_grouping_result_cache (
                    folder_path,
                    cache_key,
                    result_json
                )
                VALUES (?, ?, ?)
                ON CONFLICT(folder_path) DO UPDATE SET
                    cache_key = excluded.cache_key,
                    result_json = excluded.result_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (str(folder_path), cache_key, payload),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vibe_grouping_result_cache (
                    folder_path TEXT PRIMARY KEY,
                    cache_key TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vibe_grouping_result_cache_key
                ON vibe_grouping_result_cache (cache_key)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return self._storage.connect()

    @staticmethod
    def _serialize_result(result: VibeGroupingResult) -> dict[str, Any]:
        return {
            "groups": [
                {
                    "group_id": group.group_id,
                    "image_paths": list(group.image_paths),
                    "representative_path": group.representative_path,
                    "start_timestamp": group.start_timestamp,
                    "end_timestamp": group.end_timestamp,
                    "recognized_person_ids": list(group.recognized_person_ids),
                    "recognized_person_names": list(group.recognized_person_names),
                    "label": group.label,
                    "cohesion_score": group.cohesion_score,
                    "metadata": group.metadata,
                    "duplicate_subgroups": [
                        {
                            "subgroup_id": subgroup.subgroup_id,
                            "image_paths": list(subgroup.image_paths),
                        }
                        for subgroup in group.duplicate_subgroups
                    ],
                }
                for group in result.groups
            ],
            "ungrouped_paths": list(result.ungrouped_paths),
            "errors": [asdict(error) for error in result.errors],
            "config_snapshot": result.config_snapshot,
            "model_fingerprint": result.model_fingerprint,
            "provider": result.provider,
            "cache_hits": result.cache_hits,
            "cache_misses": result.cache_misses,
            "stage_timings": result.stage_timings,
            "used_fallback_embedder": result.used_fallback_embedder,
            "diagnostics": result.diagnostics,
        }

    @staticmethod
    def _deserialize_result(payload: dict[str, Any]) -> VibeGroupingResult:
        return VibeGroupingResult(
            groups=[
                VibeGroup(
                    group_id=item["group_id"],
                    image_paths=list(item["image_paths"]),
                    representative_path=item["representative_path"],
                    start_timestamp=item["start_timestamp"],
                    end_timestamp=item["end_timestamp"],
                    recognized_person_ids=tuple(item["recognized_person_ids"]),
                    recognized_person_names=tuple(item.get("recognized_person_names", [])),
                    label=item.get("label"),
                    cohesion_score=float(item["cohesion_score"]),
                    metadata=dict(item.get("metadata", {})),
                    duplicate_subgroups=[
                        VibeDuplicateSubgroup(
                            subgroup_id=subgroup["subgroup_id"],
                            image_paths=tuple(subgroup["image_paths"]),
                        )
                        for subgroup in item.get("duplicate_subgroups", [])
                    ],
                )
                for item in payload.get("groups", [])
            ],
            ungrouped_paths=list(payload.get("ungrouped_paths", [])),
            errors=[ScanError(**item) for item in payload.get("errors", [])],
            config_snapshot=dict(payload.get("config_snapshot", {})),
            model_fingerprint=str(payload["model_fingerprint"]),
            provider=str(payload.get("provider", "CPUExecutionProvider")),
            cache_hits=int(payload.get("cache_hits", 0)),
            cache_misses=int(payload.get("cache_misses", 0)),
            stage_timings=dict(payload.get("stage_timings", {})),
            used_fallback_embedder=bool(payload.get("used_fallback_embedder", False)),
            diagnostics=dict(payload.get("diagnostics", {})),
        )
