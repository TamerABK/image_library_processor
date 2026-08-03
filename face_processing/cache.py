"""
Persistent caches for face scan results and image-level face analysis.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from face_analyzer.models import (
    AssessmentStatus,
    EyeLabel,
    EyeMeasurement,
    EyeState,
    FaceAnalysisResult,
    FaceGeometry,
    FaceImageQuality,
    HeadPose,
    MetricResult,
    MetricScore,
    PoseQuality,
)
from image_cache_storage import SharedImageCacheDatabase

from .models import DetectedFace, EmbeddedFace


class _FaceAnalysisStorageMixin:
    _ANALYSIS_COLUMNS = (
        ("analysis_present", "INTEGER NOT NULL DEFAULT 0"),
        ("detector_confidence", "REAL"),
        ("geometry_width", "INTEGER"),
        ("geometry_height", "INTEGER"),
        ("geometry_area", "INTEGER"),
        ("geometry_relative_area", "REAL"),
        ("geometry_visible_ratio", "REAL"),
        ("geometry_minimum_dimension", "INTEGER"),
        ("sharpness_value", "REAL"),
        ("sharpness_confidence", "REAL"),
        ("focus_sharpness_quality", "REAL"),
        ("focus_sharpness_ranking_score", "REAL"),
        ("high_frequency_energy_ratio", "REAL"),
        ("detail_availability_measure", "REAL"),
        ("detail_availability_score", "REAL"),
        ("detail_availability_confidence", "REAL"),
        ("detail_availability_ranking_score", "REAL"),
        ("sharpness_ranking_score", "REAL"),
        ("exposure_value", "REAL"),
        ("exposure_confidence", "REAL"),
        ("display_exposure_score", "REAL"),
        ("shadow_detail_score", "REAL"),
        ("highlight_detail_score", "REAL"),
        ("tonal_balance_score", "REAL"),
        ("exposure_ranking_score", "REAL"),
        ("contrast_value", "REAL"),
        ("contrast_confidence", "REAL"),
        ("contrast_ranking_score", "REAL"),
        ("laplacian_variance", "REAL"),
        ("tenengrad_energy", "REAL"),
        ("median_luminance", "REAL"),
        ("p05_luminance", "REAL"),
        ("p95_luminance", "REAL"),
        ("dark_clip_ratio", "REAL"),
        ("bright_clip_ratio", "REAL"),
        ("usable_tonal_range", "REAL"),
        ("clipping_score", "REAL"),
        ("luminance_score", "REAL"),
        ("tonal_information_score", "REAL"),
        ("raw_exposure_score", "REAL"),
        ("p10_luminance", "REAL"),
        ("p25_luminance", "REAL"),
        ("p75_luminance", "REAL"),
        ("p90_luminance", "REAL"),
        ("broad_tonal_range", "REAL"),
        ("interquartile_range", "REAL"),
        ("broad_contrast_score", "REAL"),
        ("interquartile_contrast_score", "REAL"),
        ("local_contrast_raw", "REAL"),
        ("local_contrast_score", "REAL"),
        ("contrast_quality_score", "REAL"),
        ("detector_ranking_score", "REAL"),
        ("visible_face_ranking_score", "REAL"),
        ("head_pose_yaw_degrees", "REAL"),
        ("head_pose_pitch_degrees", "REAL"),
        ("head_pose_roll_degrees", "REAL"),
        ("head_pose_confidence", "REAL"),
        ("head_pose_status", "TEXT"),
        ("head_pose_source", "TEXT"),
        ("pose_score", "REAL"),
        ("pose_ranking_score", "REAL"),
        ("pose_yaw_score", "REAL"),
        ("pose_pitch_score", "REAL"),
        ("pose_roll_score", "REAL"),
        ("eye_left_open_probability", "REAL"),
        ("eye_left_label", "TEXT"),
        ("eye_left_confidence", "REAL"),
        ("eye_left_status", "TEXT"),
        ("eye_left_source_width", "INTEGER"),
        ("eye_left_source_height", "INTEGER"),
        ("eye_right_open_probability", "REAL"),
        ("eye_right_label", "TEXT"),
        ("eye_right_confidence", "REAL"),
        ("eye_right_status", "TEXT"),
        ("eye_right_source_width", "INTEGER"),
        ("eye_right_source_height", "INTEGER"),
        ("eye_combined_open_score", "REAL"),
        ("eye_confidence", "REAL"),
        ("eye_status", "TEXT"),
        ("eye_ranking_score", "REAL"),
        ("eye_weight", "REAL"),
        ("measurement_reliability_score", "REAL"),
        ("global_selection_score", "REAL"),
        ("group_relative_score", "REAL"),
        ("final_group_score", "REAL"),
        ("selection_score", "REAL"),
        ("embedding_utility_score", "REAL"),
    )
    _ANALYSIS_COLUMN_NAMES = tuple(name for name, _type in _ANALYSIS_COLUMNS)

    @staticmethod
    def _has_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
    ) -> bool:
        columns = connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
        return any(existing["name"] == column for existing in columns)

    @classmethod
    def _ensure_column(
        cls,
        connection: sqlite3.Connection,
        table: str,
        column: str,
        column_type: str,
    ) -> None:
        if cls._has_column(connection, table, column):
            return
        connection.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
        )

    @classmethod
    def _ensure_analysis_columns(
        cls,
        connection: sqlite3.Connection,
        table: str,
    ) -> None:
        for column_name, column_type in cls._ANALYSIS_COLUMNS:
            cls._ensure_column(connection, table, column_name, column_type)

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

    @staticmethod
    def _float_or_fallback(
        row: sqlite3.Row,
        column: str,
        fallback: float,
    ) -> float:
        value = row[column]
        if value is None:
            return float(fallback)
        return float(value)

    @classmethod
    def _analysis_values(
        cls,
        analysis: FaceAnalysisResult | None,
    ) -> tuple[object, ...]:
        if analysis is None:
            return (0, *([None] * (len(cls._ANALYSIS_COLUMN_NAMES) - 1)))

        return (
            1,
            analysis.detector_confidence,
            analysis.geometry.width,
            analysis.geometry.height,
            analysis.geometry.area,
            analysis.geometry.relative_area,
            analysis.geometry.visible_ratio,
            analysis.geometry.minimum_dimension,
            analysis.image_quality.sharpness.value,
            analysis.image_quality.sharpness.confidence,
            analysis.image_quality.focus_sharpness_score,
            analysis.image_quality.focus_ranking_score,
            analysis.image_quality.high_frequency_energy_ratio,
            analysis.image_quality.detail_availability_measure,
            analysis.image_quality.detail_availability_score,
            analysis.image_quality.detail_availability.confidence,
            analysis.image_quality.detail_availability_ranking_score,
            analysis.image_quality.sharpness_ranking_score,
            analysis.image_quality.exposure.value,
            analysis.image_quality.exposure.confidence,
            analysis.image_quality.display_exposure_score,
            analysis.image_quality.shadow_detail_score,
            analysis.image_quality.highlight_detail_score,
            analysis.image_quality.tonal_balance_score,
            analysis.image_quality.exposure_ranking_score,
            analysis.image_quality.contrast.value,
            analysis.image_quality.contrast.confidence,
            analysis.image_quality.contrast_ranking_score,
            analysis.image_quality.laplacian_variance,
            analysis.image_quality.tenengrad_energy,
            analysis.image_quality.median_luminance,
            analysis.image_quality.p05_luminance,
            analysis.image_quality.p95_luminance,
            analysis.image_quality.dark_clip_ratio,
            analysis.image_quality.bright_clip_ratio,
            analysis.image_quality.usable_tonal_range,
            analysis.image_quality.clipping_score,
            analysis.image_quality.luminance_score,
            analysis.image_quality.tonal_information_score,
            analysis.image_quality.raw_exposure_score,
            analysis.image_quality.p10_luminance,
            analysis.image_quality.p25_luminance,
            analysis.image_quality.p75_luminance,
            analysis.image_quality.p90_luminance,
            analysis.image_quality.broad_tonal_range,
            analysis.image_quality.interquartile_range,
            analysis.image_quality.broad_contrast_score,
            analysis.image_quality.interquartile_contrast_score,
            analysis.image_quality.local_contrast_raw,
            analysis.image_quality.local_contrast_score,
            analysis.image_quality.contrast_quality_score,
            analysis.detector_metric.ranking_score,
            analysis.visible_face_metric.ranking_score,
            analysis.head_pose.yaw_degrees,
            analysis.head_pose.pitch_degrees,
            analysis.head_pose.roll_degrees,
            analysis.head_pose.confidence,
            analysis.head_pose.status.value,
            analysis.head_pose.source,
            analysis.pose_score,
            analysis.pose_ranking_score,
            analysis.pose.yaw_score,
            analysis.pose.pitch_score,
            analysis.pose.roll_score,
            analysis.eye_state.left.open_probability,
            analysis.eye_state.left.label.value,
            analysis.eye_state.left.confidence,
            analysis.eye_state.left.status.value,
            analysis.eye_state.left.source_width,
            analysis.eye_state.left.source_height,
            analysis.eye_state.right.open_probability,
            analysis.eye_state.right.label.value,
            analysis.eye_state.right.confidence,
            analysis.eye_state.right.status.value,
            analysis.eye_state.right.source_width,
            analysis.eye_state.right.source_height,
            analysis.eye_state.combined_open_score,
            analysis.eye_state.confidence,
            analysis.eye_state.status.value,
            analysis.eye_ranking_score,
            analysis.eye_weight,
            analysis.measurement_reliability_score,
            analysis.global_selection_score,
            analysis.group_relative_score,
            analysis.final_group_score,
            analysis.selection_score,
            analysis.embedding_utility_score,
        )

    @classmethod
    def _analysis_from_row(
        cls,
        row: sqlite3.Row,
        warnings: tuple[str, ...] = (),
    ) -> FaceAnalysisResult | None:
        if not row["analysis_present"]:
            return None

        return FaceAnalysisResult(
            detector_confidence=float(row["detector_confidence"]),
            geometry=FaceGeometry(
                width=int(row["geometry_width"]),
                height=int(row["geometry_height"]),
                area=int(row["geometry_area"]),
                relative_area=float(row["geometry_relative_area"]),
                visible_ratio=float(row["geometry_visible_ratio"]),
                minimum_dimension=int(row["geometry_minimum_dimension"]),
            ),
            image_quality=FaceImageQuality(
                focus_sharpness=MetricResult(
                    raw_value=cls._float_or_fallback(
                        row,
                        "focus_sharpness_quality",
                        float(row["sharpness_value"]),
                    ),
                    quality_score=cls._float_or_fallback(
                        row,
                        "focus_sharpness_quality",
                        float(row["sharpness_value"]),
                    ),
                    confidence=float(row["sharpness_confidence"]),
                    ranking_score=cls._float_or_fallback(
                        row,
                        "focus_sharpness_ranking_score",
                        float(row["sharpness_value"]),
                    ),
                ),
                detail_availability=MetricResult(
                    raw_value=cls._float_or_fallback(
                        row,
                        "detail_availability_measure",
                        float(row["geometry_minimum_dimension"]),
                    ),
                    quality_score=cls._float_or_fallback(
                        row,
                        "detail_availability_score",
                        float(row["sharpness_value"]),
                    ),
                    confidence=cls._float_or_fallback(
                        row,
                        "detail_availability_confidence",
                        float(row["sharpness_confidence"]),
                    ),
                    ranking_score=cls._float_or_fallback(
                        row,
                        "detail_availability_ranking_score",
                        float(row["sharpness_value"]),
                    ),
                ),
                sharpness=MetricResult(
                    raw_value=cls._float_or_fallback(
                        row,
                        "focus_sharpness_quality",
                        float(row["sharpness_value"]),
                    ),
                    quality_score=float(row["sharpness_value"]),
                    confidence=float(row["sharpness_confidence"]),
                    ranking_score=cls._float_or_fallback(
                        row,
                        "sharpness_ranking_score",
                        float(row["sharpness_value"]),
                    ),
                ),
                exposure=MetricResult(
                    raw_value=cls._float_or_fallback(
                        row,
                        "raw_exposure_score",
                        float(row["exposure_value"]),
                    ),
                    quality_score=cls._float_or_fallback(
                        row,
                        "display_exposure_score",
                        float(row["exposure_value"]),
                    ),
                    confidence=float(row["exposure_confidence"]),
                    ranking_score=cls._float_or_fallback(
                        row,
                        "exposure_ranking_score",
                        float(row["exposure_value"]),
                    ),
                ),
                contrast=MetricResult(
                    raw_value=cls._float_or_fallback(
                        row,
                        "contrast_quality_score",
                        float(row["contrast_value"]),
                    ),
                    quality_score=float(row["contrast_value"]),
                    confidence=float(row["contrast_confidence"]),
                    ranking_score=cls._float_or_fallback(
                        row,
                        "contrast_ranking_score",
                        float(row["contrast_value"]),
                    ),
                ),
                laplacian_variance=float(row["laplacian_variance"]),
                tenengrad_energy=float(row["tenengrad_energy"]),
                high_frequency_energy_ratio=cls._float_or_fallback(
                    row,
                    "high_frequency_energy_ratio",
                    0.0,
                ),
                detail_availability_measure=cls._float_or_fallback(
                    row,
                    "detail_availability_measure",
                    float(row["geometry_minimum_dimension"]),
                ),
                median_luminance=float(row["median_luminance"]),
                p05_luminance=cls._float_or_fallback(
                    row,
                    "p05_luminance",
                    float(row["median_luminance"]),
                ),
                p95_luminance=cls._float_or_fallback(
                    row,
                    "p95_luminance",
                    float(row["median_luminance"]),
                ),
                dark_clip_ratio=float(row["dark_clip_ratio"]),
                bright_clip_ratio=float(row["bright_clip_ratio"]),
                usable_tonal_range=cls._float_or_fallback(
                    row,
                    "usable_tonal_range",
                    0.0,
                ),
                clipping_score=cls._float_or_fallback(
                    row,
                    "clipping_score",
                    float(row["exposure_value"]),
                ),
                luminance_score=cls._float_or_fallback(
                    row,
                    "luminance_score",
                    float(row["exposure_value"]),
                ),
                tonal_information_score=cls._float_or_fallback(
                    row,
                    "tonal_information_score",
                    float(row["exposure_value"]),
                ),
                raw_exposure_score=cls._float_or_fallback(
                    row,
                    "raw_exposure_score",
                    float(row["exposure_value"]),
                ),
                display_exposure_score=cls._float_or_fallback(
                    row,
                    "display_exposure_score",
                    float(row["exposure_value"]),
                ),
                shadow_detail_score=cls._float_or_fallback(
                    row,
                    "shadow_detail_score",
                    0.0,
                ),
                highlight_detail_score=cls._float_or_fallback(
                    row,
                    "highlight_detail_score",
                    0.0,
                ),
                tonal_balance_score=cls._float_or_fallback(
                    row,
                    "tonal_balance_score",
                    0.0,
                ),
                p10_luminance=cls._float_or_fallback(
                    row,
                    "p10_luminance",
                    float(row["median_luminance"]),
                ),
                p25_luminance=cls._float_or_fallback(
                    row,
                    "p25_luminance",
                    float(row["median_luminance"]),
                ),
                p75_luminance=cls._float_or_fallback(
                    row,
                    "p75_luminance",
                    float(row["median_luminance"]),
                ),
                p90_luminance=cls._float_or_fallback(
                    row,
                    "p90_luminance",
                    float(row["median_luminance"]),
                ),
                broad_tonal_range=cls._float_or_fallback(
                    row,
                    "broad_tonal_range",
                    0.0,
                ),
                interquartile_range=cls._float_or_fallback(
                    row,
                    "interquartile_range",
                    0.0,
                ),
                broad_contrast_score=cls._float_or_fallback(
                    row,
                    "broad_contrast_score",
                    float(row["contrast_value"]),
                ),
                interquartile_contrast_score=cls._float_or_fallback(
                    row,
                    "interquartile_contrast_score",
                    float(row["contrast_value"]),
                ),
                local_contrast_raw=cls._float_or_fallback(
                    row,
                    "local_contrast_raw",
                    0.0,
                ),
                local_contrast_score=cls._float_or_fallback(
                    row,
                    "local_contrast_score",
                    float(row["contrast_value"]),
                ),
                contrast_quality_score=cls._float_or_fallback(
                    row,
                    "contrast_quality_score",
                    float(row["contrast_value"]),
                ),
            ),
            detector_metric=MetricResult(
                raw_value=float(row["detector_confidence"]),
                quality_score=float(row["detector_confidence"]),
                confidence=1.0,
                ranking_score=cls._float_or_fallback(
                    row,
                    "detector_ranking_score",
                    float(row["detector_confidence"]),
                ),
            ),
            head_pose=HeadPose(
                yaw_degrees=(
                    None
                    if row["head_pose_yaw_degrees"] is None
                    else float(row["head_pose_yaw_degrees"])
                ),
                pitch_degrees=(
                    None
                    if row["head_pose_pitch_degrees"] is None
                    else float(row["head_pose_pitch_degrees"])
                ),
                roll_degrees=(
                    None
                    if row["head_pose_roll_degrees"] is None
                    else float(row["head_pose_roll_degrees"])
                ),
                confidence=float(row["head_pose_confidence"]),
                status=AssessmentStatus(row["head_pose_status"]),
                source=str(row["head_pose_source"]),
            ),
            visible_face_metric=MetricResult(
                raw_value=float(row["geometry_visible_ratio"]),
                quality_score=float(row["geometry_visible_ratio"]),
                confidence=1.0,
                ranking_score=cls._float_or_fallback(
                    row,
                    "visible_face_ranking_score",
                    float(row["geometry_visible_ratio"]),
                ),
            ),
            pose=PoseQuality(
                metric=MetricResult(
                    raw_value=(
                        None
                        if row["pose_score"] is None
                        else float(row["pose_score"])
                    ),
                    quality_score=(
                        None
                        if row["pose_score"] is None
                        else float(row["pose_score"])
                    ),
                    confidence=float(row["head_pose_confidence"]),
                    ranking_score=(
                        None
                        if row["pose_score"] is None
                        else cls._float_or_fallback(
                            row,
                            "pose_ranking_score",
                            float(row["pose_score"]),
                        )
                    ),
                ),
                yaw_score=(
                    None
                    if row["pose_yaw_score"] is None
                    else float(row["pose_yaw_score"])
                ),
                pitch_score=(
                    None
                    if row["pose_pitch_score"] is None
                    else float(row["pose_pitch_score"])
                ),
                roll_score=(
                    None
                    if row["pose_roll_score"] is None
                    else float(row["pose_roll_score"])
                ),
            ),
            eye_state=EyeState(
                left=EyeMeasurement(
                    open_probability=(
                        None
                        if row["eye_left_open_probability"] is None
                        else float(row["eye_left_open_probability"])
                    ),
                    label=EyeLabel(row["eye_left_label"]),
                    confidence=float(row["eye_left_confidence"]),
                    status=AssessmentStatus(row["eye_left_status"]),
                    source_width=int(row["eye_left_source_width"]),
                    source_height=int(row["eye_left_source_height"]),
                ),
                right=EyeMeasurement(
                    open_probability=(
                        None
                        if row["eye_right_open_probability"] is None
                        else float(row["eye_right_open_probability"])
                    ),
                    label=EyeLabel(row["eye_right_label"]),
                    confidence=float(row["eye_right_confidence"]),
                    status=AssessmentStatus(row["eye_right_status"]),
                    source_width=int(row["eye_right_source_width"]),
                    source_height=int(row["eye_right_source_height"]),
                ),
                combined_open_score=(
                    None
                    if row["eye_combined_open_score"] is None
                    else float(row["eye_combined_open_score"])
                ),
                confidence=float(row["eye_confidence"]),
                status=AssessmentStatus(row["eye_status"]),
            ),
            eyes=MetricResult(
                raw_value=(
                    None
                    if row["eye_combined_open_score"] is None
                    else float(row["eye_combined_open_score"])
                ),
                quality_score=(
                    None
                    if row["eye_combined_open_score"] is None
                    else float(row["eye_combined_open_score"])
                ),
                confidence=float(row["eye_confidence"]),
                ranking_score=(
                    None
                    if row["eye_combined_open_score"] is None
                    else cls._float_or_fallback(
                        row,
                        "eye_ranking_score",
                        float(row["eye_combined_open_score"]),
                    )
                ),
            ),
            eye_weight=cls._float_or_fallback(
                row,
                "eye_weight",
                0.0,
            ),
            measurement_reliability=MetricResult(
                raw_value=cls._float_or_fallback(
                    row,
                    "measurement_reliability_score",
                    0.0,
                ),
                quality_score=cls._float_or_fallback(
                    row,
                    "measurement_reliability_score",
                    0.0,
                ),
                confidence=1.0,
                ranking_score=cls._float_or_fallback(
                    row,
                    "measurement_reliability_score",
                    0.0,
                ),
            ),
            global_selection_score=cls._float_or_fallback(
                row,
                "global_selection_score",
                float(row["selection_score"]),
            ),
            group_relative_score=cls._float_or_fallback(
                row,
                "group_relative_score",
                0.5,
            ),
            final_group_score=cls._float_or_fallback(
                row,
                "final_group_score",
                float(row["selection_score"]),
            ),
            selection_score=float(row["selection_score"]),
            embedding_utility_score=float(row["embedding_utility_score"]),
            warnings=warnings,
        )

    @classmethod
    def _deserialize_analysis_json_payload(
        cls,
        payload: str,
    ) -> FaceAnalysisResult:
        data = json.loads(payload)
        return FaceAnalysisResult(
            detector_confidence=float(data["detector_confidence"]),
            geometry=FaceGeometry(
                width=int(data["geometry"]["width"]),
                height=int(data["geometry"]["height"]),
                area=int(data["geometry"]["area"]),
                relative_area=float(data["geometry"]["relative_area"]),
                visible_ratio=float(data["geometry"]["visible_ratio"]),
                minimum_dimension=int(data["geometry"]["minimum_dimension"]),
            ),
            image_quality=FaceImageQuality(
                focus_sharpness=MetricResult(
                    raw_value=float(
                        data["image_quality"].get(
                            "focus_sharpness_score",
                            data["image_quality"]["sharpness"]["value"],
                        )
                    ),
                    quality_score=float(
                        data["image_quality"].get(
                            "focus_sharpness_score",
                            data["image_quality"]["sharpness"]["value"],
                        )
                    ),
                    confidence=float(
                        data["image_quality"]["sharpness"]["confidence"]
                    ),
                    ranking_score=float(
                        data["image_quality"].get(
                            "focus_ranking_score",
                            data["image_quality"]["sharpness"]["value"],
                        )
                    ),
                ),
                detail_availability=MetricResult(
                    raw_value=float(
                        data["image_quality"].get(
                            "detail_availability_measure",
                            data["geometry"]["minimum_dimension"],
                        )
                    ),
                    quality_score=float(
                        data["image_quality"].get(
                            "detail_availability_score",
                            data["image_quality"]["sharpness"]["value"],
                        )
                    ),
                    confidence=float(
                        data["image_quality"].get(
                            "detail_availability_confidence",
                            data["image_quality"]["sharpness"]["confidence"],
                        )
                    ),
                    ranking_score=float(
                        data["image_quality"].get(
                            "detail_availability_ranking_score",
                            data["image_quality"]["sharpness"]["value"],
                        )
                    ),
                ),
                sharpness=MetricResult(
                    raw_value=float(
                        data["image_quality"].get(
                            "focus_sharpness_score",
                            data["image_quality"]["sharpness"]["value"],
                        )
                    ),
                    quality_score=float(
                        data["image_quality"]["sharpness"]["value"]
                    ),
                    confidence=float(
                        data["image_quality"]["sharpness"]["confidence"]
                    ),
                    ranking_score=float(
                        data["image_quality"].get(
                            "sharpness_ranking_score",
                            data["image_quality"]["sharpness"]["value"],
                        )
                    ),
                ),
                exposure=MetricResult(
                    raw_value=float(
                        data["image_quality"].get(
                            "raw_exposure_score",
                            data["image_quality"]["exposure"]["value"],
                        )
                    ),
                    quality_score=float(
                        data["image_quality"].get(
                            "display_exposure_score",
                            data["image_quality"]["exposure"]["value"],
                        )
                    ),
                    confidence=float(
                        data["image_quality"]["exposure"]["confidence"]
                    ),
                    ranking_score=float(
                        data["image_quality"].get(
                            "exposure_ranking_score",
                            data["image_quality"]["exposure"]["value"],
                        )
                    ),
                ),
                contrast=MetricResult(
                    raw_value=float(
                        data["image_quality"].get(
                            "contrast_quality_score",
                            data["image_quality"]["contrast"]["value"],
                        )
                    ),
                    quality_score=float(
                        data["image_quality"]["contrast"]["value"]
                    ),
                    confidence=float(
                        data["image_quality"]["contrast"]["confidence"]
                    ),
                    ranking_score=float(
                        data["image_quality"].get(
                            "contrast_ranking_score",
                            data["image_quality"]["contrast"]["value"],
                        )
                    ),
                ),
                laplacian_variance=float(data["image_quality"]["laplacian_variance"]),
                tenengrad_energy=float(data["image_quality"]["tenengrad_energy"]),
                high_frequency_energy_ratio=float(
                    data["image_quality"].get("high_frequency_energy_ratio", 0.0)
                ),
                detail_availability_measure=float(
                    data["image_quality"].get(
                        "detail_availability_measure",
                        data["geometry"]["minimum_dimension"],
                    )
                ),
                median_luminance=float(data["image_quality"]["median_luminance"]),
                p05_luminance=float(
                    data["image_quality"].get(
                        "p05_luminance",
                        data["image_quality"]["median_luminance"],
                    )
                ),
                p95_luminance=float(
                    data["image_quality"].get(
                        "p95_luminance",
                        data["image_quality"]["median_luminance"],
                    )
                ),
                dark_clip_ratio=float(data["image_quality"]["dark_clip_ratio"]),
                bright_clip_ratio=float(data["image_quality"]["bright_clip_ratio"]),
                usable_tonal_range=float(
                    data["image_quality"].get("usable_tonal_range", 0.0)
                ),
                clipping_score=float(
                    data["image_quality"].get(
                        "clipping_score",
                        data["image_quality"]["exposure"]["value"],
                    )
                ),
                luminance_score=float(
                    data["image_quality"].get(
                        "luminance_score",
                        data["image_quality"]["exposure"]["value"],
                    )
                ),
                tonal_information_score=float(
                    data["image_quality"].get(
                        "tonal_information_score",
                        data["image_quality"]["exposure"]["value"],
                    )
                ),
                raw_exposure_score=float(
                    data["image_quality"].get(
                        "raw_exposure_score",
                        data["image_quality"]["exposure"]["value"],
                    )
                ),
                display_exposure_score=float(
                    data["image_quality"].get(
                        "display_exposure_score",
                        data["image_quality"]["exposure"]["value"],
                    )
                ),
                shadow_detail_score=float(
                    data["image_quality"].get("shadow_detail_score", 0.0)
                ),
                highlight_detail_score=float(
                    data["image_quality"].get("highlight_detail_score", 0.0)
                ),
                tonal_balance_score=float(
                    data["image_quality"].get("tonal_balance_score", 0.0)
                ),
                p10_luminance=float(
                    data["image_quality"].get(
                        "p10_luminance",
                        data["image_quality"]["median_luminance"],
                    )
                ),
                p25_luminance=float(
                    data["image_quality"].get(
                        "p25_luminance",
                        data["image_quality"]["median_luminance"],
                    )
                ),
                p75_luminance=float(
                    data["image_quality"].get(
                        "p75_luminance",
                        data["image_quality"]["median_luminance"],
                    )
                ),
                p90_luminance=float(
                    data["image_quality"].get(
                        "p90_luminance",
                        data["image_quality"]["median_luminance"],
                    )
                ),
                broad_tonal_range=float(
                    data["image_quality"].get("broad_tonal_range", 0.0)
                ),
                interquartile_range=float(
                    data["image_quality"].get("interquartile_range", 0.0)
                ),
                broad_contrast_score=float(
                    data["image_quality"].get(
                        "broad_contrast_score",
                        data["image_quality"]["contrast"]["value"],
                    )
                ),
                interquartile_contrast_score=float(
                    data["image_quality"].get(
                        "interquartile_contrast_score",
                        data["image_quality"]["contrast"]["value"],
                    )
                ),
                local_contrast_raw=float(
                    data["image_quality"].get("local_contrast_raw", 0.0)
                ),
                local_contrast_score=float(
                    data["image_quality"].get(
                        "local_contrast_score",
                        data["image_quality"]["contrast"]["value"],
                    )
                ),
                contrast_quality_score=float(
                    data["image_quality"].get(
                        "contrast_quality_score",
                        data["image_quality"]["contrast"]["value"],
                    )
                ),
            ),
            detector_metric=MetricResult(
                raw_value=float(data["detector_confidence"]),
                quality_score=float(data["detector_confidence"]),
                confidence=1.0,
                ranking_score=float(
                    data.get("detector_ranking_score", data["detector_confidence"])
                ),
            ),
            head_pose=HeadPose(
                yaw_degrees=(
                    None
                    if data["head_pose"]["yaw_degrees"] is None
                    else float(data["head_pose"]["yaw_degrees"])
                ),
                pitch_degrees=(
                    None
                    if data["head_pose"]["pitch_degrees"] is None
                    else float(data["head_pose"]["pitch_degrees"])
                ),
                roll_degrees=(
                    None
                    if data["head_pose"]["roll_degrees"] is None
                    else float(data["head_pose"]["roll_degrees"])
                ),
                confidence=float(data["head_pose"]["confidence"]),
                status=AssessmentStatus(data["head_pose"]["status"]),
                source=str(data["head_pose"]["source"]),
            ),
            visible_face_metric=MetricResult(
                raw_value=float(data["geometry"]["visible_ratio"]),
                quality_score=float(data["geometry"]["visible_ratio"]),
                confidence=1.0,
                ranking_score=float(
                    data.get(
                        "visible_face_ranking_score",
                        data["geometry"]["visible_ratio"],
                    )
                ),
            ),
            pose=PoseQuality(
                metric=MetricResult(
                    raw_value=(
                        None
                        if data["pose_score"] is None
                        else float(data["pose_score"])
                    ),
                    quality_score=(
                        None
                        if data["pose_score"] is None
                        else float(data["pose_score"])
                    ),
                    confidence=float(data["head_pose"]["confidence"]),
                    ranking_score=(
                        None
                        if data["pose_score"] is None
                        else float(data.get("pose_ranking_score", data["pose_score"]))
                    ),
                ),
                yaw_score=(
                    None
                    if data.get("pose_yaw_score") is None
                    else float(data["pose_yaw_score"])
                ),
                pitch_score=(
                    None
                    if data.get("pose_pitch_score") is None
                    else float(data["pose_pitch_score"])
                ),
                roll_score=(
                    None
                    if data.get("pose_roll_score") is None
                    else float(data["pose_roll_score"])
                ),
            ),
            eye_state=EyeState(
                left=EyeMeasurement(
                    open_probability=(
                        None
                        if data["eye_state"]["left"]["open_probability"] is None
                        else float(data["eye_state"]["left"]["open_probability"])
                    ),
                    label=EyeLabel(data["eye_state"]["left"]["label"]),
                    confidence=float(data["eye_state"]["left"]["confidence"]),
                    status=AssessmentStatus(data["eye_state"]["left"]["status"]),
                    source_width=int(data["eye_state"]["left"]["source_width"]),
                    source_height=int(data["eye_state"]["left"]["source_height"]),
                ),
                right=EyeMeasurement(
                    open_probability=(
                        None
                        if data["eye_state"]["right"]["open_probability"] is None
                        else float(data["eye_state"]["right"]["open_probability"])
                    ),
                    label=EyeLabel(data["eye_state"]["right"]["label"]),
                    confidence=float(data["eye_state"]["right"]["confidence"]),
                    status=AssessmentStatus(data["eye_state"]["right"]["status"]),
                    source_width=int(data["eye_state"]["right"]["source_width"]),
                    source_height=int(data["eye_state"]["right"]["source_height"]),
                ),
                combined_open_score=(
                    None
                    if data["eye_state"]["combined_open_score"] is None
                    else float(data["eye_state"]["combined_open_score"])
                ),
                confidence=float(data["eye_state"]["confidence"]),
                status=AssessmentStatus(data["eye_state"]["status"]),
            ),
            eyes=MetricResult(
                raw_value=(
                    None
                    if data["eye_state"]["combined_open_score"] is None
                    else float(data["eye_state"]["combined_open_score"])
                ),
                quality_score=(
                    None
                    if data["eye_state"]["combined_open_score"] is None
                    else float(data["eye_state"]["combined_open_score"])
                ),
                confidence=float(data["eye_state"]["confidence"]),
                ranking_score=(
                    None
                    if data["eye_state"]["combined_open_score"] is None
                    else float(
                        data.get(
                            "eye_ranking_score",
                            data["eye_state"]["combined_open_score"],
                        )
                    )
                ),
            ),
            eye_weight=float(data.get("eye_weight", 0.0)),
            measurement_reliability=MetricResult(
                raw_value=float(data.get("measurement_reliability_score", 0.0)),
                quality_score=float(data.get("measurement_reliability_score", 0.0)),
                confidence=1.0,
                ranking_score=float(data.get("measurement_reliability_score", 0.0)),
            ),
            global_selection_score=float(
                data.get("global_selection_score", data["selection_score"])
            ),
            group_relative_score=float(data.get("group_relative_score", 0.5)),
            final_group_score=float(
                data.get("final_group_score", data["selection_score"])
            ),
            selection_score=float(data["selection_score"]),
            embedding_utility_score=float(data["embedding_utility_score"]),
            warnings=tuple(data.get("warnings", ())),
        )

    @classmethod
    def _analysis_update_sql(cls, table: str) -> str:
        assignments = ", ".join(
            f"{column_name} = ?"
            for column_name in cls._ANALYSIS_COLUMN_NAMES
        )
        return (
            f"UPDATE {table} SET {assignments}, analysis_json = NULL "
            "WHERE entry_path = ? AND face_index = ?"
        )

    @classmethod
    def _select_analysis_columns_sql(cls) -> str:
        return ",\n                    ".join(cls._ANALYSIS_COLUMN_NAMES)

    @classmethod
    def _replace_warning_rows(
        cls,
        connection: sqlite3.Connection,
        warnings_table: str,
        entry_path: str,
        faces: list[DetectedFace] | list[EmbeddedFace],
    ) -> None:
        connection.execute(
            f"DELETE FROM {warnings_table} WHERE entry_path = ?",
            (entry_path,),
        )

        warning_rows: list[tuple[str, int, int, str]] = []
        for face_index, face in enumerate(faces):
            if face.analysis is None:
                continue
            for warning_index, warning_text in enumerate(face.analysis.warnings):
                warning_rows.append(
                    (entry_path, face_index, warning_index, warning_text)
                )

        if warning_rows:
            connection.executemany(
                f"""
                INSERT INTO {warnings_table} (
                    entry_path,
                    face_index,
                    warning_index,
                    warning_text
                )
                VALUES (?, ?, ?, ?)
                """,
                warning_rows,
            )

    @classmethod
    def _load_warning_map(
        cls,
        connection: sqlite3.Connection,
        warnings_table: str,
        entry_path: str,
    ) -> dict[int, tuple[str, ...]]:
        warning_rows = connection.execute(
            f"""
            SELECT face_index, warning_text
            FROM {warnings_table}
            WHERE entry_path = ?
            ORDER BY face_index, warning_index
            """,
            (entry_path,),
        ).fetchall()

        grouped: dict[int, list[str]] = defaultdict(list)
        for row in warning_rows:
            grouped[int(row["face_index"])].append(str(row["warning_text"]))
        return {
            face_index: tuple(warnings)
            for face_index, warnings in grouped.items()
        }

    @classmethod
    def _migrate_analysis_json(
        cls,
        connection: sqlite3.Connection,
        table: str,
        warnings_table: str,
    ) -> None:
        if not cls._has_column(connection, table, "analysis_json"):
            return

        rows = connection.execute(
            f"""
            SELECT entry_path, face_index, analysis_json
            FROM {table}
            WHERE analysis_json IS NOT NULL
            """
        ).fetchall()

        if not rows:
            return

        update_sql = cls._analysis_update_sql(table)
        for row in rows:
            analysis = cls._deserialize_analysis_json_payload(row["analysis_json"])
            connection.execute(
                update_sql,
                (
                    *cls._analysis_values(analysis),
                    str(row["entry_path"]),
                    int(row["face_index"]),
                ),
            )
            connection.execute(
                f"""
                DELETE FROM {warnings_table}
                WHERE entry_path = ? AND face_index = ?
                """,
                (str(row["entry_path"]), int(row["face_index"])),
            )
            if analysis.warnings:
                connection.executemany(
                    f"""
                    INSERT INTO {warnings_table} (
                        entry_path,
                        face_index,
                        warning_index,
                        warning_text
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            str(row["entry_path"]),
                            int(row["face_index"]),
                            warning_index,
                            warning_text,
                        )
                        for warning_index, warning_text in enumerate(analysis.warnings)
                    ],
                )


class FaceScanCache(_FaceAnalysisStorageMixin):
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
        require_analysis: bool = False,
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
            if include_unknown_faces and entry["coverage"] != self.COVERAGE_ALL_FACES:
                return None
            if (
                not include_unknown_faces
                and entry["coverage"] == self.COVERAGE_RECOGNIZED_ONLY
                and entry["database_signature"] != database_signature
            ):
                return None

            rows = connection.execute(
                f"""
                SELECT
                    face_index,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                    confidence,
                    landmarks,
                    landmarks_rows,
                    landmarks_cols,
                    embedding,
                    {self._select_analysis_columns_sql()}
                FROM face_scan_cache_faces
                WHERE entry_path = ?
                ORDER BY face_index
                """,
                (str(path),),
            ).fetchall()
            warning_map = self._load_warning_map(
                connection,
                "face_scan_cache_face_warnings",
                str(path),
            )

        if require_analysis and any(not row["analysis_present"] for row in rows):
            return None

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
                analysis=self._analysis_from_row(
                    row,
                    warning_map.get(int(row["face_index"]), ()),
                ),
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
                    f"""
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
                        embedding,
                        {", ".join(self._ANALYSIS_COLUMN_NAMES)}
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {", ".join("?" for _ in self._ANALYSIS_COLUMN_NAMES)})
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
                            *self._analysis_values(face.analysis),
                        )
                        for index, face in enumerate(faces)
                    ],
                )

            self._replace_warning_rows(
                connection,
                "face_scan_cache_face_warnings",
                str(path),
                faces,
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

                CREATE TABLE IF NOT EXISTS face_scan_cache_face_warnings (
                    entry_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    warning_index INTEGER NOT NULL,
                    warning_text TEXT NOT NULL,
                    PRIMARY KEY (entry_path, face_index, warning_index),
                    FOREIGN KEY (entry_path, face_index)
                        REFERENCES face_scan_cache_faces(entry_path, face_index)
                        ON DELETE CASCADE
                );
                """
            )
            self._ensure_analysis_columns(connection, "face_scan_cache_faces")
            self._migrate_analysis_json(
                connection,
                "face_scan_cache_faces",
                "face_scan_cache_face_warnings",
            )

    def _connect(self) -> sqlite3.Connection:
        return self._storage.connect()


class ImageFaceAnalysisCache(_FaceAnalysisStorageMixin):
    CACHE_VERSION = 1

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
    ) -> list[DetectedFace] | None:
        with self._connect() as connection:
            entry = connection.execute(
                """
                SELECT image_face_analysis_cache.path
                FROM image_face_analysis_cache
                INNER JOIN image_entries
                    ON image_entries.path = image_face_analysis_cache.path
                WHERE image_entries.path = ?
                  AND image_entries.file_size = ?
                  AND image_entries.mtime_ns = ?
                  AND image_face_analysis_cache.cache_version = ?
                """,
                (str(path), file_size, mtime_ns, self.CACHE_VERSION),
            ).fetchone()

            if entry is None:
                return None

            rows = connection.execute(
                f"""
                SELECT
                    face_index,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                    confidence,
                    landmarks,
                    landmarks_rows,
                    landmarks_cols,
                    {self._select_analysis_columns_sql()}
                FROM image_face_analysis_faces
                WHERE entry_path = ?
                ORDER BY face_index
                """,
                (str(path),),
            ).fetchall()
            warning_map = self._load_warning_map(
                connection,
                "image_face_analysis_face_warnings",
                str(path),
            )

        return [
            DetectedFace(
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
                analysis=self._analysis_from_row(
                    row,
                    warning_map.get(int(row["face_index"]), ()),
                ),
            )
            for row in rows
        ]

    def put(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
        faces: list[DetectedFace],
        *,
        width: int | None = None,
        height: int | None = None,
        is_raw: bool | None = None,
    ) -> None:
        analyzed_faces = [face for face in faces if face.analysis is not None]
        best_selection_score = max(
            (face.analysis.selection_score for face in analyzed_faces),
            default=None,
        )
        best_embedding_utility_score = max(
            (face.analysis.embedding_utility_score for face in analyzed_faces),
            default=None,
        )

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
                INSERT INTO image_face_analysis_cache (
                    path,
                    cache_version,
                    face_count,
                    analyzed_face_count,
                    best_selection_score,
                    best_embedding_utility_score
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    cache_version = excluded.cache_version,
                    face_count = excluded.face_count,
                    analyzed_face_count = excluded.analyzed_face_count,
                    best_selection_score = excluded.best_selection_score,
                    best_embedding_utility_score = excluded.best_embedding_utility_score,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    str(path),
                    self.CACHE_VERSION,
                    len(faces),
                    len(analyzed_faces),
                    best_selection_score,
                    best_embedding_utility_score,
                ),
            )
            connection.execute(
                """
                DELETE FROM image_face_analysis_faces
                WHERE entry_path = ?
                """,
                (str(path),),
            )

            if faces:
                connection.executemany(
                    f"""
                    INSERT INTO image_face_analysis_faces (
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
                        {", ".join(self._ANALYSIS_COLUMN_NAMES)}
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {", ".join("?" for _ in self._ANALYSIS_COLUMN_NAMES)})
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
                            *self._analysis_values(face.analysis),
                        )
                        for index, face in enumerate(faces)
                    ],
                )

            self._replace_warning_rows(
                connection,
                "image_face_analysis_face_warnings",
                str(path),
                faces,
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS image_face_analysis_cache (
                    path TEXT PRIMARY KEY,
                    cache_version INTEGER NOT NULL DEFAULT 1,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    analyzed_face_count INTEGER NOT NULL DEFAULT 0,
                    best_selection_score REAL,
                    best_embedding_utility_score REAL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (path) REFERENCES image_entries(path) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS image_face_analysis_faces (
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
                    PRIMARY KEY (entry_path, face_index),
                    FOREIGN KEY (entry_path) REFERENCES image_face_analysis_cache(path)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS image_face_analysis_face_warnings (
                    entry_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    warning_index INTEGER NOT NULL,
                    warning_text TEXT NOT NULL,
                    PRIMARY KEY (entry_path, face_index, warning_index),
                    FOREIGN KEY (entry_path, face_index)
                        REFERENCES image_face_analysis_faces(entry_path, face_index)
                        ON DELETE CASCADE
                );
                """
            )
            self._ensure_analysis_columns(connection, "image_face_analysis_faces")
            self._ensure_column(
                connection,
                "image_face_analysis_cache",
                "analyzed_face_count",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                connection,
                "image_face_analysis_cache",
                "best_selection_score",
                "REAL",
            )
            self._ensure_column(
                connection,
                "image_face_analysis_cache",
                "best_embedding_utility_score",
                "REAL",
            )
            self._migrate_analysis_json(
                connection,
                "image_face_analysis_faces",
                "image_face_analysis_face_warnings",
            )

    def _connect(self) -> sqlite3.Connection:
        return self._storage.connect()
