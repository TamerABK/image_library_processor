from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class BlurPolicy(StrEnum):
    AUTOMATIC = "automatic"
    REVIEW_BORDERLINE = "review_borderline"
    KEEP_ALL = "keep_all"


class FaceQualityPreset(StrEnum):
    RELAXED = "relaxed"
    BALANCED = "balanced"
    STRICT = "strict"


class VibeDetail(StrEnum):
    SESSION = "session"
    BALANCED_SCENES = "balanced_scenes"
    TIGHT_SCENES = "tight_scenes"


class ExecutionProvider(StrEnum):
    AUTOMATIC = "automatic"
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True, slots=True)
class PeoplePictureThresholds:
    minimum_face_reliability: float = 0.34
    incidental_face_area_threshold: float = 0.015
    primary_face_area_threshold: float = 0.085
    primary_face_span_threshold: float = 0.02
    total_primary_face_area_threshold: float = 0.008
    primary_face_prominence_fraction: float = 0.68
    people_picture_confidence_threshold: float = 0.36
    multi_face_confidence_multiplier_per_additional_face: float = 0.12
    multi_face_confidence_multiplier_cap: float = 1.60
    multi_face_confidence_min_span: float = 0.015
    multi_face_confidence_min_selection_score: float = 0.55


@dataclass(frozen=True, slots=True)
class FaceQualityThresholds:
    weighted_average_threshold: float
    worst_primary_threshold: float
    unusable_primary_threshold: float
    minimum_primary_pick_score_threshold: float
    closed_eyes_terminal_for_small_group: bool
    dark_closed_eye_terminal_for_small_group: bool
    missing_primary_faces_terminal: bool
    minimum_primary_exposure_score: float
    minimum_primary_contrast_score: float
    minimum_primary_eye_open_score: float
    low_reliability_warning_threshold: float = 0.36
    eye_closed_confidence_threshold: float = 0.60


@dataclass(frozen=True, slots=True)
class DuplicateSelectionConfig:
    keepers_per_group: int = 1
    prefer_open_eyes: bool = True
    prioritize_primary_face_quality: bool = True
    include_exact_copies: bool = True
    candidate_time_window_seconds: int | None = None

    def __post_init__(self) -> None:
        if not (1 <= self.keepers_per_group <= 5):
            raise ValueError("keepers_per_group must be between 1 and 5.")


@dataclass(frozen=True, slots=True)
class FaceProcessingConfig:
    recognize_known_people: bool = True
    cluster_unknown_people: bool = True
    hard_exclude_dark_primary_faces: bool = False
    minimum_embedding_utility: float = 0.48
    minimum_recognition_confidence: float = 0.60
    maximum_training_prompts: int = 8
    hide_low_quality_unknown_clusters: bool = True


@dataclass(frozen=True, slots=True)
class VibeConfig:
    detail: VibeDetail = VibeDetail.BALANCED_SCENES
    include_background_embedding: bool = True
    enable_fallback: bool = True
    diagnostics_export: bool = True


@dataclass(frozen=True, slots=True)
class PerformanceConfig:
    execution_provider: ExecutionProvider = ExecutionProvider.AUTOMATIC
    batch_size: int | None = None
    worker_count: int | None = None
    memory_conservative: bool = True


@dataclass(frozen=True, slots=True)
class AutomaticScanConfig:
    file_extensions: tuple[str, ...] | None = None
    orientation_filter: str | None = None
    blur_policy: BlurPolicy = BlurPolicy.AUTOMATIC
    face_quality_preset: FaceQualityPreset = FaceQualityPreset.BALANCED
    people_picture: PeoplePictureThresholds = field(default_factory=PeoplePictureThresholds)
    duplicate_selection: DuplicateSelectionConfig = field(
        default_factory=DuplicateSelectionConfig,
    )
    faces: FaceProcessingConfig = field(default_factory=FaceProcessingConfig)
    vibe: VibeConfig = field(default_factory=VibeConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    config_version: int = 7

    def face_quality_thresholds(self) -> FaceQualityThresholds:
        if self.face_quality_preset is FaceQualityPreset.RELAXED:
            return FaceQualityThresholds(
                weighted_average_threshold=0.40,
                worst_primary_threshold=0.22,
                unusable_primary_threshold=0.16,
                minimum_primary_pick_score_threshold=0.24,
                closed_eyes_terminal_for_small_group=False,
                dark_closed_eye_terminal_for_small_group=False,
                missing_primary_faces_terminal=False,
                minimum_primary_exposure_score=0.30,
                minimum_primary_contrast_score=0.20,
                minimum_primary_eye_open_score=0.16,
            )
        if self.face_quality_preset is FaceQualityPreset.STRICT:
            return FaceQualityThresholds(
                weighted_average_threshold=0.58,
                worst_primary_threshold=0.34,
                unusable_primary_threshold=0.24,
                minimum_primary_pick_score_threshold=0.70,
                closed_eyes_terminal_for_small_group=True,
                dark_closed_eye_terminal_for_small_group=True,
                missing_primary_faces_terminal=True,
                minimum_primary_exposure_score=0.80,
                minimum_primary_contrast_score=0.28,
                minimum_primary_eye_open_score=0.30,
            )
        return FaceQualityThresholds(
            weighted_average_threshold=0.50,
            worst_primary_threshold=0.28,
            unusable_primary_threshold=0.20,
            minimum_primary_pick_score_threshold=0.34,
            closed_eyes_terminal_for_small_group=True,
            dark_closed_eye_terminal_for_small_group=True,
            missing_primary_faces_terminal=False,
            minimum_primary_exposure_score=0.40,
            minimum_primary_contrast_score=0.24,
            minimum_primary_eye_open_score=0.24,
        )

    def cache_signature(self) -> str:
        payload = {
            "config_version": self.config_version,
            "file_extensions": list(self.file_extensions) if self.file_extensions else None,
            "orientation_filter": self.orientation_filter,
            "blur_policy": self.blur_policy.value,
            "face_quality_preset": self.face_quality_preset.value,
            "people_picture": asdict(self.people_picture),
            "duplicate_selection": asdict(self.duplicate_selection),
            "faces": asdict(self.faces),
            "vibe": {
                **asdict(self.vibe),
                "detail": self.vibe.detail.value,
            },
            "performance": {
                **asdict(self.performance),
                "execution_provider": self.performance.execution_provider.value,
            },
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
