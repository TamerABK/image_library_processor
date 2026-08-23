from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from grouping.models import VibeGroup


class AutomaticStage(StrEnum):
    DISCOVERY = "discovery"
    CACHE_RESOLUTION = "cache_resolution"
    BLUR_ANALYSIS = "blur_analysis"
    DUPLICATE_CANDIDATES = "duplicate_candidates"
    FACE_DETECTION_ANALYSIS = "face_detection_analysis"
    PEOPLE_IMAGE_CLASSIFICATION = "people_image_classification"
    FACE_QUALITY_GATE = "face_quality_gate"
    DUPLICATE_FINALIZATION = "duplicate_finalization"
    FACE_EMBEDDING = "face_embedding"
    FACE_RECOGNITION = "face_recognition"
    UNKNOWN_FACE_CLUSTERING = "unknown_face_clustering"
    VIBE_GROUPING = "vibe_grouping"
    RESULT_FINALIZATION = "result_finalization"


class PipelineImageStatus(StrEnum):
    ACTIVE = "active"
    REVIEW = "review"
    EXCLUDED = "excluded"
    DUPLICATE_ALTERNATIVE = "duplicate_alternative"
    PROCESSING_ERROR = "processing_error"
    MANUALLY_RESTORED = "manually_restored"
    MANUALLY_KEPT = "manually_kept"


class StageDecision(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    TERMINAL_EXCLUSION = "terminal_exclusion"
    RETRYABLE_ERROR = "retryable_error"
    NON_RETRYABLE_ERROR = "non_retryable_error"


class ManualOverrideKind(StrEnum):
    RESTORE = "restore"
    KEEP = "keep"
    PRIMARY_KEEPER = "primary_keeper"
    ADDITIONAL_KEEPER = "additional_keeper"


@dataclass(frozen=True, slots=True)
class DiscoveryRecord:
    path: Path
    file_size: int
    mtime_ns: int
    extension: str
    orientation: str | None
    capture_timestamp: float | None
    width: int | None
    height: int | None


@dataclass(slots=True)
class StageStats:
    processed: int = 0
    skipped: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    duration_seconds: float = 0.0


@dataclass(slots=True)
class PipelineSummary:
    discovered_count: int = 0
    active_count: int = 0
    excluded_count: int = 0
    duplicate_group_count: int = 0
    keeper_count: int = 0
    duplicate_alternative_count: int = 0
    known_people_count: int = 0
    unknown_cluster_count: int = 0
    vibe_group_count: int = 0
    canceled: bool = False
    stage_stats: dict[str, StageStats] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineWarning:
    stage: str
    path: str | None
    message: str


@dataclass(frozen=True, slots=True)
class PipelineError:
    stage: str
    path: str | None
    message: str
    fatal: bool = False


@dataclass(frozen=True, slots=True)
class PeoplePictureAssessment:
    is_people_picture: bool
    confidence: float
    primary_face_indices: tuple[int, ...]
    supporting_reasons: tuple[str, ...]
    detected_face_count: int = 0
    reliable_face_count: int = 0
    largest_primary_face_area: float = 0.0
    total_primary_face_area: float = 0.0
    largest_primary_face_span: float = 0.0
    average_primary_centrality: float = 0.0
    best_primary_selection_score: float | None = None


@dataclass(frozen=True, slots=True)
class ImageFaceQualityAssessment:
    is_people_picture: bool
    primary_face_count: int
    acceptable_primary_face_count: int
    weighted_primary_average: float | None
    worst_primary_score: float | None
    best_primary_score: float | None
    closed_eye_primary_count: int
    unusable_primary_count: int
    terminal_failure: bool
    warnings: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DuplicateRankingBreakdown:
    acceptable_primary_face_count: int
    worst_primary_score: float
    weighted_primary_average: float
    eye_open_score: float
    pose_visibility_score: float
    image_sharpness_score: float
    exposure_contrast_score: float
    overall_quality_score: float
    group_visual_centrality: float
    capture_timestamp: float | None
    path_key: str
    explanation: tuple[str, ...]


@dataclass(slots=True)
class AutomaticDuplicateGroup:
    group_id: str
    image_paths: list[str]
    keeper_paths: list[str]
    excluded_paths: list[str]
    ranking: dict[str, DuplicateRankingBreakdown]


@dataclass(slots=True)
class AutomaticUnknownCluster:
    cluster_id: str
    image_paths: list[str]
    representative_path: str
    face_count: int
    suggested_names: tuple[str, ...] = ()
    hidden: bool = False
    cluster: Any | None = None


@dataclass(slots=True)
class AutomaticScanDiagnostics:
    config_signature: str
    folder: str
    stage_stats: dict[str, StageStats] = field(default_factory=dict)
    cache_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    step_timings: dict[str, float] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    total_seconds: float = 0.0
    startup_seconds: float = 0.0
    scan_seconds: float = 0.0
    images_per_second: float = 0.0
    cache_mode: str = "unknown"
    image_io: dict[str, Any] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    database: dict[str, Any] = field(default_factory=dict)
    queues: dict[str, dict[str, Any]] = field(default_factory=dict)
    startup_breakdown: dict[str, float] = field(default_factory=dict)
    discovery_breakdown: dict[str, float] = field(default_factory=dict)
    model_phase_breakdown: dict[str, float] = field(default_factory=dict)
    vibe_fallback_used: bool = False
    vibe_fallback_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineImageState:
    path: str
    file_size: int
    mtime_ns: int
    extension: str
    orientation: str | None
    capture_timestamp: float | None
    width: int | None
    height: int | None
    status: PipelineImageStatus = PipelineImageStatus.ACTIVE
    primary_exclusion_reason: str | None = None
    excluded_at_stage: str | None = None
    warnings: list[str] = field(default_factory=list)
    completed_stages: set[str] = field(default_factory=set)
    skipped_stages: set[str] = field(default_factory=set)
    cache_hits: set[str] = field(default_factory=set)
    cache_misses: set[str] = field(default_factory=set)
    selected_for_deletion: bool = False
    manually_restored: bool = False
    manually_kept: bool = False
    manual_override: ManualOverrideKind | None = None
    duplicate_group_id: str | None = None
    duplicate_rank: int | None = None
    is_duplicate_keeper: bool = False
    vibe_group_id: str | None = None
    known_person_ids: tuple[str, ...] = ()
    unknown_cluster_ids: tuple[str, ...] = ()
    people_picture_assessment: PeoplePictureAssessment | None = None
    face_quality_assessment: ImageFaceQualityAssessment | None = None

    def path_obj(self) -> Path:
        return Path(self.path)

    @property
    def is_active(self) -> bool:
        return self.status in {
            PipelineImageStatus.ACTIVE,
            PipelineImageStatus.REVIEW,
            PipelineImageStatus.MANUALLY_RESTORED,
            PipelineImageStatus.MANUALLY_KEPT,
        }

    @property
    def is_terminally_excluded(self) -> bool:
        return self.status in {
            PipelineImageStatus.EXCLUDED,
            PipelineImageStatus.DUPLICATE_ALTERNATIVE,
            PipelineImageStatus.PROCESSING_ERROR,
        }

    def add_warning(self, value: str) -> None:
        if value and value not in self.warnings:
            self.warnings.append(value)

    def mark_terminal_exclusion(
        self,
        *,
        stage: AutomaticStage,
        reason: str,
        selected_for_deletion: bool,
        duplicate_alternative: bool = False,
    ) -> None:
        if self.manually_restored or self.manually_kept:
            self.add_warning(f"manual_override_preserved:{reason}")
            return
        if self.primary_exclusion_reason is not None:
            return
        self.primary_exclusion_reason = reason
        self.excluded_at_stage = stage.value
        self.selected_for_deletion = selected_for_deletion
        if duplicate_alternative:
            self.status = PipelineImageStatus.DUPLICATE_ALTERNATIVE
        else:
            self.status = PipelineImageStatus.EXCLUDED


@dataclass(slots=True)
class AutomaticScanResult:
    folder: str
    all_images: list[PipelineImageState]
    vibe_groups: list[VibeGroup]
    vibe_ungrouped: list[PipelineImageState]
    duplicate_groups: list[AutomaticDuplicateGroup]
    known_people: dict[str, list[str]]
    unknown_clusters: list[AutomaticUnknownCluster]
    exclusions_by_reason: dict[str, list[PipelineImageState]]
    warnings: list[PipelineWarning]
    processing_errors: list[PipelineError]
    progress_summary: PipelineSummary
    diagnostics: AutomaticScanDiagnostics

    def image_map(self) -> dict[str, PipelineImageState]:
        return {image.path: image for image in self.all_images}

    @property
    def active_images(self) -> list[PipelineImageState]:
        return [image for image in self.all_images if image.is_active]

    @property
    def selected_for_deletion(self) -> list[PipelineImageState]:
        return [image for image in self.all_images if image.selected_for_deletion]

    def to_diagnostics_payload(self) -> dict[str, Any]:
        not_people_picture_images = [
            self._not_people_picture_payload(image)
            for image in self.all_images
            if image.people_picture_assessment is not None
            and not image.people_picture_assessment.is_people_picture
        ]
        return {
            "folder": self.folder,
            "file_count": len(self.all_images),
            "active_count": len(self.active_images),
            "excluded_count": len(self.all_images) - len(self.active_images),
            "keeper_count": sum(1 for image in self.all_images if image.is_duplicate_keeper),
            "duplicate_alternative_count": sum(
                1
                for image in self.all_images
                if image.status is PipelineImageStatus.DUPLICATE_ALTERNATIVE
            ),
            "known_people_count": len(self.known_people),
            "unknown_cluster_count": len(self.unknown_clusters),
            "vibe_group_count": len(self.vibe_groups),
            "vibe_fallback_used": self.diagnostics.vibe_fallback_used,
            "total_seconds": self.diagnostics.total_seconds,
            "startup_seconds": self.diagnostics.startup_seconds,
            "scan_seconds": self.diagnostics.scan_seconds,
            "images_per_second": self.diagnostics.images_per_second,
            "cache_mode": self.diagnostics.cache_mode,
            "stage_stats": {
                key: {
                    "processed": value.processed,
                    "skipped": value.skipped,
                    "cache_hits": value.cache_hits,
                    "cache_misses": value.cache_misses,
                    "duration_seconds": value.duration_seconds,
                }
                for key, value in self.diagnostics.stage_stats.items()
            },
            "cache_stats": self.diagnostics.cache_stats,
            "step_timings": self.diagnostics.step_timings,
            "image_io": self.diagnostics.image_io,
            "models": self.diagnostics.models,
            "database": self.diagnostics.database,
            "queues": self.diagnostics.queues,
            "startup_breakdown": self.diagnostics.startup_breakdown,
            "discovery_breakdown": self.diagnostics.discovery_breakdown,
            "model_phase_breakdown": self.diagnostics.model_phase_breakdown,
            "warnings": [warning.message for warning in self.warnings],
            "errors": [error.message for error in self.processing_errors],
            "not_people_picture_count": len(not_people_picture_images),
            "not_people_picture_images": not_people_picture_images,
            "images": [
                {
                    "path": image.path,
                    "status": image.status.value,
                    "primary_exclusion_reason": image.primary_exclusion_reason,
                    "excluded_at_stage": image.excluded_at_stage,
                    "warnings": list(image.warnings),
                    "completed_stages": sorted(image.completed_stages),
                    "duplicate_group_id": image.duplicate_group_id,
                    "duplicate_rank": image.duplicate_rank,
                    "is_keeper": image.is_duplicate_keeper,
                    "vibe_group_id": image.vibe_group_id,
                    "known_person_ids": list(image.known_person_ids),
                    "selected_for_deletion": image.selected_for_deletion,
                    "manual_override": (
                        None if image.manual_override is None else image.manual_override.value
                    ),
                    "people_picture_assessment": self._people_picture_assessment_payload(
                        image.people_picture_assessment
                    ),
                }
                for image in self.all_images
            ],
        }

    @staticmethod
    def _people_picture_assessment_payload(
        assessment: PeoplePictureAssessment | None,
    ) -> dict[str, Any] | None:
        if assessment is None:
            return None
        return {
            "is_people_picture": assessment.is_people_picture,
            "confidence": assessment.confidence,
            "primary_face_indices": list(assessment.primary_face_indices),
            "supporting_reasons": list(assessment.supporting_reasons),
            "detected_face_count": assessment.detected_face_count,
            "reliable_face_count": assessment.reliable_face_count,
            "primary_face_count": len(assessment.primary_face_indices),
            "largest_primary_face_area": assessment.largest_primary_face_area,
            "total_primary_face_area": assessment.total_primary_face_area,
            "largest_primary_face_span": assessment.largest_primary_face_span,
            "average_primary_centrality": assessment.average_primary_centrality,
            "best_primary_selection_score": assessment.best_primary_selection_score,
        }

    @classmethod
    def _not_people_picture_payload(cls, image: PipelineImageState) -> dict[str, Any]:
        assessment = cls._people_picture_assessment_payload(image.people_picture_assessment)
        return {
            "path": image.path,
            "warnings": list(image.warnings),
            "assessment": assessment,
        }
