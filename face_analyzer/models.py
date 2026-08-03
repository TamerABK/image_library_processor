from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .math_utils import clamp01
from .ranking import confidence_blended_score


class AssessmentStatus(str, Enum):
    ASSESSED = "assessed"
    PARTIAL = "partial"
    NOT_CONFIGURED = "not_configured"
    INVALID_INPUT = "invalid_input"
    LANDMARKS_MISSING = "landmarks_missing"
    FACE_TOO_SMALL = "face_too_small"
    POSE_TOO_EXTREME = "pose_too_extreme"
    LOW_SIGNAL = "low_signal"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_ERROR = "model_error"


class EyeLabel(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNCERTAIN = "uncertain"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class DetectedFace:
    """Detector output expected by the analyzer.

    Landmark order must be:
        left eye, right eye, nose, left mouth corner, right mouth corner.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class FaceGeometry:
    width: int
    height: int
    area: int
    relative_area: float
    visible_ratio: float
    minimum_dimension: int


@dataclass(frozen=True)
class MetricResult:
    """Keep raw measurement, absolute quality, confidence, and ranking separate."""

    raw_value: float | None
    quality_score: float | None
    confidence: float
    ranking_score: float | None

    @property
    def value(self) -> float:
        return 0.0 if self.quality_score is None else float(self.quality_score)

    def confidence_blended_ranking(self, prior: float) -> float:
        return confidence_blended_score(
            self.ranking_score,
            self.confidence,
            prior=prior,
        )


class MetricScore(MetricResult):
    """Compatibility helper for older tests and cache fixtures."""

    def __init__(self, value: float, confidence: float):
        score = clamp01(value)
        super().__init__(
            raw_value=score,
            quality_score=score,
            confidence=clamp01(confidence),
            ranking_score=score,
        )


@dataclass(frozen=True)
class FaceImageQuality:
    """Interpretable image-quality measurements for the aligned face crop."""

    focus_sharpness: MetricResult
    detail_availability: MetricResult
    sharpness: MetricResult
    exposure: MetricResult
    contrast: MetricResult

    laplacian_variance: float
    tenengrad_energy: float
    high_frequency_energy_ratio: float
    detail_availability_measure: float
    median_luminance: float
    p05_luminance: float
    p95_luminance: float
    dark_clip_ratio: float
    bright_clip_ratio: float
    usable_tonal_range: float
    clipping_score: float
    luminance_score: float
    tonal_information_score: float
    raw_exposure_score: float
    display_exposure_score: float
    shadow_detail_score: float
    highlight_detail_score: float
    tonal_balance_score: float
    p10_luminance: float
    p25_luminance: float
    p75_luminance: float
    p90_luminance: float
    broad_tonal_range: float
    interquartile_range: float
    broad_contrast_score: float
    interquartile_contrast_score: float
    local_contrast_raw: float
    local_contrast_score: float
    contrast_quality_score: float

    @property
    def focus_sharpness_score(self) -> float:
        return self.focus_sharpness.value

    @property
    def focus_ranking_score(self) -> float:
        return 0.0 if self.focus_sharpness.ranking_score is None else float(self.focus_sharpness.ranking_score)

    @property
    def detail_availability_score(self) -> float:
        return self.detail_availability.value

    @property
    def detail_availability_ranking_score(self) -> float:
        return 0.0 if self.detail_availability.ranking_score is None else float(self.detail_availability.ranking_score)

    @property
    def sharpness_score(self) -> float:
        return self.sharpness.value

    @property
    def sharpness_ranking_score(self) -> float:
        return 0.0 if self.sharpness.ranking_score is None else float(self.sharpness.ranking_score)

    @property
    def exposure_score(self) -> float:
        return self.display_exposure_score

    @property
    def exposure_ranking_score(self) -> float:
        return 0.0 if self.exposure.ranking_score is None else float(self.exposure.ranking_score)

    @property
    def contrast_score(self) -> float:
        return self.contrast_quality_score

    @property
    def contrast_ranking_score(self) -> float:
        return 0.0 if self.contrast.ranking_score is None else float(self.contrast.ranking_score)


@dataclass(frozen=True)
class HeadPose:
    yaw_degrees: float | None
    pitch_degrees: float | None
    roll_degrees: float | None
    confidence: float
    status: AssessmentStatus
    source: str

    @classmethod
    def unknown(
        cls,
        status: AssessmentStatus,
        *,
        roll_degrees: float | None = None,
        confidence: float = 0.0,
        source: str = "unavailable",
    ) -> "HeadPose":
        return cls(
            yaw_degrees=None,
            pitch_degrees=None,
            roll_degrees=roll_degrees,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            status=status,
            source=source,
        )


@dataclass(frozen=True)
class PoseQuality:
    metric: MetricResult
    yaw_score: float | None
    pitch_score: float | None
    roll_score: float | None


@dataclass(frozen=True)
class EyeMeasurement:
    open_probability: float | None
    label: EyeLabel
    confidence: float
    status: AssessmentStatus
    source_width: int = 0
    source_height: int = 0

    @classmethod
    def unknown(
        cls,
        status: AssessmentStatus,
        *,
        label: EyeLabel = EyeLabel.NOT_ASSESSED,
    ) -> "EyeMeasurement":
        return cls(
            open_probability=None,
            label=label,
            confidence=0.0,
            status=status,
        )


@dataclass(frozen=True)
class EyeState:
    left: EyeMeasurement
    right: EyeMeasurement
    combined_open_score: float | None
    confidence: float
    status: AssessmentStatus

    @property
    def has_confident_closed_eye(self) -> bool:
        return any(
            eye.label is EyeLabel.CLOSED and eye.confidence >= 0.60
            for eye in (self.left, self.right)
        )

    @classmethod
    def unknown(cls, status: AssessmentStatus) -> "EyeState":
        unknown_eye = EyeMeasurement.unknown(status)
        return cls(
            left=unknown_eye,
            right=unknown_eye,
            combined_open_score=None,
            confidence=0.0,
            status=status,
        )


@dataclass(frozen=True)
class FaceAnalysisResult:
    detector_confidence: float
    detector_metric: MetricResult
    geometry: FaceGeometry
    visible_face_metric: MetricResult
    image_quality: FaceImageQuality
    head_pose: HeadPose
    pose: PoseQuality
    eye_state: EyeState
    eyes: MetricResult
    eye_weight: float
    measurement_reliability: MetricResult

    # Global ranking score from calibrated metrics before duplicate-group context.
    global_selection_score: float

    # Duplicate-group relative ranking stays separate so unrelated folders do not
    # change the global score. Callers can blend it in only when a real group exists.
    group_relative_score: float
    final_group_score: float

    # Compatibility field. This equals final_group_score when group context exists,
    # otherwise it matches global_selection_score.
    selection_score: float

    # Whether the face is a useful recognition sample. Eye closure is excluded.
    embedding_utility_score: float

    warnings: tuple[str, ...] = ()

    @property
    def face_width(self) -> int:
        return self.geometry.width

    @property
    def face_height(self) -> int:
        return self.geometry.height

    @property
    def face_area(self) -> int:
        return self.geometry.area

    @property
    def relative_face_area(self) -> float:
        return self.geometry.relative_area

    @property
    def visible_face_ratio(self) -> float:
        return self.geometry.visible_ratio

    @property
    def sharpness_score(self) -> float:
        return self.image_quality.sharpness_score

    @property
    def sharpness_confidence(self) -> float:
        return self.image_quality.sharpness.confidence

    @property
    def exposure_score(self) -> float:
        return self.image_quality.exposure_score

    @property
    def contrast_score(self) -> float:
        return self.image_quality.contrast_score

    @property
    def laplacian_variance(self) -> float:
        return self.image_quality.laplacian_variance

    @property
    def tenengrad_energy(self) -> float:
        return self.image_quality.tenengrad_energy

    @property
    def pose_score(self) -> float | None:
        return self.pose.metric.quality_score

    @property
    def pose_ranking_score(self) -> float | None:
        return self.pose.metric.ranking_score

    @property
    def pose_confidence(self) -> float:
        return self.pose.metric.confidence

    @property
    def combined_eye_open_score(self) -> float | None:
        return self.eye_state.combined_open_score

    @property
    def eye_ranking_score(self) -> float | None:
        return self.eyes.ranking_score

    @property
    def eye_confidence(self) -> float:
        return self.eyes.confidence

    @property
    def measurement_reliability_score(self) -> float:
        return self.measurement_reliability.value
