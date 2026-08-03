from __future__ import annotations

import cv2
from pathlib import Path
from typing import Sequence

import numpy as np

from .classical_metrics import ClassicalFaceQualityAssessor
from .config import FaceAnalyzerConfig
from .eye_state import OpenClosedEyeOnnxEstimator
from .geometry import (
    aligned_or_fallback_crop,
    calculate_geometry,
    validate_image,
)
from .head_pose import FivePointRollEstimator, SixDRepNetOnnxEstimator
from .interfaces import EyeStateEstimator, FaceAnalyzer, HeadPoseEstimator
from .math_utils import clamp01, smoothstep
from .models import (
    AssessmentStatus,
    DetectedFace,
    EyeState,
    FaceAnalysisResult,
    FaceGeometry,
    HeadPose,
    MetricResult,
    PoseQuality,
)
from .onnx_session import Provider
from .ranking import (
    apply_calibration_curve,
    confidence_blended_score,
    weighted_sum,
)


def _pose_quality(
    pose: HeadPose,
    *,
    config: FaceAnalyzerConfig,
) -> PoseQuality:
    if (
        pose.yaw_degrees is None
        or pose.pitch_degrees is None
        or pose.roll_degrees is None
    ):
        return PoseQuality(
            metric=MetricResult(
                raw_value=None,
                quality_score=None,
                confidence=pose.confidence,
                ranking_score=None,
            ),
            yaw_score=None,
            pitch_score=None,
            roll_score=None,
        )

    yaw = abs(pose.yaw_degrees)
    pitch = abs(pose.pitch_degrees)
    roll = abs(pose.roll_degrees)

    # Broad plateaus preserve photographer-friendly interpretation while still
    # giving ranking enough room to separate modestly better poses.
    yaw_score = 1.0 - 0.70 * smoothstep(18.0, 68.0, yaw)
    pitch_score = 1.0 - 0.65 * smoothstep(14.0, 50.0, pitch)
    roll_score = 1.0 - 0.45 * smoothstep(12.0, 45.0, roll)
    pose_quality_score = clamp01(
        0.50 * yaw_score + 0.30 * pitch_score + 0.20 * roll_score
    )
    pose_ranking_score = apply_calibration_curve(
        pose_quality_score,
        config.calibration_profile.pose,
    )
    return PoseQuality(
        metric=MetricResult(
            raw_value=pose_quality_score,
            quality_score=pose_quality_score,
            confidence=pose.confidence,
            ranking_score=pose_ranking_score,
        ),
        yaw_score=yaw_score,
        pitch_score=pitch_score,
        roll_score=roll_score,
    )


class DefaultFaceAnalyzer(FaceAnalyzer):
    """Balanced face analysis for photo ranking and recognition gating."""

    def __init__(
        self,
        *,
        eye_state_model_path: str | Path | None = None,
        head_pose_model_path: str | Path | None = None,
        debug_output_dir: Path | None = None,
        providers: Sequence[Provider] | None = None,
        config: FaceAnalyzerConfig | None = None,
        eye_state_estimator: EyeStateEstimator | None = None,
        head_pose_estimator: HeadPoseEstimator | None = None,
    ) -> None:
        self._config = config or FaceAnalyzerConfig()
        self._quality_assessor = ClassicalFaceQualityAssessor(
            normalized_size=self._config.aligned_face_size,
            exposure_config=self._config.exposure,
            contrast_config=self._config.contrast,
            sharpness_config=self._config.sharpness,
            calibration_profile=self._config.calibration_profile,
        )

        if eye_state_estimator is not None and eye_state_model_path is not None:
            raise ValueError(
                "Pass either eye_state_estimator or eye_state_model_path, not both."
            )
        if head_pose_estimator is not None and head_pose_model_path is not None:
            raise ValueError(
                "Pass either head_pose_estimator or head_pose_model_path, not both."
            )

        self._eye_estimator = eye_state_estimator
        if self._eye_estimator is None and eye_state_model_path is not None:
            self._eye_estimator = OpenClosedEyeOnnxEstimator(
                eye_state_model_path,
                config=self._config,
                providers=providers,
                debug_output_dir=debug_output_dir,
            )

        self._head_pose_estimator = head_pose_estimator
        if self._head_pose_estimator is None:
            self._head_pose_estimator = (
                SixDRepNetOnnxEstimator(
                    head_pose_model_path,
                    providers=providers,
                )
                if head_pose_model_path is not None
                else FivePointRollEstimator()
            )

    def analyze(
        self,
        image: np.ndarray,
        face: DetectedFace,
    ) -> FaceAnalysisResult:
        validate_image(image)
        image_height, image_width = image.shape[:2]

        geometry = calculate_geometry(
            face.bbox,
            image_width=image_width,
            image_height=image_height,
        )
        aligned_face, alignment_confidence = aligned_or_fallback_crop(
            image,
            face.bbox,
            face.landmarks,
            output_size=self._config.aligned_face_size,
        )
        image_quality = self._quality_assessor.assess(
            aligned_face,
            original_face_minimum_dimension=geometry.minimum_dimension,
            alignment_confidence=alignment_confidence,
        )

        head_pose = self._safe_head_pose(image, face)
        pose = _pose_quality(head_pose, config=self._config)
        eye_state = self._safe_eye_state(
            image,
            face,
            head_pose=head_pose,
        )
        eyes, eye_weight = self._eye_metric(
            eye_state=eye_state,
            head_pose=head_pose,
        )

        detector_metric = MetricResult(
            raw_value=clamp01(float(face.confidence)),
            quality_score=clamp01(float(face.confidence)),
            confidence=1.0,
            ranking_score=apply_calibration_curve(
                clamp01(float(face.confidence)),
                self._config.calibration_profile.detector_confidence,
            ),
        )
        visible_face_metric = MetricResult(
            raw_value=geometry.visible_ratio,
            quality_score=geometry.visible_ratio,
            confidence=1.0,
            ranking_score=apply_calibration_curve(
                geometry.visible_ratio,
                self._config.calibration_profile.visible_face_ratio,
            ),
        )
        measurement_reliability = self._measurement_reliability(
            image_quality=image_quality,
            pose=pose,
            eyes=eyes,
        )
        global_selection_score = self._selection_score(
            image_quality=image_quality,
            pose=pose,
            eyes=eyes,
            eye_weight=eye_weight,
            detector_metric=detector_metric,
            visible_face_metric=visible_face_metric,
            measurement_reliability=measurement_reliability,
        )
        embedding_utility_score = self._embedding_score(
            face=face,
            geometry=geometry,
            image_quality=image_quality,
            pose=pose,
        )

        return FaceAnalysisResult(
            detector_confidence=clamp01(float(face.confidence)),
            detector_metric=detector_metric,
            geometry=geometry,
            visible_face_metric=visible_face_metric,
            image_quality=image_quality,
            head_pose=head_pose,
            pose=pose,
            eye_state=eye_state,
            eyes=eyes,
            eye_weight=eye_weight,
            measurement_reliability=measurement_reliability,
            global_selection_score=global_selection_score,
            group_relative_score=0.5,
            final_group_score=global_selection_score,
            selection_score=global_selection_score,
            embedding_utility_score=embedding_utility_score,
            warnings=self._warnings(
                geometry=geometry,
                image_quality=image_quality,
                pose=pose,
                eye_state=eye_state,
            ),
        )

    def _safe_head_pose(
        self,
        image: np.ndarray,
        face: DetectedFace,
    ) -> HeadPose:
        try:
            return self._head_pose_estimator.estimate(image, face)
        except (ValueError, RuntimeError, cv2.error):
            return HeadPose.unknown(
                AssessmentStatus.MODEL_ERROR,
                source=type(self._head_pose_estimator).__name__,
            )

    def _safe_eye_state(
        self,
        image: np.ndarray,
        face: DetectedFace,
        *,
        head_pose: HeadPose,
    ) -> EyeState:
        if self._eye_estimator is None:
            return EyeState.unknown(AssessmentStatus.NOT_CONFIGURED)
        try:
            return self._eye_estimator.estimate(
                image,
                face,
                head_pose=head_pose,
                face_index=getattr(face, "index", None),
            )
        except (ValueError, RuntimeError, cv2.error):
            return EyeState.unknown(AssessmentStatus.MODEL_ERROR)

    def _selection_score(
        self,
        *,
        image_quality,
        pose: PoseQuality,
        eyes: MetricResult,
        eye_weight: float,
        detector_metric: MetricResult,
        visible_face_metric: MetricResult,
        measurement_reliability: MetricResult,
    ) -> float:
        weights = self._config.ranking_weights
        priors = self._config.metric_priors

        entries = [
            (
                image_quality.sharpness.confidence_blended_ranking(priors.sharpness),
                weights.sharpness,
            ),
            (
                eyes.confidence_blended_ranking(priors.eyes),
                eye_weight,
            ),
            (
                pose.metric.confidence_blended_ranking(priors.pose),
                weights.pose,
            ),
            (
                image_quality.exposure.confidence_blended_ranking(priors.exposure),
                weights.exposure,
            ),
            (
                image_quality.contrast.confidence_blended_ranking(priors.contrast),
                weights.contrast,
            ),
            (
                detector_metric.confidence_blended_ranking(priors.detector_confidence),
                weights.detector_confidence,
            ),
            (
                measurement_reliability.confidence_blended_ranking(
                    priors.measurement_reliability
                ),
                weights.measurement_reliability,
            ),
            (
                visible_face_metric.confidence_blended_ranking(priors.visible_face),
                weights.visible_face,
            ),
            (
                image_quality.detail_availability.confidence_blended_ranking(
                    priors.detail_availability
                ),
                weights.detail_availability,
            ),
        ]
        return weighted_sum(entries)

    def _eye_metric(
        self,
        *,
        eye_state: EyeState,
        head_pose: HeadPose,
    ) -> tuple[MetricResult, float]:
        neutral_score = self._config.eye_neutral_score
        assessed_probabilities = [
            eye.open_probability
            for eye in (eye_state.left, eye_state.right)
            if eye.status is AssessmentStatus.ASSESSED
            and eye.open_probability is not None
        ]
        rounded_probabilities = [
            round(float(probability), self._config.eye_probability_rounding_decimals)
            for probability in assessed_probabilities
        ]
        if rounded_probabilities:
            combined_open_score = clamp01(
                0.70 * min(rounded_probabilities)
                + 0.30 * float(np.mean(rounded_probabilities))
            )
            confidence = (
                float(np.mean([eye.confidence for eye in (eye_state.left, eye_state.right) if eye.status is AssessmentStatus.ASSESSED]))
                if len(rounded_probabilities) > 1
                else float(
                    next(
                        eye.confidence
                        for eye in (eye_state.left, eye_state.right)
                        if eye.status is AssessmentStatus.ASSESSED
                    )
                )
            )
            ranking_score = apply_calibration_curve(
                combined_open_score,
                self._config.calibration_profile.eyes,
            )
            metric = MetricResult(
                raw_value=combined_open_score,
                quality_score=combined_open_score,
                confidence=confidence,
                ranking_score=ranking_score,
            )
        else:
            metric = MetricResult(
                raw_value=None,
                quality_score=None,
                confidence=0.0,
                ranking_score=None,
            )

        eye_weight = self._eye_weight_scale(head_pose)
        if metric.quality_score is None:
            blended_quality = confidence_blended_score(
                None,
                0.0,
                prior=neutral_score,
            )
            blended_ranking = apply_calibration_curve(
                blended_quality,
                self._config.calibration_profile.eyes,
            )
            metric = MetricResult(
                raw_value=None,
                quality_score=None,
                confidence=0.0,
                ranking_score=blended_ranking,
            )
        return metric, eye_weight

    def _eye_weight_scale(
        self,
        head_pose: HeadPose,
    ) -> float:
        yaw = 0.0 if head_pose.yaw_degrees is None else abs(head_pose.yaw_degrees)
        if yaw <= self._config.eye_full_weight_yaw_degrees:
            return self._config.ranking_weights.eyes
        if yaw >= self._config.eye_minimal_weight_yaw_degrees:
            return (
                self._config.ranking_weights.eyes
                * self._config.eye_minimum_weight_at_high_yaw
            )

        reduced_scale = 1.0 - (
            1.0 - self._config.eye_minimum_weight_at_high_yaw
        ) * smoothstep(
            self._config.eye_full_weight_yaw_degrees,
            self._config.eye_minimal_weight_yaw_degrees,
            yaw,
        )
        return self._config.ranking_weights.eyes * reduced_scale

    def _measurement_reliability(
        self,
        *,
        image_quality,
        pose: PoseQuality,
        eyes: MetricResult,
    ) -> MetricResult:
        reliability_score = weighted_sum(
            [
                (image_quality.sharpness.confidence, 0.32),
                (image_quality.exposure.confidence, 0.20),
                (image_quality.contrast.confidence, 0.12),
                (pose.metric.confidence, 0.18),
                (eyes.confidence, 0.18),
            ]
        )
        return MetricResult(
            raw_value=reliability_score,
            quality_score=reliability_score,
            confidence=1.0,
            ranking_score=reliability_score,
        )

    def _embedding_score(
        self,
        *,
        face: DetectedFace,
        geometry: FaceGeometry,
        image_quality,
        pose: PoseQuality,
    ) -> float:
        weights = self._config.embedding_weights
        return weighted_sum(
            [
                (image_quality.sharpness.value, weights.sharpness),
                (pose.metric.quality_score, weights.pose),
                (geometry.visible_ratio, weights.visible_face),
                (
                    clamp01(float(face.confidence)),
                    weights.detector_confidence,
                ),
            ]
        )

    def _warnings(
        self,
        *,
        geometry: FaceGeometry,
        image_quality,
        pose: PoseQuality,
        eye_state: EyeState,
    ) -> tuple[str, ...]:
        warnings: list[str] = []

        if geometry.visible_ratio < 0.82:
            warnings.append("face_partially_outside_image")
        if geometry.minimum_dimension < self._config.minimum_reliable_face_size:
            warnings.append("face_too_small_for_reliable_analysis")

        if (
            image_quality.sharpness.confidence >= 0.45
            and image_quality.focus_sharpness_score < 0.24
        ):
            warnings.append("face_soft_or_blurred")
        if (
            image_quality.exposure.confidence >= 0.45
            and image_quality.exposure_score < 0.27
        ):
            warnings.append("face_exposure_problem")

        if (
            pose.yaw_score is not None
            and pose.yaw_score < 0.40
        ) or (
            pose.pitch_score is not None
            and pose.pitch_score < 0.45
        ):
            warnings.append("extreme_head_pose")

        if eye_state.has_confident_closed_eye:
            warnings.append("closed_eye_detected")

        return tuple(warnings)
