from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .config import FaceAnalyzerConfig
from .default_face_analyzer import DefaultFaceAnalyzer
from .eye_state import OpenClosedEyeOnnxEstimator
from .geometry import extract_level_eye_crops, level_eye_crop_geometry
from .head_pose import SixDRepNetOnnxEstimator
from .models import (
    AssessmentStatus,
    DetectedFace,
    EyeLabel,
    EyeMeasurement,
    EyeState,
    FaceImageQuality,
    HeadPose,
    MetricResult,
    MetricScore,
    PoseQuality,
)
from .ranking import apply_group_relative_ranking, MetricPriorConfig, RankingWeights


class _FakeModel:
    def __init__(self, outputs: list[np.ndarray]):
        self._outputs = list(outputs)
        self._last_output: np.ndarray | None = None
        self.inputs: list[np.ndarray] = []
        self.input = SimpleNamespace(shape=["batch", 3, 32, 32])

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        self.inputs.append(np.asarray(tensor))
        if self._outputs:
            output = self._outputs.pop(0)
            self._last_output = np.asarray(output, dtype=np.float32)
        elif self._last_output is not None:
            output = self._last_output
        else:
            raise IndexError("No fake model outputs configured.")
        return [np.asarray(output, dtype=np.float32)]

    def run_all_outputs(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.run(tensor)


class _StubHeadPoseEstimator:
    def __init__(self, pose: HeadPose):
        self._pose = pose

    def estimate(self, image: np.ndarray, face: DetectedFace) -> HeadPose:
        return self._pose


class _StubEyeStateEstimator:
    def __init__(self, eye_state: EyeState):
        self._eye_state = eye_state

    def estimate(
        self,
        image: np.ndarray,
        face: DetectedFace,
        *,
        head_pose: HeadPose,
        face_index: int | None = None,
    ) -> EyeState:
        return self._eye_state


class _BatchOnlyHeadPoseEstimator:
    def __init__(self, pose: HeadPose):
        self._pose = pose
        self.calls: list[int] = []

    def estimate(self, image: np.ndarray, face: DetectedFace) -> HeadPose:
        raise AssertionError("estimate() should not be called when estimate_many() is available")

    def estimate_many(self, image: np.ndarray, faces: list[DetectedFace]) -> list[HeadPose]:
        self.calls.append(len(faces))
        return [self._pose for _face in faces]


class _BatchOnlyEyeStateEstimator:
    def __init__(self, eye_state: EyeState):
        self._eye_state = eye_state
        self.calls: list[int] = []

    def estimate(
        self,
        image: np.ndarray,
        face: DetectedFace,
        *,
        head_pose: HeadPose,
        face_index: int | None = None,
    ) -> EyeState:
        raise AssertionError("estimate() should not be called when estimate_many() is available")

    def estimate_many(
        self,
        image: np.ndarray,
        faces: list[DetectedFace],
        *,
        head_poses: list[HeadPose],
        face_indices: list[int | None] | None = None,
    ) -> list[EyeState]:
        self.calls.append(len(faces))
        return [self._eye_state for _face in faces]


class _StubQualityAssessor:
    def assess(
        self,
        aligned_face: np.ndarray,
        *,
        original_face_minimum_dimension: int,
        alignment_confidence: float,
    ) -> FaceImageQuality:
        return FaceImageQuality(
            focus_sharpness=MetricResult(
                raw_value=0.80,
                quality_score=0.80,
                confidence=1.0,
                ranking_score=0.78,
            ),
            detail_availability=MetricResult(
                raw_value=90.0,
                quality_score=0.74,
                confidence=1.0,
                ranking_score=0.70,
            ),
            sharpness=MetricResult(
                raw_value=0.80,
                quality_score=0.79,
                confidence=1.0,
                ranking_score=0.76,
            ),
            exposure=MetricResult(
                raw_value=0.76,
                quality_score=0.70,
                confidence=1.0,
                ranking_score=0.72,
            ),
            contrast=MetricResult(
                raw_value=0.65,
                quality_score=0.65,
                confidence=1.0,
                ranking_score=0.67,
            ),
            laplacian_variance=100.0,
            tenengrad_energy=50.0,
            high_frequency_energy_ratio=0.31,
            detail_availability_measure=90.0,
            median_luminance=0.45,
            p05_luminance=0.22,
            p95_luminance=0.78,
            dark_clip_ratio=0.02,
            bright_clip_ratio=0.01,
            usable_tonal_range=56.0,
            clipping_score=0.91,
            luminance_score=0.74,
            tonal_information_score=0.69,
            raw_exposure_score=0.76,
            display_exposure_score=0.70,
            shadow_detail_score=0.93,
            highlight_detail_score=0.95,
            tonal_balance_score=0.88,
            p10_luminance=0.24,
            p25_luminance=0.34,
            p75_luminance=0.62,
            p90_luminance=0.72,
            broad_tonal_range=48.0,
            interquartile_range=28.0,
            broad_contrast_score=0.50,
            interquartile_contrast_score=0.52,
            local_contrast_raw=14.0,
            local_contrast_score=0.54,
            contrast_quality_score=0.65,
        )


def _sample_face() -> DetectedFace:
    return DetectedFace(
        bbox=(20, 20, 70, 90),
        confidence=0.95,
        landmarks=np.asarray(
            [
                [40.0, 46.0],
                [74.0, 52.0],
                [58.0, 66.0],
                [44.0, 90.0],
                [72.0, 92.0],
            ],
            dtype=np.float32,
        ),
    )


def _sample_image() -> np.ndarray:
    image = np.zeros((140, 140, 3), dtype=np.uint8)
    image[:] = (120, 120, 120)
    cv2.circle(image, (40, 46), 7, (255, 255, 255), -1)
    cv2.circle(image, (74, 52), 7, (255, 255, 255), -1)
    return image


def _assessed_eye(
    probability: float,
    *,
    confidence: float = 0.8,
) -> EyeMeasurement:
    is_open = probability >= 0.5
    return EyeMeasurement(
        open_probability=probability,
        label=EyeLabel.OPEN if is_open else EyeLabel.CLOSED,
        confidence=confidence,
        status=AssessmentStatus.ASSESSED,
        source_width=18,
        source_height=13,
    )


def _sample_eye_crop(
    *,
    height: int = 13,
    width: int = 18,
) -> np.ndarray:
    source_eye_crop = np.full((height, width, 3), 126, dtype=np.uint8)
    source_eye_crop[:, : width // 3] = (84, 84, 84)
    source_eye_crop[:, (2 * width) // 3 :] = (176, 176, 176)
    cv2.line(
        source_eye_crop,
        (2, height // 2),
        (width - 3, height // 2),
        (220, 220, 220),
        1,
    )
    return source_eye_crop


class EyeStateTests(unittest.TestCase):
    def test_crop_sizing_uses_inter_eye_distance(self) -> None:
        face = _sample_face()
        image = _sample_image()
        eye_distance = float(
            np.linalg.norm(face.landmarks[1] - face.landmarks[0])
        )
        crop_geometry = level_eye_crop_geometry(
            face.landmarks,
            width_ratio=0.58,
            height_ratio=0.38,
        )

        self.assertIsNotNone(crop_geometry)
        self.assertEqual(
            crop_geometry.source_width,
            max(12, int(round(eye_distance * 0.58))),
        )
        self.assertEqual(
            crop_geometry.source_height,
            max(8, int(round(eye_distance * 0.38))),
        )
        crops = extract_level_eye_crops(
            image,
            face.landmarks,
            width_ratio=0.58,
            height_ratio=0.38,
        )
        self.assertIsNotNone(crops)
        left_source_eye_crop, right_source_eye_crop = crops or (None, None)
        self.assertEqual(left_source_eye_crop.shape[:2], (crop_geometry.source_height, crop_geometry.source_width))
        self.assertEqual(right_source_eye_crop.shape[:2], (crop_geometry.source_height, crop_geometry.source_width))

    def test_roll_correction_levels_eye_centers(self) -> None:
        face = _sample_face()
        crop_geometry = level_eye_crop_geometry(
            face.landmarks,
            width_ratio=0.58,
            height_ratio=0.38,
        )

        self.assertIsNotNone(crop_geometry)
        self.assertAlmostEqual(
            float(crop_geometry.level_eye_centers[0][1]),
            float(crop_geometry.level_eye_centers[1][1]),
            places=4,
        )

    def test_preserves_source_dimensions_before_resize(self) -> None:
        model = _FakeModel([np.asarray([0.95, 0.05], dtype=np.float32)])
        estimator = OpenClosedEyeOnnxEstimator(
            "onnx_models/open_closed_eye.onnx",
            config=FaceAnalyzerConfig(),
            model=model,
        )
        source_eye_crop = np.full((13, 18, 3), 130, dtype=np.uint8)

        measurement, _debug = estimator._estimate_one(
            source_eye_crop=source_eye_crop,
            side_name="left",
        )

        self.assertEqual(measurement.source_width, 18)
        self.assertEqual(measurement.source_height, 13)
        self.assertEqual(model.inputs[0].shape, (1, 3, 32, 32))

    def test_model_input_preserves_bgr_order_and_normalization(self) -> None:
        model = _FakeModel([np.asarray([0.95, 0.05], dtype=np.float32)])
        estimator = OpenClosedEyeOnnxEstimator(
            "onnx_models/open_closed_eye.onnx",
            config=FaceAnalyzerConfig(),
            model=model,
        )
        source_eye_crop = np.zeros((32, 32, 3), dtype=np.uint8)
        source_eye_crop[:, :, 0] = 10
        source_eye_crop[:, :, 1] = 20
        source_eye_crop[:, :, 2] = 30

        estimator._estimate_one(
            source_eye_crop=source_eye_crop,
            side_name="left",
        )

        tensor = model.inputs[0]
        self.assertEqual(tensor.shape, (1, 3, 32, 32))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), (10.0 - 127.0) / 255.0, places=6)
        self.assertAlmostEqual(float(tensor[0, 1, 0, 0]), (20.0 - 127.0) / 255.0, places=6)
        self.assertAlmostEqual(float(tensor[0, 2, 0, 0]), (30.0 - 127.0) / 255.0, places=6)
        self.assertEqual(len(model.inputs), 1)

    def test_output_class_order_treats_index_one_as_open(self) -> None:
        model = _FakeModel([np.asarray([0.03, 0.97], dtype=np.float32)])
        estimator = OpenClosedEyeOnnxEstimator(
            "onnx_models/open_closed_eye.onnx",
            config=FaceAnalyzerConfig(),
            model=model,
        )
        measurement, _debug = estimator._estimate_one(
            source_eye_crop=_sample_eye_crop(),
            side_name="left",
        )

        self.assertEqual(measurement.label, EyeLabel.OPEN)
        self.assertGreater(measurement.open_probability, 0.90)

    def test_logits_and_probabilities_are_both_supported(self) -> None:
        estimator_from_logits = OpenClosedEyeOnnxEstimator(
            "onnx_models/open_closed_eye.onnx",
            config=FaceAnalyzerConfig(),
            model=_FakeModel([np.asarray([-3.0, 3.0], dtype=np.float32)]),
        )
        estimator_from_probabilities = OpenClosedEyeOnnxEstimator(
            "onnx_models/open_closed_eye.onnx",
            config=FaceAnalyzerConfig(),
            model=_FakeModel([np.asarray([0.04, 0.96], dtype=np.float32)]),
        )

        logits_measurement, _debug_logits = estimator_from_logits._estimate_one(
            source_eye_crop=_sample_eye_crop(),
            side_name="left",
        )
        probability_measurement, _debug_probs = estimator_from_probabilities._estimate_one(
            source_eye_crop=_sample_eye_crop(),
            side_name="left",
        )

        self.assertEqual(logits_measurement.label, EyeLabel.OPEN)
        self.assertEqual(probability_measurement.label, EyeLabel.OPEN)
        self.assertGreater(logits_measurement.open_probability, 0.90)
        self.assertGreater(probability_measurement.open_probability, 0.90)

    def test_low_resolution_crop_defaults_to_low_confidence(self) -> None:
        estimator = OpenClosedEyeOnnxEstimator(
            "onnx_models/open_closed_eye.onnx",
            config=FaceAnalyzerConfig(),
            model=_FakeModel([np.asarray([0.01, 0.99], dtype=np.float32)]),
        )

        measurement, _debug = estimator._estimate_one(
            source_eye_crop=np.full((7, 11, 3), 130, dtype=np.uint8),
            side_name="left",
        )

        self.assertEqual(measurement.status, AssessmentStatus.LOW_CONFIDENCE)
        self.assertEqual(measurement.label, EyeLabel.UNCERTAIN)
        self.assertNotEqual(measurement.label, EyeLabel.CLOSED)

    def test_combined_eye_state_uses_only_assessed_eyes(self) -> None:
        estimator = OpenClosedEyeOnnxEstimator(
            "onnx_models/open_closed_eye.onnx",
            config=FaceAnalyzerConfig(),
            model=_FakeModel([]),
        )
        face = _sample_face()
        image = _sample_image()
        assessed_eye = _assessed_eye(0.96)
        uncertain_eye = EyeMeasurement(
            open_probability=0.52,
            label=EyeLabel.UNCERTAIN,
            confidence=0.20,
            status=AssessmentStatus.LOW_CONFIDENCE,
            source_width=18,
            source_height=13,
        )
        outputs = iter([assessed_eye, uncertain_eye])
        estimator._estimate_one = (  # type: ignore[method-assign]
            lambda **kwargs: (next(outputs), None)
        )

        eye_state = estimator.estimate(
            image,
            face,
            head_pose=HeadPose(0.0, 0.0, 0.0, 1.0, AssessmentStatus.ASSESSED, "test"),
        )

        self.assertEqual(eye_state.status, AssessmentStatus.PARTIAL)
        self.assertAlmostEqual(eye_state.combined_open_score, assessed_eye.open_probability)

    def test_combined_eye_state_without_assessed_eyes_has_no_score(self) -> None:
        estimator = OpenClosedEyeOnnxEstimator(
            "onnx_models/open_closed_eye.onnx",
            config=FaceAnalyzerConfig(),
            model=_FakeModel([]),
        )
        face = _sample_face()
        image = _sample_image()
        low_confidence_eye = EyeMeasurement(
            open_probability=0.52,
            label=EyeLabel.UNCERTAIN,
            confidence=0.20,
            status=AssessmentStatus.LOW_CONFIDENCE,
            source_width=18,
            source_height=13,
        )
        outputs = iter([low_confidence_eye, low_confidence_eye])
        estimator._estimate_one = (  # type: ignore[method-assign]
            lambda **kwargs: (next(outputs), None)
        )

        eye_state = estimator.estimate(
            image,
            face,
            head_pose=HeadPose(0.0, 0.0, 0.0, 1.0, AssessmentStatus.ASSESSED, "test"),
        )

        self.assertEqual(eye_state.status, AssessmentStatus.LOW_CONFIDENCE)
        self.assertIsNone(eye_state.combined_open_score)

    def test_debug_output_saves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _FakeModel([
                np.asarray([0.96, 0.04], dtype=np.float32),
                np.asarray([0.05, 0.95], dtype=np.float32),
            ])
            estimator = OpenClosedEyeOnnxEstimator(
                "onnx_models/open_closed_eye.onnx",
                config=FaceAnalyzerConfig(),
                model=model,
                debug_output_dir=Path(tmpdir),
            )
            estimator.estimate(
                _sample_image(),
                _sample_face(),
                head_pose=HeadPose(0.0, 0.0, 0.0, 1.0, AssessmentStatus.ASSESSED, "test"),
                face_index=3,
            )

            generated = sorted(path.name for path in Path(tmpdir).glob("*.png"))
            self.assertTrue(any("face3_left" in name for name in generated))
            self.assertTrue(any("face3_right" in name for name in generated))
            self.assertTrue(any("annotated" in name for name in generated))

    def test_estimate_many_returns_matching_eye_states(self) -> None:
        model = _FakeModel([
            np.asarray([0.05, 0.95], dtype=np.float32),
            np.asarray([0.04, 0.96], dtype=np.float32),
            np.asarray([0.97, 0.03], dtype=np.float32),
            np.asarray([0.96, 0.04], dtype=np.float32),
        ])
        model.input = SimpleNamespace(shape=[1, 3, 32, 32])
        estimator = OpenClosedEyeOnnxEstimator(
            "onnx_models/open_closed_eye.onnx",
            config=FaceAnalyzerConfig(),
            model=model,
        )

        eye_states = estimator.estimate_many(
            _sample_image(),
            [_sample_face(), _sample_face()],
            head_poses=[
                HeadPose(0.0, 0.0, 0.0, 1.0, AssessmentStatus.ASSESSED, "test"),
                HeadPose(0.0, 0.0, 0.0, 1.0, AssessmentStatus.ASSESSED, "test"),
            ],
        )

        self.assertEqual(len(eye_states), 2)
        self.assertEqual(len(model.inputs), 4)
        self.assertEqual(eye_states[0].left.label, EyeLabel.OPEN)
        self.assertEqual(eye_states[0].right.label, EyeLabel.OPEN)
        self.assertEqual(eye_states[1].left.label, EyeLabel.CLOSED)
        self.assertEqual(eye_states[1].right.label, EyeLabel.CLOSED)


class HeadPoseBatchTests(unittest.TestCase):
    def test_estimate_many_batches_dynamic_model_inputs(self) -> None:
        identity = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
        model = _FakeModel([np.repeat(identity, 2, axis=0)])
        model.input = SimpleNamespace(shape=["batch", 3, 224, 224])
        estimator = SixDRepNetOnnxEstimator(
            None,
            model=model,
        )

        poses = estimator.estimate_many(
            _sample_image(),
            [_sample_face(), _sample_face()],
        )

        self.assertEqual(len(poses), 2)
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.inputs[0].shape[0], 2)
        self.assertTrue(all(pose.status is AssessmentStatus.ASSESSED for pose in poses))


class AnalyzerScoreTests(unittest.TestCase):
    def _build_analyzer(self, eye_state: EyeState, *, yaw_degrees: float = 0.0) -> DefaultFaceAnalyzer:
        analyzer = DefaultFaceAnalyzer(
            eye_state_estimator=_StubEyeStateEstimator(eye_state),
            head_pose_estimator=_StubHeadPoseEstimator(
                HeadPose(
                    yaw_degrees=yaw_degrees,
                    pitch_degrees=0.0,
                    roll_degrees=0.0,
                    confidence=1.0,
                    status=AssessmentStatus.ASSESSED,
                    source="stub",
                )
            ),
        )
        analyzer._quality_assessor = _StubQualityAssessor()
        return analyzer

    def test_unavailable_eye_state_does_not_improve_selection_score(self) -> None:
        open_eye_state = EyeState(
            left=_assessed_eye(0.95),
            right=_assessed_eye(0.94),
            combined_open_score=0.945,
            confidence=0.85,
            status=AssessmentStatus.ASSESSED,
        )
        unavailable_eye_state = EyeState.unknown(AssessmentStatus.NOT_CONFIGURED)

        open_result = self._build_analyzer(open_eye_state).analyze(
            _sample_image(),
            _sample_face(),
        )
        unavailable_result = self._build_analyzer(unavailable_eye_state).analyze(
            _sample_image(),
            _sample_face(),
        )

        self.assertLess(unavailable_result.selection_score, open_result.selection_score)

    def test_embedding_utility_is_independent_of_eye_state(self) -> None:
        open_eye_state = EyeState(
            left=_assessed_eye(0.95),
            right=_assessed_eye(0.94),
            combined_open_score=0.945,
            confidence=0.85,
            status=AssessmentStatus.ASSESSED,
        )
        closed_eye_state = EyeState(
            left=_assessed_eye(0.05),
            right=_assessed_eye(0.06),
            combined_open_score=0.055,
            confidence=0.85,
            status=AssessmentStatus.ASSESSED,
        )

        open_result = self._build_analyzer(open_eye_state).analyze(
            _sample_image(),
            _sample_face(),
        )
        closed_result = self._build_analyzer(closed_eye_state).analyze(
            _sample_image(),
            _sample_face(),
        )

        self.assertAlmostEqual(
            open_result.embedding_utility_score,
            closed_result.embedding_utility_score,
            places=6,
        )

    def test_eye_weight_reduces_with_high_yaw(self) -> None:
        open_eye_state = EyeState(
            left=_assessed_eye(0.95),
            right=_assessed_eye(0.94),
            combined_open_score=0.945,
            confidence=0.85,
            status=AssessmentStatus.ASSESSED,
        )
        low_yaw_result = self._build_analyzer(open_eye_state, yaw_degrees=10.0).analyze(
            _sample_image(),
            _sample_face(),
        )
        high_yaw_result = self._build_analyzer(open_eye_state, yaw_degrees=55.0).analyze(
            _sample_image(),
            _sample_face(),
        )

        self.assertLess(high_yaw_result.selection_score, low_yaw_result.selection_score)

    def test_group_relative_ranking_only_changes_scores_within_group(self) -> None:
        open_eye_state = EyeState(
            left=_assessed_eye(0.95),
            right=_assessed_eye(0.94),
            combined_open_score=0.945,
            confidence=0.85,
            status=AssessmentStatus.ASSESSED,
        )
        better = self._build_analyzer(open_eye_state).analyze(
            _sample_image(),
            _sample_face(),
        )

        class _SofterQualityAssessor(_StubQualityAssessor):
            def assess(self, *args, **kwargs) -> FaceImageQuality:
                quality = super().assess(*args, **kwargs)
                return FaceImageQuality(
                    focus_sharpness=MetricResult(
                        raw_value=0.52,
                        quality_score=0.52,
                        confidence=1.0,
                        ranking_score=0.50,
                    ),
                    detail_availability=quality.detail_availability,
                    sharpness=MetricResult(
                        raw_value=0.52,
                        quality_score=0.58,
                        confidence=1.0,
                        ranking_score=0.56,
                    ),
                    exposure=quality.exposure,
                    contrast=quality.contrast,
                    laplacian_variance=quality.laplacian_variance,
                    tenengrad_energy=quality.tenengrad_energy,
                    high_frequency_energy_ratio=quality.high_frequency_energy_ratio,
                    detail_availability_measure=quality.detail_availability_measure,
                    median_luminance=quality.median_luminance,
                    p05_luminance=quality.p05_luminance,
                    p95_luminance=quality.p95_luminance,
                    dark_clip_ratio=quality.dark_clip_ratio,
                    bright_clip_ratio=quality.bright_clip_ratio,
                    usable_tonal_range=quality.usable_tonal_range,
                    clipping_score=quality.clipping_score,
                    luminance_score=quality.luminance_score,
                    tonal_information_score=quality.tonal_information_score,
                    raw_exposure_score=quality.raw_exposure_score,
                    display_exposure_score=quality.display_exposure_score,
                    shadow_detail_score=quality.shadow_detail_score,
                    highlight_detail_score=quality.highlight_detail_score,
                    tonal_balance_score=quality.tonal_balance_score,
                    p10_luminance=quality.p10_luminance,
                    p25_luminance=quality.p25_luminance,
                    p75_luminance=quality.p75_luminance,
                    p90_luminance=quality.p90_luminance,
                    broad_tonal_range=quality.broad_tonal_range,
                    interquartile_range=quality.interquartile_range,
                    broad_contrast_score=quality.broad_contrast_score,
                    interquartile_contrast_score=quality.interquartile_contrast_score,
                    local_contrast_raw=quality.local_contrast_raw,
                    local_contrast_score=quality.local_contrast_score,
                    contrast_quality_score=quality.contrast_quality_score,
                )

        softer_analyzer = self._build_analyzer(open_eye_state)
        softer_analyzer._quality_assessor = _SofterQualityAssessor()
        softer = softer_analyzer.analyze(
            _sample_image(),
            _sample_face(),
        )

        group_scores = apply_group_relative_ranking(
            [better, softer],
            weights=RankingWeights(),
            priors=MetricPriorConfig(),
        )
        self.assertEqual(len(group_scores), 2)
        self.assertNotEqual(group_scores[0].group_relative_score, 0.5)
        self.assertNotEqual(group_scores[1].group_relative_score, 0.5)
        self.assertAlmostEqual(better.global_selection_score, group_scores[0].global_selection_score)
        self.assertAlmostEqual(softer.global_selection_score, group_scores[1].global_selection_score)

    def test_analyze_many_uses_batch_estimators(self) -> None:
        pose = HeadPose(
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            confidence=1.0,
            status=AssessmentStatus.ASSESSED,
            source="stub",
        )
        eye_state = EyeState(
            left=_assessed_eye(0.95),
            right=_assessed_eye(0.94),
            combined_open_score=0.945,
            confidence=0.85,
            status=AssessmentStatus.ASSESSED,
        )
        head_pose_estimator = _BatchOnlyHeadPoseEstimator(pose)
        eye_state_estimator = _BatchOnlyEyeStateEstimator(eye_state)
        analyzer = DefaultFaceAnalyzer(
            eye_state_estimator=eye_state_estimator,
            head_pose_estimator=head_pose_estimator,
        )
        analyzer._quality_assessor = _StubQualityAssessor()

        results = analyzer.analyze_many(
            _sample_image(),
            [_sample_face(), _sample_face()],
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(head_pose_estimator.calls, [2])
        self.assertEqual(eye_state_estimator.calls, [2])


if __name__ == "__main__":
    unittest.main()
