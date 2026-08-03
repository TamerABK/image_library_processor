from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import FaceAnalyzerConfig
from .geometry import extract_level_eye_crops, normalize_bbox, to_gray
from .math_utils import EPSILON, clamp01, smoothstep, softmax
from .models import (
    AssessmentStatus,
    DetectedFace,
    EyeLabel,
    EyeMeasurement,
    EyeState,
    HeadPose,
)
from .onnx_session import OnnxModel, Provider


@dataclass(frozen=True)
class EyeInferenceDebug:
    model_input_image: np.ndarray
    tensor: np.ndarray
    raw_outputs_bgr: tuple[np.ndarray, ...]
    raw_outputs_rgb: tuple[np.ndarray, ...]


class OpenClosedEyeOnnxEstimator:
    """Open Model Zoo open-closed-eye-0001 adapter.

    Input:  1 x 3 x 32 x 32 BGR
    Output observed from the shipped ONNX: [closed, open]
    Normalization: (pixel - 127) / 255
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        config: FaceAnalyzerConfig,
        providers: Sequence[Provider] | None = None,
        debug_output_dir: Path | None = None,
        model: OnnxModel | None = None,
    ) -> None:
        self._model = model or OnnxModel(model_path, providers=providers)
        self._config = config
        self._debug_output_dir = debug_output_dir

    def estimate(
        self,
        image: np.ndarray,
        face: DetectedFace,
        *,
        head_pose: HeadPose,
        face_index: int | None = None,
    ) -> EyeState:
        _, _, face_width, face_height = normalize_bbox(face.bbox)
        if min(face_width, face_height) < self._config.minimum_reliable_face_size:
            return EyeState.unknown(AssessmentStatus.FACE_TOO_SMALL)

        if self._pose_is_too_extreme(head_pose):
            return EyeState.unknown(AssessmentStatus.POSE_TOO_EXTREME)

        crops = extract_level_eye_crops(
            image,
            face.landmarks,
            width_ratio=self._config.eye_crop_width_ratio,
            height_ratio=self._config.eye_crop_height_ratio,
        )
        if crops is None:
            return EyeState.unknown(AssessmentStatus.LANDMARKS_MISSING)

        left, left_debug = self._estimate_one(
            source_eye_crop=crops[0],
        )
        right, right_debug = self._estimate_one(
            source_eye_crop=crops[1],
        )

        if self._debug_output_dir is not None:
            self._save_debug_outputs(
                image=image,
                face=face,
                face_index=face_index,
                left_source_eye_crop=crops[0],
                right_source_eye_crop=crops[1],
                left_measurement=left,
                right_measurement=right,
                left_debug=left_debug,
                right_debug=right_debug,
            )

        return self._combine_eye_measurements(left=left, right=right)

    def _pose_is_too_extreme(self, pose: HeadPose) -> bool:
        return bool(
            pose.yaw_degrees is not None
            and abs(pose.yaw_degrees) > self._config.maximum_eye_yaw_degrees
        ) or bool(
            pose.pitch_degrees is not None
            and abs(pose.pitch_degrees) > self._config.maximum_eye_pitch_degrees
        )

    def _estimate_one(
        self,
        *,
        source_eye_crop: np.ndarray,
    ) -> tuple[EyeMeasurement, EyeInferenceDebug | None]:
        if source_eye_crop.size == 0:
            return EyeMeasurement.unknown(AssessmentStatus.INVALID_INPUT), None

        source_height, source_width = source_eye_crop.shape[:2]
        resolution_confidence, exposure_confidence, contrast_confidence = (
            self._estimate_crop_quality_components(source_eye_crop)
        )
        crop_quality_confidence = clamp01(
            0.36 * resolution_confidence
            + 0.32 * exposure_confidence
            + 0.32 * contrast_confidence
        )

        if (
            source_width < self._config.eye_source_min_width
            or source_height < self._config.eye_source_min_height
        ):
            return (
                EyeMeasurement(
                    open_probability=None,
                    label=EyeLabel.UNCERTAIN,
                    confidence=crop_quality_confidence,
                    status=AssessmentStatus.LOW_CONFIDENCE,
                    source_width=source_width,
                    source_height=source_height,
                ),
                None,
            )

        if (
            crop_quality_confidence
            < self._config.eye_low_signal_confidence_threshold
        ):
            return (
                EyeMeasurement(
                    open_probability=None,
                    label=EyeLabel.UNCERTAIN,
                    confidence=crop_quality_confidence,
                    status=AssessmentStatus.LOW_CONFIDENCE,
                    source_width=source_width,
                    source_height=source_height,
                ),
                None,
            )

        model_input_image, tensor_bgr = self._prepare_model_input(
            source_eye_crop,
            convert_to_rgb=False,
        )
        _, tensor_rgb = self._prepare_model_input(
            source_eye_crop,
            convert_to_rgb=True,
        )

        raw_outputs_bgr = tuple(
            np.asarray(output)
            for output in self._run_model_outputs(tensor_bgr)
        )
        raw_outputs_rgb = tuple(
            np.asarray(output)
            for output in self._run_model_outputs(tensor_rgb)
        )

        if not raw_outputs_bgr or np.asarray(raw_outputs_bgr[0]).size < 2:
            return EyeMeasurement.unknown(AssessmentStatus.MODEL_ERROR), None

        open_probability, closed_probability, _ = (
            self._interpret_output(np.asarray(raw_outputs_bgr[0]).reshape(-1)[:2])
        )

        model_margin_confidence = abs(open_probability - closed_probability)
        confidence = clamp01(
            0.24 * resolution_confidence
            + 0.22 * exposure_confidence
            + 0.20 * contrast_confidence
            + 0.34 * model_margin_confidence
        )
        label, status = self._classify_prediction(
            open_probability=open_probability,
            closed_probability=closed_probability,
            confidence=confidence,
        )

        return (
            EyeMeasurement(
                open_probability=open_probability,
                label=label,
                confidence=confidence,
                status=status,
                source_width=source_width,
                source_height=source_height,
            ),
            EyeInferenceDebug(
                model_input_image=model_input_image,
                tensor=tensor_bgr,
                raw_outputs_bgr=raw_outputs_bgr,
                raw_outputs_rgb=raw_outputs_rgb,
            ),
        )

    def _estimate_crop_quality_components(
        self,
        source_eye_crop: np.ndarray,
    ) -> tuple[float, float, float]:
        source_height, source_width = source_eye_crop.shape[:2]
        gray = to_gray(source_eye_crop).astype(np.float32)
        median_luminance = float(np.median(gray))
        percentile_range = float(
            np.percentile(gray, 90) - np.percentile(gray, 10)
        )

        resolution_confidence = smoothstep(
            float(self._config.eye_source_min_width),
            28.0,
            float(min(source_width, source_height) * 1.35),
        )
        exposure_confidence = smoothstep(12.0, 42.0, median_luminance) * (
            1.0 - smoothstep(215.0, 248.0, median_luminance)
        )
        contrast_confidence = smoothstep(8.0, 32.0, percentile_range)
        return (
            resolution_confidence,
            exposure_confidence,
            contrast_confidence,
        )

    def _prepare_model_input(
        self,
        source_eye_crop: np.ndarray,
        *,
        convert_to_rgb: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        model_input_size = self._config.eye_model_input_size
        # The crop is intentionally smaller than the 32 x 32 model input so we
        # isolate the eyelids and iris before interpolation adds context.
        model_input_image = cv2.resize(
            source_eye_crop,
            (model_input_size, model_input_size),
            interpolation=self._resize_interpolation(
                source_eye_crop=source_eye_crop,
                model_input_size=model_input_size,
            ),
        )
        if convert_to_rgb:
            model_input_image = cv2.cvtColor(model_input_image, cv2.COLOR_BGR2RGB)
        tensor = (model_input_image.astype(np.float32) - 127.0) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        tensor = np.ascontiguousarray(tensor)
        self._validate_model_input(tensor, model_input_size=model_input_size)
        return model_input_image, tensor

    @staticmethod
    def _resize_interpolation(
        *,
        source_eye_crop: np.ndarray,
        model_input_size: int,
    ) -> int:
        return (
            cv2.INTER_AREA
            if max(source_eye_crop.shape[:2]) > model_input_size
            else cv2.INTER_CUBIC
        )

    def _interpret_output(
        self,
        two_class_output: np.ndarray,
    ) -> tuple[float, float, str]:
        output = np.asarray(two_class_output, dtype=np.float32).reshape(-1)
        if output.size < 2 or not np.isfinite(output[:2]).all():
            raise ValueError("Eye model returned invalid output.")

        probabilities = output[:2]
        interpretation_mode = "softmax_from_logits"
        if (
            np.all(probabilities >= 0.0)
            and np.all(probabilities <= 1.0)
            and abs(float(probabilities.sum()) - 1.0) < 0.08
        ):
            interpretation_mode = "already_probabilities"
            probabilities = probabilities / max(
                float(probabilities.sum()),
                EPSILON,
            )
        else:
            probabilities = softmax(probabilities)

        closed_probability = clamp01(float(probabilities[0]))
        open_probability = clamp01(float(probabilities[1]))
        normalization = max(open_probability + closed_probability, EPSILON)
        return (
            clamp01(open_probability / normalization),
            clamp01(closed_probability / normalization),
            interpretation_mode,
        )

    def _classify_prediction(
        self,
        *,
        open_probability: float,
        closed_probability: float,
        confidence: float,
    ) -> tuple[EyeLabel, AssessmentStatus]:
        if (
            open_probability >= self._config.eye_open_threshold
            and confidence >= self._config.eye_minimum_decision_confidence
        ):
            return EyeLabel.OPEN, AssessmentStatus.ASSESSED

        if (
            closed_probability >= self._config.eye_closed_threshold
            and confidence >= self._config.eye_minimum_decision_confidence
        ):
            return EyeLabel.CLOSED, AssessmentStatus.ASSESSED

        return EyeLabel.UNCERTAIN, AssessmentStatus.LOW_CONFIDENCE

    def _combine_eye_measurements(
        self,
        *,
        left: EyeMeasurement,
        right: EyeMeasurement,
    ) -> EyeState:
        assessed_measurements = [
            eye
            for eye in (left, right)
            if eye.status is AssessmentStatus.ASSESSED
            and eye.open_probability is not None
        ]
        if not assessed_measurements:
            return EyeState(
                left=left,
                right=right,
                combined_open_score=None,
                confidence=0.0,
                status=AssessmentStatus.LOW_CONFIDENCE,
            )

        if len(assessed_measurements) == 1:
            only_eye = assessed_measurements[0]
            return EyeState(
                left=left,
                right=right,
                combined_open_score=float(only_eye.open_probability),
                confidence=float(only_eye.confidence),
                status=AssessmentStatus.PARTIAL,
            )

        probabilities = [
            float(eye.open_probability)
            for eye in assessed_measurements
        ]
        combined_open_score = clamp01(
            0.70 * min(probabilities) + 0.30 * float(np.mean(probabilities))
        )
        return EyeState(
            left=left,
            right=right,
            combined_open_score=combined_open_score,
            confidence=float(
                np.mean([eye.confidence for eye in assessed_measurements])
            ),
            status=AssessmentStatus.ASSESSED,
        )

    @staticmethod
    def _validate_model_input(
        tensor: np.ndarray,
        *,
        model_input_size: int,
    ) -> None:
        if tensor.shape != (1, 3, model_input_size, model_input_size):
            raise ValueError(f"Unexpected eye model input shape: {tensor.shape}")
        if tensor.dtype != np.float32:
            raise ValueError(f"Unexpected eye model input dtype: {tensor.dtype}")
        if not np.isfinite(tensor).all():
            raise ValueError("Eye model input contains non-finite values.")

    def _save_debug_outputs(
        self,
        *,
        image: np.ndarray,
        face: DetectedFace,
        face_index: int | None,
        left_source_eye_crop: np.ndarray,
        right_source_eye_crop: np.ndarray,
        left_measurement: EyeMeasurement,
        right_measurement: EyeMeasurement,
        left_debug: EyeInferenceDebug | None,
        right_debug: EyeInferenceDebug | None,
    ) -> None:
        debug_output_dir = self._debug_output_dir
        if debug_output_dir is None:
            return

        debug_output_dir.mkdir(parents=True, exist_ok=True)
        face_index_text = "na" if face_index is None else str(face_index)
        face_path = getattr(face, "path", Path("eye"))
        stem = self._safe_filename(Path(face_path).stem)

        self._save_eye_debug_pair(
            debug_output_dir=debug_output_dir,
            stem=stem,
            face_index_text=face_index_text,
            side_name="left",
            source_eye_crop=left_source_eye_crop,
            measurement=left_measurement,
            debug=left_debug,
        )
        self._save_eye_debug_pair(
            debug_output_dir=debug_output_dir,
            stem=stem,
            face_index_text=face_index_text,
            side_name="right",
            source_eye_crop=right_source_eye_crop,
            measurement=right_measurement,
            debug=right_debug,
        )

        annotated_face_image = image.copy()
        x, y, width, height = normalize_bbox(face.bbox)
        cv2.rectangle(
            annotated_face_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )
        if face.landmarks is not None:
            for point in np.asarray(face.landmarks, dtype=np.float32).reshape(-1, 2)[:5]:
                cv2.circle(
                    annotated_face_image,
                    (int(round(point[0])), int(round(point[1]))),
                    2,
                    (0, 255, 255),
                    -1,
                )

        cv2.imwrite(
            str(debug_output_dir / f"{stem}_face{face_index_text}_annotated.png"),
            annotated_face_image,
        )

    def _save_eye_debug_pair(
        self,
        *,
        debug_output_dir: Path,
        stem: str,
        face_index_text: str,
        side_name: str,
        source_eye_crop: np.ndarray,
        measurement: EyeMeasurement,
        debug: EyeInferenceDebug | None,
    ) -> None:
        source_height, source_width = source_eye_crop.shape[:2]
        open_probability = (
            -1.0 if measurement.open_probability is None else measurement.open_probability
        )
        label_text = measurement.label.value
        filename_prefix = (
            f"{stem}_face{face_index_text}_{side_name}"
            f"_{source_width}x{source_height}"
            f"_p{open_probability:.3f}_{label_text}"
        )
        cv2.imwrite(
            str(debug_output_dir / f"{filename_prefix}_source.png"),
            source_eye_crop,
        )
        if debug is not None:
            cv2.imwrite(
                str(debug_output_dir / f"{filename_prefix}_model_input.png"),
                debug.model_input_image,
            )

    @staticmethod
    def _safe_filename(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "eye"

    def _run_model_outputs(self, tensor: np.ndarray) -> list[np.ndarray]:
        if hasattr(self._model, "run_all_outputs"):
            return self._model.run_all_outputs(tensor)
        return self._model.run(tensor)
