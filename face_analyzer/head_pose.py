from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .geometry import ensure_bgr, extract_square_crop, valid_five_landmarks
from .math_utils import EPSILON, clamp01, smoothstep
from .models import AssessmentStatus, DetectedFace, HeadPose
from .onnx_session import OnnxModel, Provider


class FivePointRollEstimator:
    """Fallback that reports only roll; yaw and pitch remain unknown."""

    def estimate(self, image: np.ndarray, face: DetectedFace) -> HeadPose:
        del image
        points = valid_five_landmarks(face.landmarks)
        if points is None:
            return HeadPose.unknown(AssessmentStatus.LANDMARKS_MISSING)

        left_eye, right_eye = points[0], points[1]
        roll = math.degrees(
            math.atan2(
                float(right_eye[1] - left_eye[1]),
                float(right_eye[0] - left_eye[0]),
            )
        )
        return HeadPose.unknown(
            AssessmentStatus.PARTIAL,
            roll_degrees=roll,
            confidence=0.75,
            source="five_point_roll_only",
        )


class SixDRepNetOnnxEstimator:
    """Head-pose estimator for an ONNX-exported official 6DRepNet model.

    The official network returns a 3 x 3 rotation matrix. Raw six-dimensional
    output is also accepted to make custom exports easier.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[Provider] | None = None,
        crop_expansion: float = 1.25,
    ) -> None:
        self._model = OnnxModel(model_path, providers=providers)
        self._crop_expansion = max(1.0, crop_expansion)

        shape = self._model.input.shape
        self._height = int(shape[2]) if isinstance(shape[2], int) else 224
        self._width = int(shape[3]) if isinstance(shape[3], int) else 224

    def estimate(self, image: np.ndarray, face: DetectedFace) -> HeadPose:
        crop = extract_square_crop(
            ensure_bgr(image),
            face.bbox,
            expansion=self._crop_expansion,
        )
        if crop.size == 0:
            return HeadPose.unknown(AssessmentStatus.INVALID_INPUT)

        source_minimum_dimension = min(crop.shape[:2])
        if source_minimum_dimension < 40:
            return HeadPose.unknown(AssessmentStatus.FACE_TOO_SMALL)

        tensor = self._preprocess(crop)
        output = np.asarray(self._model.run(tensor)[0], dtype=np.float64).squeeze()
        rotation = self._rotation_matrix(output)
        if rotation is None:
            return HeadPose.unknown(
                AssessmentStatus.MODEL_ERROR,
                source="sixdrepnet_onnx",
            )

        pitch, yaw, roll = self._euler_xyz(rotation)
        confidence = smoothstep(48.0, 128.0, float(source_minimum_dimension))

        return HeadPose(
            yaw_degrees=float(math.degrees(yaw)),
            pitch_degrees=float(math.degrees(pitch)),
            roll_degrees=float(math.degrees(roll)),
            confidence=clamp01(confidence),
            status=AssessmentStatus.ASSESSED,
            source="sixdrepnet_onnx",
        )

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Official evaluation transform: Resize(224), CenterCrop(224),
        # ImageNet normalization. Since the detector crop is square, direct
        # resizing is equivalent to that final geometry.
        resized = cv2.resize(
            rgb,
            (self._width, self._height),
            interpolation=cv2.INTER_AREA,
        )
        tensor = resized.astype(np.float32) / 255.0
        tensor = (
            tensor - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return np.transpose(tensor, (2, 0, 1))[None, ...]

    @staticmethod
    def _rotation_matrix(output: np.ndarray) -> np.ndarray | None:
        if output.shape == (3, 3):
            rotation = output
        elif output.size == 9:
            rotation = output.reshape(3, 3)
        elif output.size == 6:
            x_raw = output.reshape(6)[:3]
            y_raw = output.reshape(6)[3:]

            x_norm = float(np.linalg.norm(x_raw))
            if x_norm < EPSILON:
                return None
            x_axis = x_raw / x_norm

            z_axis = np.cross(x_axis, y_raw)
            z_norm = float(np.linalg.norm(z_axis))
            if z_norm < EPSILON:
                return None
            z_axis /= z_norm
            y_axis = np.cross(z_axis, x_axis)
            rotation = np.stack((x_axis, y_axis, z_axis), axis=1)
        else:
            return None

        if not np.isfinite(rotation).all():
            return None

        # Project numerical noise onto a proper rotation matrix.
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        return rotation

    @staticmethod
    def _euler_xyz(rotation: np.ndarray) -> tuple[float, float, float]:
        # Matches 6DRepNet's compute_euler_angles_from_rotation_matrices.
        sy = math.sqrt(float(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            yaw = math.atan2(float(-rotation[2, 0]), sy)
            roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        else:
            pitch = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
            yaw = math.atan2(float(-rotation[2, 0]), sy)
            roll = 0.0

        return pitch, yaw, roll
