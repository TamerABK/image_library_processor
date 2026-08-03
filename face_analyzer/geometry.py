from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .models import FaceGeometry


# Standard five-point ArcFace alignment template for a 112 x 112 face.
_ALIGNMENT_TEMPLATE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class EyeCropGeometry:
    level_eye_centers: np.ndarray
    source_width: int
    source_height: int
    rotation_angle_degrees: float


def validate_image(image: np.ndarray) -> None:
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"Invalid image shape: {image.shape}")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape: {image.shape}")


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_bbox(
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    return (
        int(round(x)),
        int(round(y)),
        max(0, int(round(width))),
        max(0, int(round(height))),
    )


def calculate_geometry(
    bbox: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
) -> FaceGeometry:
    x, y, width, height = normalize_bbox(bbox)
    area = width * height
    image_area = max(1, image_width * image_height)

    clipped_x1 = max(0, min(x, image_width))
    clipped_y1 = max(0, min(y, image_height))
    clipped_x2 = max(0, min(x + width, image_width))
    clipped_y2 = max(0, min(y + height, image_height))
    visible_area = max(0, clipped_x2 - clipped_x1) * max(
        0, clipped_y2 - clipped_y1
    )

    return FaceGeometry(
        width=width,
        height=height,
        area=area,
        relative_area=float(area / image_area),
        visible_ratio=float(visible_area / area) if area > 0 else 0.0,
        minimum_dimension=min(width, height),
    )


def valid_five_landmarks(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None:
        return None

    points = np.asarray(landmarks, dtype=np.float32)
    if points.size < 10:
        return None

    points = points.reshape(-1, 2)[:5]
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return None

    if float(np.linalg.norm(points[1] - points[0])) < 2.0:
        return None

    return points


def extract_square_crop(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    expansion: float = 1.0,
) -> np.ndarray:
    x, y, width, height = normalize_bbox(bbox)
    if width <= 0 or height <= 0:
        return np.empty((0, 0, 3), dtype=image.dtype)

    side = max(width, height) * max(1.0, expansion)
    center_x = x + width / 2.0
    center_y = y + height / 2.0

    x1 = int(math.floor(center_x - side / 2.0))
    y1 = int(math.floor(center_y - side / 2.0))
    x2 = int(math.ceil(center_x + side / 2.0))
    y2 = int(math.ceil(center_y + side / 2.0))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - image.shape[1])
    pad_bottom = max(0, y2 - image.shape[0])

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x1 >= x2 or y1 >= y2:
        return np.empty((0, 0, 3), dtype=image.dtype)

    crop = image[y1:y2, x1:x2]
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE,
        )
    return crop


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray | None,
    *,
    output_size: int,
) -> np.ndarray | None:
    points = valid_five_landmarks(landmarks)
    if points is None:
        return None

    destination = _ALIGNMENT_TEMPLATE_112 * (float(output_size) / 112.0)
    transform, _ = cv2.estimateAffinePartial2D(
        points,
        destination,
        method=cv2.LMEDS,
    )
    if transform is None or not np.isfinite(transform).all():
        return None

    return cv2.warpAffine(
        ensure_bgr(image),
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def aligned_or_fallback_crop(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarks: np.ndarray | None,
    *,
    output_size: int,
) -> tuple[np.ndarray, float]:
    aligned = align_face(image, landmarks, output_size=output_size)
    if aligned is not None:
        return aligned, 1.0

    crop = extract_square_crop(ensure_bgr(image), bbox, expansion=1.05)
    if crop.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8), 0.0

    resized = cv2.resize(
        crop,
        (output_size, output_size),
        interpolation=(
            cv2.INTER_AREA if max(crop.shape[:2]) > output_size else cv2.INTER_CUBIC
        ),
    )
    return resized, 0.35


def extract_level_eye_crops(
    image: np.ndarray,
    landmarks: np.ndarray | None,
    *,
    width_ratio: float,
    height_ratio: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    crop_geometry = level_eye_crop_geometry(
        landmarks,
        width_ratio=width_ratio,
        height_ratio=height_ratio,
    )
    if crop_geometry is None:
        return None

    bgr_image = ensure_bgr(image)
    left_eye = valid_five_landmarks(landmarks)[0]
    right_eye = valid_five_landmarks(landmarks)[1]
    eye_midpoint = (left_eye + right_eye) * 0.5
    transform = cv2.getRotationMatrix2D(
        (float(eye_midpoint[0]), float(eye_midpoint[1])),
        crop_geometry.rotation_angle_degrees,
        1.0,
    )

    level_image = cv2.warpAffine(
        bgr_image,
        transform,
        (bgr_image.shape[1], bgr_image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    left_crop = cv2.getRectSubPix(
        level_image,
        (crop_geometry.source_width, crop_geometry.source_height),
        tuple(crop_geometry.level_eye_centers[0].astype(float)),
    )
    right_crop = cv2.getRectSubPix(
        level_image,
        (crop_geometry.source_width, crop_geometry.source_height),
        tuple(crop_geometry.level_eye_centers[1].astype(float)),
    )
    return left_crop, right_crop


def level_eye_crop_geometry(
    landmarks: np.ndarray | None,
    *,
    width_ratio: float,
    height_ratio: float,
) -> EyeCropGeometry | None:
    points = valid_five_landmarks(landmarks)
    if points is None:
        return None

    left_eye = points[0]
    right_eye = points[1]
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance < 8.0:
        return None

    eye_midpoint = (left_eye + right_eye) * 0.5
    rotation_angle_degrees = math.degrees(
        math.atan2(float(eye_vector[1]), float(eye_vector[0]))
    )
    transform = cv2.getRotationMatrix2D(
        (float(eye_midpoint[0]), float(eye_midpoint[1])),
        rotation_angle_degrees,
        1.0,
    )

    homogeneous = np.concatenate(
        [points[:2], np.ones((2, 1), dtype=np.float32)],
        axis=1,
    )
    level_eye_centers = (transform @ homogeneous.T).T

    source_width = max(12, int(round(eye_distance * width_ratio)))
    source_height = max(8, int(round(eye_distance * height_ratio)))

    return EyeCropGeometry(
        level_eye_centers=level_eye_centers,
        source_width=source_width,
        source_height=source_height,
        rotation_angle_degrees=rotation_angle_degrees,
    )
