from dataclasses import dataclass

import cv2
import numpy as np

from image_loader import default_image_loader

from .interfaces import FacePreviewRenderer
from .models import EmbeddedFace


@dataclass(slots=True)
class FacePreviewRenderer(FacePreviewRenderer):
    output_size: int = 256
    padding_factor: float = 2.0
    circle_color: tuple[int, int, int] = (0, 255, 0)
    circle_thickness: int = 4
    dim_factor: float = 0.40
    _crop_offset: tuple[int, int] = (0, 0)

    def render(
        self,
        face: EmbeddedFace,
    ) -> np.ndarray:
        """
        Returns a square preview centered around the face with the
        representative circled.
        """

        image = default_image_loader.load_for_detection(face.path)

        if image is None:
            raise RuntimeError(f"Could not load image: {face.path}")

        x, y, w, h = face.bbox

        image = self._crop(image, x, y, w, h)

        self._highlight(image, x, y, w, h)

        return cv2.resize(
            image,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA,
        )

    def _crop(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> np.ndarray:

        pad = int(max(w, h) * self.padding_factor)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)

        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        self._crop_offset = (x1, y1)

        return image[y1:y2, x1:x2].copy()

    def _highlight(
        self,
        image: np.ndarray,
        original_x: int,
        original_y: int,
        w: int,
        h: int,
    ) -> None:

        crop_x, crop_y = self._crop_offset

        x = original_x - crop_x
        y = original_y - crop_y

        center = (
            x + w // 2,
            y + h // 2,
        )

        radius = int(max(w, h) * 0.65)

        # Darken entire crop
        overlay = (image.astype(np.float32) * self.dim_factor).astype(np.uint8)

        # Restore circle region
        mask = np.zeros(image.shape[:2], np.uint8)

        cv2.circle(
            mask,
            center,
            radius,
            255,
            -1,
        )

        image[:] = np.where(
            mask[..., None] == 255,
            image,
            overlay,
        )

        # Draw highlight
        cv2.circle(
            image,
            center,
            radius,
            self.circle_color,
            self.circle_thickness,
        )
