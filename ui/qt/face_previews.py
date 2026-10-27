"""Lazy, bounded face crops on the existing thumbnail scheduling pattern."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

from image_loader import ImageLoader
from .thumbnails import ThumbnailService

CARD_ASPECT = 1.55


def face_crop_bounds(image_size, bbox, aspect=CARD_ASPECT):
    """Frame a detector-space xywh box with context in the card's aspect ratio."""
    image_width, image_height = image_size
    x, y, width, height = bbox
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None
    # Fit the entire face even at an edge. Give the vertical span enough room
    # for hair and chin; the horizontal span follows the card aspect ratio.
    desired_height = max(height * 1.85, width * 2.1 / aspect)
    # Clamp each axis independently. Scaling both axes to keep the aspect
    # ratio on a narrow source can make the crop smaller than the face itself.
    crop_width = min(image_width, desired_height * aspect)
    crop_height = min(image_height, desired_height)
    center_x = x + width / 2
    center_y = y + height / 2
    left = min(max(0, center_x - crop_width / 2), image_width - crop_width)
    top = min(max(0, center_y - crop_height / 2), image_height - crop_height)
    return (int(left), int(top), min(image_width, int(left + crop_width + 0.5)),
            min(image_height, int(top + crop_height + 0.5)))


def decode_face_preview(request: str, edge: int) -> str:
    try:
        path, bbox = json.loads(request)
        x, y, width, height = (int(value) for value in bbox)
        if width <= 0 or height <= 0:
            return ""
        if not Path(path).is_file():
            return ""
        # Face detection uses ImageLoader's 2048-pixel oriented decode, so its
        # stored boxes are in these same coordinates. Never retain that decode.
        image = ImageLoader(max_decode_dimension=2048).load_pil_for_hashing(
            Path(path), max_dimension=2048)
        if image is None:
            return ""
        bounds = face_crop_bounds(image.size, (x, y, width, height))
        if bounds is None:
            return ""
        left, top, right, bottom = bounds
        if right <= left or bottom <= top:
            return ""
        crop = image.crop((left, top, right, bottom))
        crop.thumbnail((edge, edge))
        with BytesIO() as output:
            crop.save(output, format="PNG")
            return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return ""


class FacePreviewService(ThumbnailService):
    def __init__(self, parent=None, **kwargs):
        options = dict(workers=2, max_pending=64, cache_bytes=16 * 1024 * 1024,
                       cache_entries=96, max_edge=256, decoder=decode_face_preview)
        options.update(kwargs)
        super().__init__(parent, **options)

    def request_face(self, path: str, bbox, edge: int, revision: str) -> int:
        request = json.dumps([path, list(bbox)], separators=(",", ":"))
        return self.request(request, edge, revision)
