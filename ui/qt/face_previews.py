"""Lazy, bounded face crops on the existing thumbnail scheduling pattern."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

from image_loader import ImageLoader
from .thumbnails import ThumbnailService


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
        side = max(width, height) * 1.36  # 18% context on each side.
        center_x, center_y = x + width / 2, y + height / 2
        left = max(0, int(center_x - side / 2))
        top = max(0, int(center_y - side / 2))
        right = min(image.width, int(center_x + side / 2 + 0.5))
        bottom = min(image.height, int(center_y + side / 2 + 0.5))
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
