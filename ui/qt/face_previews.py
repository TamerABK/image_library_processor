"""Lazy, bounded face crops on the existing thumbnail scheduling pattern."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from image_loader import ImageLoader
from .thumbnails import ThumbnailService

CARD_ASPECT = 1.55
DETECTION_EDGE = 2048


def preview_bbox(loader: ImageLoader, path: Path, image_size, bbox):
    """Map xywh from the detector decode onto a reduced JPEG preview decode."""
    metadata = loader.read_metadata(path)
    if metadata is None or metadata.is_raw or not metadata.width or not metadata.height:
        return bbox
    original_width, original_height = metadata.width, metadata.height
    longest = max(original_width, original_height)
    factor = min(1.0, DETECTION_EDGE / longest)
    detector_width = max(1, round(original_width * factor))
    detector_height = max(1, round(original_height * factor))
    image_width, image_height = image_size
    scale_x = image_width / detector_width
    scale_y = image_height / detector_height
    x, y, width, height = bbox
    return (x * scale_x, y * scale_y, width * scale_x, height * scale_y)


def select_face_candidate(loader: ImageLoader, candidates):
    """Prefer a fully visible detector box among a bounded set of faces."""
    best = None
    for index, (path_text, bbox) in enumerate(candidates):
        path = Path(path_text)
        if not path.is_file():
            continue
        metadata = loader.read_metadata(path)
        if metadata is None or not metadata.width or not metadata.height:
            continue
        longest = max(metadata.width, metadata.height)
        factor = min(1.0, DETECTION_EDGE / longest)
        detector_width = max(1, round(metadata.width * factor))
        detector_height = max(1, round(metadata.height * factor))
        x, y, width, height = (float(value) for value in bbox)
        if width <= 0 or height <= 0:
            continue
        visible_width = max(0, min(detector_width, x + width) - max(0, x))
        visible_height = max(0, min(detector_height, y + height) - max(0, y))
        visible_ratio = visible_width * visible_height / (width * height)
        if visible_ratio <= 0:
            continue
        margin = min(x, y, detector_width - x - width, detector_height - y - height)
        context_ratio = margin / min(width, height)
        score = (visible_ratio >= 0.9 and context_ratio >= 0.25,
                 visible_ratio >= 0.9, visible_ratio * width * height,
                 context_ratio, -index)
        if best is None or score > best[0]:
            best = (score, str(path), (x, y, width, height))
    return None if best is None else (best[1], best[2])


def face_crop_bounds(image_size, bbox, aspect=CARD_ASPECT):
    """Centered detector-space frame; space beyond image edges is padded."""
    image_width, image_height = image_size
    x, y, width, height = bbox
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None
    # Do not move the frame at an image edge: that would move the face away
    # from the center. The compositor fills out-of-image pixels at card size.
    crop_height = max(height * 1.75, width * 2.0 / aspect)
    crop_width = crop_height * aspect
    center_x = x + width / 2
    center_y = y + height / 2
    left = center_x - crop_width / 2
    top = center_y - crop_height / 2
    return (left, top, left + crop_width, top + crop_height)


def compose_face_preview(image: Image.Image, bbox, edge: int) -> Image.Image | None:
    bounds = face_crop_bounds(image.size, bbox)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    output_width = max(32, int(edge))
    output_height = max(1, round(output_width / CARD_ASPECT))
    scale_x = (right - left) / output_width
    scale_y = (bottom - top) / output_height
    # Render directly at thumbnail size. This bounds intermediate image memory
    # even when a face touches the source edge or fills the whole photograph.
    sharp = image.transform((output_width, output_height), Image.Transform.AFFINE,
        (scale_x, 0, left, 0, scale_y, top), resample=Image.Resampling.BICUBIC,
        fillcolor=(12, 15, 20)).convert("RGB")
    background = ImageEnhance.Brightness(sharp.filter(ImageFilter.GaussianBlur(
        max(3, output_height * 0.045)))).enhance(0.28)
    x, y, width, height = bbox
    center_x = (x + width / 2 - left) / scale_x
    center_y = (y + height / 2 - top) / scale_y
    face_width = width / scale_x
    face_height = height / scale_y
    mask = Image.new("L", sharp.size, 0)
    ImageDraw.Draw(mask).ellipse((center_x - face_width * 0.78,
        center_y - face_height * 0.76, center_x + face_width * 0.78,
        center_y + face_height * 0.76), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(2, min(face_width, face_height) * 0.16)))
    return Image.composite(sharp, background, mask)


def decode_face_preview(request: str, edge: int) -> str:
    try:
        payload = json.loads(request)
        candidates = payload["candidates"] if isinstance(payload, dict) else [payload]
        # Detector boxes refer to the oriented 2048-pixel scan image. An
        # explicit preview bound can select JPEG 1/2, 1/4 or 1/8 decoding,
        # which has different dimensions even with the same maximum edge.
        loader = ImageLoader(max_decode_dimension=DETECTION_EDGE)
        selected = select_face_candidate(loader, candidates[:8])
        if selected is None:
            return ""
        path, bbox = selected
        photo = Path(path)
        image = loader.load_pil_for_hashing(photo, max_dimension=DETECTION_EDGE)
        if image is None:
            return ""
        bbox = preview_bbox(loader, photo, image.size, bbox)
        crop = compose_face_preview(image, bbox, edge)
        if crop is None:
            return ""
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

    def request_face(self, path: str, bbox, edge: int, revision: str, *, candidates=()) -> int:
        options = [[str(face["path"]), list(face["bbox"])] for face in candidates]
        request = json.dumps({"candidates": options} if options else [path, list(bbox)],
                             separators=(",", ":"))
        return self.request(request, edge, revision)
