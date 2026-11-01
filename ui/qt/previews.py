"""Independent, tightly bounded inspector cache using the existing scheduler."""
import base64
from io import BytesIO
from pathlib import Path

from image_loader import ImageLoader
from .thumbnails import ThumbnailService


def decode_preview(path: str, edge: int) -> str:
    """Use a bounded JPEG for the large inspector image, including EXIF handling."""
    if not Path(path).is_file():
        return ""
    image = ImageLoader(max_decode_dimension=edge).load_pil_for_hashing(
        Path(path), max_dimension=edge)
    if image is None:
        return ""
    with BytesIO() as output:
        image.convert("RGB").save(output, format="JPEG", quality=88, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class PreviewService(ThumbnailService):
    def __init__(self, parent=None, **kwargs):
        kwargs.setdefault("decoder", decode_preview)
        super().__init__(parent, workers=1, max_pending=3,
            cache_bytes=24 * 1024 * 1024, cache_entries=3, max_edge=1600, **kwargs)
