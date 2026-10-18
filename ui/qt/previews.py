"""Independent, tightly bounded inspector cache using the existing scheduler."""
from .thumbnails import ThumbnailService


class PreviewService(ThumbnailService):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, workers=1, max_pending=3,
            cache_bytes=24 * 1024 * 1024, cache_entries=3, max_edge=1600, **kwargs)
