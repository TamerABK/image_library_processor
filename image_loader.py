from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import rawpy
except ImportError:  # pragma: no cover - optional dependency
    rawpy = None


STANDARD_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

RAW_EXTENSIONS = (
    ".arw",
    ".cr2",
    ".cr3",
    ".dng",
    ".erf",
    ".kdc",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rw2",
    ".sr2",
)

_JPEG_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    width: int
    height: int
    is_raw: bool


@dataclass(frozen=True, slots=True)
class _MetadataCacheEntry:
    file_size: int
    mtime_ns: int
    metadata: ImageMetadata | None


class ImageLoader:
    def __init__(
        self,
        *,
        max_decode_dimension: int | None = 2048,
    ) -> None:
        self._max_decode_dimension = max_decode_dimension
        self._metadata_cache: dict[Path, _MetadataCacheEntry] = {}
        self._lock = threading.Lock()

    @property
    def max_decode_dimension(self) -> int | None:
        return self._max_decode_dimension

    def supported_extensions(self) -> tuple[str, ...]:
        if rawpy is None:
            return STANDARD_EXTENSIONS
        return STANDARD_EXTENSIONS + RAW_EXTENSIONS

    def read_metadata(
        self,
        path: Path,
    ) -> ImageMetadata | None:
        normalized_path = path.resolve()
        try:
            stat = normalized_path.stat()
        except OSError:
            return None

        with self._lock:
            cached = self._metadata_cache.get(normalized_path)
            if (
                cached is not None
                and cached.file_size == stat.st_size
                and cached.mtime_ns == stat.st_mtime_ns
            ):
                return cached.metadata

        metadata = self._read_metadata_uncached(normalized_path)
        with self._lock:
            self._metadata_cache[normalized_path] = _MetadataCacheEntry(
                file_size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                metadata=metadata,
            )
        return metadata

    def load_for_detection(
        self,
        path: Path,
    ) -> np.ndarray | None:
        return self.load_for_scan(path)

    def load_for_scan(
        self,
        path: Path,
        *,
        max_dimension: int | None = None,
        grayscale: bool = False,
    ) -> np.ndarray | None:
        suffix = path.suffix.lower()
        if suffix in RAW_EXTENSIONS and rawpy is not None:
            return self._load_raw_preview(
                path,
                max_dimension=max_dimension,
                grayscale=grayscale,
            )

        metadata = self.read_metadata(path)
        return self._load_raster(
            path,
            metadata,
            max_dimension=max_dimension,
            grayscale=grayscale,
        )

    def load_grayscale(
        self,
        path: Path,
        *,
        max_dimension: int | None = None,
    ) -> np.ndarray | None:
        return self.load_for_scan(
            path,
            max_dimension=max_dimension,
            grayscale=True,
        )

    def load_pil_for_hashing(
        self,
        path: Path,
        *,
        max_dimension: int | None = None,
    ) -> Image.Image | None:
        image = self.load_for_scan(
            path,
            max_dimension=max_dimension,
            grayscale=False,
        )
        if image is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _read_metadata_uncached(
        self,
        path: Path,
    ) -> ImageMetadata | None:
        suffix = path.suffix.lower()
        if suffix in RAW_EXTENSIONS and rawpy is not None:
            try:
                with rawpy.imread(str(path)) as raw:
                    return ImageMetadata(
                        width=int(raw.sizes.width),
                        height=int(raw.sizes.height),
                        is_raw=True,
                    )
            except Exception:
                return None

        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
        except Exception:
            return None

        return ImageMetadata(width=width, height=height, is_raw=False)

    def _load_raster(
        self,
        path: Path,
        metadata: ImageMetadata | None,
        *,
        max_dimension: int | None,
        grayscale: bool,
    ) -> np.ndarray | None:
        read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        if metadata is not None and metadata.width and metadata.height:
            read_flag = self._raster_read_flag(
                path.suffix.lower(),
                metadata,
                max_dimension=max_dimension,
                grayscale=grayscale,
            )

        image = cv2.imread(str(path), read_flag)
        if image is None:
            return None
        return self._resize_if_needed(image, max_dimension)

    def _raster_read_flag(
        self,
        suffix: str,
        metadata: ImageMetadata,
        *,
        max_dimension: int | None,
        grayscale: bool,
    ) -> int:
        base_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        if suffix not in _JPEG_EXTENSIONS or max_dimension is None:
            return base_flag

        largest = max(metadata.width, metadata.height)
        if grayscale:
            if largest > max_dimension * 4:
                return cv2.IMREAD_REDUCED_GRAYSCALE_8
            if largest > max_dimension * 2:
                return cv2.IMREAD_REDUCED_GRAYSCALE_4
            if largest > max_dimension:
                return cv2.IMREAD_REDUCED_GRAYSCALE_2
            return cv2.IMREAD_GRAYSCALE

        if largest > max_dimension * 4:
            return cv2.IMREAD_REDUCED_COLOR_8
        if largest > max_dimension * 2:
            return cv2.IMREAD_REDUCED_COLOR_4
        if largest > max_dimension:
            return cv2.IMREAD_REDUCED_COLOR_2
        return cv2.IMREAD_COLOR

    def _load_raw_preview(
        self,
        path: Path,
        *,
        max_dimension: int | None,
        grayscale: bool,
    ) -> np.ndarray | None:
        try:
            with rawpy.imread(str(path)) as raw:
                image = self._extract_raw_thumbnail(raw)
                if image is None:
                    rgb = raw.postprocess(
                        half_size=True,
                        no_auto_bright=True,
                        use_camera_wb=True,
                    )
                    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

        image = self._resize_if_needed(image, max_dimension)
        if grayscale and image is not None and image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def _extract_raw_thumbnail(
        raw: rawpy.RawPy,
    ) -> np.ndarray | None:
        try:
            thumb = raw.extract_thumb()
        except Exception:
            return None

        if thumb.format == rawpy.ThumbFormat.JPEG:
            payload = np.frombuffer(thumb.data, dtype=np.uint8)
            return cv2.imdecode(payload, cv2.IMREAD_COLOR)

        if thumb.format == rawpy.ThumbFormat.BITMAP:
            image = np.asarray(thumb.data)
            if image.ndim == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image

        return None

    def _resize_if_needed(
        self,
        image: np.ndarray,
        max_dimension: int | None,
    ) -> np.ndarray:
        if max_dimension is None:
            max_dimension = self._max_decode_dimension

        if max_dimension is None:
            return image

        height, width = image.shape[:2]
        largest = max(width, height)
        if largest <= max_dimension:
            return image

        scale = max_dimension / largest
        return cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )


default_image_loader = ImageLoader()
