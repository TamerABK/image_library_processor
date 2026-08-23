from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from automatic_scan.perf import get_active_profiler

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
_EXIF_ORIENTATION_TAG = 274
_EXIF_DATETIME_TAGS = (
    36867,
    36868,
    306,
)


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    width: int
    height: int
    is_raw: bool
    orientation: str | None = None
    capture_timestamp: float | None = None
    timestamp_source: str = "filesystem"


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

        convert_started = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        profiler = get_active_profiler()
        if profiler is not None:
            profiler.record_color_conversion(path, time.perf_counter() - convert_started)
        return Image.fromarray(rgb)

    def _read_metadata_uncached(
        self,
        path: Path,
    ) -> ImageMetadata | None:
        suffix = path.suffix.lower()
        if suffix in RAW_EXTENSIONS and rawpy is not None:
            started = time.perf_counter()
            try:
                with rawpy.imread(str(path)) as raw:
                    metadata = ImageMetadata(
                        width=int(raw.sizes.width),
                        height=int(raw.sizes.height),
                        is_raw=True,
                    )
                    profiler = get_active_profiler()
                    if profiler is not None:
                        profiler.record_path_open(path)
                        profiler.record_metadata_read(
                            path,
                            time.perf_counter() - started,
                            exif=False,
                        )
                    return metadata
            except Exception:
                return None

        header_started = time.perf_counter()
        try:
            with Image.open(path) as image:
                width, height = image.size
                profiler = get_active_profiler()
                if profiler is not None:
                    profiler.record_path_open(path)
                    profiler.record_header_read(
                        path,
                        time.perf_counter() - header_started,
                    )
                metadata_started = time.perf_counter()
                exif = image.getexif()
                orientation_value = exif.get(_EXIF_ORIENTATION_TAG)
                if orientation_value in {5, 6, 7, 8}:
                    width, height = height, width
                capture_timestamp = None
                for tag_id in _EXIF_DATETIME_TAGS:
                    capture_timestamp = _parse_exif_datetime(exif.get(tag_id))
                    if capture_timestamp is not None:
                        break
                if profiler is not None:
                    profiler.record_metadata_read(
                        path,
                        time.perf_counter() - metadata_started,
                        exif=True,
                    )
        except Exception:
            return None

        orientation = None
        if width > height:
            orientation = "landscape"
        elif height > width:
            orientation = "portrait"
        elif width == height:
            orientation = "square"

        return ImageMetadata(
            width=width,
            height=height,
            is_raw=False,
            orientation=orientation,
            capture_timestamp=capture_timestamp,
            timestamp_source="exif" if capture_timestamp is not None else "filesystem",
        )

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

        decode_started = time.perf_counter()
        image = cv2.imread(str(path), read_flag)
        if image is None:
            return None
        profiler = get_active_profiler()
        if profiler is not None:
            profiler.record_path_open(path)
            bounded = False
            if max_dimension is not None and metadata is not None:
                bounded = max(metadata.width, metadata.height) > max_dimension
            profiler.record_decode(
                path,
                time.perf_counter() - decode_started,
                bounded=bounded,
            )
        return self._resize_if_needed(image, max_dimension, path=path)

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
        decode_started = time.perf_counter()
        try:
            with rawpy.imread(str(path)) as raw:
                image = self._extract_raw_thumbnail(raw)
                if image is None:
                    rgb = raw.postprocess(
                        half_size=True,
                        no_auto_bright=True,
                        use_camera_wb=True,
                    )
                    convert_started = time.perf_counter()
                    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    profiler = get_active_profiler()
                    if profiler is not None:
                        profiler.record_color_conversion(
                            path,
                            time.perf_counter() - convert_started,
                        )
        except Exception:
            return None

        profiler = get_active_profiler()
        if profiler is not None:
            profiler.record_path_open(path)
            profiler.record_decode(
                path,
                time.perf_counter() - decode_started,
                bounded=max_dimension is not None,
            )

        image = self._resize_if_needed(image, max_dimension, path=path)
        if grayscale and image is not None and image.ndim == 3:
            convert_started = time.perf_counter()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if profiler is not None:
                profiler.record_grayscale_conversion(
                    path,
                    time.perf_counter() - convert_started,
                )
            return gray
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
        *,
        path: Path | None = None,
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
        resize_started = time.perf_counter()
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        profiler = get_active_profiler()
        if profiler is not None:
            profiler.record_resize(path, time.perf_counter() - resize_started)
        return resized


def _parse_exif_datetime(raw_value: object) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bytes):
        try:
            raw_value = raw_value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if not isinstance(raw_value, str):
        return None

    value = raw_value.strip()
    if not value:
        return None

    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc).timestamp()
    return None


default_image_loader = ImageLoader()
