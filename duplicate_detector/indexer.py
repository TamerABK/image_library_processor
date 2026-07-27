"""
Filesystem indexing and perceptual hash computation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import imagehash

from image_file_utils import find_supported_files
from image_loader import default_image_loader

from .cache import DuplicatePhotoCache
from .config import DetectorConfig
from .models import PhotoInfo


ProgressCallback = Callable[[int, int], None]


class ImageIndexer:
    """
    Scans folders and builds a PhotoInfo index.

    Responsibilities
    ----------------
    - Discover images
    - Compute perceptual hashes
    - Extract metadata
    - Parallelize expensive work
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._cache = DuplicatePhotoCache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(
        self,
        folder: str | Path,
        progress_callback: ProgressCallback | None = None,
        file_extensions: tuple[str, ...] | None = None,
        orientation_filter: str | None = None,
    ) -> list[PhotoInfo]:
        """
        Index every supported image inside a folder.
        """

        folder = Path(folder)

        image_paths = self._find_images(folder, file_extensions, orientation_filter)

        total = len(image_paths)

        if total == 0:
            return []

        photos: list[PhotoInfo] = []

        with ThreadPoolExecutor(
            max_workers=self.config.workers
        ) as executor:

            futures = {
                executor.submit(self._index_image, path): path
                for path in image_paths
            }

            completed = 0

            for future in as_completed(futures):

                photo = future.result()

                if photo is not None:
                    photos.append(photo)

                completed += 1

                if progress_callback:
                    progress_callback(completed, total)

        photos.sort(key=lambda p: p.path)

        return photos

    # ------------------------------------------------------------------
    # Image discovery
    # ------------------------------------------------------------------

    def _find_images(
        self,
        folder: Path,
        file_extensions: tuple[str, ...] | None = None,
        orientation_filter: str | None = None,
    ) -> list[Path]:
        return find_supported_files(
            folder,
            self.config.supported_extensions,
            file_extensions,
            orientation_filter=orientation_filter,
        )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _index_image(
        self,
        path: Path,
    ) -> PhotoInfo | None:

        try:
            stat = path.stat()
            normalized_path = path.resolve()
            cached = self._get_cached_photo(
                normalized_path,
                stat.st_size,
                stat.st_mtime_ns,
            )

            if cached is not None:
                return cached

            metadata = default_image_loader.read_metadata(normalized_path)
            image = default_image_loader.load_pil_for_hashing(
                normalized_path,
                max_dimension=self.config.hash_decode_dimension,
            )
            if image is None:
                raise ValueError(f"Could not open image: {normalized_path}")

            if metadata is None:
                width, height = image.size
            else:
                width, height = metadata.width, metadata.height

            phash = self._compute_phash(image)
            dhash = self._compute_dhash(image)

            photo = PhotoInfo(
                path=normalized_path,
                width=width,
                height=height,
                file_size=stat.st_size,
                phash=phash,
                dhash=dhash,
            )

            self._store_cached_photo(
                photo,
                stat.st_mtime_ns,
                is_raw=metadata.is_raw if metadata is not None else None,
            )

            return photo

        except Exception:

            if self.config.ignore_load_errors:
                return None

            raise

    def _get_cached_photo(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
    ) -> PhotoInfo | None:
        try:
            return self._cache.get(path, file_size, mtime_ns)
        except Exception:
            return None

    def _store_cached_photo(
        self,
        photo: PhotoInfo,
        mtime_ns: int,
        *,
        is_raw: bool | None = None,
    ) -> None:
        try:
            self._cache.put(photo, mtime_ns, is_raw=is_raw)
        except Exception:
            # Cache writes should never make duplicate detection fail.
            return

    # ------------------------------------------------------------------
    # Hashes
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_phash(image) -> int:

        return int(
            str(
                imagehash.phash(
                    image,
                    hash_size=16,
                )
            ),
            16,
        )

    @staticmethod
    def _compute_dhash(image) -> int:

        return int(
            str(
                imagehash.dhash(
                    image,
                    hash_size=16,
                )
            ),
            16,
        )
