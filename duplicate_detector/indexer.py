"""
Filesystem indexing and perceptual hash computation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from PIL import Image
import imagehash

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(
        self,
        folder: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> list[PhotoInfo]:
        """
        Index every supported image inside a folder.
        """

        folder = Path(folder)

        image_paths = self._find_images(folder)

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

    def _find_images(self, folder: Path) -> list[Path]:

        return [
            path
            for path in folder.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in self.config.supported_extensions
            )
        ]

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _index_image(
        self,
        path: Path,
    ) -> PhotoInfo | None:

        try:

            with Image.open(path) as image:

                width, height = image.size

                phash = self._compute_phash(image)
                dhash = self._compute_dhash(image)

            return PhotoInfo(
                path=path,
                width=width,
                height=height,
                file_size=path.stat().st_size,
                phash=phash,
                dhash=dhash,
            )

        except Exception:

            if self.config.ignore_load_errors:
                return None

            raise

    # ------------------------------------------------------------------
    # Hashes
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_phash(image: Image.Image) -> int:

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
    def _compute_dhash(image: Image.Image) -> int:

        return int(
            str(
                imagehash.dhash(
                    image,
                    hash_size=16,
                )
            ),
            16,
        )