from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np

from image_file_utils import find_supported_files
from image_loader import default_image_loader

from .cache import BlurScanCache


def _detect_worker(detector, path):

    try:
        score = detector.detect(path)
    except Exception:
        return path, None

    return path, score


@dataclass
class BlurResult:
    laplacian: float
    sobel: float
    local_contrast: float

    lap_norm: float
    sobel_norm: float
    contrast_norm: float

    final_score: float
    status: str


@dataclass(slots=True)
class BlurScanResult:
    path: Path
    result: BlurResult


class BlurDetector:
    supported_extensions = default_image_loader.supported_extensions()

    def __init__(
            self,
            grid_size=8,
            lap_max=400.0,
            sobel_max=60.0,
            contrast_max=50.0,
            review_threshold=0.45,
            sharp_threshold=0.70,
            max_dimension=1000,
    ):
        self.grid_size = grid_size

        self.lap_max = lap_max
        self.sobel_max = sobel_max
        self.contrast_max = contrast_max

        self.review_threshold = review_threshold
        self.sharp_threshold = sharp_threshold

        self.max_dimension = max_dimension

        cv2.setUseOptimized(True)
        self._cache = BlurScanCache()

    def detect(self, image_path: str | Path) -> BlurResult:
        gray = default_image_loader.load_grayscale(
            Path(image_path),
            max_dimension=self.max_dimension,
        )

        if gray is None:
            raise ValueError(f"Could not open image: {image_path}")

        lap_score = self._laplacian(gray)
        sobel_score = self._sobel(gray)
        contrast_score = self._local_contrast(gray)

        lap_norm = min(lap_score / self.lap_max, 1.0)
        sobel_norm = min(sobel_score / self.sobel_max, 1.0)
        contrast_norm = min(contrast_score / self.contrast_max, 1.0)

        final_score = (
            0.45 * lap_norm +
            0.35 * sobel_norm +
            0.20 * contrast_norm
        )

        if final_score >= self.sharp_threshold:
            status = "Sharp"
        elif final_score >= self.review_threshold:
            status = "Review"
        else:
            status = "Blurry"

        return BlurResult(
            laplacian=lap_score,
            sobel=sobel_score,
            local_contrast=contrast_score,
            lap_norm=lap_norm,
            sobel_norm=sobel_norm,
            contrast_norm=contrast_norm,
            final_score=final_score,
            status=status,
        )

    @staticmethod
    def _laplacian(gray):
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return lap.var()

    @staticmethod
    def _sobel(gray):
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        return magnitude.mean()

    def _local_contrast(self, gray):
        h, w = gray.shape

        tile_h = h // self.grid_size
        tile_w = w // self.grid_size

        if tile_h == 0 or tile_w == 0:
            return 0.0

        gray = gray[
            : tile_h * self.grid_size,
            : tile_w * self.grid_size,
        ]

        tiles = gray.reshape(
            self.grid_size,
            tile_h,
            self.grid_size,
            tile_w,
        ).swapaxes(1, 2)

        return float(tiles.std(axis=(2, 3)).mean())


    def scan_folder(
            self,
            folder_path: str | Path,
            progress_callback: Callable[[int, int], None] | None = None,
            file_extensions: tuple[str, ...] | None = None,
            orientation_filter: str | None = None,
    ) -> List[BlurScanResult]:
        folder = Path(folder_path)

        paths = find_supported_files(
            folder,
            self.supported_extensions,
            file_extensions,
            orientation_filter=orientation_filter,
        )

        total = len(paths)

        if progress_callback is not None:
            progress_callback(0, total)

        if total == 0:
            return []

        blurry_results: list[BlurScanResult] = []
        pending_paths: list[tuple[Path, int, int]] = []
        completed = 0
        pending_metadata: dict[Path, tuple[int | None, int | None, bool | None]] = {}

        for path in paths:
            normalized_path = path.resolve()

            try:
                stat = normalized_path.stat()
            except OSError:
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
                continue

            cached = self._get_cached_result(
                normalized_path,
                stat.st_size,
                stat.st_mtime_ns,
            )

            if cached is not None:
                if cached.status == "Blurry":
                    blurry_results.append(
                        BlurScanResult(path=normalized_path, result=cached)
                    )

                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
                continue

            metadata = default_image_loader.read_metadata(normalized_path)
            pending_paths.append((normalized_path, stat.st_size, stat.st_mtime_ns))
            pending_metadata[normalized_path] = (
                metadata.width if metadata is not None else None,
                metadata.height if metadata is not None else None,
                metadata.is_raw if metadata is not None else None,
            )

        executor_cls = ThreadPoolExecutor if os.name == "nt" else ProcessPoolExecutor

        with executor_cls(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(_detect_worker, self, path)
                for path, _, _ in pending_paths
            ]

            pending_meta = {
                path: (file_size, mtime_ns)
                for path, file_size, mtime_ns in pending_paths
            }

            for future in as_completed(futures):
                path, result = future.result()
                if result is not None:
                    file_size, mtime_ns = pending_meta[path]
                    width, height, is_raw = pending_metadata.get(path, (None, None, None))
                    self._store_cached_result(
                        path,
                        file_size,
                        mtime_ns,
                        result,
                        width=width,
                        height=height,
                        is_raw=is_raw,
                    )

                    if result.status == "Blurry":
                        blurry_results.append(
                            BlurScanResult(path=path, result=result)
                        )

                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)

        return blurry_results

    def find_blurry_photos(self, folder_path: str | Path) -> List[str]:
        return [str(result.path) for result in self.scan_folder(folder_path)]

    def _get_cached_result(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
    ) -> BlurResult | None:
        try:
            return self._cache.get(path, file_size, mtime_ns)
        except Exception:
            return None

    def _store_cached_result(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
        result: BlurResult,
        *,
        width: int | None = None,
        height: int | None = None,
        is_raw: bool | None = None,
    ) -> None:
        try:
            self._cache.put(
                path,
                file_size,
                mtime_ns,
                result,
                width=width,
                height=height,
                is_raw=is_raw,
            )
        except Exception:
            return
