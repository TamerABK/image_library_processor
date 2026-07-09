from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np


def _detect_worker(args):
    detector, path = args

    try:
        score = detector.detect(path)

        if score.status == "Blurry":
            return BlurScanResult(path=path, result=score)

    except Exception:
        pass

    return None


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
    supported_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )

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

    def detect(self, image_path: str | Path) -> BlurResult:
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not open image: {image_path}")

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            raise ValueError(f"Could not open image: {image_path}")

        h, w = gray.shape

        largest = max(h, w)

        if largest > self.max_dimension:
            scale = self.max_dimension / largest

            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

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
    ) -> List[BlurScanResult]:
        folder = Path(folder_path)

        paths = [
            p
            for p in folder.rglob("*")
            if p.is_file()
               and p.suffix.lower() in self.supported_extensions
        ]

        total = len(paths)

        if progress_callback is not None:
            progress_callback(0, total)

        if total == 0:
            return []

        blurry_results: list[BlurScanResult] = []

        executor_cls = ThreadPoolExecutor if os.name == "nt" else ProcessPoolExecutor

        with executor_cls(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(_detect_worker, (self, path))
                for path in paths
            ]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    blurry_results.append(result)

                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)

        return blurry_results

    def find_blurry_photos(self, folder_path: str | Path) -> List[str]:
        return [str(result.path) for result in self.scan_folder(folder_path)]




if __name__=="__main__":
    detector = BlurDetector()

    result = detector.detect("/home/tamer/PycharmProjects/image_deduplicator/data/sharp.jpeg")

    print(f"Laplacian      : {result.laplacian:.2f}")
    print(f"Sobel          : {result.sobel:.2f}")
    print(f"Local Contrast : {result.local_contrast:.2f}")

    print(f"Final Score    : {result.final_score:.3f}")
    print(f"Status         : {result.status}")