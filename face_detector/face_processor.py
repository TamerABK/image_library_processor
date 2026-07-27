import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from image_file_utils import find_supported_files
from image_loader import default_image_loader

from .face_scan_cache import FaceScanCache
from .interfaces import (
    FaceClusterer,
    FaceDetector,
    FaceRecognizer,
    FaceDatabase, FaceEmbedder
)
from .models import (
    DetectedFace,
    EmbeddedFace,
    FaceProcessorResult,
    KnownPersonResult,
    RecognizedFace,
)


class FaceProcessor:

    SUPPORTED_EXTENSIONS = default_image_loader.supported_extensions()

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        recognizer: FaceRecognizer,
        clusterer: FaceClusterer,
        database: FaceDatabase,
        worker_factory: Callable[[], FaceDetector] | None = None,
        max_workers: int | None = None,
        embed_batch_size: int = 64,
    ):
        self._detector = detector
        self._embedder = embedder
        self._recognizer = recognizer
        self._clusterer = clusterer
        self._database = database
        self._worker_factory = worker_factory
        self._max_workers = max_workers
        self._embed_batch_size = max(1, embed_batch_size)
        self._thread_local = threading.local()
        self._cache = FaceScanCache()

    def scan_folder(
        self,
        folder: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
        file_extensions: tuple[str, ...] | None = None,
        orientation_filter: str | None = None,
        known_people_only: bool = False,
    ) -> FaceProcessorResult:

        folder = Path(folder)

        known_faces = []
        unknown_faces: list[EmbeddedFace] = []
        cache_signature = (
            self._database.cache_signature()
            if known_people_only
            else None
        )

        image_files = find_supported_files(
            folder,
            self.SUPPORTED_EXTENSIONS,
            file_extensions,
            orientation_filter=orientation_filter,
        )
        total_files = len(image_files)

        if progress_callback:
            progress_callback(0, total_files)

        processed_results: list[_ProcessedImageResult | None] = [None] * total_files
        pending: list[tuple[int, Path, int, int]] = []
        completed = 0

        for index, path in enumerate(image_files):
            normalized_path = path.resolve()

            try:
                stat = normalized_path.stat()
            except OSError:
                processed_results[index] = _ProcessedImageResult(
                    recognized=[],
                    unknown=[],
                )
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_files)
                continue

            cached_faces = self._get_cached_faces(
                normalized_path,
                stat.st_size,
                stat.st_mtime_ns,
                include_unknown_faces=not known_people_only,
                database_signature=cache_signature,
            )

            if cached_faces is not None:
                processed_results[index] = self._classify_faces(cached_faces)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_files)
                continue

            pending.append(
                (index, normalized_path, stat.st_size, stat.st_mtime_ns)
            )

        max_workers = self._resolve_max_workers(len(pending))
        pending_detected: dict[int, _PendingDetectedImage] = {}
        pending_face_requests: list[_PendingFaceRequest] = []

        def finalize_detected_image(detected_image: "_PendingDetectedImage") -> None:
            nonlocal completed
            processed_result = self._classify_faces(detected_image.embedded_faces)
            self._store_cached_faces(
                detected_image.path,
                detected_image.file_size,
                detected_image.mtime_ns,
                self._faces_for_cache(
                    detected_image.embedded_faces,
                    processed_result.recognized,
                    known_people_only,
                ),
                known_people_only=known_people_only,
                database_signature=cache_signature,
            )
            processed_results[detected_image.index] = processed_result
            completed += 1
            if progress_callback:
                progress_callback(completed, total_files)

        def flush_face_requests(force: bool = False) -> None:
            while pending_face_requests and (
                force or len(pending_face_requests) >= self._embed_batch_size
            ):
                batch_size = min(self._embed_batch_size, len(pending_face_requests))
                batch = pending_face_requests[:batch_size]
                del pending_face_requests[:batch_size]

                embedded_faces = self._embed_requests(batch)
                for request, embedded_face in zip(batch, embedded_faces):
                    detected_image = pending_detected[request.owner_index]
                    detected_image.embedded_faces.append(embedded_face)
                    if len(detected_image.embedded_faces) == len(detected_image.detected_faces):
                        finalize_detected_image(detected_image)
                        pending_detected.pop(request.owner_index, None)
                if len(embedded_faces) != len(batch):
                    raise RuntimeError(
                        f"Expected {len(batch)} embedded faces but got {len(embedded_faces)}."
                    )

        def consume_detected_image(
            index: int,
            path: Path,
            file_size: int,
            mtime_ns: int,
            image: np.ndarray | None,
            detected_faces: list[DetectedFace],
        ) -> None:
            if image is None or not detected_faces:
                detected_image = _PendingDetectedImage(
                    index=index,
                    path=path,
                    file_size=file_size,
                    mtime_ns=mtime_ns,
                    detected_faces=[],
                    embedded_faces=[],
                )
                finalize_detected_image(detected_image)
                return

            pending_detected[index] = _PendingDetectedImage(
                index=index,
                path=path,
                file_size=file_size,
                mtime_ns=mtime_ns,
                detected_faces=detected_faces,
                embedded_faces=[],
            )
            pending_face_requests.extend(
                _PendingFaceRequest(
                    owner_index=index,
                    image=image,
                    face=face,
                )
                for face in detected_faces
            )
            flush_face_requests()

        if max_workers == 1:
            for index, path, file_size, mtime_ns in pending:
                image, detected_faces = self._load_and_detect(path)
                consume_detected_image(index, path, file_size, mtime_ns, image, detected_faces)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._load_and_detect, path): (
                        index,
                        path,
                        file_size,
                        mtime_ns,
                    )
                    for index, path, file_size, mtime_ns in pending
                }

                for future in as_completed(futures):
                    index, path, file_size, mtime_ns = futures[future]
                    image, detected_faces = future.result()
                    consume_detected_image(index, path, file_size, mtime_ns, image, detected_faces)

        flush_face_requests(force=True)

        for result in processed_results:
            if result is None:
                continue

            known_faces.extend(result.recognized)
            if not known_people_only:
                unknown_faces.extend(result.unknown)

        if known_people_only:
            unknown_clusters = []
        else:
            unknown_clusters = self._clusterer.cluster(
                unknown_faces,
            )

        people = self._group_known_people(
            known_faces,
        )

        if progress_callback:
            progress_callback(total_files, total_files)

        return FaceProcessorResult(
            known_people=people,
            unknown_clusters=unknown_clusters,
        )

    def _load_image(
        self,
        path: Path,
    ) -> np.ndarray | None:
        return default_image_loader.load_for_detection(path)

    def _detect_faces(
        self,
        detector: FaceDetector,
        image: np.ndarray,
        path: Path,
    ) -> list[DetectedFace]:
        return detector.detect(image, path)

    def _load_and_detect(
        self,
        path: Path,
    ) -> tuple[np.ndarray | None, list[DetectedFace]]:
        image = self._load_image(path)
        if image is None:
            return None, []
        detector = self._get_worker_detector()
        return image, self._detect_faces(detector, image, path)

    def _embed_requests(
        self,
        face_requests: list["_PendingFaceRequest"],
    ) -> list[EmbeddedFace]:
        return self._embedder.embed_requests(
            [(request.image, request.face) for request in face_requests]
        )

    def _classify_faces(
        self,
        faces: list[EmbeddedFace],
    ) -> "_ProcessedImageResult":
        recognized, unknown = self._recognizer.recognize(faces)

        return _ProcessedImageResult(
            recognized=recognized,
            unknown=unknown,
        )

    def _get_cached_faces(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
        include_unknown_faces: bool,
        database_signature: str | None,
    ) -> list[EmbeddedFace] | None:
        try:
            return self._cache.get(
                path,
                file_size,
                mtime_ns,
                include_unknown_faces=include_unknown_faces,
                database_signature=database_signature,
            )
        except Exception:
            return None

    def _store_cached_faces(
        self,
        path: Path,
        file_size: int,
        mtime_ns: int,
        faces: list[EmbeddedFace],
        known_people_only: bool,
        database_signature: str | None,
    ) -> None:
        try:
            metadata = default_image_loader.read_metadata(path)
            coverage = (
                FaceScanCache.COVERAGE_RECOGNIZED_ONLY
                if known_people_only
                else FaceScanCache.COVERAGE_ALL_FACES
            )
            self._cache.put(
                path,
                file_size,
                mtime_ns,
                faces,
                coverage=coverage,
                database_signature=database_signature,
                width=metadata.width if metadata is not None else None,
                height=metadata.height if metadata is not None else None,
                is_raw=metadata.is_raw if metadata is not None else None,
            )
        except Exception:
            return

    @staticmethod
    def _faces_for_cache(
        faces: list[EmbeddedFace],
        recognized_faces: list[RecognizedFace],
        known_people_only: bool,
    ) -> list[EmbeddedFace]:
        if not known_people_only:
            return faces

        return [
            EmbeddedFace(
                bbox=face.bbox,
                confidence=face.confidence,
                landmarks=face.landmarks,
                embedding=face.embedding,
                path=face.path,
            )
            for face in recognized_faces
        ]

    def _get_worker_detector(
        self,
    ) -> FaceDetector:
        if self._worker_factory is None:
            return self._detector

        detector = getattr(self._thread_local, "detector", None)
        if detector is None:
            detector = self._worker_factory()
            self._thread_local.detector = detector

        return detector

    def _resolve_max_workers(
        self,
        total_files: int,
    ) -> int:
        if total_files <= 1:
            return 1

        if self._max_workers is not None:
            return max(1, min(self._max_workers, total_files))

        cpu_count = os.cpu_count() or 1
        return max(1, min(total_files, cpu_count))

    def _group_known_people(
        self,
        faces,
    ) -> list[KnownPersonResult]:

        grouped = defaultdict(set)

        for face in faces:

            grouped[face.person_id].add(face.path)

        results = []

        for person_id, photos in grouped.items():

            person = self._database.get_person(
                person_id,
            )

            if person is None:
                continue

            results.append(
                KnownPersonResult(
                    person_id=person.id,
                    name=person.name,
                    photos=sorted(photos),
                )
            )

        results.sort(
            key=lambda p: len(p.photos),
            reverse=True,
        )

        return results


@dataclass(slots=True)
class _ProcessedImageResult:
    recognized: list
    unknown: list


@dataclass(slots=True)
class _PendingFaceRequest:
    owner_index: int
    image: np.ndarray
    face: DetectedFace


@dataclass(slots=True)
class _PendingDetectedImage:
    index: int
    path: Path
    file_size: int
    mtime_ns: int
    detected_faces: list[DetectedFace]
    embedded_faces: list[EmbeddedFace]
