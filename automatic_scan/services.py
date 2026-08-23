from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from app_paths import app_data_path, model_path
from blur_detector.blur_detector import BlurDetector, BlurResult
from blur_detector.cache import BlurScanCache
from duplicate_detector.cache import DuplicatePhotoCache
from duplicate_detector.candidate_generator import CandidateGenerator
from duplicate_detector.config import DetectorConfig
from duplicate_detector.indexer import ImageIndexer
from duplicate_detector.models import CandidatePair, PhotoInfo
from duplicate_detector.orb import OrbVerifier
from duplicate_detector.union_find import UnionFind
from face_analyzer.default_face_analyzer import DefaultFaceAnalyzer
from face_detector.arc_embedder import ArcFaceEmbedder
from face_detector.connected_face_clusterer import ConnectedComponentFaceClusterer
from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_detector.face_aligner import FaceAligner
from face_detector.face_database_sqlite import SQLiteFaceDatabase
from face_detector.onnx_runtime import select_providers
from face_detector.scrfd_face_detector import SCRFDFaceDetector
from face_processing.cache import FaceScanCache, ImageFaceAnalysisCache
from face_processing.models import DetectedFace, EmbeddedFace, Match, RecognizedFace, UnknownCluster
from face_processing.recognition import DefaultFaceRecognizer
from grouping.models import VibeGroupingResult
from grouping.vibe import VibeGroupingPreset, VibeGroupingProcessor, preset_config
from grouping.vibe.embedder import load_embedder
from image_loader import default_image_loader
from scan_controls import CancellationToken

from .config import AutomaticScanConfig, ExecutionProvider, VibeDetail
from .models import DiscoveryRecord
from .perf import get_active_profiler


def _resolve_onnx_providers(execution_provider: ExecutionProvider) -> list[str] | None:
    if execution_provider is ExecutionProvider.AUTOMATIC:
        return None
    if execution_provider is ExecutionProvider.CPU:
        return ["CPUExecutionProvider"]
    return select_providers(
        [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "OpenVINOExecutionProvider",
        "ROCMExecutionProvider",
        "CPUExecutionProvider",
        ]
    )


class AutomaticScanServices:
    def __init__(
        self,
        config: AutomaticScanConfig,
        *,
        duplicate_config: DetectorConfig | None = None,
        blur_detector: BlurDetector | None = None,
    ) -> None:
        profiler = get_active_profiler()
        self._config = config

        database_started = time.perf_counter()
        self._blur_detector = blur_detector or BlurDetector()
        self._blur_cache = BlurScanCache()

        self._duplicate_config = duplicate_config or DetectorConfig()
        self._duplicate_cache = DuplicatePhotoCache()
        self._duplicate_indexer = ImageIndexer(self._duplicate_config)
        self._duplicate_candidates = CandidateGenerator(self._duplicate_config)
        self._duplicate_verifier = OrbVerifier(self._duplicate_config)

        self._analysis_cache = ImageFaceAnalysisCache()
        self._face_scan_cache = FaceScanCache()
        if profiler is not None:
            profiler.record_startup(
                "startup.open_databases",
                time.perf_counter() - database_started,
            )

        model_setup_started = time.perf_counter()
        providers = _resolve_onnx_providers(config.performance.execution_provider)
        detector_model = model_path("scrfd_10g_bnkps.onnx")
        embedder_model = model_path("glintr100.onnx")
        eye_state_model = model_path("open_closed_eye.onnx")
        head_pose_model = model_path("sixdrepnet.onnx")
        similarity = CosineEmbeddingSimilarity()
        if profiler is not None:
            profiler.record_startup(
                "startup.load_models",
                time.perf_counter() - model_setup_started,
            )

        session_started = time.perf_counter()
        self._face_detector = SCRFDFaceDetector(
            model_path=detector_model,
            providers=providers,
        )
        self._face_embedder = ArcFaceEmbedder(
            model_path=embedder_model,
            aligner=FaceAligner(),
            providers=providers,
        )
        self._face_analyzer = DefaultFaceAnalyzer(
            eye_state_model_path=eye_state_model,
            head_pose_model_path=head_pose_model,
            providers=providers,
        )
        self._face_database = SQLiteFaceDatabase(
            app_data_path("face_embeddings.sqlite3"),
            similarity,
        )
        self._face_recognizer = DefaultFaceRecognizer(
            self._face_database,
            similarity.default_threshold,
        )
        self._face_clusterer = ConnectedComponentFaceClusterer(
            similarity,
            strong_threshold=0.63,
            weak_threshold=0.55,
        )
        if profiler is not None:
            profiler.record_startup(
                "startup.create_onnx_sessions",
                time.perf_counter() - session_started,
            )
        self._providers = tuple(providers or ())
        self._vibe_providers = providers
        self._vibe_processors: dict[str, VibeGroupingProcessor] = {}

    def is_compatible(self, config: AutomaticScanConfig) -> bool:
        return tuple(_resolve_onnx_providers(config.performance.execution_provider) or ()) == self._providers

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self._blur_detector.supported_extensions)
                | set(self._duplicate_config.supported_extensions)
                | set(default_image_loader.supported_extensions())
                | set(VibeGroupingProcessor.supported_extensions)
            )
        )

    def discover(self, folder: Path) -> list[DiscoveryRecord]:
        records: list[DiscoveryRecord] = []
        allowed_extensions = set(self.supported_extensions)
        profiler = get_active_profiler()
        enumerate_started = time.perf_counter()
        paths = self._enumerate_paths(folder, allowed_extensions)
        if profiler is not None:
            profiler.record_discovery(
                "discovery.enumerate_paths",
                time.perf_counter() - enumerate_started,
            )
        for path in paths:
            try:
                stat_started = time.perf_counter()
                stat = path.stat()
                if profiler is not None:
                    profiler.record_discovery(
                        "discovery.stat_files",
                        time.perf_counter() - stat_started,
                    )
            except OSError:
                continue
            metadata_started = time.perf_counter()
            metadata = default_image_loader.read_metadata(path)
            if profiler is not None:
                profiler.record_discovery(
                    "discovery.read_headers",
                    time.perf_counter() - metadata_started,
                )
                if metadata is not None:
                    profiler.record_discovery(
                        "discovery.read_exif",
                        time.perf_counter() - metadata_started,
                    )
            width = None if metadata is None else metadata.width
            height = None if metadata is None else metadata.height
            orientation = None if metadata is None else metadata.orientation
            capture_timestamp = None if metadata is None else metadata.capture_timestamp
            if capture_timestamp is None:
                capture_timestamp = stat.st_mtime_ns / 1_000_000_000
            elif profiler is not None:
                profiler.record_discovery(
                    "discovery.read_capture_time",
                    time.perf_counter() - metadata_started,
                )
            records.append(
                DiscoveryRecord(
                    path=path,
                    file_size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    extension=path.suffix.lower(),
                    orientation=orientation,
                    capture_timestamp=capture_timestamp,
                    width=width,
                    height=height,
                )
            )
        return records

    def _enumerate_paths(
        self,
        folder: Path,
        allowed_extensions: set[str],
    ) -> list[Path]:
        discovered: list[Path] = []
        pending = [folder]
        while pending:
            current = pending.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        entry_path = Path(entry.path)
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry_path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        if entry_path.suffix.lower() not in allowed_extensions:
                            continue
                        discovered.append(entry_path.resolve())
            except OSError:
                continue
        discovered.sort()
        return discovered

    def get_blur_cached(self, record: DiscoveryRecord) -> BlurResult | None:
        return self._blur_cache.get(record.path, record.file_size, record.mtime_ns)

    def analyze_blur(self, record: DiscoveryRecord) -> BlurResult:
        result = self._blur_detector.detect(record.path)
        self._store_blur(record, result)
        return result

    def analyze_blur_from_image(
        self,
        record: DiscoveryRecord,
        image,
    ) -> BlurResult:
        result = self._blur_detector.detect_image(
            image,
            path=record.path,
        )
        self._store_blur(record, result)
        return result

    def _store_blur(
        self,
        record: DiscoveryRecord,
        result: BlurResult,
    ) -> None:
        self._blur_cache.put(
            record.path,
            record.file_size,
            record.mtime_ns,
            result,
            width=record.width,
            height=record.height,
            is_raw=record.extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
        )

    def get_duplicate_photo_cached(self, record: DiscoveryRecord) -> PhotoInfo | None:
        return self._duplicate_cache.get(record.path, record.file_size, record.mtime_ns)

    def index_duplicate_photo(self, record: DiscoveryRecord) -> PhotoInfo | None:
        return self._duplicate_indexer._index_image(record.path)

    def index_duplicate_photo_from_image(
        self,
        record: DiscoveryRecord,
        image,
    ) -> PhotoInfo | None:
        return self._duplicate_indexer.index_loaded_image(
            record.path,
            file_size=record.file_size,
            mtime_ns=record.mtime_ns,
            image_bgr=image,
            width=record.width,
            height=record.height,
            is_raw=record.extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
        )

    def generate_duplicate_candidates(
        self,
        photos: list[PhotoInfo],
        *,
        progress_callback: callable | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[CandidatePair]:
        return list(
            self._duplicate_candidates.generate(
                photos,
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
            )
        )

    def verify_duplicate_groups(
        self,
        photos: list[PhotoInfo],
        candidate_pairs: list[CandidatePair],
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> list[list[PhotoInfo]]:
        photo_to_index = {
            photo: index
            for index, photo in enumerate(photos)
        }
        union_find = UnionFind(len(photos))
        for pair in candidate_pairs:
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            if self._duplicate_verifier.verify(pair):
                union_find.union(photo_to_index[pair.left], photo_to_index[pair.right])
        groups: list[list[PhotoInfo]] = []
        for component in union_find.non_trivial_components():
            group = [photos[index] for index in component]
            group.sort(key=lambda item: item.path)
            groups.append(group)
        groups.sort(key=len, reverse=True)
        self._duplicate_verifier.clear_cache()
        return groups

    def get_face_analysis_cached(self, record: DiscoveryRecord) -> list[DetectedFace] | None:
        return self._analysis_cache.get(record.path, record.file_size, record.mtime_ns)

    def analyze_faces(self, record: DiscoveryRecord) -> list[DetectedFace]:
        image = default_image_loader.load_for_detection(record.path)
        if image is None:
            raise ValueError(f"Could not decode image for face analysis: {record.path}")
        return self.analyze_faces_from_image(record, image)

    def analyze_faces_from_image(
        self,
        record: DiscoveryRecord,
        image,
    ) -> list[DetectedFace]:
        detected_faces = self._face_detector.detect(image, record.path)
        for face_index, face in enumerate(detected_faces):
            face.index = face_index

        analysis_results = None
        analyze_many = getattr(self._face_analyzer, "analyze_many", None)
        if callable(analyze_many):
            try:
                analysis_results = list(analyze_many(image, detected_faces))
            except Exception:
                analysis_results = None

        if analysis_results is not None and len(analysis_results) == len(detected_faces):
            for face, analysis in zip(detected_faces, analysis_results):
                face.analysis = analysis
            analyzed_faces = list(detected_faces)
        else:
            analyzed_faces = []
            for face in detected_faces:
                try:
                    face.analysis = self._face_analyzer.analyze(image, face)
                except Exception:
                    pass
                analyzed_faces.append(face)
        self._analysis_cache.put(
            record.path,
            record.file_size,
            record.mtime_ns,
            analyzed_faces,
            width=record.width,
            height=record.height,
            is_raw=record.extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
        )
        return analyzed_faces

    def get_embedded_faces_cached(
        self,
        record: DiscoveryRecord,
    ) -> list[EmbeddedFace] | None:
        return self._face_scan_cache.get(
            record.path,
            record.file_size,
            record.mtime_ns,
            include_unknown_faces=True,
            require_analysis=True,
        )

    def embed_faces(
        self,
        record: DiscoveryRecord,
        faces: list[DetectedFace],
        *,
        minimum_embedding_utility: float,
    ) -> list[EmbeddedFace]:
        image = default_image_loader.load_for_detection(record.path)
        if image is None:
            raise ValueError(f"Could not decode image for face embedding: {record.path}")
        eligible_faces = [
            face
            for face in faces
            if face.analysis is not None
            and face.analysis.embedding_utility_score >= minimum_embedding_utility
        ]
        embedded_faces = self._face_embedder.embed(image, eligible_faces)
        self._face_scan_cache.put(
            record.path,
            record.file_size,
            record.mtime_ns,
            embedded_faces,
            coverage=FaceScanCache.COVERAGE_ALL_FACES,
            width=record.width,
            height=record.height,
            is_raw=record.extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
        )
        return embedded_faces

    def align_faces_for_embedding(
        self,
        image,
        faces: list[DetectedFace],
    ) -> list[np.ndarray]:
        return self._face_embedder.align_requests(
            [(image, face) for face in faces]
        )

    def embed_faces_batched(
        self,
        records: list[DiscoveryRecord],
        faces_by_path: dict[str, list[DetectedFace]],
        *,
        minimum_embedding_utility: float,
        prealigned_faces_by_path: dict[str, dict[int, np.ndarray]] | None = None,
    ) -> dict[str, list[EmbeddedFace]]:
        batch_size = self.embedding_batch_size()
        pending_aligned = []
        pending_faces: list[DetectedFace] = []
        embedded_by_path: dict[str, list[EmbeddedFace]] = {
            str(record.path): []
            for record in records
        }

        def flush() -> None:
            nonlocal pending_aligned, pending_faces
            if not pending_aligned:
                return
            embedded_faces = self._face_embedder.embed_aligned_faces(
                pending_aligned,
                pending_faces,
            )
            for embedded_face in embedded_faces:
                embedded_by_path[str(embedded_face.path)].append(embedded_face)
            pending_aligned = []
            pending_faces = []

        for record in records:
            faces = [
                face
                for face in faces_by_path.get(str(record.path), [])
                if face.analysis is not None
                and face.analysis.embedding_utility_score >= minimum_embedding_utility
            ]
            if not faces:
                continue
            aligned_faces_by_index = (
                prealigned_faces_by_path.pop(str(record.path), None)
                if prealigned_faces_by_path is not None
                else None
            )
            if aligned_faces_by_index is not None and all(
                face.index is not None and int(face.index) in aligned_faces_by_index
                for face in faces
            ):
                aligned_faces = [
                    aligned_faces_by_index[int(face.index)]
                    for face in faces
                ]
            else:
                image = default_image_loader.load_for_detection(record.path)
                if image is None:
                    raise ValueError(f"Could not decode image for face embedding: {record.path}")
                aligned_faces = self._face_embedder.align_requests(
                    [(image, face) for face in faces]
                )
            for face, aligned in zip(faces, aligned_faces):
                pending_faces.append(face)
                pending_aligned.append(aligned)
                if len(pending_aligned) >= batch_size:
                    flush()

        flush()

        for record in records:
            self._face_scan_cache.put(
                record.path,
                record.file_size,
                record.mtime_ns,
                embedded_by_path[str(record.path)],
                coverage=FaceScanCache.COVERAGE_ALL_FACES,
                width=record.width,
                height=record.height,
                is_raw=record.extension not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
            )
        return embedded_by_path

    def recognize_faces(
        self,
        embedded_faces: list[EmbeddedFace],
    ) -> tuple[list[RecognizedFace], list[EmbeddedFace]]:
        return self._face_recognizer.recognize(embedded_faces)

    def cluster_unknown_faces(
        self,
        embedded_faces: list[EmbeddedFace],
    ) -> list[UnknownCluster]:
        return self._face_clusterer.cluster(embedded_faces)

    def suggest_known_names(self, embedding_match: Match | None) -> tuple[str, ...]:
        if embedding_match is None:
            return tuple(self._face_database.list_people_names()[:12])
        person = self._face_database.get_person(embedding_match.person_id)
        names = self._face_database.list_people_names()
        if person is None or person.name not in names:
            return tuple(names[:12])
        names.remove(person.name)
        names.insert(0, person.name)
        return tuple(names[:12])

    def nearest_known_match(self, embedded_face: EmbeddedFace) -> Match | None:
        return self._face_database.find_nearest_embedding(embedded_face.embedding)

    def add_known_person(self, name: str, faces: list[EmbeddedFace]) -> int:
        person = self._face_database.add_person(name)
        for face in faces:
            self._face_database.add_embedding(person.id, face.embedding)
        return person.id

    def get_person_name(self, person_id: int) -> str | None:
        person = self._face_database.get_person(person_id)
        if person is None:
            return None
        return person.name

    def group_vibes(
        self,
        image_paths: list[str],
        *,
        progress_callback: callable | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> VibeGroupingResult:
        vibe_config = self._build_vibe_config()
        processor_key = json.dumps(vibe_config.cache_signature(), sort_keys=True)
        processor = self._vibe_processors.get(processor_key)
        if processor is None:
            embedder = load_embedder(vibe_config, providers=self._vibe_providers)
            processor = VibeGroupingProcessor(
                vibe_config,
                embedder=embedder,
                face_database=self._face_database,
            )
            self._vibe_processors[processor_key] = processor
        return processor.group(
            image_paths,
            folder_path=None,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
        )

    def runtime_info(self) -> dict[str, Any]:
        return {
            "face_detector": self._face_detector.runtime_info(),
            "face_embedder": self._face_embedder.runtime_info(),
            "embedding_batch_size": self.embedding_batch_size(),
            "vibe_batch_size": self.vibe_batch_size(),
        }

    def _build_vibe_config(self):
        if self._config.vibe.detail is VibeDetail.SESSION:
            preset = VibeGroupingPreset.SESSION
        elif self._config.vibe.detail is VibeDetail.TIGHT_SCENES:
            preset = VibeGroupingPreset.TIGHT_SCENES
        else:
            preset = VibeGroupingPreset.BALANCED_SCENES
        return preset_config(
            preset,
            include_background_embedding=self._config.vibe.include_background_embedding,
            batch_size=self.vibe_batch_size(),
        )

    def embedding_batch_size(self) -> int:
        configured = self._config.performance.batch_size
        if configured is not None and configured > 0:
            return configured
        return 64 if self._uses_gpu() else 32

    def vibe_batch_size(self) -> int:
        configured = self._config.performance.batch_size
        if configured is not None and configured > 0:
            return configured
        return 32 if self._uses_gpu() else 8

    def _uses_gpu(self) -> bool:
        if not self._providers:
            return False
        return any(
            provider.startswith(("CUDA", "Tensorrt", "ROCM", "Dml", "CoreML", "OpenVINO"))
            for provider in self._providers
        )
