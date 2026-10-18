from pathlib import Path
from dataclasses import replace
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from automatic_scan.config import AutomaticScanConfig, PerformanceConfig
from automatic_scan.services import AutomaticScanServices
from automatic_scan.orchestrator import AutomaticScanOrchestrator
from automatic_scan.models import DiscoveryRecord
from automatic_scan.project_pipeline import MAX_UNKNOWN_FACES, StageState
from automatic_scan.project_cache import ProjectStageCache
from automatic_scan.perf import AutomaticScanProfiler, active_profiler
from automatic_scan.scan_logging import LOG_FILENAME, get_scan_logger
from automatic_scan.state_store import AutomaticStateStore
from blur_detector.blur_detector import BlurDetector
from scan_controls import CancellationToken, ScanCancelledError
from ui.phase6_test_support import NoFacesServices


class FaceFixtureServices(NoFacesServices):
    """Controlled detections/embeddings; real face cache, quality, matching/clustering."""
    def initialize_faces(self):
        from face_detector.cosine_similarity import CosineEmbeddingSimilarity
        from face_detector.face_database_sqlite import SQLiteFaceDatabase
        from face_detector.connected_face_clusterer import ConnectedComponentFaceClusterer
        from face_processing.recognition import DefaultFaceRecognizer
        similarity = CosineEmbeddingSimilarity()
        self._face_database = SQLiteFaceDatabase(self._face_database_path, similarity)
        self._face_recognizer = DefaultFaceRecognizer(self._face_database, similarity.default_threshold)
        self._face_clusterer = ConnectedComponentFaceClusterer(similarity, .63, .55)

    def analyze_faces(self, record, **kwargs):
        from face_processing.models import DetectedFace
        from face_processing.test_processing import _sample_analysis
        face = DetectedFace((8, 4, 50, 30), .91,
            np.array([[15, 10], [45, 10], [30, 20], [20, 30], [40, 30]], dtype=np.float32),
            record.path, analysis=_sample_analysis())
        self._analysis_cache.put(record.path, record.file_size, record.mtime_ns, [face])
        return [face]

    def embed_faces(self, record, faces, **kwargs):
        from face_processing.models import EmbeddedFace
        vector = np.zeros(512, dtype=np.float32)
        vector[0 if record.path.name == "known.jpg" else 1] = 1
        embedded = [EmbeddedFace(f.bbox, f.confidence, f.landmarks, vector, f.path, f.analysis) for f in faces]
        self._face_scan_cache.put(record.path, record.file_size, record.mtime_ns, embedded,
            coverage=self._face_scan_cache.COVERAGE_ALL_FACES)
        return embedded


class BatchedFaceFixtureServices(FaceFixtureServices):
    """Use the production batch entry point with deterministic face data."""
    embed_faces = AutomaticScanServices.embed_faces

    def analyze_faces(self, record, **kwargs):
        from face_processing.models import DetectedFace
        from face_processing.test_processing import _sample_analysis
        faces = [DetectedFace((8, 4, 50, 30), .91,
            np.array([[15, 10], [45, 10], [30, 20], [20, 30], [40, 30]], dtype=np.float32),
            record.path, analysis=_sample_analysis()) for _ in range(3)]
        self._analysis_cache.put(record.path, record.file_size, record.mtime_ns, faces)
        return faces


class ProjectPipelineTests(unittest.TestCase):
    @staticmethod
    def _close_logger(logger):
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photos = self.root / "photos"
        self.photos.mkdir()
        self.path = self.photos / "flat.jpg"
        Image.new("RGB", (80, 40), "gray").save(self.path)
        self.service = NoFacesServices(AutomaticScanConfig(), lazy_models=True,
            cache_path=self.root / "analysis.sqlite3", face_database_path=self.root / "faces.sqlite3")
        self.orchestrator = AutomaticScanOrchestrator(state_store=AutomaticStateStore(self.root / "state"))
        self.events = []
        self.token = CancellationToken()

    def run_project(self, paths=None):
        self.events.clear()
        return self.orchestrator.process_project(self.photos, paths or [self.path], services=self.service,
            cache_path=self.root / "analysis.sqlite3", cancellation_token=self.token,
            emit=lambda *args: self.events.append(args))

    def event(self, stage, state=StageState.COMPLETE):
        return next(e for e in self.events if e[0] == stage and e[1] == state)

    def test_real_blur_semantics_and_diagnostics(self):
        stats = self.run_project()
        row = self.event("blur")[5][0]
        self.assertEqual("Blurry", row["status"])
        self.assertEqual(0, row["final_score"])
        self.assertEqual("UNDECIDED", row["reviewState"])
        self.assertIn("sobel", row)
        self.assertEqual(1, stats["blur"]["computed"])
        detector = self.service._blur_detector
        sharp = detector.detect_image(np.random.default_rng(123).integers(0, 256, (256, 256, 3), dtype=np.uint8))
        self.assertEqual("Sharp", sharp.status)
        self.assertGreaterEqual(sharp.final_score, detector.sharp_threshold)

    def test_valid_cache_reuses_pixels_and_project_aggregates(self):
        self.run_project()
        with patch.object(self.service, "analyze_blur", side_effect=AssertionError("recomputed blur")), \
             patch.object(self.service, "analyze_faces", side_effect=AssertionError("recomputed faces")), \
             patch.object(self.service, "index_duplicate_photo_from_image", side_effect=AssertionError("recomputed hash")), \
             patch.object(self.service, "verify_duplicate_groups", side_effect=AssertionError("recomputed ORB")):
            stats = self.run_project()
        for stage in ("blur", "duplicates", "faces", "embeddings"):
            self.assertEqual(1, stats[stage]["cache_hits"], stage)
        for stage in ("duplicates", "people", "ranking"):
            self.assertEqual(1, stats[stage]["aggregate_reused"], stage)
        self.assertTrue(all(self.event(stage) for stage in stats))

    def test_hot_cache_reuses_one_read_connection_per_stage(self):
        paths = [self.path]
        for index in range(29):
            path = self.photos / f"cached_{index}.jpg"
            Image.new("RGB", (80, 40), "gray").save(path)
            paths.append(path)
        self.run_project(paths)

        profiler = AutomaticScanProfiler()
        with active_profiler(profiler):
            stats = self.run_project(paths)
        for stage in ("blur", "duplicates", "faces", "embeddings"):
            self.assertEqual(30, stats[stage]["cache_hits"], stage)
            self.assertEqual(0, stats[stage]["computed"], stage)
        self.assertLess(profiler.database_metrics()["connect_count"], 100)

        changed = paths[-1]
        stat = changed.stat()
        os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        stats = self.run_project(paths)
        for stage in ("blur", "duplicates", "faces", "embeddings"):
            self.assertEqual(29, stats[stage]["cache_hits"], stage)
            self.assertEqual(1, stats[stage]["computed"], stage)

    def test_compatible_paths_chunks_queries_without_cross_project_matches(self):
        cache = ProjectStageCache(self.root / "compat.sqlite3")
        paths = [str(self.root / f"image_{index}.jpg") for index in range(405)]
        with cache.storage.connect() as db:
            db.executemany("INSERT INTO image_entries(path,file_size,mtime_ns) VALUES (?,?,?)",
                           [(path, 1, 1) for path in paths])
            db.executemany("INSERT INTO processor_compatibility(path,stage,signature) VALUES (?,?,?)",
                           [(path, "blur", "current" if index != 404 else "old")
                            for index, path in enumerate(paths)])
        with cache.storage.connect() as db:
            matches = cache.compatible_paths(paths + [str(self.root / "other.jpg")],
                                             "blur", "current", connection=db)
        self.assertEqual(set(paths[:-1]), matches)

    def test_modified_file_invalidates_all_dependent_stage_rows(self):
        self.run_project()
        stat = self.path.stat()
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        stats = self.run_project()
        for name in ("blur", "duplicates", "faces", "embeddings"):
            self.assertEqual(0, stats[name]["cache_hits"], name)
            self.assertEqual(1, stats[name]["computed"], name)

    def test_configuration_change_invalidates_only_affected_pixel_stage(self):
        self.run_project()
        self.service._blur_detector.review_threshold = 0.40
        stats = self.run_project()
        self.assertEqual(1, stats["blur"]["computed"])
        self.assertEqual(1, stats["duplicates"]["cache_hits"])
        self.assertEqual(1, stats["faces"]["cache_hits"])

    def test_model_fingerprint_invalidates_face_results(self):
        model = self.root / "model.onnx"
        model.write_bytes(b"model version 1")
        with patch("automatic_scan.project_pipeline.model_path", return_value=model):
            self.run_project()
            model.write_bytes(b"model version 2, different size")
            stats = self.run_project()
        self.assertEqual(1, stats["faces"]["computed"])
        self.assertEqual(1, stats["embeddings"]["computed"])
        self.assertEqual(1, stats["blur"]["cache_hits"])

    def test_corrupt_cache_is_recomputed(self):
        self.run_project()
        with patch.object(self.service, "get_blur_cached", side_effect=ValueError("bad persisted metric")):
            stats = self.run_project()
        self.assertEqual(1, stats["blur"]["computed"])
        self.assertEqual(0, stats["blur"]["file_errors"])

    def test_custom_cache_getter_keeps_original_call_contract(self):
        self.run_project()
        original = self.service.get_blur_cached
        with patch.object(self.service, "get_blur_cached", side_effect=lambda record: original(record)):
            stats = self.run_project()
        self.assertEqual(1, stats["blur"]["cache_hits"])
        self.assertEqual(0, stats["blur"]["computed"])

    def test_hashes_round_trip_without_sqlite_numeric_coercion(self):
        self.run_project()
        stat = self.path.stat()
        record = self.service.get_duplicate_photo_cached(DiscoveryRecord(self.path, stat.st_size,
            stat.st_mtime_ns, ".jpg", None, None, 80, 40))
        for value in (0, 16, 0x8000000000000000000000000000000000000000000000000000000000000000,
                      int("12345678901234567890" * 3, 16)):
            record = replace(record, phash=value, dhash=value)
            self.service._duplicate_cache.put(record, self.path.stat().st_mtime_ns)
            saved = self.service._duplicate_cache.get(self.path, record.file_size, self.path.stat().st_mtime_ns)
            self.assertEqual(value, saved.phash)
            self.assertEqual(value, saved.dhash)
        with self.service._duplicate_cache._storage.connect() as db:
            self.assertEqual("text", db.execute("SELECT typeof(phash) FROM duplicate_photo_cache").fetchone()[0])

    def test_new_file_reuses_existing_photo_results(self):
        self.run_project()
        new = self.photos / "new.png"
        Image.new("RGB", (90, 50), "white").save(new)
        stats = self.run_project([self.path, new])
        self.assertEqual(1, stats["blur"]["cache_hits"])
        self.assertEqual(1, stats["blur"]["computed"])
        self.assertEqual(2, len(self.event("blur")[5]))

    def test_removed_and_corrupt_files_are_isolated(self):
        corrupt = self.photos / "corrupt.jpg"
        corrupt.write_bytes(b"not an image")
        missing = self.photos / "missing.png"
        stats = self.run_project([self.path, corrupt, missing])
        self.assertEqual(2, stats["blur"]["file_errors"])
        self.assertEqual(1, len(self.event("blur")[5]))
        self.assertTrue(self.path.is_file())

    def test_all_corrupt_is_failed_and_retryable(self):
        self.path.write_bytes(b"corrupt")
        self.run_project()
        self.assertIn("could not finish", self.event("blur", StageState.FAILED)[4])

    def test_later_failure_preserves_progressive_blur(self):
        with patch.object(self.service, "initialize_faces", side_effect=RuntimeError("missing model")):
            self.run_project()
        blur = self.event("blur")
        face = self.event("faces", StageState.FAILED)
        self.assertLess(self.events.index(blur), self.events.index(face))
        self.assertEqual(1, len(blur[5]))
        self.event("duplicates")
        self.event("embeddings", StageState.CANCELLED)

    def test_scan_log_explains_people_matching_failure(self):
        self.service = FaceFixtureServices(AutomaticScanConfig(), lazy_models=True,
            cache_path=self.root / "analysis.sqlite3", face_database_path=self.root / "faces.sqlite3")
        log_root = self.root / "logs"
        with patch.dict(os.environ, {"IMAGE_DEDUPLICATOR_APP_DATA_ROOT": str(log_root)}), \
             patch.object(self.service, "recognize_faces", side_effect=RuntimeError("matching backend unavailable")):
            logger = get_scan_logger()
            self.addCleanup(self._close_logger, logger)
            self.run_project()
            self.event("blur")
            failed = self.event("people", StageState.FAILED)
            self.assertNotIn("matching backend unavailable", failed[4])
            log = (log_root / LOG_FILENAME).read_text(encoding="utf-8")
        self.assertIn("stage_started", log)
        self.assertIn("stage=people", log)
        self.assertIn("stage_failed", log)
        self.assertIn("RuntimeError: matching backend unavailable", log)
        self.assertIn("stage_completed", log)
        self.assertIn("stage=blur", log)
        self.assertIn("Traceback", log)

    def test_scan_log_identifies_failed_photo_and_operation(self):
        log_root = self.root / "logs"
        with patch.dict(os.environ, {"IMAGE_DEDUPLICATOR_APP_DATA_ROOT": str(log_root)}), \
             patch.object(self.service, "analyze_blur", side_effect=RuntimeError("decode failed")):
            logger = get_scan_logger()
            self.addCleanup(self._close_logger, logger)
            self.run_project()
            self.event("blur", StageState.FAILED)
            log = (log_root / LOG_FILENAME).read_text(encoding="utf-8")
        self.assertIn("photo_failed", log)
        self.assertIn("stage=blur operation=compute", log)
        self.assertIn(str(self.path), log)
        self.assertIn("RuntimeError: decode failed", log)

    def test_candidate_capacity_fails_honestly(self):
        new = self.photos / "other.jpg"
        Image.new("RGB", (80, 40), "gray").save(new)
        with patch("automatic_scan.project_pipeline.MAX_CANDIDATE_PAIRS", 0):
            self.run_project([self.path, new])
        self.event("duplicates", StageState.FAILED)
        self.event("blur")

    def test_cooperative_cancel_does_not_start_later_stages(self):
        original = self.service.analyze_blur
        def compute(record):
            result = original(record)
            self.token.cancel()
            return result
        with patch.object(self.service, "analyze_blur", side_effect=compute):
            self.run_project()
        self.assertFalse(any(e[0] == "duplicates" and e[1] == StageState.RUNNING for e in self.events))
        self.event("blur", StageState.CANCELLED)

    def test_shared_cache_connection_context_closes_handles(self):
        storage = self.service._blur_cache._storage
        with storage.connect() as connection:
            connection.execute("SELECT 1")
        with self.assertRaises(Exception):
            connection.execute("SELECT 1")

    def test_threshold_boundaries_owned_by_detector(self):
        detector = self.service._blur_detector
        # Metrics normalize to the same value, so their weighted sum equals it.
        for value, expected in ((0.449, "Blurry"), (0.45, "Review"), (0.699, "Review"), (0.70, "Sharp")):
            with patch.object(detector, "_laplacian", return_value=value * detector.lap_max), \
                 patch.object(detector, "_sobel", return_value=value * detector.sobel_max), \
                 patch.object(detector, "_local_contrast", return_value=value * detector.contrast_max):
                result = detector.detect_image(np.zeros((64, 64, 3), dtype=np.uint8))
            self.assertEqual(expected, result.status)

    def test_face_quality_matching_clustering_and_cache_reuse(self):
        from face_detector.cosine_similarity import CosineEmbeddingSimilarity
        from face_detector.face_database_sqlite import SQLiteFaceDatabase
        database = SQLiteFaceDatabase(self.root / "faces.sqlite3", CosineEmbeddingSimilarity())
        person = database.add_person("Fixture person")
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1
        database.add_embedding(person.id, vector)
        self.service = FaceFixtureServices(AutomaticScanConfig(), lazy_models=True,
            cache_path=self.root / "analysis.sqlite3", face_database_path=self.root / "faces.sqlite3")
        known = self.photos / "known.jpg"
        other = self.photos / "other.jpg"
        Image.new("RGB", (80, 40), "gray").save(known)
        Image.new("RGB", (80, 40), "gray").save(other)
        self.run_project([self.path, known, other])
        self.assertEqual(3, self.event("face_quality")[5]["photo_count"])
        payload = self.event("people")[5]
        self.assertEqual(person.id, payload["known"][0]["person_id"])
        self.assertEqual(2, payload["unknown"][0]["face_count"])
        self.assertEqual(2, len(payload["unknown"][0]["faces"]))
        matching_progress = [(event[2], event[3]) for event in self.events
                             if event[0] == "people" and event[1] == StageState.RUNNING]
        self.assertIn((3, 3), matching_progress)
        self.assertEqual((0, 0), matching_progress[-1])  # Unknown-face clustering is indeterminate.
        with patch.object(self.service, "analyze_faces", side_effect=AssertionError("recomputed")), \
             patch.object(self.service, "embed_faces", side_effect=AssertionError("recomputed")), \
             patch.object(self.service, "cluster_unknown_faces", side_effect=AssertionError("recomputed")):
            stats = self.run_project([self.path, known, other])
        self.assertEqual(3, stats["faces"]["cache_hits"])
        self.assertEqual(3, stats["embeddings"]["cache_hits"])
        self.assertEqual(1, stats["people"]["aggregate_reused"])

    def test_unknown_face_limit_fails_only_people_stage(self):
        from face_processing.models import EmbeddedFace

        self.service = FaceFixtureServices(AutomaticScanConfig(), lazy_models=True,
            cache_path=self.root / "analysis.sqlite3", face_database_path=self.root / "faces.sqlite3")
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1
        face = EmbeddedFace((8, 4, 50, 30), .91, np.zeros((5, 2), dtype=np.float32),
                            vector, self.path)
        faces = [face] * (MAX_UNKNOWN_FACES + 1)
        with patch.object(self.service, "embed_faces", return_value=faces), \
             patch.object(self.service, "recognize_faces", side_effect=lambda batch: ([], batch)), \
             patch.object(self.service, "cluster_unknown_faces", return_value=[]) as cluster:
            self.run_project()
        failed = self.event("people", StageState.FAILED)
        self.assertIn("too many unknown faces", failed[4])
        self.assertEqual(MAX_UNKNOWN_FACES + 1, self.event("embeddings")[5]["face_count"])
        cluster.assert_not_called()
        for stage in ("blur", "duplicates", "faces", "face_quality", "embeddings", "ranking"):
            self.event(stage)

    def test_people_matching_handles_1500_unknown_faces(self):
        from face_processing.models import EmbeddedFace

        self.service = FaceFixtureServices(AutomaticScanConfig(), lazy_models=True,
            cache_path=self.root / "analysis.sqlite3", face_database_path=self.root / "faces.sqlite3")
        vectors = np.random.default_rng(36).normal(size=(1500, 512)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        faces = [EmbeddedFace((8, 4, 50, 30), .91, np.zeros((5, 2), dtype=np.float32),
            vector, self.path) for vector in vectors]
        with patch.object(self.service, "embed_faces", return_value=faces):
            self.run_project()

        self.assertEqual(1500, self.event("embeddings")[5]["face_count"])
        self.assertEqual(1500, len(self.event("people")[5]["unknown"]))

    def test_embedding_batches_are_bounded_and_cancel_between_native_calls(self):
        from face_processing.models import DetectedFace, EmbeddedFace
        from face_processing.test_processing import _sample_analysis
        from types import SimpleNamespace
        self.service._config = replace(self.service._config, performance=PerformanceConfig(batch_size=8))
        faces = [DetectedFace((0, 0, 40, 40), .9, np.zeros((5, 2)), self.path,
            analysis=_sample_analysis()) for _ in range(18)]
        stat = self.path.stat()
        record = DiscoveryRecord(self.path, stat.st_size, stat.st_mtime_ns, ".jpg", None, None, 80, 40)
        calls = []
        def embed(image, batch):
            calls.append(len(batch))
            return [EmbeddedFace(f.bbox, f.confidence, f.landmarks, np.zeros(512), f.path, f.analysis) for f in batch]
        self.service._face_embedder = SimpleNamespace(embed=embed)
        result = AutomaticScanServices.embed_faces(self.service, record, faces, minimum_embedding_utility=0)
        self.assertEqual([8, 8, 2], calls)
        self.assertEqual(18, len(result))
        token = CancellationToken()
        calls.clear()
        def cancelled_embed(image, batch):
            token.cancel()
            return embed(image, batch)
        self.service._face_embedder.embed = cancelled_embed
        with self.assertRaises(ScanCancelledError):
            AutomaticScanServices.embed_faces(self.service, record, faces,
                minimum_embedding_utility=0, cancellation_token=token)
        self.assertEqual([8], calls)

    def test_project_embeddings_batch_across_photos_and_reuse_per_photo_cache(self):
        from face_processing.models import EmbeddedFace
        from types import SimpleNamespace

        self.service = BatchedFaceFixtureServices(
            AutomaticScanConfig(performance=PerformanceConfig(batch_size=8)),
            lazy_models=True, cache_path=self.root / "analysis.sqlite3",
            face_database_path=self.root / "faces.sqlite3")
        paths = [self.path]
        for index in range(3):
            path = self.photos / f"extra_{index}.jpg"
            Image.new("RGB", (80, 40), "gray").save(path)
            paths.append(path)
        calls = []
        def embed(aligned, faces):
            calls.append(len(faces))
            self.assertLessEqual(len(aligned), 8)
            return [EmbeddedFace(f.bbox, f.confidence, f.landmarks,
                np.eye(1, 512, dtype=np.float32)[0], f.path, f.analysis) for f in faces]
        self.service._face_embedder = SimpleNamespace(
            align_requests=lambda requests: [np.zeros((112, 112, 3), dtype=np.uint8)
                                             for _ in requests],
            embed_aligned_faces=embed)

        stats = self.run_project(paths)
        self.event("embeddings")
        self.assertEqual([8, 4], calls)
        self.assertEqual(4, stats["embeddings"]["computed"])
        self.assertEqual(12, self.event("embeddings")[5]["face_count"])

        calls.clear()
        stats = self.run_project(paths)
        self.assertEqual([], calls)
        self.assertEqual(4, stats["embeddings"]["cache_hits"])

        changed = paths[1]
        stat = changed.stat()
        os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        stats = self.run_project(paths)
        self.assertEqual([3], calls)
        self.assertEqual(3, stats["embeddings"]["cache_hits"])
        self.assertEqual(1, stats["embeddings"]["computed"])

    def test_cross_photo_batch_isolates_decode_failure_and_cancellation(self):
        from face_processing.models import DetectedFace, EmbeddedFace
        from face_processing.test_processing import _sample_analysis
        from types import SimpleNamespace

        paths = [self.path]
        for index in range(2):
            path = self.photos / f"extra_{index}.jpg"
            Image.new("RGB", (80, 40), "gray").save(path)
            paths.append(path)
        records, faces_by_path = [], {}
        for path in paths:
            stat = path.stat()
            records.append(DiscoveryRecord(path, stat.st_size, stat.st_mtime_ns,
                                           ".jpg", None, None, 80, 40))
            faces_by_path[str(path)] = [DetectedFace((0, 0, 40, 40), .9,
                np.zeros((5, 2)), path, analysis=_sample_analysis())]
        self.service._config = replace(self.service._config,
            performance=PerformanceConfig(batch_size=8))
        calls = []
        def embed(aligned, faces):
            calls.append(len(faces))
            return [EmbeddedFace(f.bbox, f.confidence, f.landmarks,
                np.eye(1, 512, dtype=np.float32)[0], f.path, f.analysis) for f in faces]
        self.service._face_embedder = SimpleNamespace(
            align_requests=lambda requests: [np.zeros((112, 112, 3), dtype=np.uint8)
                                             for _ in requests],
            embed_aligned_faces=embed)
        from image_loader import default_image_loader
        original_load = default_image_loader.load_for_detection
        def load(path):
            return None if path == paths[1] else original_load(path)
        errors = []
        with patch.object(default_image_loader, "load_for_detection", side_effect=load):
            results = self.service.embed_faces_batched(records, faces_by_path,
                minimum_embedding_utility=0,
                on_error=lambda record, exc: errors.append((record.path, type(exc))))
        self.assertEqual({str(paths[0]), str(paths[2])}, set(results))
        self.assertEqual([(paths[1], ValueError)], errors)
        self.assertEqual([2], calls)
        self.assertIsNone(self.service.get_embedded_faces_cached(records[1]))

        token = CancellationToken()
        def cancel_embed(aligned, faces):
            token.cancel()
            return embed(aligned, faces)
        self.service._face_embedder.embed_aligned_faces = cancel_embed
        with self.assertRaises(ScanCancelledError):
            self.service.embed_faces_batched(records, faces_by_path,
                minimum_embedding_utility=0, cancellation_token=token)
        self.assertEqual([2, 3], calls)

    def test_native_embedding_batch_failure_recovers_per_photo(self):
        from face_processing.models import EmbeddedFace
        from types import SimpleNamespace

        self.service = BatchedFaceFixtureServices(
            AutomaticScanConfig(performance=PerformanceConfig(batch_size=8)),
            lazy_models=True, cache_path=self.root / "analysis.sqlite3",
            face_database_path=self.root / "faces.sqlite3")
        other = self.photos / "other.jpg"
        Image.new("RGB", (80, 40), "gray").save(other)
        calls = []
        def embedded(faces):
            return [EmbeddedFace(f.bbox, f.confidence, f.landmarks,
                np.eye(1, 512, dtype=np.float32)[0], f.path, f.analysis) for f in faces]
        def fail_batch(aligned, faces):
            calls.append(("batch", len(faces)))
            raise RuntimeError("native batch unavailable")
        def single(image, faces):
            calls.append(("photo", len(faces)))
            return embedded(faces)
        self.service._face_embedder = SimpleNamespace(
            align_requests=lambda requests: [np.zeros((112, 112, 3), dtype=np.uint8)
                                             for _ in requests],
            embed_aligned_faces=fail_batch, embed=single)
        stats = self.run_project([self.path, other])
        self.event("embeddings")
        self.assertEqual([("batch", 6), ("photo", 3), ("photo", 3)], calls)
        self.assertEqual(2, stats["embeddings"]["computed"])
        self.assertEqual(0, stats["embeddings"]["file_errors"])

    def test_quality_batches_cancel_without_scheduling_more_native_work(self):
        from face_processing.models import DetectedFace
        from face_processing.test_processing import _sample_analysis
        from types import SimpleNamespace
        self.service._config = replace(self.service._config, performance=PerformanceConfig(batch_size=8))
        faces = [DetectedFace((0, 0, 40, 40), .9, np.zeros((5, 2)), self.path) for _ in range(18)]
        stat = self.path.stat()
        record = DiscoveryRecord(self.path, stat.st_size, stat.st_mtime_ns, ".jpg", None, None, 80, 40)
        token, calls = CancellationToken(), []
        self.service._face_detector = SimpleNamespace(detect=lambda *_: faces)
        def analyze(image, batch):
            calls.append(len(batch))
            token.cancel()
            return [_sample_analysis() for _ in batch]
        self.service._face_analyzer = SimpleNamespace(analyze_many=analyze)
        with self.assertRaises(ScanCancelledError):
            AutomaticScanServices.analyze_faces_from_image(self.service, record,
                np.zeros((40, 80, 3), dtype=np.uint8), cancellation_token=token)
        self.assertEqual([8], calls)

    def test_ranking_preserves_backend_exif_capture_time_semantics(self):
        other = self.photos / "other.jpg"
        for path, date, modified in ((self.path, "2005:01:01 12:00:00", 1700000000000000000),
                                     (other, "1990:01:01 12:00:00", 1800000000000000000)):
            exif = Image.Exif()
            exif[36867] = date
            Image.new("RGB", (80, 40), "gray").save(path, exif=exif)
            os.utime(path, ns=(modified, modified))
        with patch.object(self.service, "verify_duplicate_groups", side_effect=lambda photos, *args, **kwargs: [photos]):
            self.run_project([self.path, other])
        self.assertEqual([str(other)], self.event("ranking")[5][0]["keeper_paths"])
