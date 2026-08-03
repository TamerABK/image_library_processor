from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from face_analyzer.models import (
    AssessmentStatus,
    EyeLabel,
    EyeMeasurement,
    EyeState,
    FaceAnalysisResult,
    FaceGeometry,
    FaceImageQuality,
    HeadPose,
    MetricResult,
    MetricScore,
    PoseQuality,
)

from .cache import FaceScanCache, ImageFaceAnalysisCache
from .models import DetectedFace, EmbeddedFace, Match, Person, UnknownCluster
from .processor import FaceProcessor


def _sample_analysis() -> FaceAnalysisResult:
    return FaceAnalysisResult(
        detector_confidence=0.91,
        detector_metric=MetricResult(
            raw_value=0.91,
            quality_score=0.91,
            confidence=1.0,
            ranking_score=0.87,
        ),
        geometry=FaceGeometry(
            width=120,
            height=120,
            area=14400,
            relative_area=0.25,
            visible_ratio=0.94,
            minimum_dimension=120,
        ),
        visible_face_metric=MetricResult(
            raw_value=0.94,
            quality_score=0.94,
            confidence=1.0,
            ranking_score=0.91,
        ),
        image_quality=FaceImageQuality(
            focus_sharpness=MetricResult(
                raw_value=0.88,
                quality_score=0.88,
                confidence=0.95,
                ranking_score=0.86,
            ),
            detail_availability=MetricResult(
                raw_value=120.0,
                quality_score=0.84,
                confidence=0.95,
                ranking_score=0.82,
            ),
            sharpness=MetricResult(
                raw_value=0.88,
                quality_score=0.87,
                confidence=0.95,
                ranking_score=0.85,
            ),
            exposure=MetricResult(
                raw_value=0.74,
                quality_score=0.61,
                confidence=0.8,
                ranking_score=0.66,
            ),
            contrast=MetricResult(
                raw_value=0.72,
                quality_score=0.72,
                confidence=0.77,
                ranking_score=0.70,
            ),
            laplacian_variance=122.0,
            tenengrad_energy=38.0,
            high_frequency_energy_ratio=0.28,
            detail_availability_measure=120.0,
            median_luminance=0.48,
            p05_luminance=0.21,
            p95_luminance=0.79,
            dark_clip_ratio=0.01,
            bright_clip_ratio=0.02,
            usable_tonal_range=58.0,
            clipping_score=0.93,
            luminance_score=0.68,
            tonal_information_score=0.73,
            raw_exposure_score=0.74,
            display_exposure_score=0.61,
            shadow_detail_score=0.95,
            highlight_detail_score=0.94,
            tonal_balance_score=0.87,
            p10_luminance=0.25,
            p25_luminance=0.33,
            p75_luminance=0.66,
            p90_luminance=0.74,
            broad_tonal_range=49.0,
            interquartile_range=33.0,
            broad_contrast_score=0.51,
            interquartile_contrast_score=0.56,
            local_contrast_raw=16.0,
            local_contrast_score=0.58,
            contrast_quality_score=0.72,
        ),
        head_pose=HeadPose(
            yaw_degrees=1.5,
            pitch_degrees=-0.5,
            roll_degrees=0.25,
            confidence=0.87,
            status=AssessmentStatus.ASSESSED,
            source="test",
        ),
        pose=PoseQuality(
            metric=MetricResult(
                raw_value=0.93,
                quality_score=0.93,
                confidence=0.87,
                ranking_score=0.90,
            ),
            yaw_score=0.95,
            pitch_score=0.94,
            roll_score=0.97,
        ),
        eye_state=EyeState(
            left=EyeMeasurement(
                open_probability=0.98,
                label=EyeLabel.OPEN,
                confidence=0.9,
                status=AssessmentStatus.ASSESSED,
                source_width=24,
                source_height=12,
            ),
            right=EyeMeasurement(
                open_probability=0.97,
                label=EyeLabel.OPEN,
                confidence=0.88,
                status=AssessmentStatus.ASSESSED,
                source_width=24,
                source_height=12,
            ),
            combined_open_score=0.975,
            confidence=0.89,
            status=AssessmentStatus.ASSESSED,
        ),
        eyes=MetricResult(
            raw_value=0.975,
            quality_score=0.975,
            confidence=0.89,
            ranking_score=0.93,
        ),
        eye_weight=0.23,
        measurement_reliability=MetricResult(
            raw_value=0.88,
            quality_score=0.88,
            confidence=1.0,
            ranking_score=0.88,
        ),
        global_selection_score=0.81,
        group_relative_score=0.55,
        final_group_score=0.81,
        selection_score=0.81,
        embedding_utility_score=0.84,
        warnings=("ok",),
    )


class _StubDetector:
    def detect(self, image: np.ndarray, path: Path) -> list[DetectedFace]:
        return [
            DetectedFace(
                bbox=(10, 10, 40, 40),
                confidence=0.9,
                landmarks=np.asarray(
                    [
                        [20.0, 20.0],
                        [35.0, 20.0],
                        [28.0, 28.0],
                        [22.0, 38.0],
                        [34.0, 38.0],
                    ],
                    dtype=np.float32,
                ),
                path=path,
            )
        ]


class _StubAnalyzer:
    def analyze(self, image: np.ndarray, face: DetectedFace) -> FaceAnalysisResult:
        return _sample_analysis()


class _FailingDetector:
    def detect(self, image: np.ndarray, path: Path) -> list[DetectedFace]:
        raise AssertionError("detector should not be called when face scan cache is used")


class _AssertingEmbedder:
    def embed(self, image: np.ndarray, faces: list[DetectedFace]) -> list[EmbeddedFace]:
        return self.embed_requests([(image, face) for face in faces])

    def embed_requests(
        self,
        face_requests: list[tuple[np.ndarray, DetectedFace]],
    ) -> list[EmbeddedFace]:
        embedded: list[EmbeddedFace] = []
        for _image, face in face_requests:
            assert face.analysis is not None
            embedded.append(
                EmbeddedFace(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    landmarks=face.landmarks,
                    embedding=np.ones(4, dtype=np.float32),
                    path=face.path,
                    analysis=face.analysis,
                )
            )
        return embedded


class _AssertingRecognizer:
    def recognize(
        self,
        faces: list[EmbeddedFace],
    ) -> tuple[list[object], list[EmbeddedFace]]:
        assert faces
        assert all(face.analysis is not None for face in faces)
        return [], faces


class _PassthroughClusterer:
    def cluster(self, faces: list[EmbeddedFace]) -> list[UnknownCluster]:
        if not faces:
            return []
        return [
            UnknownCluster(
                id=0,
                faces=faces,
                representative=faces[0],
                preview=None,
            )
        ]


class _EmptyDatabase:
    def contains_people(self) -> bool:
        return False

    def cache_signature(self) -> str:
        return "0:0:0:0"

    def find_nearest_embedding(self, embedding: np.ndarray) -> Match | None:
        return None

    def add_person(self, name: str) -> Person:
        raise NotImplementedError

    def add_embedding(self, person_id: int, embedding: np.ndarray):
        raise NotImplementedError

    def get_person(self, person_id: int) -> Person | None:
        return None

    def list_people_names(self) -> list[str]:
        return []


class FaceProcessingTests(unittest.TestCase):
    def test_processor_runs_analysis_before_embedding_and_recognition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.jpg"
            image = np.full((80, 80, 3), 127, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            processor = FaceProcessor(
                detector=_StubDetector(),
                embedder=_AssertingEmbedder(),
                analyzer=_StubAnalyzer(),
                recognizer=_AssertingRecognizer(),
                clusterer=_PassthroughClusterer(),
                database=_EmptyDatabase(),
                max_workers=1,
                embed_batch_size=1,
            )

            result = processor.scan_folder(tmpdir)

            self.assertEqual(len(result.unknown_clusters), 1)
            self.assertIsNotNone(result.unknown_clusters[0].faces[0].analysis)
            self.assertAlmostEqual(
                result.unknown_clusters[0].faces[0].analysis.selection_score,
                0.81,
            )

    def test_cache_round_trips_analysis_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FaceScanCache(Path(tmpdir) / "cache.sqlite3")
            image_path = Path(tmpdir) / "cached.jpg"
            image_path.write_bytes(b"cached")
            analysis = _sample_analysis()
            face = EmbeddedFace(
                bbox=(1, 2, 3, 4),
                confidence=0.95,
                landmarks=np.ones((5, 2), dtype=np.float32),
                embedding=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                path=image_path,
                analysis=analysis,
            )

            cache.put(
                image_path,
                file_size=7,
                mtime_ns=11,
                faces=[face],
            )

            cached = cache.get(
                image_path,
                file_size=7,
                mtime_ns=11,
                require_analysis=True,
            )

            self.assertIsNotNone(cached)
            self.assertEqual(len(cached or []), 1)
            cached_analysis = cached[0].analysis
            self.assertIsNotNone(cached_analysis)
            self.assertAlmostEqual(cached_analysis.selection_score, analysis.selection_score)
            self.assertEqual(cached_analysis.head_pose.status, AssessmentStatus.ASSESSED)
            self.assertEqual(cached_analysis.eye_state.left.label, EyeLabel.OPEN)

    def test_cache_requires_analysis_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FaceScanCache(Path(tmpdir) / "cache.sqlite3")
            image_path = Path(tmpdir) / "cached.jpg"
            image_path.write_bytes(b"cached")
            face = EmbeddedFace(
                bbox=(1, 2, 3, 4),
                confidence=0.95,
                landmarks=np.ones((5, 2), dtype=np.float32),
                embedding=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                path=image_path,
                analysis=None,
            )

            cache.put(
                image_path,
                file_size=7,
                mtime_ns=11,
                faces=[face],
            )

            self.assertIsNone(
                cache.get(
                    image_path,
                    file_size=7,
                    mtime_ns=11,
                    require_analysis=True,
                )
            )

    def test_image_face_analysis_cache_round_trips_face_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ImageFaceAnalysisCache(Path(tmpdir) / "cache.sqlite3")
            image_path = Path(tmpdir) / "faces.jpg"
            image_path.write_bytes(b"cached")
            analysis = _sample_analysis()
            detected_face = DetectedFace(
                bbox=(1, 2, 30, 40),
                confidence=0.95,
                landmarks=np.ones((5, 2), dtype=np.float32),
                path=image_path,
                analysis=analysis,
            )

            cache.put(
                image_path,
                file_size=7,
                mtime_ns=11,
                faces=[detected_face],
            )

            cached = cache.get(
                image_path,
                file_size=7,
                mtime_ns=11,
            )

            self.assertIsNotNone(cached)
            self.assertEqual(len(cached or []), 1)
            self.assertIsNotNone(cached[0].analysis)
            self.assertAlmostEqual(
                cached[0].analysis.embedding_utility_score,
                analysis.embedding_utility_score,
            )

    def test_processor_persists_face_analysis_in_image_analysis_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.jpg"
            image = np.full((80, 80, 3), 127, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            processor = FaceProcessor(
                detector=_StubDetector(),
                embedder=_AssertingEmbedder(),
                analyzer=_StubAnalyzer(),
                recognizer=_AssertingRecognizer(),
                clusterer=_PassthroughClusterer(),
                database=_EmptyDatabase(),
                max_workers=1,
                embed_batch_size=1,
            )
            analysis_cache = ImageFaceAnalysisCache(Path(tmpdir) / "analysis.sqlite3")
            processor._analysis_cache = analysis_cache

            result = processor.scan_folder(tmpdir)
            self.assertEqual(len(result.unknown_clusters), 1)

            stat = image_path.stat()
            cached_faces = analysis_cache.get(
                image_path.resolve(),
                file_size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )

            self.assertIsNotNone(cached_faces)
            self.assertEqual(len(cached_faces or []), 1)
            self.assertIsNotNone(cached_faces[0].analysis)
            self.assertAlmostEqual(cached_faces[0].analysis.selection_score, 0.81)

    def test_processor_backfills_image_analysis_cache_from_face_scan_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.jpg"
            image = np.full((80, 80, 3), 127, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            resolved_path = image_path.resolve()
            stat = resolved_path.stat()
            db_path = Path(tmpdir) / "analysis.sqlite3"

            cached_face = EmbeddedFace(
                bbox=(10, 10, 40, 40),
                confidence=0.9,
                landmarks=np.asarray(
                    [
                        [20.0, 20.0],
                        [35.0, 20.0],
                        [28.0, 28.0],
                        [22.0, 38.0],
                        [34.0, 38.0],
                    ],
                    dtype=np.float32,
                ),
                embedding=np.ones(4, dtype=np.float32),
                path=resolved_path,
                analysis=_sample_analysis(),
            )

            face_scan_cache = FaceScanCache(db_path)
            face_scan_cache.put(
                resolved_path,
                stat.st_size,
                stat.st_mtime_ns,
                [cached_face],
            )

            processor = FaceProcessor(
                detector=_FailingDetector(),
                embedder=_AssertingEmbedder(),
                analyzer=_StubAnalyzer(),
                recognizer=_AssertingRecognizer(),
                clusterer=_PassthroughClusterer(),
                database=_EmptyDatabase(),
                max_workers=1,
                embed_batch_size=1,
            )
            analysis_cache = ImageFaceAnalysisCache(db_path)
            processor._cache = face_scan_cache
            processor._analysis_cache = analysis_cache

            result = processor.scan_folder(tmpdir)
            self.assertEqual(len(result.unknown_clusters), 1)

            cached_faces = analysis_cache.get(
                resolved_path,
                file_size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )

            self.assertIsNotNone(cached_faces)
            self.assertEqual(len(cached_faces or []), 1)
            self.assertIsNotNone(cached_faces[0].analysis)
            self.assertAlmostEqual(cached_faces[0].analysis.selection_score, 0.81)


if __name__ == "__main__":
    unittest.main()
