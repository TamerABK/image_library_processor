from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from automatic_scan.config import (
    AutomaticScanConfig,
    DuplicateSelectionConfig,
    FaceProcessingConfig,
    FaceQualityPreset,
)
from automatic_scan.models import AutomaticStage, DiscoveryRecord, PipelineImageStatus
from automatic_scan.orchestrator import AutomaticScanOrchestrator
from face_analyzer.models import AssessmentStatus, EyeLabel, EyeMeasurement, EyeState
from face_processing.models import DetectedFace, EmbeddedFace, Match, RecognizedFace, UnknownCluster
from face_processing.test_processing import _sample_analysis
from blur_detector.blur_detector import BlurResult
from duplicate_detector.models import CandidatePair, PhotoInfo
from grouping.models import VibeGroup, VibeGroupingResult


def _blur(status: str, score: float) -> BlurResult:
    return BlurResult(
        laplacian=score * 100.0,
        sobel=score * 50.0,
        local_contrast=score * 25.0,
        lap_norm=score,
        sobel_norm=score,
        contrast_norm=score,
        final_score=score,
        status=status,
    )


def _analysis(
    *,
    score: float = 0.75,
    relative_area: float = 0.16,
    reliability: float = 0.88,
    embedding_utility: float | None = None,
    closed_eyes: bool = False,
    one_eye_closed: bool = False,
    pose_score: float = 0.78,
    exposure_score: float = 0.61,
    contrast_score: float = 0.72,
) :
    base = _sample_analysis()
    eye_state = base.eye_state
    eyes = base.eyes
    if closed_eyes:
        closed = EyeMeasurement(
            open_probability=0.05,
            label=EyeLabel.CLOSED,
            confidence=0.92,
            status=AssessmentStatus.ASSESSED,
            source_width=24,
            source_height=12,
        )
        eye_state = EyeState(
            left=closed,
            right=closed,
            combined_open_score=0.05,
            confidence=0.92,
            status=AssessmentStatus.ASSESSED,
        )
        eyes = replace(
            base.eyes,
            raw_value=0.05,
            quality_score=0.05,
            ranking_score=0.05,
        )
    elif one_eye_closed:
        closed = EyeMeasurement(
            open_probability=0.05,
            label=EyeLabel.CLOSED,
            confidence=0.92,
            status=AssessmentStatus.ASSESSED,
            source_width=24,
            source_height=12,
        )
        open_eye = EyeMeasurement(
            open_probability=0.96,
            label=EyeLabel.OPEN,
            confidence=0.94,
            status=AssessmentStatus.ASSESSED,
            source_width=24,
            source_height=12,
        )
        eye_state = EyeState(
            left=open_eye,
            right=closed,
            combined_open_score=0.20,
            confidence=0.92,
            status=AssessmentStatus.ASSESSED,
        )
        eyes = replace(
            base.eyes,
            raw_value=0.20,
            quality_score=0.20,
            ranking_score=0.20,
        )
    return replace(
        base,
        geometry=replace(
            base.geometry,
            relative_area=relative_area,
        ),
        pose=replace(
            base.pose,
            metric=replace(
                base.pose.metric,
                raw_value=pose_score,
                quality_score=pose_score,
                ranking_score=pose_score,
            ),
        ),
        eye_state=eye_state,
        eyes=eyes,
        measurement_reliability=replace(
            base.measurement_reliability,
            raw_value=reliability,
            quality_score=reliability,
            ranking_score=reliability,
        ),
        image_quality=replace(
            base.image_quality,
            exposure=replace(
                base.image_quality.exposure,
                raw_value=exposure_score,
                quality_score=exposure_score,
                ranking_score=exposure_score,
            ),
            contrast=replace(
                base.image_quality.contrast,
                raw_value=contrast_score,
                quality_score=contrast_score,
                ranking_score=contrast_score,
            ),
            display_exposure_score=exposure_score,
            contrast_quality_score=contrast_score,
        ),
        global_selection_score=score,
        final_group_score=score,
        selection_score=score,
        embedding_utility_score=score if embedding_utility is None else embedding_utility,
    )


def _detected_face(
    path: Path,
    *,
    score: float = 0.75,
    relative_area: float = 0.16,
    reliability: float = 0.88,
    embedding_utility: float | None = None,
    closed_eyes: bool = False,
    one_eye_closed: bool = False,
    bbox: tuple[int, int, int, int] = (20, 20, 120, 120),
    exposure_score: float = 0.61,
    contrast_score: float = 0.72,
) -> DetectedFace:
    return DetectedFace(
        bbox=bbox,
        confidence=0.95,
        landmarks=np.asarray(
            [
                [48.0, 48.0],
                [84.0, 48.0],
                [66.0, 70.0],
                [50.0, 94.0],
                [82.0, 94.0],
            ],
            dtype=np.float32,
        ),
        path=path,
        analysis=_analysis(
            score=score,
            relative_area=relative_area,
            reliability=reliability,
            embedding_utility=embedding_utility,
            closed_eyes=closed_eyes,
            one_eye_closed=one_eye_closed,
            exposure_score=exposure_score,
            contrast_score=contrast_score,
        ),
    )


def _embedded_face(path: Path, face: DetectedFace) -> EmbeddedFace:
    return EmbeddedFace(
        bbox=face.bbox,
        confidence=face.confidence,
        landmarks=face.landmarks,
        embedding=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        path=path,
        analysis=face.analysis,
    )


class _FakeServices:
    supported_extensions = (".jpg",)

    def __init__(self, folder: Path, paths: list[Path]) -> None:
        self._folder = folder
        self._records = [
            DiscoveryRecord(
                path=path.resolve(),
                file_size=1,
                mtime_ns=1,
                extension=path.suffix.lower(),
                orientation="landscape",
                capture_timestamp=float(index + 1),
                width=1600,
                height=1000,
            )
            for index, path in enumerate(paths)
        ]
        self.blur_results: dict[str, BlurResult] = {}
        self.face_results: dict[str, list[DetectedFace]] = {}
        self.embedded_results: dict[str, list[EmbeddedFace]] = {}
        self.recognized_by_path: dict[str, list[RecognizedFace]] = {}
        self.unknown_by_path: dict[str, list[EmbeddedFace]] = {}
        self.candidate_pairs: list[CandidatePair] = []
        self.duplicate_groups: list[list[str]] = []
        self.vibe_raise: Exception | None = None
        self.vibe_group_paths: list[list[str]] = []
        self.blur_calls: list[str] = []
        self.face_calls: list[str] = []
        self.embed_calls: list[str] = []
        self.vibe_inputs: list[list[str]] = []

    def runtime_info(self) -> dict[str, object]:
        return {}

    def discover(self, folder: Path) -> list[DiscoveryRecord]:
        assert folder == self._folder.resolve()
        return list(self._records)

    def get_blur_cached(self, record: DiscoveryRecord):
        return None

    def analyze_blur(self, record: DiscoveryRecord):
        self.blur_calls.append(str(record.path))
        return self.blur_results[str(record.path)]

    def get_duplicate_photo_cached(self, record: DiscoveryRecord):
        return None

    def index_duplicate_photo(self, record: DiscoveryRecord):
        return PhotoInfo(
            path=record.path,
            width=record.width or 1600,
            height=record.height or 1000,
            file_size=record.file_size,
            phash=hash(record.path.name) & ((1 << 32) - 1),
            dhash=(hash(record.path.name) >> 1) & ((1 << 32) - 1),
        )

    def generate_duplicate_candidates(self, photos, *, progress_callback=None, cancellation_token=None):
        return list(self.candidate_pairs)

    def verify_duplicate_groups(self, photos, candidate_pairs, *, cancellation_token=None):
        photo_map = {str(photo.path): photo for photo in photos}
        return [
            [photo_map[path] for path in group if path in photo_map]
            for group in self.duplicate_groups
        ]

    def get_face_analysis_cached(self, record: DiscoveryRecord):
        return None

    def analyze_faces(self, record: DiscoveryRecord):
        self.face_calls.append(str(record.path))
        return list(self.face_results.get(str(record.path), []))

    def get_embedded_faces_cached(self, record: DiscoveryRecord):
        return None

    def embed_faces(self, record: DiscoveryRecord, faces, *, minimum_embedding_utility: float):
        self.embed_calls.append(str(record.path))
        return list(self.embedded_results.get(str(record.path), []))

    def recognize_faces(self, embedded_faces):
        if not embedded_faces:
            return [], []
        path = str(embedded_faces[0].path)
        return (
            list(self.recognized_by_path.get(path, [])),
            list(self.unknown_by_path.get(path, [])),
        )

    def get_person_name(self, person_id: int):
        if person_id == 1:
            return "Alex"
        return None

    def cluster_unknown_faces(self, embedded_faces):
        if not embedded_faces:
            return []
        return [
            UnknownCluster(
                id=7,
                faces=list(embedded_faces),
                representative=embedded_faces[0],
                preview=None,
            )
        ]

    def nearest_known_match(self, embedded_face: EmbeddedFace):
        return Match(person_id=1, score=0.81)

    def suggest_known_names(self, embedding_match):
        return ("Alex", "Jordan")

    def group_vibes(self, image_paths, *, progress_callback=None, cancellation_token=None):
        self.vibe_inputs.append(list(image_paths))
        if self.vibe_raise is not None:
            raise self.vibe_raise
        groups = [
            VibeGroup(
                group_id=f"scene-{index + 1}",
                image_paths=list(group_paths),
                representative_path=group_paths[0],
                start_timestamp=None,
                end_timestamp=None,
                recognized_person_ids=(),
                recognized_person_names=(),
                label=None,
                cohesion_score=0.9,
                metadata={},
            )
            for index, group_paths in enumerate(self.vibe_group_paths)
        ]
        return VibeGroupingResult(
            groups=groups,
            ungrouped_paths=[],
            errors=[],
            config_snapshot={},
            model_fingerprint="fake",
            provider="CPUExecutionProvider",
            diagnostics={},
        )


class AutomaticOrchestratorTests(unittest.TestCase):
    def _make_services(self, names: list[str]):
        temp_dir = tempfile.TemporaryDirectory()
        folder = Path(temp_dir.name)
        paths = []
        for name in names:
            path = folder / name
            path.write_bytes(b"data")
            paths.append(path)
        services = _FakeServices(folder, paths)
        return temp_dir, folder, paths, services

    def _make_virtual_services(self, names: list[str]):
        temp_dir = tempfile.TemporaryDirectory()
        folder = Path(temp_dir.name)
        paths = [folder / name for name in names]
        services = _FakeServices(folder, paths)
        return temp_dir, folder, paths, services

    def test_blurry_image_short_circuits_downstream_processing(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["blurry.jpg", "sharp.jpg"])
        try:
            blurry_path, sharp_path = [path.resolve() for path in paths]
            services.blur_results[str(blurry_path)] = _blur("Blurry", 0.20)
            services.blur_results[str(sharp_path)] = _blur("Sharp", 0.90)
            services.face_results[str(sharp_path)] = []
            services.vibe_group_paths = [[str(sharp_path)]]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(),
            )
            image_map = result.image_map()

            blurry = image_map[str(blurry_path)]
            self.assertEqual(blurry.primary_exclusion_reason, "blurry")
            self.assertEqual(blurry.excluded_at_stage, AutomaticStage.BLUR_ANALYSIS.value)
            self.assertTrue(blurry.selected_for_deletion)
            self.assertIn(AutomaticStage.FACE_DETECTION_ANALYSIS.value, blurry.skipped_stages)
            self.assertNotIn(str(blurry_path), services.face_calls)
            self.assertNotIn(str(blurry_path), services.embed_calls)
            self.assertEqual(services.vibe_inputs, [[str(sharp_path)]])
        finally:
            temp_dir.cleanup()

    def test_poor_people_picture_face_quality_short_circuits_embeddings(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["portrait.jpg"])
        try:
            portrait_path = paths[0].resolve()
            services.blur_results[str(portrait_path)] = _blur("Sharp", 0.88)
            face = _detected_face(
                portrait_path,
                score=0.14,
                relative_area=0.22,
                reliability=0.92,
                embedding_utility=0.80,
                closed_eyes=True,
            )
            services.face_results[str(portrait_path)] = [face]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(),
            )
            image = result.image_map()[str(portrait_path)]

            self.assertEqual(image.primary_exclusion_reason, "poor_primary_face_quality")
            self.assertEqual(image.excluded_at_stage, AutomaticStage.FACE_QUALITY_GATE.value)
            self.assertNotIn(str(portrait_path), services.embed_calls)
            self.assertEqual(services.vibe_inputs, [[]])
        finally:
            temp_dir.cleanup()

    def test_tiny_incidental_face_does_not_exclude_non_people_picture(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["detail.jpg"])
        try:
            detail_path = paths[0].resolve()
            services.blur_results[str(detail_path)] = _blur("Sharp", 0.90)
            face = _detected_face(
                detail_path,
                score=0.10,
                relative_area=0.008,
                reliability=0.55,
                embedding_utility=0.70,
                bbox=(20, 20, 14, 18),
            )
            services.face_results[str(detail_path)] = [face]
            embedded = _embedded_face(detail_path, face)
            services.embedded_results[str(detail_path)] = [embedded]
            services.unknown_by_path[str(detail_path)] = [embedded]
            services.vibe_group_paths = [[str(detail_path)]]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(),
            )
            image = result.image_map()[str(detail_path)]

            self.assertTrue(image.is_active)
            self.assertIsNone(image.primary_exclusion_reason)
            self.assertEqual(services.embed_calls, [str(detail_path)])
            self.assertEqual(services.vibe_inputs, [[str(detail_path)]])
        finally:
            temp_dir.cleanup()

    def test_diagnostics_payload_records_not_people_picture_images(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["detail.jpg"])
        try:
            detail_path = paths[0].resolve()
            services.blur_results[str(detail_path)] = _blur("Sharp", 0.90)
            face = _detected_face(
                detail_path,
                score=0.10,
                relative_area=0.008,
                reliability=0.55,
                embedding_utility=0.70,
                bbox=(20, 20, 14, 18),
            )
            services.face_results[str(detail_path)] = [face]
            embedded = _embedded_face(detail_path, face)
            services.embedded_results[str(detail_path)] = [embedded]
            services.unknown_by_path[str(detail_path)] = [embedded]
            services.vibe_group_paths = [[str(detail_path)]]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(),
            )
            payload = result.to_diagnostics_payload()

            self.assertEqual(payload["not_people_picture_count"], 1)
            self.assertIn("total_seconds", payload)
            self.assertIn("startup_seconds", payload)
            self.assertIn("scan_seconds", payload)
            self.assertIn("images_per_second", payload)
            self.assertIn("cache_mode", payload)
            self.assertIn("image_io", payload)
            self.assertIn("models", payload)
            self.assertIn("database", payload)
            self.assertEqual(
                payload["not_people_picture_images"][0]["path"],
                str(detail_path),
            )
            assessment = payload["not_people_picture_images"][0]["assessment"]
            self.assertFalse(assessment["is_people_picture"])
            self.assertEqual(assessment["detected_face_count"], 1)
            self.assertEqual(assessment["reliable_face_count"], 1)
            self.assertEqual(assessment["supporting_reasons"], ["faces_too_small_or_incidental"])
        finally:
            temp_dir.cleanup()

    def test_full_body_portrait_below_old_span_threshold_is_people_picture(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["full_body.jpg"])
        try:
            image_path = paths[0].resolve()
            services._records[0] = replace(
                services._records[0],
                width=4160,
                height=6240,
            )
            services.blur_results[str(image_path)] = _blur("Sharp", 0.93)
            face = _detected_face(
                image_path,
                score=0.86,
                relative_area=0.006,
                reliability=0.91,
                bbox=(600, 900, 84, 120),
                exposure_score=0.68,
                contrast_score=0.72,
            )
            services.face_results[str(image_path)] = [face]
            embedded = _embedded_face(image_path, face)
            services.embedded_results[str(image_path)] = [embedded]
            services.unknown_by_path[str(image_path)] = [embedded]
            services.vibe_group_paths = [[str(image_path)]]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(),
            )
            image = result.image_map()[str(image_path)]

            self.assertTrue(image.people_picture_assessment is not None)
            self.assertTrue(image.people_picture_assessment.is_people_picture)
            self.assertIn("face_scale_prominent", image.people_picture_assessment.supporting_reasons)
            self.assertTrue(image.is_active)
        finally:
            temp_dir.cleanup()

    def test_group_photo_with_small_faces_is_people_picture(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["group.jpg"])
        try:
            image_path = paths[0].resolve()
            services._records[0] = replace(
                services._records[0],
                width=4160,
                height=6240,
            )
            services.blur_results[str(image_path)] = _blur("Sharp", 0.94)
            faces = [
                _detected_face(
                    image_path,
                    score=0.74,
                    relative_area=0.0025,
                    reliability=0.90,
                    bbox=(800, 1300, 64, 96),
                    exposure_score=0.66,
                    contrast_score=0.71,
                ),
                _detected_face(
                    image_path,
                    score=0.71,
                    relative_area=0.0024,
                    reliability=0.88,
                    bbox=(1700, 1250, 64, 96),
                    exposure_score=0.64,
                    contrast_score=0.69,
                ),
                _detected_face(
                    image_path,
                    score=0.69,
                    relative_area=0.0022,
                    reliability=0.87,
                    bbox=(2500, 1320, 64, 96),
                    exposure_score=0.65,
                    contrast_score=0.68,
                ),
            ]
            services.face_results[str(image_path)] = faces
            services.embedded_results[str(image_path)] = [
                _embedded_face(image_path, face)
                for face in faces
            ]
            services.vibe_group_paths = [[str(image_path)]]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(),
            )
            image = result.image_map()[str(image_path)]

            self.assertTrue(image.people_picture_assessment is not None)
            self.assertTrue(image.people_picture_assessment.is_people_picture)
            self.assertIn(
                "additional_face_count_multiplier",
                image.people_picture_assessment.supporting_reasons,
            )
            self.assertTrue(image.is_active)
        finally:
            temp_dir.cleanup()

    def test_group_photo_passes_when_summed_face_area_is_large_enough(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["group_sum.jpg"])
        try:
            image_path = paths[0].resolve()
            services._records[0] = replace(
                services._records[0],
                width=4160,
                height=6240,
            )
            services.blur_results[str(image_path)] = _blur("Sharp", 0.94)
            faces = [
                _detected_face(
                    image_path,
                    score=0.62,
                    relative_area=0.0022,
                    reliability=0.87,
                    bbox=(700, 1400, 50, 72),
                    exposure_score=0.64,
                    contrast_score=0.69,
                ),
                _detected_face(
                    image_path,
                    score=0.60,
                    relative_area=0.0021,
                    reliability=0.86,
                    bbox=(1150, 1380, 50, 72),
                    exposure_score=0.63,
                    contrast_score=0.68,
                ),
                _detected_face(
                    image_path,
                    score=0.61,
                    relative_area=0.0020,
                    reliability=0.86,
                    bbox=(1600, 1410, 50, 72),
                    exposure_score=0.62,
                    contrast_score=0.67,
                ),
                _detected_face(
                    image_path,
                    score=0.59,
                    relative_area=0.0020,
                    reliability=0.85,
                    bbox=(2050, 1390, 50, 72),
                    exposure_score=0.62,
                    contrast_score=0.67,
                ),
            ]
            services.face_results[str(image_path)] = faces
            services.embedded_results[str(image_path)] = [
                _embedded_face(image_path, face)
                for face in faces
            ]
            services.vibe_group_paths = [[str(image_path)]]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(),
            )
            image = result.image_map()[str(image_path)]

            self.assertTrue(image.people_picture_assessment is not None)
            self.assertTrue(image.people_picture_assessment.is_people_picture)
            self.assertIn(
                "total_face_area_prominent",
                image.people_picture_assessment.supporting_reasons,
            )
        finally:
            temp_dir.cleanup()

    def test_strict_mode_excludes_dark_eye_compromised_full_body_people_picture(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["portrait_full_body.jpg"])
        try:
            image_path = paths[0].resolve()
            services._records[0] = replace(
                services._records[0],
                width=4160,
                height=6240,
            )
            services.blur_results[str(image_path)] = _blur("Sharp", 0.92)
            face = _detected_face(
                image_path,
                score=0.76,
                relative_area=0.018,
                reliability=0.90,
                one_eye_closed=True,
                bbox=(220, 260, 180, 240),
                exposure_score=0.18,
                contrast_score=0.24,
            )
            services.face_results[str(image_path)] = [face]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(face_quality_preset=FaceQualityPreset.STRICT),
            )
            image = result.image_map()[str(image_path)]

            self.assertTrue(image.people_picture_assessment is not None)
            self.assertTrue(image.people_picture_assessment.is_people_picture)
            self.assertEqual(image.primary_exclusion_reason, "poor_primary_face_quality")
            self.assertEqual(image.excluded_at_stage, AutomaticStage.FACE_QUALITY_GATE.value)
            self.assertIn("closed_eye_primary_face", image.warnings)
            self.assertIn("low_primary_face_exposure", image.warnings)
            self.assertNotIn(str(image_path), services.embed_calls)
        finally:
            temp_dir.cleanup()

    def test_strict_mode_excludes_no_face_burst_neighbor_of_people_picture(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["IMG_8200.jpg", "IMG_8201.jpg"])
        try:
            no_face_path, people_path = [path.resolve() for path in paths]
            services._records[0] = replace(
                services._records[0],
                width=4160,
                height=6240,
            )
            services._records[1] = replace(
                services._records[1],
                width=4160,
                height=6240,
            )
            services.blur_results[str(no_face_path)] = _blur("Sharp", 0.90)
            services.blur_results[str(people_path)] = _blur("Sharp", 0.91)

            people_face = _detected_face(
                people_path,
                score=0.76,
                relative_area=0.0105,
                reliability=0.93,
                one_eye_closed=True,
                bbox=(620, 820, 144, 197),
                exposure_score=0.49,
                contrast_score=0.61,
            )
            services.face_results[str(no_face_path)] = []
            services.face_results[str(people_path)] = [people_face]

            no_face_photo = services.index_duplicate_photo(services._records[0])
            people_photo = services.index_duplicate_photo(services._records[1])
            services.candidate_pairs = [
                CandidatePair(
                    left=no_face_photo,
                    right=people_photo,
                    phash_distance=3,
                    dhash_distance=4,
                )
            ]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(face_quality_preset=FaceQualityPreset.STRICT),
            )
            image_map = result.image_map()

            self.assertEqual(
                image_map[str(people_path)].primary_exclusion_reason,
                "poor_primary_face_quality",
            )
            no_face_image = image_map[str(no_face_path)]
            self.assertTrue(no_face_image.people_picture_assessment is not None)
            self.assertTrue(no_face_image.people_picture_assessment.is_people_picture)
            self.assertIn(
                "inferred_from_related_people_picture",
                no_face_image.people_picture_assessment.supporting_reasons,
            )
            self.assertEqual(no_face_image.primary_exclusion_reason, "poor_primary_face_quality")
            self.assertEqual(no_face_image.excluded_at_stage, AutomaticStage.FACE_QUALITY_GATE.value)
            self.assertIn("no_primary_faces_detected_for_people_picture", no_face_image.warnings)
        finally:
            temp_dir.cleanup()

    def test_hard_exclude_dark_faces_option_excludes_dark_primary_faces(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["dark_face.jpg"])
        try:
            image_path = paths[0].resolve()
            services._records[0] = replace(
                services._records[0],
                width=4160,
                height=6240,
            )
            services.blur_results[str(image_path)] = _blur("Sharp", 0.94)
            face = _detected_face(
                image_path,
                score=0.72,
                relative_area=0.014,
                reliability=0.91,
                bbox=(620, 820, 144, 197),
                exposure_score=0.18,
                contrast_score=0.42,
            )
            services.face_results[str(image_path)] = [face]

            default_result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(face_quality_preset=FaceQualityPreset.BALANCED),
            )
            default_image = default_result.image_map()[str(image_path)]
            self.assertTrue(default_image.is_active)
            self.assertIn("low_primary_face_exposure", default_image.warnings)
            services.embed_calls.clear()

            hard_exclude_result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(
                    face_quality_preset=FaceQualityPreset.BALANCED,
                    faces=FaceProcessingConfig(hard_exclude_dark_primary_faces=True),
                ),
            )
            hard_exclude_image = hard_exclude_result.image_map()[str(image_path)]

            self.assertEqual(hard_exclude_image.primary_exclusion_reason, "dark_primary_faces")
            self.assertEqual(
                hard_exclude_image.excluded_at_stage,
                AutomaticStage.FACE_QUALITY_GATE.value,
            )
            self.assertNotIn(str(image_path), services.embed_calls)
        finally:
            temp_dir.cleanup()

    def test_low_pick_scores_are_terminally_excluded(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["low_pick.jpg"])
        try:
            image_path = paths[0].resolve()
            services.blur_results[str(image_path)] = _blur("Sharp", 0.93)
            face = _detected_face(
                image_path,
                score=0.32,
                relative_area=0.16,
                reliability=0.92,
                exposure_score=0.67,
                contrast_score=0.70,
                bbox=(40, 40, 220, 220),
            )
            services.face_results[str(image_path)] = [face]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(face_quality_preset=FaceQualityPreset.BALANCED),
            )
            image = result.image_map()[str(image_path)]

            self.assertEqual(image.primary_exclusion_reason, "low_primary_pick_scores")
            self.assertEqual(image.excluded_at_stage, AutomaticStage.FACE_QUALITY_GATE.value)
            self.assertIn("low_primary_pick_score", image.warnings)
            self.assertNotIn(str(image_path), services.embed_calls)
        finally:
            temp_dir.cleanup()

    def test_duplicate_selection_uses_keeper_count_and_excludes_alternatives(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["a.jpg", "b.jpg", "c.jpg"])
        try:
            resolved_paths = [path.resolve() for path in paths]
            for path in resolved_paths:
                services.blur_results[str(path)] = _blur("Sharp", 0.95)
            faces = [
                _detected_face(resolved_paths[0], score=0.85, relative_area=0.20),
                _detected_face(resolved_paths[1], score=0.72, relative_area=0.18),
                _detected_face(resolved_paths[2], score=0.35, relative_area=0.18),
            ]
            for path, face in zip(resolved_paths, faces):
                services.face_results[str(path)] = [face]
                services.embedded_results[str(path)] = [_embedded_face(path, face)]
                services.unknown_by_path[str(path)] = services.embedded_results[str(path)]
            services.duplicate_groups = [[str(path) for path in resolved_paths]]
            services.vibe_group_paths = [[str(resolved_paths[0]), str(resolved_paths[1])]]

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(
                    duplicate_selection=DuplicateSelectionConfig(keepers_per_group=2),
                ),
            )
            image_map = result.image_map()

            self.assertTrue(image_map[str(resolved_paths[0])].is_duplicate_keeper)
            self.assertTrue(image_map[str(resolved_paths[1])].is_duplicate_keeper)
            self.assertEqual(
                image_map[str(resolved_paths[2])].status,
                PipelineImageStatus.DUPLICATE_ALTERNATIVE,
            )
            self.assertEqual(
                image_map[str(resolved_paths[2])].primary_exclusion_reason,
                "duplicate_alternative",
            )
            self.assertEqual(services.vibe_inputs, [[str(resolved_paths[0]), str(resolved_paths[1])]])
        finally:
            temp_dir.cleanup()

    def test_vibe_fallback_always_returns_groups(self) -> None:
        temp_dir, folder, paths, services = self._make_services(["one.jpg", "two.jpg"])
        try:
            resolved_paths = [path.resolve() for path in paths]
            for path in resolved_paths:
                services.blur_results[str(path)] = _blur("Sharp", 0.95)
                services.face_results[str(path)] = []
            services.vibe_raise = RuntimeError("primary vibe failure")

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(),
            )

            self.assertTrue(result.diagnostics.vibe_fallback_used)
            self.assertEqual(len(result.vibe_groups), 2)
        finally:
            temp_dir.cleanup()

    def test_large_folder_short_circuits_terminal_blur_exclusions(self) -> None:
        names = [f"img_{index:05d}.jpg" for index in range(20_000)]
        temp_dir, folder, paths, services = self._make_virtual_services(names)
        try:
            resolved_paths = [path.resolve() for path in paths]
            for path in resolved_paths:
                services.blur_results[str(path)] = _blur("Blurry", 0.18)

            result = AutomaticScanOrchestrator(services=services).scan_folder(
                folder,
                AutomaticScanConfig(),
            )

            self.assertEqual(len(result.all_images), 20_000)
            self.assertEqual(len(services.blur_calls), 20_000)
            self.assertEqual(services.face_calls, [])
            self.assertEqual(services.embed_calls, [])
            self.assertEqual(len(result.exclusions_by_reason.get("blurry", [])), 20_000)
        finally:
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
