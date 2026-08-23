from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from duplicate_detector.models import CandidatePair, PhotoInfo
from face_processing.models import DetectedFace, EmbeddedFace
from image_loader import default_image_loader
from scan_controls import CancellationToken, ScanCancelledError

from .config import AutomaticScanConfig, BlurPolicy
from .dependency_graph import AutomaticDependencyGraph
from .duplicate_selection import select_duplicate_keepers
from .eligibility import assess_face_quality, assess_people_picture
from .fallbacks import group_vibes_with_fallback
from .models import (
    AutomaticScanDiagnostics,
    AutomaticScanResult,
    AutomaticStage,
    AutomaticUnknownCluster,
    DiscoveryRecord,
    PipelineError,
    PipelineImageState,
    PipelineImageStatus,
    PipelineSummary,
    PipelineWarning,
    StageStats,
)
from .perf import AutomaticScanProfiler, active_profiler, get_active_profiler
from .state_store import AutomaticStateStore

if TYPE_CHECKING:
    from .services import AutomaticScanServices


ProgressCallback = callable


def _get_automatic_scan_timing_logger() -> logging.Logger:
    logger = logging.getLogger("image_deduplicator.automatic_scan_timing")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    from app_paths import app_data_path

    handler = logging.FileHandler(
        app_data_path("automatic_scan_timings.log"),
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _get_automatic_people_picture_logger() -> logging.Logger:
    logger = logging.getLogger("image_deduplicator.automatic_people_picture")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    from app_paths import app_data_path

    handler = logging.FileHandler(
        app_data_path("automatic_people_picture.log"),
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger


class AutomaticScanOrchestrator:
    def __init__(
        self,
        *,
        services: AutomaticScanServices | None = None,
        state_store: AutomaticStateStore | None = None,
        dependency_graph: AutomaticDependencyGraph | None = None,
    ) -> None:
        self._services = services
        self._state_store = state_store or AutomaticStateStore()
        self._dependency_graph = dependency_graph or AutomaticDependencyGraph()

    def scan_folder(
        self,
        folder: str | Path,
        config: AutomaticScanConfig,
        *,
        progress_callback: callable | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> AutomaticScanResult:
        scan_started_at = time.perf_counter()
        folder_path = Path(folder).expanduser().resolve()
        profiler = AutomaticScanProfiler()
        startup_started_at = time.perf_counter()
        with active_profiler(profiler):
            services = self._resolve_services(config)
        startup_seconds = time.perf_counter() - startup_started_at
        profiler.record_startup("startup.initialize_services", startup_seconds)
        diagnostics = AutomaticScanDiagnostics(
            config_signature=config.cache_signature(),
            folder=str(folder_path),
            runtime=services.runtime_info(),
        )
        warnings: list[PipelineWarning] = []
        errors: list[PipelineError] = []

        with active_profiler(profiler):
            discovered_records = self._run_stage(
                AutomaticStage.DISCOVERY,
                diagnostics,
                lambda: self._discover_records(services, folder_path, config),
            )
            images = [
                self._record_to_state(record)
                for record in discovered_records
            ]
            image_map = {image.path: image for image in images}
            folder_state = self._state_store.load(folder_path)
            self._state_store.apply(folder_state, images)

            face_analysis_by_path: dict[str, list[DetectedFace]] = {}
            embedded_faces_by_path: dict[str, list[EmbeddedFace]] = {}
            recognized_person_ids_by_path: dict[str, tuple[str, ...]] = {}
            unknown_faces_by_path: dict[str, list[EmbeddedFace]] = {}
            candidate_pairs: list[CandidatePair] = []
            duplicate_photos: dict[str, PhotoInfo] = {}
            duplicate_groups = []
            vibe_groups = []
            vibe_ungrouped: list[PipelineImageState] = []
            unknown_clusters: list[AutomaticUnknownCluster] = []
            known_people: dict[str, list[str]] = defaultdict(list)
            canceled = False

            try:
                self._resolve_caches(
                    services,
                    discovered_records,
                    image_map,
                    diagnostics,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token,
                )

                (
                    duplicate_photos,
                    face_analysis_by_path,
                    aligned_embedding_crops_by_path,
                ) = self._run_initial_analysis_bundle(
                    services,
                    discovered_records,
                    image_map,
                    config,
                    diagnostics,
                    warnings,
                    errors,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token,
                )

                blur_survivor_records = self._active_records(discovered_records, image_map)
                candidate_pairs = self._run_duplicate_candidate_pair_stage(
                    services,
                    list(duplicate_photos.values()),
                    diagnostics,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token,
                )

                self._run_people_and_face_quality_stages(
                    blur_survivor_records,
                    image_map,
                    face_analysis_by_path,
                    candidate_pairs,
                    config,
                    diagnostics,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token,
                )

                active_records = self._active_records(discovered_records, image_map)
                duplicate_groups = self._run_duplicate_finalization_stage(
                    services,
                    active_records,
                    image_map,
                    duplicate_photos,
                    candidate_pairs,
                    config,
                    diagnostics,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token,
                )

                embedding_records = self._active_records(discovered_records, image_map)
                embedded_faces_by_path = self._run_embedding_stage(
                    services,
                    embedding_records,
                    image_map,
                    face_analysis_by_path,
                    config,
                    aligned_embedding_crops_by_path,
                    diagnostics,
                    warnings,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token,
                )

                recognized_person_ids_by_path, unknown_faces_by_path, known_people = (
                    self._run_recognition_stage(
                        services,
                        embedding_records,
                        image_map,
                        embedded_faces_by_path,
                        config,
                        diagnostics,
                        warnings,
                        progress_callback=progress_callback,
                        cancellation_token=cancellation_token,
                    )
                )

                unknown_clusters = self._run_unknown_clustering_stage(
                    services,
                    image_map,
                    unknown_faces_by_path,
                    config,
                    diagnostics,
                    warnings,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token,
                )

                vibe_groups, vibe_ungrouped = self._run_vibe_grouping_stage(
                    services,
                    discovered_records,
                    image_map,
                    config,
                    diagnostics,
                    warnings,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token,
                )
            except ScanCancelledError:
                canceled = True
                warnings.append(
                    PipelineWarning(
                        stage=AutomaticStage.RESULT_FINALIZATION.value,
                        path=None,
                        message="Scan canceled. Partial results are available.",
                    )
                )
                duplicate_groups = []

        exclusions_by_reason = self._group_exclusions(images)
        summary = self._build_summary(
            images,
            duplicate_groups=duplicate_groups,
            known_people=known_people,
            unknown_clusters=unknown_clusters,
            vibe_group_count=len(vibe_groups),
            diagnostics=diagnostics,
            canceled=canceled,
        )
        result = self._run_stage(
            AutomaticStage.RESULT_FINALIZATION,
            diagnostics,
            lambda: AutomaticScanResult(
                folder=str(folder_path),
                all_images=images,
                vibe_groups=vibe_groups,
                vibe_ungrouped=vibe_ungrouped,
                duplicate_groups=duplicate_groups,
                known_people={key: sorted(set(value)) for key, value in known_people.items()},
                unknown_clusters=unknown_clusters,
                exclusions_by_reason=exclusions_by_reason,
                warnings=warnings,
                processing_errors=errors,
                progress_summary=summary,
                diagnostics=diagnostics,
            ),
        )
        folder_state.current_tab = folder_state.current_tab or "Scenes"
        self._state_store.capture(folder_state, images)
        self._state_store.save(folder_state)
        total_seconds = time.perf_counter() - scan_started_at
        self._finalize_diagnostics(
            diagnostics,
            profiler=profiler,
            total_seconds=total_seconds,
            startup_seconds=startup_seconds,
            image_count=len(images),
        )
        self._log_scan_timing(
            folder=folder_path,
            config=config,
            result=result,
            total_seconds=total_seconds,
        )
        self._log_people_picture_decisions(
            folder=folder_path,
            config=config,
            result=result,
        )
        return result

    def _resolve_services(self, config: AutomaticScanConfig):
        services = self._services
        if services is not None:
            is_compatible = getattr(services, "is_compatible", None)
            if callable(is_compatible):
                if is_compatible(config):
                    return services
            else:
                return services

        from .services import AutomaticScanServices

        services = AutomaticScanServices(config)
        self._services = services
        return services

    def _discover_records(
        self,
        services: AutomaticScanServices,
        folder_path: Path,
        config: AutomaticScanConfig,
    ) -> list[DiscoveryRecord]:
        records = services.discover(folder_path)
        discover_started = time.perf_counter()
        selected_extensions = None if not config.file_extensions else {
            value.lower()
            for value in config.file_extensions
        }
        selected_orientation = config.orientation_filter
        filtered: list[DiscoveryRecord] = []
        for record in records:
            if selected_extensions is not None and record.extension.lower() not in selected_extensions:
                continue
            if selected_orientation is not None and record.orientation != selected_orientation:
                continue
            filtered.append(record)
        profiler = get_active_profiler()
        if profiler is not None:
            profiler.record_discovery(
                "discovery.apply_filters",
                time.perf_counter() - discover_started,
            )
        return filtered

    @staticmethod
    def _record_to_state(record: DiscoveryRecord) -> PipelineImageState:
        return PipelineImageState(
            path=str(record.path),
            file_size=record.file_size,
            mtime_ns=record.mtime_ns,
            extension=record.extension,
            orientation=record.orientation,
            capture_timestamp=record.capture_timestamp,
            width=record.width,
            height=record.height,
        )

    def _resolve_caches(
        self,
        services: AutomaticScanServices,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        diagnostics: AutomaticScanDiagnostics,
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> None:
        stage_name = AutomaticStage.CACHE_RESOLUTION.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        total = max(len(records), 1)
        for index, record in enumerate(records, start=1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            image = image_map[str(record.path)]

            step_started = time.perf_counter()
            if services.get_blur_cached(record) is not None:
                image.cache_hits.add(AutomaticStage.BLUR_ANALYSIS.value)
                stats.cache_hits += 1
            else:
                image.cache_misses.add(AutomaticStage.BLUR_ANALYSIS.value)
                stats.cache_misses += 1
            self._record_step_timing(
                diagnostics,
                "cache_resolution.blur_lookup",
                time.perf_counter() - step_started,
            )

            step_started = time.perf_counter()
            if services.get_duplicate_photo_cached(record) is not None:
                image.cache_hits.add(AutomaticStage.DUPLICATE_CANDIDATES.value)
            else:
                image.cache_misses.add(AutomaticStage.DUPLICATE_CANDIDATES.value)
            self._record_step_timing(
                diagnostics,
                "cache_resolution.duplicate_lookup",
                time.perf_counter() - step_started,
            )

            step_started = time.perf_counter()
            if services.get_face_analysis_cached(record) is not None:
                image.cache_hits.add(AutomaticStage.FACE_DETECTION_ANALYSIS.value)
            else:
                image.cache_misses.add(AutomaticStage.FACE_DETECTION_ANALYSIS.value)
            self._record_step_timing(
                diagnostics,
                "cache_resolution.face_analysis_lookup",
                time.perf_counter() - step_started,
            )

            step_started = time.perf_counter()
            if services.get_embedded_faces_cached(record) is not None:
                image.cache_hits.add(AutomaticStage.FACE_EMBEDDING.value)
            else:
                image.cache_misses.add(AutomaticStage.FACE_EMBEDDING.value)
            self._record_step_timing(
                diagnostics,
                "cache_resolution.face_embedding_lookup",
                time.perf_counter() - step_started,
            )

            stats.processed += 1
            if progress_callback is not None:
                progress_callback(stage_name, index, total)
        stats.duration_seconds = round(time.perf_counter() - started_at, 4)
        diagnostics.cache_stats[stage_name] = {
            "hits": stats.cache_hits,
            "misses": stats.cache_misses,
        }

    def _run_initial_analysis_bundle(
        self,
        services: AutomaticScanServices,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        config: AutomaticScanConfig,
        diagnostics: AutomaticScanDiagnostics,
        warnings: list[PipelineWarning],
        errors: list[PipelineError],
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> tuple[
        dict[str, PhotoInfo],
        dict[str, list[DetectedFace]],
        dict[str, dict[int, Any]],
    ]:
        blur_stage = AutomaticStage.BLUR_ANALYSIS.value
        duplicate_stage = AutomaticStage.DUPLICATE_CANDIDATES.value
        face_stage = AutomaticStage.FACE_DETECTION_ANALYSIS.value
        blur_stats = diagnostics.stage_stats.setdefault(blur_stage, StageStats())
        duplicate_stats = diagnostics.stage_stats.setdefault(duplicate_stage, StageStats())
        face_stats = diagnostics.stage_stats.setdefault(face_stage, StageStats())
        photos: dict[str, PhotoInfo] = {}
        analysis_by_path: dict[str, list[DetectedFace]] = {}
        aligned_embedding_crops_by_path: dict[str, dict[int, Any]] = {}
        aligned_embedding_cache_limit = (
            128 * 1024 * 1024
            if config.performance.memory_conservative
            else 256 * 1024 * 1024
        )
        aligned_embedding_cache_bytes = 0
        blur_duration = 0.0
        duplicate_duration = 0.0
        face_duration = 0.0
        total = max(len(records), 1)

        for index, record in enumerate(records, start=1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            image_state = image_map[str(record.path)]
            shared_image = None

            def ensure_image():
                nonlocal shared_image
                if shared_image is None:
                    shared_image = default_image_loader.load_for_detection(record.path)
                    if shared_image is None:
                        raise ValueError(f"Could not decode image for scan analysis: {record.path}")
                return shared_image

            blur_started = time.perf_counter()
            try:
                step_started = time.perf_counter()
                cached_blur = services.get_blur_cached(record)
                self._record_step_timing(
                    diagnostics,
                    "blur_analysis.cache_lookup",
                    time.perf_counter() - step_started,
                )
                if cached_blur is not None:
                    blur_stats.cache_hits += 1
                    blur_result = cached_blur
                else:
                    step_started = time.perf_counter()
                    if hasattr(services, "analyze_blur_from_image"):
                        blur_result = services.analyze_blur_from_image(
                            record,
                            ensure_image(),
                        )
                    else:
                        blur_result = services.analyze_blur(record)
                    self._record_step_timing(
                        diagnostics,
                        "blur_analysis.run_detector",
                        time.perf_counter() - step_started,
                    )
                    blur_stats.cache_misses += 1
                image_state.completed_stages.add(blur_stage)
                step_started = time.perf_counter()
                self._apply_blur_policy(
                    image_state,
                    blur_result,
                    config=config,
                )
                self._record_step_timing(
                    diagnostics,
                    "blur_analysis.apply_policy",
                    time.perf_counter() - step_started,
                )
                blur_stats.processed += 1
            except Exception as exc:
                image_state.status = PipelineImageStatus.PROCESSING_ERROR
                image_state.primary_exclusion_reason = "processing_error"
                image_state.excluded_at_stage = blur_stage
                errors.append(
                    PipelineError(
                        stage=blur_stage,
                        path=image_state.path,
                        message=str(exc),
                    )
                )
            blur_duration += time.perf_counter() - blur_started
            if progress_callback is not None:
                progress_callback(blur_stage, index, total)

            duplicate_started = time.perf_counter()
            try:
                if not image_state.is_active:
                    duplicate_stats.skipped += 1
                else:
                    step_started = time.perf_counter()
                    photo = services.get_duplicate_photo_cached(record)
                    self._record_step_timing(
                        diagnostics,
                        "duplicate_candidates.cache_lookup",
                        time.perf_counter() - step_started,
                    )
                    if photo is not None:
                        duplicate_stats.cache_hits += 1
                    else:
                        step_started = time.perf_counter()
                        if hasattr(services, "index_duplicate_photo_from_image"):
                            photo = services.index_duplicate_photo_from_image(
                                record,
                                ensure_image(),
                            )
                        else:
                            photo = services.index_duplicate_photo(record)
                        self._record_step_timing(
                            diagnostics,
                            "duplicate_candidates.index_photo",
                            time.perf_counter() - step_started,
                        )
                        duplicate_stats.cache_misses += 1
                    if photo is not None:
                        photos[str(record.path)] = photo
                        image_state.completed_stages.add(duplicate_stage)
                        duplicate_stats.processed += 1
            except Exception as exc:
                errors.append(
                    PipelineError(
                        stage=duplicate_stage,
                        path=image_state.path,
                        message=str(exc),
                    )
                )
            duplicate_duration += time.perf_counter() - duplicate_started
            if progress_callback is not None:
                progress_callback(duplicate_stage, index, total)

            face_started = time.perf_counter()
            try:
                if not image_state.is_active:
                    face_stats.skipped += 1
                    analysis_by_path[str(record.path)] = []
                else:
                    step_started = time.perf_counter()
                    faces = services.get_face_analysis_cached(record)
                    self._record_step_timing(
                        diagnostics,
                        "face_detection_analysis.cache_lookup",
                        time.perf_counter() - step_started,
                    )
                    if faces is not None:
                        face_stats.cache_hits += 1
                    else:
                        step_started = time.perf_counter()
                        if hasattr(services, "analyze_faces_from_image"):
                            faces = services.analyze_faces_from_image(
                                record,
                                ensure_image(),
                            )
                        else:
                            faces = services.analyze_faces(record)
                        self._record_step_timing(
                            diagnostics,
                            "face_detection_analysis.run_detector_and_analyzer",
                            time.perf_counter() - step_started,
                        )
                        face_stats.cache_misses += 1
                    analysis_by_path[str(record.path)] = faces or []
                    image_state.completed_stages.add(face_stage)
                    face_stats.processed += 1
                    if (
                        shared_image is not None
                        and faces
                        and hasattr(services, "align_faces_for_embedding")
                    ):
                        embedding_faces = [
                            face
                            for face in faces
                            if face.index is not None
                            and face.analysis is not None
                            and face.analysis.embedding_utility_score
                            >= config.faces.minimum_embedding_utility
                        ]
                        if embedding_faces:
                            step_started = time.perf_counter()
                            aligned_faces = services.align_faces_for_embedding(
                                shared_image,
                                embedding_faces,
                            )
                            cache_bytes = sum(
                                getattr(aligned_face, "nbytes", 0)
                                for aligned_face in aligned_faces
                            )
                            if (
                                cache_bytes > 0
                                and aligned_embedding_cache_bytes + cache_bytes
                                <= aligned_embedding_cache_limit
                            ):
                                aligned_embedding_crops_by_path[str(record.path)] = {
                                    int(face.index): aligned_face
                                    for face, aligned_face in zip(embedding_faces, aligned_faces)
                                }
                                aligned_embedding_cache_bytes += cache_bytes
                            self._record_step_timing(
                                diagnostics,
                                "face_embedding.prepare_aligned_crops",
                                time.perf_counter() - step_started,
                            )
            except Exception as exc:
                warnings.append(
                    PipelineWarning(
                        stage=face_stage,
                        path=image_state.path,
                        message=f"Face analysis degraded: {exc}",
                    )
                )
                analysis_by_path[str(record.path)] = []
            face_duration += time.perf_counter() - face_started
            if progress_callback is not None:
                progress_callback(face_stage, index, total)

        blur_stats.duration_seconds = round(blur_duration, 4)
        duplicate_stats.duration_seconds = round(duplicate_duration, 4)
        face_stats.duration_seconds = round(face_duration, 4)
        return photos, analysis_by_path, aligned_embedding_crops_by_path

    def _apply_blur_policy(
        self,
        image: PipelineImageState,
        result,
        *,
        config: AutomaticScanConfig,
    ) -> None:
        status = result.status.lower()
        if status == "blurry":
            if config.blur_policy is BlurPolicy.KEEP_ALL:
                image.add_warning("blur_detected_manual_review")
            else:
                image.mark_terminal_exclusion(
                    stage=AutomaticStage.BLUR_ANALYSIS,
                    reason="blurry",
                    selected_for_deletion=True,
                )
                self._mark_downstream_skipped(
                    image,
                    AutomaticStage.BLUR_ANALYSIS,
                )
        elif status == "review":
            image.add_warning("borderline_blur")

    def _run_blur_stage(
        self,
        services: AutomaticScanServices,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        config: AutomaticScanConfig,
        diagnostics: AutomaticScanDiagnostics,
        warnings: list[PipelineWarning],
        errors: list[PipelineError],
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> None:
        stage_name = AutomaticStage.BLUR_ANALYSIS.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        total = max(len(records), 1)
        for index, record in enumerate(records, start=1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            image = image_map[str(record.path)]
            try:
                step_started = time.perf_counter()
                cached = services.get_blur_cached(record)
                self._record_step_timing(
                    diagnostics,
                    "blur_analysis.cache_lookup",
                    time.perf_counter() - step_started,
                )
                if cached is not None:
                    result = cached
                    stats.cache_hits += 1
                else:
                    step_started = time.perf_counter()
                    result = services.analyze_blur(record)
                    self._record_step_timing(
                        diagnostics,
                        "blur_analysis.run_detector",
                        time.perf_counter() - step_started,
                    )
                    stats.cache_misses += 1
                image.completed_stages.add(stage_name)
                step_started = time.perf_counter()
                status = result.status.lower()
                if status == "blurry":
                    if config.blur_policy is BlurPolicy.KEEP_ALL:
                        image.add_warning("blur_detected_manual_review")
                    else:
                        image.mark_terminal_exclusion(
                            stage=AutomaticStage.BLUR_ANALYSIS,
                            reason="blurry",
                            selected_for_deletion=True,
                        )
                        self._mark_downstream_skipped(
                            image,
                            AutomaticStage.BLUR_ANALYSIS,
                        )
                elif status == "review":
                    image.add_warning("borderline_blur")
                self._record_step_timing(
                    diagnostics,
                    "blur_analysis.apply_policy",
                    time.perf_counter() - step_started,
                )
                stats.processed += 1
            except Exception as exc:
                image.status = PipelineImageStatus.PROCESSING_ERROR
                image.primary_exclusion_reason = "processing_error"
                image.excluded_at_stage = stage_name
                errors.append(
                    PipelineError(
                        stage=stage_name,
                        path=image.path,
                        message=str(exc),
                    )
                )
            if progress_callback is not None:
                progress_callback(stage_name, index, total)
        stats.duration_seconds = round(time.perf_counter() - started_at, 4)

    def _run_duplicate_candidate_stage(
        self,
        services: AutomaticScanServices,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        diagnostics: AutomaticScanDiagnostics,
        errors: list[PipelineError],
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> dict[str, PhotoInfo]:
        stage_name = AutomaticStage.DUPLICATE_CANDIDATES.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        total = max(len(records), 1)
        photos: dict[str, PhotoInfo] = {}
        for index, record in enumerate(records, start=1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            image = image_map[str(record.path)]
            try:
                step_started = time.perf_counter()
                photo = services.get_duplicate_photo_cached(record)
                self._record_step_timing(
                    diagnostics,
                    "duplicate_candidates.cache_lookup",
                    time.perf_counter() - step_started,
                )
                if photo is not None:
                    stats.cache_hits += 1
                else:
                    step_started = time.perf_counter()
                    photo = services.index_duplicate_photo(record)
                    self._record_step_timing(
                        diagnostics,
                        "duplicate_candidates.index_photo",
                        time.perf_counter() - step_started,
                    )
                    stats.cache_misses += 1
                if photo is not None:
                    photos[str(record.path)] = photo
                    image.completed_stages.add(stage_name)
                    stats.processed += 1
            except Exception as exc:
                errors.append(
                    PipelineError(
                        stage=stage_name,
                        path=image.path,
                        message=str(exc),
                    )
                )
            if progress_callback is not None:
                progress_callback(stage_name, index, total)
        stats.duration_seconds = round(time.perf_counter() - started_at, 4)
        return photos

    def _run_duplicate_candidate_pair_stage(
        self,
        services: AutomaticScanServices,
        photos: list[PhotoInfo],
        diagnostics: AutomaticScanDiagnostics,
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> list[CandidatePair]:
        if not photos:
            return []
        stage_name = AutomaticStage.DUPLICATE_CANDIDATES.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        pair_started = time.perf_counter()
        pairs = services.generate_duplicate_candidates(
            photos,
            progress_callback=None,
            cancellation_token=cancellation_token,
        )
        self._record_step_timing(
            diagnostics,
            "duplicate_candidates.generate_pairs",
            time.perf_counter() - pair_started,
        )
        stats.processed += len(photos)
        stats.duration_seconds += round(time.perf_counter() - started_at, 4)
        if progress_callback is not None:
            progress_callback(stage_name, len(photos), len(photos))
        return pairs

    def _run_face_analysis_stage(
        self,
        services: AutomaticScanServices,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        diagnostics: AutomaticScanDiagnostics,
        warnings: list[PipelineWarning],
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> dict[str, list[DetectedFace]]:
        stage_name = AutomaticStage.FACE_DETECTION_ANALYSIS.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        total = max(len(records), 1)
        analysis_by_path: dict[str, list[DetectedFace]] = {}
        for index, record in enumerate(records, start=1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            image = image_map[str(record.path)]
            try:
                step_started = time.perf_counter()
                faces = services.get_face_analysis_cached(record)
                self._record_step_timing(
                    diagnostics,
                    "face_detection_analysis.cache_lookup",
                    time.perf_counter() - step_started,
                )
                if faces is not None:
                    stats.cache_hits += 1
                else:
                    step_started = time.perf_counter()
                    faces = services.analyze_faces(record)
                    self._record_step_timing(
                        diagnostics,
                        "face_detection_analysis.run_detector_and_analyzer",
                        time.perf_counter() - step_started,
                    )
                    stats.cache_misses += 1
                analysis_by_path[str(record.path)] = faces or []
                image.completed_stages.add(stage_name)
                stats.processed += 1
            except Exception as exc:
                warnings.append(
                    PipelineWarning(
                        stage=stage_name,
                        path=image.path,
                        message=f"Face analysis degraded: {exc}",
                    )
                )
                analysis_by_path[str(record.path)] = []
            if progress_callback is not None:
                progress_callback(stage_name, index, total)
        stats.duration_seconds = round(time.perf_counter() - started_at, 4)
        return analysis_by_path

    def _run_people_and_face_quality_stages(
        self,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        face_analysis_by_path: dict[str, list[DetectedFace]],
        candidate_pairs: list[CandidatePair],
        config: AutomaticScanConfig,
        diagnostics: AutomaticScanDiagnostics,
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> None:
        stage_stats_people = diagnostics.stage_stats.setdefault(
            AutomaticStage.PEOPLE_IMAGE_CLASSIFICATION.value,
            StageStats(),
        )
        stage_stats_quality = diagnostics.stage_stats.setdefault(
            AutomaticStage.FACE_QUALITY_GATE.value,
            StageStats(),
        )
        started_people = time.perf_counter()
        total = max(len(records), 1)
        for index, record in enumerate(records, start=1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            image = image_map[str(record.path)]
            faces = face_analysis_by_path.get(str(record.path), [])
            step_started = time.perf_counter()
            people_picture = assess_people_picture(
                image.width,
                image.height,
                faces,
                config,
            )
            self._record_step_timing(
                diagnostics,
                "people_image_classification.assess",
                time.perf_counter() - step_started,
            )
            image.people_picture_assessment = people_picture
            image.completed_stages.add(AutomaticStage.PEOPLE_IMAGE_CLASSIFICATION.value)
            stage_stats_people.processed += 1
            if progress_callback is not None:
                progress_callback(
                    AutomaticStage.PEOPLE_IMAGE_CLASSIFICATION.value,
                    index,
                    total,
                )
        stage_stats_people.duration_seconds = round(time.perf_counter() - started_people, 4)

        inference_started = time.perf_counter()
        self._infer_related_people_pictures(
            records,
            image_map,
            face_analysis_by_path,
            candidate_pairs,
            config,
        )
        self._record_step_timing(
            diagnostics,
            "people_image_classification.infer_related_people_pictures",
            time.perf_counter() - inference_started,
        )

        started_quality = time.perf_counter()
        for index, record in enumerate(records, start=1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            image = image_map[str(record.path)]
            faces = face_analysis_by_path.get(str(record.path), [])
            step_started = time.perf_counter()
            quality = assess_face_quality(
                faces,
                image.people_picture_assessment or assess_people_picture(
                    image.width,
                    image.height,
                    faces,
                    config,
                ),
                config,
            )
            self._record_step_timing(
                diagnostics,
                "face_quality_gate.assess",
                time.perf_counter() - step_started,
            )
            image.face_quality_assessment = quality
            image.completed_stages.add(AutomaticStage.FACE_QUALITY_GATE.value)
            step_started = time.perf_counter()
            for warning in quality.warnings:
                image.add_warning(warning)
            if quality.terminal_failure:
                exclusion_reason = "poor_primary_face_quality"
                if "dark_primary_faces" in quality.reasons:
                    exclusion_reason = "dark_primary_faces"
                elif "low_primary_pick_scores" in quality.reasons:
                    exclusion_reason = "low_primary_pick_scores"
                image.mark_terminal_exclusion(
                    stage=AutomaticStage.FACE_QUALITY_GATE,
                    reason=exclusion_reason,
                    selected_for_deletion=True,
                )
                self._mark_downstream_skipped(image, AutomaticStage.FACE_QUALITY_GATE)
            self._record_step_timing(
                diagnostics,
                "face_quality_gate.apply_policy",
                time.perf_counter() - step_started,
            )
            stage_stats_quality.processed += 1
            if progress_callback is not None:
                progress_callback(AutomaticStage.FACE_QUALITY_GATE.value, index, total)
        stage_stats_quality.duration_seconds = round(time.perf_counter() - started_quality, 4)

    def _infer_related_people_pictures(
        self,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        face_analysis_by_path: dict[str, list[DetectedFace]],
        candidate_pairs: list[CandidatePair],
        config: AutomaticScanConfig,
    ) -> None:
        related_paths: dict[str, set[str]] = defaultdict(set)
        for pair in candidate_pairs:
            left = str(pair.left.path)
            right = str(pair.right.path)
            related_paths[left].add(right)
            related_paths[right].add(left)

        burst_neighbors = self._burst_neighbor_paths(records)
        for path, neighbors in burst_neighbors.items():
            related_paths[path].update(neighbors)

        qualifying_paths: set[str] = set()
        for record in records:
            path = str(record.path)
            assessment = image_map[path].people_picture_assessment
            if assessment is not None and assessment.is_people_picture:
                qualifying_paths.add(path)

        for record in records:
            path = str(record.path)
            image = image_map[path]
            assessment = image.people_picture_assessment
            if assessment is None or assessment.is_people_picture:
                continue
            faces = face_analysis_by_path.get(path, [])
            if faces:
                continue

            related_people = sorted(
                other_path
                for other_path in related_paths.get(path, set())
                if other_path in qualifying_paths
            )
            if not related_people:
                continue

            image.people_picture_assessment = type(assessment)(
                is_people_picture=True,
                confidence=min(
                    0.72,
                    max(assessment.confidence, 0.62),
                ),
                primary_face_indices=(),
                supporting_reasons=tuple(
                    dict.fromkeys(
                        (
                            *assessment.supporting_reasons,
                            "inferred_from_related_people_picture",
                            f"related_count:{len(related_people)}",
                        )
                    )
                ),
            )

    @staticmethod
    def _burst_neighbor_paths(
        records: list[DiscoveryRecord],
    ) -> dict[str, set[str]]:
        ordered_records = sorted(
            records,
            key=lambda record: (
                float("inf") if record.capture_timestamp is None else record.capture_timestamp,
                str(record.path),
            ),
        )
        neighbors: dict[str, set[str]] = defaultdict(set)
        for left_record, right_record in zip(ordered_records, ordered_records[1:]):
            if left_record.path.parent != right_record.path.parent:
                continue
            if not AutomaticScanOrchestrator._looks_like_burst_neighbor(left_record, right_record):
                continue
            left_path = str(left_record.path)
            right_path = str(right_record.path)
            neighbors[left_path].add(right_path)
            neighbors[right_path].add(left_path)
        return neighbors

    @staticmethod
    def _looks_like_burst_neighbor(
        left_record: DiscoveryRecord,
        right_record: DiscoveryRecord,
    ) -> bool:
        if (
            left_record.capture_timestamp is not None
            and right_record.capture_timestamp is not None
            and abs(right_record.capture_timestamp - left_record.capture_timestamp) > 3.0
        ):
            return False

        left_index = AutomaticScanOrchestrator._trailing_numeric_suffix(left_record.path.stem)
        right_index = AutomaticScanOrchestrator._trailing_numeric_suffix(right_record.path.stem)
        if left_index is None or right_index is None:
            return False
        return abs(right_index - left_index) <= 2

    @staticmethod
    def _trailing_numeric_suffix(value: str) -> int | None:
        match = re.search(r"(\d+)$", value)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _run_duplicate_finalization_stage(
        self,
        services: AutomaticScanServices,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        duplicate_photos: dict[str, PhotoInfo],
        candidate_pairs: list[CandidatePair],
        config: AutomaticScanConfig,
        diagnostics: AutomaticScanDiagnostics,
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ):
        stage_name = AutomaticStage.DUPLICATE_FINALIZATION.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        active_paths = {str(record.path) for record in records}
        active_photos = [
            duplicate_photos[path]
            for path in sorted(active_paths)
            if path in duplicate_photos
        ]
        filtered_pairs = [
            pair
            for pair in candidate_pairs
            if str(pair.left.path) in active_paths and str(pair.right.path) in active_paths
        ]
        step_started = time.perf_counter()
        grouped_photos = services.verify_duplicate_groups(
            active_photos,
            filtered_pairs,
            cancellation_token=cancellation_token,
        )
        self._record_step_timing(
            diagnostics,
            "duplicate_finalization.verify_groups",
            time.perf_counter() - step_started,
        )
        step_started = time.perf_counter()
        duplicate_groups = select_duplicate_keepers(
            grouped_photos,
            image_states=image_map,
            config=config,
        )
        self._record_step_timing(
            diagnostics,
            "duplicate_finalization.rank_and_select_keepers",
            time.perf_counter() - step_started,
        )
        grouped_paths = set()
        step_started = time.perf_counter()
        for group in duplicate_groups:
            keeper_set = set(group.keeper_paths)
            for rank, path in enumerate(group.image_paths, start=1):
                grouped_paths.add(path)
                image = image_map[path]
                image.duplicate_group_id = group.group_id
                image.duplicate_rank = rank
                if path in keeper_set:
                    image.is_duplicate_keeper = True
                    image.selected_for_deletion = False
                    image.completed_stages.add(stage_name)
                    if image.manually_restored:
                        image.status = PipelineImageStatus.MANUALLY_RESTORED
                    elif image.manually_kept:
                        image.status = PipelineImageStatus.MANUALLY_KEPT
                    else:
                        image.status = PipelineImageStatus.ACTIVE
                else:
                    image.is_duplicate_keeper = False
                    image.mark_terminal_exclusion(
                        stage=AutomaticStage.DUPLICATE_FINALIZATION,
                        reason="duplicate_alternative",
                        selected_for_deletion=True,
                        duplicate_alternative=True,
                    )
                    self._mark_downstream_skipped(image, AutomaticStage.DUPLICATE_FINALIZATION)
            stats.processed += len(group.image_paths)
        for record in records:
            path = str(record.path)
            if path in grouped_paths:
                continue
            image_map[path].completed_stages.add(stage_name)
            stats.processed += 1
        self._record_step_timing(
            diagnostics,
            "duplicate_finalization.apply_group_results",
            time.perf_counter() - step_started,
        )
        stats.duration_seconds = round(time.perf_counter() - started_at, 4)
        if progress_callback is not None:
            progress_callback(stage_name, len(active_photos), max(len(active_photos), 1))
        return duplicate_groups

    def _run_embedding_stage(
        self,
        services: AutomaticScanServices,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        face_analysis_by_path: dict[str, list[DetectedFace]],
        config: AutomaticScanConfig,
        aligned_embedding_crops_by_path: dict[str, dict[int, Any]],
        diagnostics: AutomaticScanDiagnostics,
        warnings: list[PipelineWarning],
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> dict[str, list[EmbeddedFace]]:
        stage_name = AutomaticStage.FACE_EMBEDDING.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        total = max(len(records), 1)
        embedded_by_path: dict[str, list[EmbeddedFace]] = {}
        pending_records: list[DiscoveryRecord] = []
        pending_order: list[tuple[int, DiscoveryRecord]] = []

        for index, record in enumerate(records, start=1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            image = image_map[str(record.path)]
            faces = face_analysis_by_path.get(str(record.path), [])
            if not faces:
                image.skipped_stages.add(stage_name)
                stats.skipped += 1
                continue
            try:
                step_started = time.perf_counter()
                embedded_faces = services.get_embedded_faces_cached(record)
                self._record_step_timing(
                    diagnostics,
                    "face_embedding.cache_lookup",
                    time.perf_counter() - step_started,
                )
                if embedded_faces is not None:
                    stats.cache_hits += 1
                    embedded_by_path[str(record.path)] = [
                        face
                        for face in embedded_faces
                        if face.analysis is not None
                        and face.analysis.embedding_utility_score >= config.faces.minimum_embedding_utility
                    ]
                else:
                    pending_records.append(record)
                    pending_order.append((index, record))
            except Exception as exc:
                warnings.append(
                    PipelineWarning(
                        stage=stage_name,
                        path=image.path,
                        message=f"Embedding skipped: {exc}",
                    )
                )

        if pending_records:
            try:
                step_started = time.perf_counter()
                embedded_by_path.update(
                    services.embed_faces_batched(
                        pending_records,
                        face_analysis_by_path,
                        minimum_embedding_utility=config.faces.minimum_embedding_utility,
                        prealigned_faces_by_path=aligned_embedding_crops_by_path,
                    )
                )
                self._record_step_timing(
                    diagnostics,
                    "face_embedding.run_embedder",
                    time.perf_counter() - step_started,
                )
                stats.cache_misses += len(pending_records)
            except Exception:
                for record in pending_records:
                    image = image_map[str(record.path)]
                    try:
                        step_started = time.perf_counter()
                        embedded_by_path[str(record.path)] = services.embed_faces(
                            record,
                            face_analysis_by_path.get(str(record.path), []),
                            minimum_embedding_utility=config.faces.minimum_embedding_utility,
                        )
                        self._record_step_timing(
                            diagnostics,
                            "face_embedding.run_embedder",
                            time.perf_counter() - step_started,
                        )
                        stats.cache_misses += 1
                    except Exception as exc:
                        warnings.append(
                            PipelineWarning(
                                stage=stage_name,
                                path=image.path,
                                message=f"Embedding skipped: {exc}",
                            )
                        )

        for index, record in enumerate(records, start=1):
            image = image_map[str(record.path)]
            faces = face_analysis_by_path.get(str(record.path), [])
            if not faces:
                if progress_callback is not None:
                    progress_callback(stage_name, index, total)
                continue
            step_started = time.perf_counter()
            embedded_faces = [
                face
                for face in embedded_by_path.get(str(record.path), [])
                if face.analysis is not None
                and face.analysis.embedding_utility_score >= config.faces.minimum_embedding_utility
            ]
            self._record_step_timing(
                diagnostics,
                "face_embedding.filter_embedding_utility",
                time.perf_counter() - step_started,
            )
            embedded_by_path[str(record.path)] = embedded_faces
            image.completed_stages.add(stage_name)
            stats.processed += 1
            if progress_callback is not None:
                progress_callback(stage_name, index, total)
        stats.duration_seconds = round(time.perf_counter() - started_at, 4)
        return embedded_by_path

    def _run_recognition_stage(
        self,
        services: AutomaticScanServices,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        embedded_by_path: dict[str, list[EmbeddedFace]],
        config: AutomaticScanConfig,
        diagnostics: AutomaticScanDiagnostics,
        warnings: list[PipelineWarning],
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, list[EmbeddedFace]], dict[str, list[str]]]:
        stage_name = AutomaticStage.FACE_RECOGNITION.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        total = max(len(records), 1)
        recognized_by_path: dict[str, tuple[str, ...]] = {}
        unknown_by_path: dict[str, list[EmbeddedFace]] = {}
        known_people: dict[str, list[str]] = defaultdict(list)
        for index, record in enumerate(records, start=1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            image = image_map[str(record.path)]
            embedded_faces = embedded_by_path.get(str(record.path), [])
            if not embedded_faces:
                image.skipped_stages.add(stage_name)
                stats.skipped += 1
                if progress_callback is not None:
                    progress_callback(stage_name, index, total)
                continue
            try:
                if config.faces.recognize_known_people:
                    step_started = time.perf_counter()
                    recognized_faces, unknown_faces = services.recognize_faces(embedded_faces)
                    self._record_step_timing(
                        diagnostics,
                        "face_recognition.run_recognizer",
                        time.perf_counter() - step_started,
                    )
                else:
                    recognized_faces, unknown_faces = [], list(embedded_faces)
                    self._record_step_timing(
                        diagnostics,
                        "face_recognition.skip_recognizer",
                        0.0,
                    )
                step_started = time.perf_counter()
                person_ids = sorted({str(face.person_id) for face in recognized_faces})
                recognized_by_path[str(record.path)] = tuple(person_ids)
                image.known_person_ids = tuple(person_ids)
                unknown_by_path[str(record.path)] = list(unknown_faces)
                for face in recognized_faces:
                    person_name = services.get_person_name(face.person_id)
                    if person_name is None:
                        continue
                    known_people[person_name].append(image.path)
                self._record_step_timing(
                    diagnostics,
                    "face_recognition.materialize_results",
                    time.perf_counter() - step_started,
                )
                image.completed_stages.add(stage_name)
                stats.processed += 1
            except Exception as exc:
                unknown_by_path[str(record.path)] = list(embedded_faces)
                warnings.append(
                    PipelineWarning(
                        stage=stage_name,
                        path=image.path,
                        message=f"Recognition degraded: {exc}",
                    )
                )
            if progress_callback is not None:
                progress_callback(stage_name, index, total)
        stats.duration_seconds = round(time.perf_counter() - started_at, 4)
        return recognized_by_path, unknown_by_path, known_people

    def _run_unknown_clustering_stage(
        self,
        services: AutomaticScanServices,
        image_map: dict[str, PipelineImageState],
        unknown_faces_by_path: dict[str, list[EmbeddedFace]],
        config: AutomaticScanConfig,
        diagnostics: AutomaticScanDiagnostics,
        warnings: list[PipelineWarning],
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ) -> list[AutomaticUnknownCluster]:
        stage_name = AutomaticStage.UNKNOWN_FACE_CLUSTERING.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        all_unknown_faces: list[EmbeddedFace] = []
        for faces in unknown_faces_by_path.values():
            all_unknown_faces.extend(faces)
        if not config.faces.cluster_unknown_people or not all_unknown_faces:
            stats.skipped += len(image_map)
            stats.duration_seconds = round(time.perf_counter() - started_at, 4)
            if progress_callback is not None:
                progress_callback(stage_name, len(image_map), max(len(image_map), 1))
            return []
        try:
            step_started = time.perf_counter()
            clusters = services.cluster_unknown_faces(all_unknown_faces)
            self._record_step_timing(
                diagnostics,
                "unknown_face_clustering.cluster_faces",
                time.perf_counter() - step_started,
            )
            result: list[AutomaticUnknownCluster] = []
            step_started = time.perf_counter()
            for cluster in clusters:
                representative = cluster.representative
                nearest_match = services.nearest_known_match(representative)
                suggestions = services.suggest_known_names(nearest_match)
                cluster_id = f"unknown-{cluster.id}"
                result.append(
                    AutomaticUnknownCluster(
                        cluster_id=cluster_id,
                        image_paths=sorted({str(face.path) for face in cluster.faces}),
                        representative_path=str(representative.path),
                        face_count=len(cluster.faces),
                        suggested_names=suggestions,
                        hidden=(
                            config.faces.hide_low_quality_unknown_clusters
                            and representative.analysis is not None
                            and representative.analysis.embedding_utility_score
                            < config.faces.minimum_embedding_utility
                        ),
                        cluster=cluster,
                    )
                )
                for face in cluster.faces:
                    image = image_map[str(face.path)]
                    image.unknown_cluster_ids = tuple(
                        sorted({*image.unknown_cluster_ids, cluster_id})
                    )
            self._record_step_timing(
                diagnostics,
                "unknown_face_clustering.materialize_clusters",
                time.perf_counter() - step_started,
            )
            stats.processed = len(result)
            if progress_callback is not None:
                progress_callback(stage_name, len(result), max(len(result), 1))
            stats.duration_seconds = round(time.perf_counter() - started_at, 4)
            return result
        except Exception as exc:
            warnings.append(
                PipelineWarning(
                    stage=stage_name,
                    path=None,
                    message=f"Unknown clustering degraded: {exc}",
                )
            )
            stats.duration_seconds = round(time.perf_counter() - started_at, 4)
            return []

    def _run_vibe_grouping_stage(
        self,
        services: AutomaticScanServices,
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
        config: AutomaticScanConfig,
        diagnostics: AutomaticScanDiagnostics,
        warnings: list[PipelineWarning],
        *,
        progress_callback: callable | None,
        cancellation_token: CancellationToken | None,
    ):
        stage_name = AutomaticStage.VIBE_GROUPING.value
        stats = diagnostics.stage_stats.setdefault(stage_name, StageStats())
        started_at = time.perf_counter()
        active_paths = [
            record.path.as_posix()
            for record in records
            if image_map[str(record.path)].is_active
        ]
        step_started = time.perf_counter()
        vibe_result, fallback_used, fallback_reason = group_vibes_with_fallback(
            services,
            active_paths,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
        )
        self._record_step_timing(
            diagnostics,
            "vibe_grouping.run_grouping",
            time.perf_counter() - step_started,
        )
        diagnostics.vibe_fallback_used = fallback_used
        diagnostics.vibe_fallback_reason = fallback_reason
        if fallback_used and fallback_reason:
            warnings.append(
                PipelineWarning(
                    stage=stage_name,
                    path=None,
                    message=f"Vibe grouping degraded to fallback mode: {fallback_reason}",
                )
            )
        step_started = time.perf_counter()
        for group in vibe_result.groups:
            for path in group.image_paths:
                image = image_map.get(str(Path(path).resolve()))
                if image is not None:
                    image.vibe_group_id = group.group_id
        ungrouped = [
            image_map[str(Path(path).resolve())]
            for path in vibe_result.ungrouped_paths
            if str(Path(path).resolve()) in image_map
        ]
        self._record_step_timing(
            diagnostics,
            "vibe_grouping.apply_group_assignments",
            time.perf_counter() - step_started,
        )
        stats.processed = len(active_paths)
        stats.duration_seconds = round(time.perf_counter() - started_at, 4)
        return vibe_result.groups, ungrouped

    def _mark_downstream_skipped(
        self,
        image: PipelineImageState,
        from_stage: AutomaticStage,
    ) -> None:
        for stage in self._dependency_graph.required_downstream_stages(from_stage):
            image.skipped_stages.add(stage.value)

    @staticmethod
    def _active_records(
        records: list[DiscoveryRecord],
        image_map: dict[str, PipelineImageState],
    ) -> list[DiscoveryRecord]:
        return [
            record
            for record in records
            if image_map[str(record.path)].is_active
        ]

    @staticmethod
    def _group_exclusions(
        images: list[PipelineImageState],
    ) -> dict[str, list[PipelineImageState]]:
        grouped: dict[str, list[PipelineImageState]] = defaultdict(list)
        for image in images:
            if image.primary_exclusion_reason is None:
                continue
            grouped[image.primary_exclusion_reason].append(image)
        return dict(grouped)

    @staticmethod
    def _build_summary(
        images: list[PipelineImageState],
        *,
        duplicate_groups,
        known_people: dict[str, list[str]],
        unknown_clusters: list[AutomaticUnknownCluster],
        vibe_group_count: int,
        diagnostics: AutomaticScanDiagnostics,
        canceled: bool,
    ) -> PipelineSummary:
        active_images = [image for image in images if image.is_active]
        duplicate_alternatives = [
            image
            for image in images
            if image.status is PipelineImageStatus.DUPLICATE_ALTERNATIVE
        ]
        return PipelineSummary(
            discovered_count=len(images),
            active_count=len(active_images),
            excluded_count=len(images) - len(active_images),
            duplicate_group_count=len(duplicate_groups),
            keeper_count=sum(1 for image in images if image.is_duplicate_keeper),
            duplicate_alternative_count=len(duplicate_alternatives),
            known_people_count=len(known_people),
            unknown_cluster_count=len(unknown_clusters),
            vibe_group_count=vibe_group_count,
            canceled=canceled,
            stage_stats=diagnostics.stage_stats,
        )

    def _run_stage(
        self,
        stage: AutomaticStage,
        diagnostics: AutomaticScanDiagnostics,
        func,
    ):
        stats = diagnostics.stage_stats.setdefault(stage.value, StageStats())
        started_at = time.perf_counter()
        result = func()
        stats.duration_seconds += round(time.perf_counter() - started_at, 4)
        return result

    @staticmethod
    def _record_step_timing(
        diagnostics: AutomaticScanDiagnostics,
        key: str,
        seconds: float,
    ) -> None:
        diagnostics.step_timings[key] = round(
            diagnostics.step_timings.get(key, 0.0) + seconds,
            6,
        )

    @staticmethod
    def _infer_cache_mode(diagnostics: AutomaticScanDiagnostics) -> str:
        cache_hits = 0
        cache_misses = 0
        for stats in diagnostics.stage_stats.values():
            cache_hits += stats.cache_hits
            cache_misses += stats.cache_misses
        if cache_hits and not cache_misses:
            return "warm"
        if cache_misses and not cache_hits:
            return "cold"
        if cache_hits or cache_misses:
            return "mixed"
        return "unknown"

    def _finalize_diagnostics(
        self,
        diagnostics: AutomaticScanDiagnostics,
        *,
        profiler: AutomaticScanProfiler,
        total_seconds: float,
        startup_seconds: float,
        image_count: int,
    ) -> None:
        exported = profiler.export(
            total_seconds=total_seconds,
            startup_seconds=startup_seconds,
            image_count=image_count,
            cache_mode=self._infer_cache_mode(diagnostics),
        )
        diagnostics.total_seconds = exported["total_seconds"]
        diagnostics.startup_seconds = exported["startup_seconds"]
        diagnostics.scan_seconds = exported["scan_seconds"]
        diagnostics.images_per_second = exported["images_per_second"]
        diagnostics.cache_mode = exported["cache_mode"]
        diagnostics.image_io = exported["image_io"]
        diagnostics.models = exported["models"]
        diagnostics.database = exported["database"]
        diagnostics.queues = exported["queues"]
        diagnostics.startup_breakdown = exported["startup_breakdown"]
        diagnostics.discovery_breakdown = exported["discovery_breakdown"]
        diagnostics.model_phase_breakdown = exported["model_phase_breakdown"]

    def _log_scan_timing(
        self,
        *,
        folder: Path,
        config: AutomaticScanConfig,
        result: AutomaticScanResult,
        total_seconds: float,
    ) -> None:
        payload = {
            "event": "automatic_scan_timing",
            "folder": str(folder),
            "status": "canceled" if result.progress_summary.canceled else "completed",
            "total_seconds": round(total_seconds, 4),
            "startup_seconds": result.diagnostics.startup_seconds,
            "scan_seconds": result.diagnostics.scan_seconds,
            "images_per_second": result.diagnostics.images_per_second,
            "cache_mode": result.diagnostics.cache_mode,
            "config_signature": result.diagnostics.config_signature,
            "runtime": result.diagnostics.runtime,
            "summary": {
                "discovered_count": result.progress_summary.discovered_count,
                "active_count": result.progress_summary.active_count,
                "excluded_count": result.progress_summary.excluded_count,
                "duplicate_group_count": result.progress_summary.duplicate_group_count,
                "keeper_count": result.progress_summary.keeper_count,
                "unknown_cluster_count": result.progress_summary.unknown_cluster_count,
                "vibe_group_count": result.progress_summary.vibe_group_count,
            },
            "stage_stats": {
                key: {
                    "processed": value.processed,
                    "skipped": value.skipped,
                    "cache_hits": value.cache_hits,
                    "cache_misses": value.cache_misses,
                    "duration_seconds": value.duration_seconds,
                }
                for key, value in sorted(result.diagnostics.stage_stats.items())
            },
            "step_timings": dict(sorted(result.diagnostics.step_timings.items())),
            "cache_stats": result.diagnostics.cache_stats,
            "image_io": result.diagnostics.image_io,
            "models": result.diagnostics.models,
            "database": result.diagnostics.database,
            "queues": result.diagnostics.queues,
            "startup_breakdown": result.diagnostics.startup_breakdown,
            "discovery_breakdown": result.diagnostics.discovery_breakdown,
            "model_phase_breakdown": result.diagnostics.model_phase_breakdown,
            "warnings": [warning.message for warning in result.warnings],
            "errors": [error.message for error in result.processing_errors],
            "vibe_fallback_used": result.diagnostics.vibe_fallback_used,
            "vibe_fallback_reason": result.diagnostics.vibe_fallback_reason,
            "diagnostics_export_enabled": config.vibe.diagnostics_export,
        }
        _get_automatic_scan_timing_logger().info(json.dumps(payload, sort_keys=True))

    def _log_people_picture_decisions(
        self,
        *,
        folder: Path,
        config: AutomaticScanConfig,
        result: AutomaticScanResult,
    ) -> None:
        not_people_picture_images = [
            image
            for image in result.all_images
            if image.people_picture_assessment is not None
            and not image.people_picture_assessment.is_people_picture
        ]
        payload = {
            "event": "automatic_people_picture",
            "folder": str(folder),
            "config_signature": result.diagnostics.config_signature,
            "face_quality_preset": config.face_quality_preset.value,
            "not_people_picture_count": len(not_people_picture_images),
            "not_people_picture_images": [
                AutomaticScanResult._not_people_picture_payload(image)
                for image in not_people_picture_images
            ],
        }
        _get_automatic_people_picture_logger().info(json.dumps(payload, sort_keys=True))
