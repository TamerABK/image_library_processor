from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable

import cv2
import numpy as np

from app_paths import app_data_path, model_path
from blur_detector.blur_detector import BlurDetector
from duplicate_detector.config import DetectorConfig
from duplicate_detector.duplicate_detector import DuplicateDetector
from duplicate_detector.models import DuplicateGroup
from face_detector.connected_face_clusterer import ConnectedComponentFaceClusterer
from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_detector.face_aligner import FaceAligner
from face_detector.face_database_sqlite import SQLiteFaceDatabase
from face_detector.preview_renderer import FacePreviewRenderer
from face_processing.interfaces import FaceDatabase, FaceDetector
from face_processing.models import DetectedFace
from face_processing.processor import FaceProcessor
from face_processing.recognition import DefaultFaceRecognizer
from grouping.models import VibeGroupingResult
from grouping.vibe import VibeGroupingPreset, VibeGroupingProcessor, preset_config
from grouping.vibe.config import VibeGroupingConfig
from image_file_utils import discover_supported_extensions, normalize_extensions
from scan_controls import CancellationToken, ScanCancelledError

from .models import (
    AppState,
    BackgroundMessage,
    ConfirmedUnknownPerson,
    DeleteResult,
    ExportResult,
    PreviewItem,
    ResultGroup,
    ResultItem,
    ScanErrorMessage,
    ScanProgressMessage,
    ScanResultMessage,
    UnknownFacePrompt,
    UnknownFacesMessage,
)


def _get_face_scan_timing_logger() -> logging.Logger:
    logger = logging.getLogger("image_deduplicator.face_scan_timing")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(app_data_path("face_scan_timings.log"), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger


class _FaceScanTimingCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.load_seconds = 0.0
        self.detect_seconds = 0.0
        self.analyze_seconds = 0.0
        self.embed_seconds = 0.0
        self.align_seconds = 0.0
        self.embed_preprocess_seconds = 0.0
        self.embed_infer_seconds = 0.0
        self.embed_postprocess_seconds = 0.0
        self.loaded_image_count = 0
        self.failed_load_count = 0
        self.detect_call_count = 0
        self.analyze_call_count = 0
        self.embed_call_count = 0
        self.detected_face_count = 0
        self.analyzed_face_count = 0
        self.embedded_face_count = 0

    def record_load(self, seconds: float, success: bool) -> None:
        with self._lock:
            self.load_seconds += seconds
            self.loaded_image_count += 1
            if not success:
                self.failed_load_count += 1

    def record_detect(self, seconds: float, face_count: int) -> None:
        with self._lock:
            self.detect_seconds += seconds
            self.detect_call_count += 1
            self.detected_face_count += face_count

    def record_analyze(self, seconds: float, face_count: int) -> None:
        with self._lock:
            self.analyze_seconds += seconds
            self.analyze_call_count += 1
            self.analyzed_face_count += face_count

    def record_embed(self, seconds: float, face_count: int) -> None:
        with self._lock:
            self.embed_seconds += seconds
            self.embed_call_count += 1
            self.embedded_face_count += face_count

    def record_embed_phase(self, phase: str, seconds: float) -> None:
        with self._lock:
            if phase == "align":
                self.align_seconds += seconds
            elif phase == "embed_preprocess":
                self.embed_preprocess_seconds += seconds
            elif phase == "embed_infer":
                self.embed_infer_seconds += seconds
            elif phase == "embed_postprocess":
                self.embed_postprocess_seconds += seconds

    def summary(
        self,
        *,
        total_seconds: float,
        init_seconds: float,
    ) -> dict[str, Any]:
        with self._lock:
            load_seconds = self.load_seconds
            detect_seconds = self.detect_seconds
            analyze_seconds = self.analyze_seconds
            embed_seconds = self.embed_seconds
            align_seconds = self.align_seconds
            embed_preprocess_seconds = self.embed_preprocess_seconds
            embed_infer_seconds = self.embed_infer_seconds
            embed_postprocess_seconds = self.embed_postprocess_seconds
            loaded_image_count = self.loaded_image_count
            failed_load_count = self.failed_load_count
            detect_call_count = self.detect_call_count
            analyze_call_count = self.analyze_call_count
            embed_call_count = self.embed_call_count
            detected_face_count = self.detected_face_count
            analyzed_face_count = self.analyzed_face_count
            embedded_face_count = self.embedded_face_count

        other_seconds = max(
            0.0,
            total_seconds
            - init_seconds
            - load_seconds
            - detect_seconds
            - analyze_seconds
            - embed_seconds,
        )
        category_seconds = {
            "load": load_seconds,
            "detect": detect_seconds,
            "analyze": analyze_seconds,
            "embed": embed_seconds,
            "init": init_seconds,
            "other": other_seconds,
        }
        largest_stage = max(category_seconds, key=category_seconds.get)

        def pct(seconds: float) -> float:
            if total_seconds <= 0:
                return 0.0
            return round((seconds / total_seconds) * 100, 2)

        return {
            "total_seconds": round(total_seconds, 4),
            "largest_stage": largest_stage,
            "load_seconds": round(load_seconds, 4),
            "load_pct": pct(load_seconds),
            "detect_seconds": round(detect_seconds, 4),
            "detect_pct": pct(detect_seconds),
            "analyze_seconds": round(analyze_seconds, 4),
            "analyze_pct": pct(analyze_seconds),
            "embed_seconds": round(embed_seconds, 4),
            "embed_pct": pct(embed_seconds),
            "align_seconds": round(align_seconds, 4),
            "align_pct": pct(align_seconds),
            "embed_preprocess_seconds": round(embed_preprocess_seconds, 4),
            "embed_preprocess_pct": pct(embed_preprocess_seconds),
            "embed_infer_seconds": round(embed_infer_seconds, 4),
            "embed_infer_pct": pct(embed_infer_seconds),
            "embed_postprocess_seconds": round(embed_postprocess_seconds, 4),
            "embed_postprocess_pct": pct(embed_postprocess_seconds),
            "init_seconds": round(init_seconds, 4),
            "init_pct": pct(init_seconds),
            "other_seconds": round(other_seconds, 4),
            "other_pct": pct(other_seconds),
            "loaded_image_count": loaded_image_count,
            "failed_load_count": failed_load_count,
            "detect_call_count": detect_call_count,
            "analyze_call_count": analyze_call_count,
            "embed_call_count": embed_call_count,
            "detected_face_count": detected_face_count,
            "analyzed_face_count": analyzed_face_count,
            "embedded_face_count": embedded_face_count,
        }


class _TimedFaceProcessor(FaceProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._timing_collector: _FaceScanTimingCollector | None = None
        self._timing_lock = threading.Lock()
        if hasattr(self._embedder, "set_timing_callback"):
            self._embedder.set_timing_callback(self._record_embed_phase)

    def set_timing_collector(
        self,
        collector: _FaceScanTimingCollector | None,
    ) -> None:
        with self._timing_lock:
            self._timing_collector = collector

    def _load_image(
        self,
        path: Path,
    ) -> np.ndarray | None:
        collector = self._timing_collector

        load_started = time.perf_counter()
        image = super()._load_image(path)
        load_elapsed = time.perf_counter() - load_started
        if collector is not None:
            collector.record_load(load_elapsed, success=image is not None)

        return image

    def _detect_faces(
        self,
        detector: FaceDetector,
        image: np.ndarray,
        path: Path,
    ) -> list[DetectedFace]:
        collector = self._timing_collector
        detect_started = time.perf_counter()
        detected = super()._detect_faces(detector, image, path)
        detect_elapsed = time.perf_counter() - detect_started
        if collector is not None:
            collector.record_detect(detect_elapsed, len(detected))
        return detected

    def _embed_requests(
        self,
        face_requests: list[Any],
    ) -> list[Any]:
        collector = self._timing_collector
        embed_started = time.perf_counter()
        embedded = super()._embed_requests(face_requests)
        embed_elapsed = time.perf_counter() - embed_started
        if collector is not None:
            collector.record_embed(embed_elapsed, len(embedded))
        return embedded

    def _analyze_faces(
        self,
        image: np.ndarray,
        faces: list[DetectedFace],
    ) -> list[DetectedFace]:
        collector = self._timing_collector
        analyze_started = time.perf_counter()
        analyzed = super()._analyze_faces(image, faces)
        analyze_elapsed = time.perf_counter() - analyze_started
        if collector is not None and self._analyzer is not None and faces:
            collector.record_analyze(analyze_elapsed, len(analyzed))
        return analyzed

    def runtime_info(self) -> dict[str, Any]:
        info = {
            "max_workers": self._max_workers,
            "embed_batch_size": self._embed_batch_size,
        }
        if hasattr(self._detector, "runtime_info"):
            info["detector"] = self._detector.runtime_info()
        if hasattr(self._embedder, "runtime_info"):
            info["embedder"] = self._embedder.runtime_info()
        return info

    def _record_embed_phase(
        self,
        phase: str,
        seconds: float,
    ) -> None:
        collector = self._timing_collector
        if collector is not None:
            collector.record_embed_phase(phase, seconds)


class PhotoCleanerViewModel:
    all_file_types_label = "All supported"
    all_orientations_label = "All pictures"
    landscape_orientation_label = "Landscape only"
    portrait_orientation_label = "Portrait only"
    results_page_size = 48

    def __init__(
        self,
        *,
        duplicate_detector_factory: Callable[[], DuplicateDetector] | None = None,
        blur_detector_factory: Callable[[], BlurDetector] | None = None,
        vibe_processor_factory: Callable[[VibeGroupingConfig], VibeGroupingProcessor] | None = None,
    ) -> None:
        self.state = AppState(
            file_type=self.all_file_types_label,
            available_file_types=(self.all_file_types_label,),
            orientation=self.all_orientations_label,
            available_orientations=(
                self.all_orientations_label,
                self.landscape_orientation_label,
                self.portrait_orientation_label,
            ),
            available_vibe_presets=tuple(
                preset.display_name
                for preset in (
                    VibeGroupingPreset.SESSION,
                    VibeGroupingPreset.BALANCED_SCENES,
                    VibeGroupingPreset.TIGHT_SCENES,
                )
            ),
        )

        self._queue: Queue[BackgroundMessage] = Queue()
        self._scan_thread: threading.Thread | None = None
        self._cancellation_token: CancellationToken | None = None
        self._scan_start_time: float | None = None
        self._last_progress_percent = -1
        self._results: list[ResultGroup] = []
        self._selection_state: dict[Path, bool] = {}
        self._duplicate_detector_factory = duplicate_detector_factory or DuplicateDetector
        self._blur_detector_factory = blur_detector_factory or BlurDetector
        self._vibe_processor_factory = vibe_processor_factory or (
            lambda config: VibeGroupingProcessor(config)
        )
        self._face_processor: _TimedFaceProcessor | None = None
        self._face_database: FaceDatabase | None = None
        self._face_groups: list[ResultGroup] = []
        self._face_group_labels: list[str] = []
        self._active_face_group_index: int | None = None
        self._named_unknown_faces: dict[int, ConfirmedUnknownPerson] = {}
        self._latest_face_result: Any = None
        self._latest_vibe_debug_payload: dict[str, Any] | None = None
        self._latest_vibe_folder: Path | None = None
        self._unknown_face_preview_renderer = FacePreviewRenderer()
        self._results_page_index = 0
        self._supported_file_types = self._collect_supported_file_types()
        self._apply_mode_state()
        self._update_action_state()

    def set_folder(self, folder: str) -> None:
        self.state.folder = folder

    def set_file_type(self, file_type: str) -> None:
        self.state.file_type = file_type or self.all_file_types_label

    def set_mode(self, mode: str) -> None:
        self.state.mode = mode
        self._apply_mode_state()
        self.refresh_file_types()
        self._update_action_state()

    def set_known_people_only(self, known_people_only: bool) -> None:
        self.state.known_people_only = known_people_only

    def set_auto_export_faces(self, auto_export_faces: bool) -> None:
        self.state.auto_export_faces = auto_export_faces

    def set_vibe_preset(self, preset: str) -> None:
        self.state.vibe_preset = preset or "Balanced Scenes"

    def set_vibe_include_people(self, enabled: bool) -> None:
        self.state.vibe_include_people = enabled

    def set_vibe_include_color(self, enabled: bool) -> None:
        self.state.vibe_include_color = enabled

    def set_vibe_include_composition(self, enabled: bool) -> None:
        self.state.vibe_include_composition = enabled

    def set_vibe_show_advanced(self, show_advanced: bool) -> None:
        self.state.vibe_show_advanced = show_advanced

    def set_vibe_session_gap_minutes(self, value: str) -> None:
        self.state.vibe_session_gap_minutes = value.strip() or self.state.vibe_session_gap_minutes

    def set_vibe_minimum_similarity(self, value: str) -> None:
        self.state.vibe_minimum_similarity = value.strip() or self.state.vibe_minimum_similarity

    def set_vibe_minimum_cohesion(self, value: str) -> None:
        self.state.vibe_minimum_cohesion = value.strip() or self.state.vibe_minimum_cohesion

    def set_vibe_maximum_group_size(self, value: str) -> None:
        self.state.vibe_maximum_group_size = value.strip() or self.state.vibe_maximum_group_size

    def set_vibe_batch_size(self, value: str) -> None:
        self.state.vibe_batch_size = value.strip() or self.state.vibe_batch_size

    def set_orientation(self, orientation: str) -> None:
        if orientation in self.state.available_orientations:
            self.state.orientation = orientation
        else:
            self.state.orientation = self.all_orientations_label

    def is_scanning(self) -> bool:
        return self._scan_start_time is not None

    def refresh_file_types(self, folder: Path | None = None) -> None:
        if folder is None:
            folder = Path(self.state.folder).expanduser()

        supported_file_types = self._supported_file_types_for_mode(self.state.mode)
        if not folder.exists() or not folder.is_dir():
            values = (self.all_file_types_label,)
        else:
            found_types = discover_supported_extensions(folder, supported_file_types)
            values = (self.all_file_types_label, *found_types)

        current_value = self.state.file_type
        self.state.available_file_types = values
        if current_value in values:
            self.state.file_type = current_value
        else:
            self.state.file_type = self.all_file_types_label

    def selected_file_extensions(self) -> tuple[str, ...] | None:
        selected_value = self.state.file_type.strip().lower()
        if not selected_value or selected_value == self.all_file_types_label.lower():
            return None
        return (selected_value,)

    def selected_orientation_filter(self) -> str | None:
        if self.state.orientation == self.landscape_orientation_label:
            return "landscape"
        if self.state.orientation == self.portrait_orientation_label:
            return "portrait"
        return None

    def start_scan(self) -> str | None:
        if self._scan_thread and self._scan_thread.is_alive():
            return None

        folder = Path(self.state.folder).expanduser()
        if not folder.exists() or not folder.is_dir():
            return "Choose a valid folder to scan."

        self.refresh_file_types(folder)
        if self.state.mode == "vibe":
            try:
                self._current_vibe_config()
            except Exception as exc:
                return f"Invalid vibe settings: {exc}"
        self._reset_results()
        self.state.can_scan = False
        self.state.can_delete = False
        self.state.can_export = False
        self.state.can_export_vibe_debug = False
        self.state.can_cancel = True
        self.state.progress_mode = "indeterminate"
        self.state.progress_value = 0
        self.state.progress_max = 100
        self._last_progress_percent = -1
        self._scan_start_time = time.monotonic()
        self._cancellation_token = CancellationToken()
        self.refresh_elapsed()

        mode = self.state.mode
        file_extensions = self.selected_file_extensions()
        orientation_filter = self.selected_orientation_filter()
        known_people_only = self.state.known_people_only
        self.state.status = f"Scanning {self._scan_target_label(mode, known_people_only)}..."
        self.state.count_text = ""

        self._scan_thread = threading.Thread(
            target=self._scan_worker,
            args=(
                folder,
                mode,
                file_extensions,
                orientation_filter,
                known_people_only,
                self._cancellation_token,
            ),
            daemon=True,
        )
        self._scan_thread.start()
        return None

    def cancel_scan(self) -> None:
        if self._cancellation_token is None:
            return
        self._cancellation_token.cancel()
        self.state.status = "Canceling scan..."
        self.state.can_cancel = False

    def poll_background_message(self) -> BackgroundMessage | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def handle_progress_message(self, message: ScanProgressMessage) -> None:
        target = self._scan_target_label(message.mode, message.known_people_only)
        if not message.total or message.total <= 0:
            self.state.status = f"Scanning {target}..."
            return

        progress_percent = self._compute_progress_percent(
            message.mode,
            message.phase,
            message.done,
            message.total,
        )
        progress_percent = max(self._last_progress_percent, progress_percent)
        if progress_percent <= self._last_progress_percent:
            return

        self.state.progress_mode = "determinate"
        self.state.progress_max = 100
        self.state.progress_value = progress_percent
        self._last_progress_percent = progress_percent

        if message.phase == "scanning":
            self.state.status = f"Scanning {target} {message.done}/{message.total}"
        else:
            phase_label = message.phase.replace("_", " ").title()
            self.state.status = (
                f"Scanning {target}: {phase_label} {message.done}/{message.total}"
            )

    def handle_unknown_faces_message(self, message: UnknownFacesMessage) -> None:
        self._latest_face_result = message.face_result

    def handle_scan_result_message(self, message: ScanResultMessage) -> None:
        if message.mode == "vibe":
            self._latest_vibe_debug_payload = (
                None if message.debug_payload is None else dict(message.debug_payload)
            )
            input_folder = None if message.debug_payload is None else message.debug_payload.get("input_folder")
            self._latest_vibe_folder = None if not input_folder else Path(str(input_folder))
        else:
            self._latest_vibe_debug_payload = None
            self._latest_vibe_folder = None

        results = message.results
        if message.mode == "faces" and self._latest_face_result is not None:
            results = self._build_face_results(self._latest_face_result)

        total_items = sum(len(group.items) for group in results)
        if total_items == 0:
            self._results = []
            self._clear_face_groups()
            if message.mode == "duplicates":
                self.finish_scan("No near duplicates found.")
            elif message.mode == "vibe":
                self.finish_scan(message.summary or "No vibe groups found.")
            elif message.mode == "blurry":
                self.finish_scan("No blurry photos found.")
            elif message.known_people_only:
                self.finish_scan("No known faces found.")
            else:
                self.finish_scan("No faces found.")
            return

        if message.mode == "faces":
            self._configure_face_groups(results)
            self.show_face_group(0)
        else:
            self._results = results
            self._results_page_index = 0
            self._initialize_selection_state()
            self._update_results_view_state()

        if message.mode == "duplicates":
            self.finish_scan(
                message.summary or f"Found {len(results)} duplicate groups with {total_items} photos."
            )
        elif message.mode == "vibe":
            self.finish_scan(
                message.summary or f"Found {len(results)} vibe groups with {total_items} photos."
            )
        elif message.mode == "blurry":
            self.finish_scan(message.summary or f"Found {total_items} blurry photos.")
        elif message.known_people_only:
            self.finish_scan(message.summary or f"Found {total_items} photos for known people.")
        else:
            self.finish_scan(message.summary or f"Found {total_items} photos grouped by face.")

    def handle_scan_error_message(self, message: ScanErrorMessage) -> None:
        self.finish_scan("Scan canceled." if message.canceled else message.message)

    def refresh_elapsed(self, final: bool = False) -> bool:
        if self._scan_start_time is None:
            return False

        elapsed_seconds = max(0, int(time.monotonic() - self._scan_start_time))
        self.state.elapsed_text = f"Elapsed: {self._format_elapsed(elapsed_seconds)}"
        return not final

    def finish_scan(self, status: str) -> None:
        self.state.progress_mode = "determinate"
        self.state.progress_value = 0
        self.state.progress_max = 100
        self._last_progress_percent = -1
        self.state.can_scan = True
        self.state.can_cancel = False
        self.state.status = status
        self.refresh_elapsed(final=True)
        self._scan_start_time = None
        self._cancellation_token = None
        self._update_action_state()

    def current_page_groups(self) -> list[ResultGroup]:
        total_items = len(self._iter_result_items())
        if total_items <= self.results_page_size:
            return self._results
        return self._paginated_results()

    def current_preview_items(self) -> list[PreviewItem]:
        preview_items: list[PreviewItem] = []
        for group in self.current_page_groups():
            for item in group.items:
                preview_items.append(
                    PreviewItem(
                        path=item.path,
                        title=item.title,
                        detail=item.detail,
                        group_title=group.title,
                    )
                )
        return preview_items

    def selection_state_snapshot(self) -> dict[Path, bool]:
        return dict(self._selection_state)

    def set_item_selected(self, path: Path, selected: bool) -> None:
        self._selection_state[path] = selected
        self._update_action_state()

    def selected_item_count(self) -> int:
        valid_paths = {item.path for item in self._iter_result_items()}
        return sum(
            1
            for path, selected in self._selection_state.items()
            if path in valid_paths and selected
        )

    def selected_paths(self) -> list[Path]:
        valid_paths = {item.path for item in self._iter_result_items()}
        return [
            path
            for path in valid_paths
            if self._selection_state.get(path, False)
        ]

    def delete_selected(self) -> DeleteResult:
        selected_paths = self.selected_paths()
        errors: list[str] = []
        deleted_paths: set[Path] = set()

        for path in selected_paths:
            try:
                path.unlink()
                deleted_paths.add(path)
            except FileNotFoundError:
                errors.append(f"Missing: {path}")
            except OSError as exc:
                errors.append(f"{path}: {exc}")

        if deleted_paths:
            updated_results: list[ResultGroup] = []
            for group in self._results:
                items = [item for item in group.items if item.path not in deleted_paths]
                if items:
                    updated_results.append(
                        ResultGroup(
                            title=group.title,
                            items=items,
                            group_type=group.group_type,
                        )
                    )
            self._results = updated_results
            for deleted_path in deleted_paths:
                self._selection_state.pop(deleted_path, None)
            self._results_page_index = min(self._results_page_index, self._max_page_index())
            self._update_results_view_state()

        return DeleteResult(deleted_count=len(deleted_paths), errors=errors)

    def export_selected(self, dest_dir: str, export_type: str) -> ExportResult:
        dest_path = Path(dest_dir)
        errors: list[str] = []
        exported_count = 0

        for source_path in self.selected_paths():
            try:
                dest_file = dest_path / source_path.name
                if dest_file.exists():
                    counter = 1
                    stem = source_path.stem
                    suffix = source_path.suffix
                    while dest_file.exists():
                        dest_file = dest_path / f"{stem}_{counter}{suffix}"
                        counter += 1

                if export_type == "copy":
                    shutil.copy2(source_path, dest_file)
                elif export_type == "symlink":
                    dest_file.symlink_to(source_path.resolve())
                else:
                    os.link(source_path, dest_file)

                exported_count += 1
            except OSError as exc:
                errors.append(f"{source_path.name}: {exc}")

        return ExportResult(exported_count=exported_count, errors=errors)

    def can_export_vibe_debug(self) -> bool:
        return (
            self.state.mode == "vibe"
            and self._latest_vibe_debug_payload is not None
            and self._scan_start_time is None
        )

    def suggest_vibe_debug_filename(self) -> str:
        folder_name = "vibe"
        if self._latest_vibe_folder is not None:
            folder_name = self._latest_vibe_folder.name or folder_name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{folder_name}_vibe_debug_{timestamp}.json"

    def export_vibe_debug(self, output_path: str | Path) -> Path:
        if self._latest_vibe_debug_payload is None:
            raise ValueError("No vibe debug data is available to export.")

        path = Path(output_path).expanduser()
        payload = dict(self._latest_vibe_debug_payload)
        payload["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def export_face_groups(self, dest_dir: str) -> ExportResult:
        base_dir = Path(dest_dir)
        errors: list[str] = []
        exported_count = 0
        groups = self._face_groups or self._results

        for group in groups:
            group_dir = base_dir / self._safe_group_folder_name(group.title)
            try:
                group_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                errors.append(f"{group.title}: {exc}")
                continue

            for item in group.items:
                try:
                    dest_file = self._unique_destination_file(group_dir, item.path.name)
                    dest_file.symlink_to(item.path.resolve())
                    exported_count += 1
                except OSError as exc:
                    errors.append(f"{group.title} / {item.path.name}: {exc}")

        return ExportResult(exported_count=exported_count, errors=errors)

    def show_face_group_by_label(self, label: str) -> None:
        if not self._face_group_labels:
            return
        try:
            index = self._face_group_labels.index(label)
        except ValueError:
            return
        self.show_face_group(index)

    def show_face_group(self, index: int) -> None:
        if not self._face_groups:
            return

        index = max(0, min(index, len(self._face_groups) - 1))
        self._active_face_group_index = index
        self.state.face_group_label = self._face_group_labels[index]
        self._results = [self._face_groups[index]]
        self._results_page_index = 0
        self._initialize_selection_state()
        self._update_results_view_state()

    def show_previous_page(self) -> None:
        if self._results_page_index <= 0:
            return
        self._results_page_index -= 1
        self._update_results_view_state()

    def show_next_page(self) -> None:
        if self._results_page_index >= self._max_page_index():
            return
        self._results_page_index += 1
        self._update_results_view_state()

    def build_unknown_face_prompt(self, cluster: Any) -> UnknownFacePrompt:
        if self._face_database is None:
            raise RuntimeError("Face database is not initialized.")

        known_names = self._face_database.list_people_names()
        nearest_match = self._face_database.find_nearest_embedding(
            cluster.representative.embedding,
        )
        preferred_name: str | None = None

        if nearest_match is not None:
            nearest_person = self._face_database.get_person(nearest_match.person_id)
            if nearest_person is not None:
                preferred_name = nearest_person.name

        suggestion_names = known_names[:]
        if preferred_name and preferred_name in suggestion_names:
            suggestion_names.remove(preferred_name)
            suggestion_names.insert(0, preferred_name)

        if cluster.preview is None:
            cluster.preview = self._unknown_face_preview_renderer.render(
                cluster.representative,
            )

        return UnknownFacePrompt(
            cluster_id=cluster.id,
            face_count=len(cluster.faces),
            representative_path=cluster.representative.path,
            preview_array=cluster.preview,
            suggestion_names=tuple(suggestion_names),
            cluster=cluster,
        )

    def apply_unknown_face_name(self, prompt: UnknownFacePrompt, name: str) -> None:
        if self._face_database is None:
            raise RuntimeError("Face database is not initialized.")

        cleaned_name = name.strip()
        if not cleaned_name:
            return

        person = self._face_database.add_person(cleaned_name)
        self._named_unknown_faces[prompt.cluster.id] = ConfirmedUnknownPerson(
            person_id=person.id,
            name=person.name,
        )

        for face in prompt.cluster.faces:
            self._face_database.add_embedding(person.id, face.embedding)

    def _scan_worker(
        self,
        folder: Path,
        mode: str,
        file_extensions: tuple[str, ...] | None,
        orientation_filter: str | None,
        known_people_only: bool,
        cancellation_token: CancellationToken | None,
    ) -> None:
        try:
            debug_payload: dict[str, Any] | None = None
            if mode == "duplicates":
                detector = self._duplicate_detector_factory()

                def progress_callback(phase: Any, done: int, total: int) -> None:
                    if cancellation_token is not None:
                        cancellation_token.raise_if_canceled()
                    self._queue.put(
                        ScanProgressMessage(
                            mode=mode,
                            phase=phase.value,
                            done=done,
                            total=total,
                            known_people_only=known_people_only,
                        )
                    )

                groups = detector.find_duplicates(
                    folder,
                    progress_callback,
                    file_extensions=file_extensions,
                    orientation_filter=orientation_filter,
                    cancellation_token=cancellation_token,
                )
                results = self._build_duplicate_results(groups)
                summary = f"Found {len(results)} duplicate groups with {sum(len(group.items) for group in results)} photos."
                warning = None
            elif mode == "blurry":
                detector = self._blur_detector_factory()

                def progress_callback(done: int, total: int) -> None:
                    if cancellation_token is not None:
                        cancellation_token.raise_if_canceled()
                    self._queue.put(
                        ScanProgressMessage(
                            mode=mode,
                            phase="scanning",
                            done=done,
                            total=total,
                            known_people_only=known_people_only,
                        )
                    )

                blur_results = detector.scan_folder(
                    folder,
                    progress_callback,
                    file_extensions=file_extensions,
                    orientation_filter=orientation_filter,
                    cancellation_token=cancellation_token,
                )
                results = self._build_blurry_results(blur_results)
                summary = f"Found {sum(len(group.items) for group in results)} blurry photos."
                warning = None
            elif mode == "vibe":
                processor = self._vibe_processor_factory(self._current_vibe_config())

                def progress_callback(phase: str, done: int, total: int | None) -> None:
                    if cancellation_token is not None:
                        cancellation_token.raise_if_canceled()
                    self._queue.put(
                        ScanProgressMessage(
                            mode=mode,
                            phase=phase,
                            done=done,
                            total=total,
                            known_people_only=known_people_only,
                        )
                    )

                vibe_result = processor.scan_folder(
                    folder,
                    file_extensions=file_extensions,
                    orientation_filter=orientation_filter,
                    progress_callback=progress_callback,
                    cancellation_token=cancellation_token,
                )
                results = self._build_vibe_results(vibe_result)
                total_grouped = sum(len(group.items) for group in results if group.group_type != "vibe_ungrouped")
                summary_parts = [
                    f"Found {len(vibe_result.groups)} vibe group(s)",
                ]
                if vibe_result.ungrouped_paths:
                    summary_parts.append(f"{len(vibe_result.ungrouped_paths)} ungrouped photo(s)")
                summary = " with ".join(summary_parts) + f" across {total_grouped + len(vibe_result.ungrouped_paths)} photo(s)."
                warning_lines: list[str] = []
                if vibe_result.errors:
                    warning_lines.append(f"{len(vibe_result.errors)} image(s) could not be analyzed.")
                if vibe_result.used_fallback_embedder:
                    warning_lines.append(
                        "The optional ONNX vibe model is not installed, so grouping used the built-in visual fallback embedder."
                    )
                warning = "\n".join(warning_lines) if warning_lines else None
                debug_payload = self._build_vibe_debug_payload(
                    folder=folder,
                    vibe_result=vibe_result,
                    summary=summary,
                    warning=warning,
                )
            else:
                def progress_callback(done: int, total: int) -> None:
                    if cancellation_token is not None:
                        cancellation_token.raise_if_canceled()
                    self._queue.put(
                        ScanProgressMessage(
                            mode=mode,
                            phase="scanning",
                            done=done,
                            total=total,
                            known_people_only=known_people_only,
                        )
                    )

                timing_collector = _FaceScanTimingCollector()
                init_started = time.perf_counter()
                processor = self._init_face_processor()
                processor.set_timing_collector(timing_collector)
                init_elapsed = time.perf_counter() - init_started
                scan_started = time.perf_counter()

                error_message: str | None = None
                try:
                    face_result = processor.scan_folder(
                        folder,
                        progress_callback,
                        file_extensions=file_extensions,
                        orientation_filter=orientation_filter,
                        known_people_only=known_people_only,
                        cancellation_token=cancellation_token,
                    )
                except Exception as exc:
                    error_message = str(exc)
                    raise
                finally:
                    scan_elapsed = time.perf_counter() - scan_started
                    processor.set_timing_collector(None)
                    self._log_face_scan_timing(
                        folder=folder,
                        known_people_only=known_people_only,
                        collector=timing_collector,
                        total_seconds=init_elapsed + scan_elapsed,
                        init_seconds=init_elapsed,
                        status="error" if error_message else "ok",
                        error=error_message,
                        runtime_info=processor.runtime_info(),
                    )
                if known_people_only:
                    results = self._build_face_results(face_result)
                    summary = f"Found {sum(len(group.items) for group in results)} photos for known people."
                else:
                    results = []
                    self._queue.put(UnknownFacesMessage(face_result=face_result))
                    summary = "Found photos grouped by face."
                warning = None

            self._queue.put(
                ScanResultMessage(
                    mode=mode,
                    results=results,
                    known_people_only=known_people_only,
                    summary=summary,
                    warning=warning,
                    debug_payload=debug_payload,
                )
            )
        except ScanCancelledError as exc:
            self._queue.put(ScanErrorMessage(message=str(exc), canceled=True))
        except Exception as exc:
            self._queue.put(ScanErrorMessage(message=str(exc)))

    def _apply_mode_state(self) -> None:
        self.state.show_face_options = self.state.mode == "faces"
        self.state.show_vibe_options = self.state.mode == "vibe"

    @classmethod
    def _collect_supported_file_types(cls) -> tuple[str, ...]:
        return normalize_extensions(
            sorted(
                set(BlurDetector.supported_extensions)
                | set(FaceProcessor.SUPPORTED_EXTENSIONS)
                | set(DetectorConfig().supported_extensions)
                | set(VibeGroupingProcessor.supported_extensions)
            )
        )

    @staticmethod
    def _supported_file_types_for_mode(mode: str) -> tuple[str, ...]:
        if mode == "duplicates":
            return DetectorConfig().supported_extensions
        if mode == "vibe":
            return VibeGroupingProcessor.supported_extensions
        if mode == "blurry":
            return BlurDetector.supported_extensions
        return FaceProcessor.SUPPORTED_EXTENSIONS

    @staticmethod
    def _scan_target_label(mode: str, known_people_only: bool = False) -> str:
        if mode == "duplicates":
            return "near duplicates"
        if mode == "vibe":
            return "vibe groups"
        if mode == "blurry":
            return "blurry photos"
        if known_people_only:
            return "known faces"
        return "faces"

    @staticmethod
    def _compute_progress_percent(mode: str, phase: str, done: int, total: int) -> int:
        clamped_done = min(max(done, 0), total)
        if mode == "vibe":
            ranges = {
                "loading_visual_features": (0, 44),
                "analyzing_actions_and_scenes": (44, 58),
                "detecting_scene_changes": (58, 70),
                "building_scene_groups": (70, 86),
                "checking_group_coherence": (86, 92),
                "choosing_previews": (92, 97),
                "finalizing_results": (97, 100),
            }
            start, end = ranges.get(phase, (0, 100))
            return start + int((clamped_done / total) * max(end - start, 1))
        if mode != "duplicates":
            return int((clamped_done / total) * 100)
        if phase == "indexing":
            return int((clamped_done / total) * 50)
        if phase == "matching":
            return 50 + int((clamped_done / total) * 50)
        return int((clamped_done / total) * 100)

    def _build_duplicate_results(self, groups: list[DuplicateGroup]) -> list[ResultGroup]:
        results: list[ResultGroup] = []
        for index, group in enumerate(groups, start=1):
            items: list[ResultItem] = []
            best = group.best
            for photo in group.photos:
                items.append(
                    ResultItem(
                        path=photo.path,
                        title=photo.filename,
                        detail=f"{photo.width}x{photo.height} • {photo.file_size // 1024} KB",
                        recommended_delete=photo.path != best.path,
                    )
                )
            results.append(
                ResultGroup(
                    title=f"Duplicate group {index} ({len(items)} photos)",
                    items=items,
                )
            )
        return results

    def _build_blurry_results(self, blur_results: Any) -> list[ResultGroup]:
        items = [
            ResultItem(
                path=result.path,
                title=result.path.name,
                detail=f"Score {result.result.final_score:.3f} • {result.result.status}",
                recommended_delete=True,
            )
            for result in sorted(
                blur_results,
                key=lambda item: (item.result.final_score, item.path.name),
            )
        ]
        if not items:
            return []
        return [ResultGroup(title=f"Blurry photos ({len(items)})", items=items)]

    def _build_vibe_results(self, vibe_result: VibeGroupingResult) -> list[ResultGroup]:
        results: list[ResultGroup] = []
        for index, group in enumerate(vibe_result.groups, start=1):
            ordered_items: list[ResultItem] = []
            features_for_group = list(group.image_paths)
            for position, path_text in enumerate(features_for_group, start=1):
                path = Path(path_text)
                ordered_items.append(
                    ResultItem(
                        path=path,
                        title=path.name,
                        detail=self._format_vibe_item_detail(path, group),
                        recommended_delete=False,
                    )
                )

            metadata_lines = [
                f"{len(group.image_paths)} photo(s)",
            ]
            time_range = self._format_vibe_time_range(group.start_timestamp, group.end_timestamp)
            if time_range:
                metadata_lines.append(time_range)
            if group.recognized_person_names:
                metadata_lines.append(
                    "People: " + ", ".join(group.recognized_person_names[:3]) + (
                        "" if len(group.recognized_person_names) <= 3 else f" +{len(group.recognized_person_names) - 3}"
                    )
                )
            if group.metadata.get("duplicate_subgroup_count"):
                metadata_lines.append(
                    f"Near-duplicate sets: {group.metadata['duplicate_subgroup_count']}"
                )

            title = group.label or f"Vibe group {index}"
            results.append(
                ResultGroup(
                    title=title,
                    items=ordered_items,
                    group_type="vibe",
                    representative_path=Path(group.representative_path),
                    subtitle=f"Group {index}",
                    metadata_lines=tuple(metadata_lines),
                    cohesion_text=f"Cohesion {group.cohesion_score:.2f}",
                )
            )

        if vibe_result.ungrouped_paths:
            ungrouped_items = [
                ResultItem(
                    path=Path(path_text),
                    title=Path(path_text).name,
                    detail="Ungrouped",
                    recommended_delete=False,
                )
                for path_text in vibe_result.ungrouped_paths
            ]
            results.append(
                ResultGroup(
                    title=f"Ungrouped ({len(ungrouped_items)} photo(s))",
                    items=ungrouped_items,
                    group_type="vibe_ungrouped",
                    subtitle="Images that did not fit a confident vibe cluster",
                    metadata_lines=("Review or keep these as standalone images.",),
                )
            )
        return results

    def _build_face_results(self, face_result: Any) -> list[ResultGroup]:
        grouped_people: OrderedDict[int, tuple[str, OrderedDict[Path, ResultItem]]] = OrderedDict()
        unknown_results: list[ResultGroup] = []

        def add_person_item(person_id: int, name: str, path: Path) -> None:
            if person_id not in grouped_people:
                grouped_people[person_id] = (name, OrderedDict())
            _, items_by_path = grouped_people[person_id]
            if path not in items_by_path:
                items_by_path[path] = ResultItem(
                    path=path,
                    title=path.name,
                    detail=f"Known: {name}",
                    person_id=person_id,
                )

        for person in face_result.known_people:
            for photo in person.photos:
                add_person_item(person.person_id, person.name, photo)

        for cluster in face_result.unknown_clusters:
            confirmed_person = self._named_unknown_faces.get(cluster.id)
            if confirmed_person is None:
                items = [
                    ResultItem(
                        path=face.path,
                        title=face.path.name,
                        detail=f"Unknown person {cluster.id}",
                        person_id=cluster.id,
                    )
                    for face in cluster.faces
                ]
                unknown_results.append(
                    ResultGroup(
                        title=f"Unknown person {cluster.id} ({len(items)} photos)",
                        items=items,
                        group_type="face",
                    )
                )
                continue

            for face in cluster.faces:
                add_person_item(
                    confirmed_person.person_id,
                    confirmed_person.name,
                    face.path,
                )

        results: list[ResultGroup] = []
        for _person_id, (name, items_by_path) in grouped_people.items():
            items = list(items_by_path.values())
            results.append(
                ResultGroup(
                    title=f"{name} ({len(items)} photos)",
                    items=items,
                    group_type="face",
                )
            )
        results.extend(unknown_results)
        return results

    def _init_face_processor(self) -> _TimedFaceProcessor:
        if self._face_processor is None:
            from face_analyzer.default_face_analyzer import DefaultFaceAnalyzer
            from face_detector.arc_embedder import ArcFaceEmbedder
            from face_detector.scrfd_face_detector import SCRFDFaceDetector

            detector_model = model_path("scrfd_10g_bnkps.onnx")
            embedder_model = model_path("glintr100.onnx")
            eye_state_model = model_path("open_closed_eye.onnx")
            head_pose_model = model_path("sixdrepnet.onnx")
            database_path = app_data_path("face_embeddings.sqlite3")

            detector = SCRFDFaceDetector(model_path=detector_model)
            embedder = ArcFaceEmbedder(model_path=embedder_model, aligner=FaceAligner())
            analyzer = DefaultFaceAnalyzer(
                eye_state_model_path=eye_state_model,
                head_pose_model_path=head_pose_model,
            )
            similarity = CosineEmbeddingSimilarity()
            self._face_database = SQLiteFaceDatabase(database_path, similarity)
            recognizer = DefaultFaceRecognizer(self._face_database, similarity.default_threshold)
            clusterer = ConnectedComponentFaceClusterer(
                similarity,
                strong_threshold=0.63,
                weak_threshold=0.55,
            )
            self._face_processor = _TimedFaceProcessor(
                detector=detector,
                embedder=embedder,
                analyzer=analyzer,
                recognizer=recognizer,
                clusterer=clusterer,
                database=self._face_database,
                worker_factory=lambda: SCRFDFaceDetector(model_path=detector_model),
                max_workers=4,
                embed_batch_size=32,
            )
        return self._face_processor

    def _log_face_scan_timing(
        self,
        *,
        folder: Path,
        known_people_only: bool,
        collector: _FaceScanTimingCollector,
        total_seconds: float,
        init_seconds: float,
        status: str,
        error: str | None,
        runtime_info: dict[str, Any],
    ) -> None:
        payload = {
            "event": "face_scan_timing",
            "folder": str(folder),
            "known_people_only": known_people_only,
            "status": status,
            "runtime": runtime_info,
            **collector.summary(
                total_seconds=total_seconds,
                init_seconds=init_seconds,
            ),
        }
        if error is not None:
            payload["error"] = error
        _get_face_scan_timing_logger().info(json.dumps(payload, sort_keys=True))

    def _initialize_selection_state(self) -> None:
        valid_paths: set[Path] = set()
        for group in self._results:
            for item in group.items:
                valid_paths.add(item.path)
                if item.path not in self._selection_state:
                    self._selection_state[item.path] = (
                        True if group.group_type == "face" else item.recommended_delete
                    )

        stale_paths = [path for path in self._selection_state if path not in valid_paths]
        for path in stale_paths:
            self._selection_state.pop(path, None)

    def _iter_result_items(self) -> list[ResultItem]:
        items: list[ResultItem] = []
        for group in self._results:
            items.extend(group.items)
        return items

    def _page_count(self) -> int:
        total_items = len(self._iter_result_items())
        if total_items == 0:
            return 0
        return (total_items + self.results_page_size - 1) // self.results_page_size

    def _max_page_index(self) -> int:
        return max(0, self._page_count() - 1)

    def _paginated_results(self) -> list[ResultGroup]:
        start = self._results_page_index * self.results_page_size
        end = start + self.results_page_size
        consumed = 0
        page_groups: list[ResultGroup] = []

        for group in self._results:
            group_count = len(group.items)
            group_start = max(0, start - consumed)
            group_end = min(group_count, end - consumed)

            if group_start < group_end:
                title = group.title
                if group_start > 0 or group_end < group_count:
                    title = f"{group.title} ({group_start + 1}-{group_end} of {group_count})"
                page_groups.append(
                    ResultGroup(
                        title=title,
                        items=group.items[group_start:group_end],
                        group_type=group.group_type,
                        representative_path=group.representative_path,
                        subtitle=group.subtitle,
                        metadata_lines=group.metadata_lines,
                        cohesion_text=group.cohesion_text,
                    )
                )

            consumed += group_count
            if consumed >= end:
                break

        return page_groups

    def _configure_face_groups(self, groups: list[ResultGroup]) -> None:
        self._face_groups = groups
        self._face_group_labels = [
            f"{index + 1}. {group.title}"
            for index, group in enumerate(groups)
        ]
        self.state.face_group_labels = tuple(self._face_group_labels)
        self.state.show_face_selector = True

    def _clear_face_groups(self) -> None:
        self._face_groups = []
        self._face_group_labels = []
        self._active_face_group_index = None
        self.state.face_group_label = ""
        self.state.face_group_labels = ()
        self.state.show_face_selector = False

    def _update_results_view_state(self) -> None:
        self._update_pagination_state()
        self._update_action_state()

    def _update_pagination_state(self) -> None:
        page_count = self._page_count()
        if page_count <= 1:
            self.state.page_label = ""
            self.state.show_pagination = False
            self.state.can_show_previous_page = False
            self.state.can_show_next_page = False
            return

        self._results_page_index = max(0, min(self._results_page_index, page_count - 1))
        start = self._results_page_index * self.results_page_size + 1
        end = min(
            (self._results_page_index + 1) * self.results_page_size,
            len(self._iter_result_items()),
        )
        self.state.page_label = (
            f"Showing {start}-{end} of {len(self._iter_result_items())} photos "
            f"(page {self._results_page_index + 1}/{page_count})"
        )
        self.state.show_pagination = True
        self.state.can_show_previous_page = self._results_page_index > 0
        self.state.can_show_next_page = self._results_page_index < page_count - 1

    def _update_action_state(self) -> None:
        selected = self.selected_item_count()
        total = len(self._iter_result_items())
        shown = len(self.current_preview_items())

        if shown < total:
            self.state.count_text = f"{selected} selected / {total} total ({shown} shown)"
        else:
            self.state.count_text = f"{selected} selected / {total} total"

        delete_enabled = bool(selected and total and self.state.mode != "faces")
        export_enabled = bool(selected and total)
        self.state.can_delete = delete_enabled
        self.state.can_export = export_enabled
        self.state.can_export_vibe_debug = self.can_export_vibe_debug()

    def _reset_results(self) -> None:
        self._results = []
        self._selection_state.clear()
        self._clear_face_groups()
        self._named_unknown_faces.clear()
        self._latest_face_result = None
        self._latest_vibe_debug_payload = None
        self._latest_vibe_folder = None
        self._results_page_index = 0
        self.state.page_label = ""
        self.state.show_pagination = False
        self.state.can_show_previous_page = False
        self.state.can_show_next_page = False
        self._update_action_state()

    def _build_vibe_debug_payload(
        self,
        *,
        folder: Path,
        vibe_result: VibeGroupingResult,
        summary: str,
        warning: str | None,
    ) -> dict[str, Any]:
        return {
            "source": "gui_app",
            "input_folder": str(folder.resolve()),
            "summary": summary,
            "warning": warning,
            "provider": vibe_result.provider,
            "used_fallback_embedder": vibe_result.used_fallback_embedder,
            "cache_hits": vibe_result.cache_hits,
            "cache_misses": vibe_result.cache_misses,
            "model_fingerprint": vibe_result.model_fingerprint,
            "config_snapshot": vibe_result.config_snapshot,
            "stage_timings": vibe_result.stage_timings,
            "groups": [
                {
                    "group_id": group.group_id,
                    "label": group.label,
                    "representative_path": group.representative_path,
                    "image_paths": list(group.image_paths),
                    "recognized_person_ids": list(group.recognized_person_ids),
                    "recognized_person_names": list(group.recognized_person_names),
                    "start_timestamp": group.start_timestamp,
                    "end_timestamp": group.end_timestamp,
                    "cohesion_score": group.cohesion_score,
                    "metadata": group.metadata,
                }
                for group in vibe_result.groups
            ],
            "ungrouped_paths": list(vibe_result.ungrouped_paths),
            "errors": [
                {
                    "path": error.path,
                    "message": error.message,
                    "fatal": error.fatal,
                }
                for error in vibe_result.errors
            ],
            "diagnostics": vibe_result.diagnostics,
        }

    @staticmethod
    def _safe_group_folder_name(title: str) -> str:
        if title.endswith(" photos)") and " (" in title:
            title = title.rsplit(" (", 1)[0]

        cleaned = "".join(
            character if character not in '<>:"/\\|?*' else "_"
            for character in title.strip()
        ).strip(" .")
        return cleaned or "Unnamed"

    @staticmethod
    def _unique_destination_file(folder: Path, filename: str) -> Path:
        destination = folder / filename
        if not destination.exists() and not destination.is_symlink():
            return destination

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        counter = 1
        while True:
            candidate = folder / f"{stem}_{counter}{suffix}"
            if not candidate.exists() and not candidate.is_symlink():
                return candidate
            counter += 1

    def _current_vibe_config(self) -> VibeGroupingConfig:
        preset_name = (self.state.vibe_preset or "Balanced Scenes").strip().lower()
        preset = VibeGroupingPreset.BALANCED_SCENES
        if preset_name == "session":
            preset = VibeGroupingPreset.SESSION
        elif preset_name == "tight scenes":
            preset = VibeGroupingPreset.TIGHT_SCENES

        overrides: dict[str, object] = {
            "include_people": self.state.vibe_include_people,
            "include_color": self.state.vibe_include_color,
            "include_composition": self.state.vibe_include_composition,
        }
        if self.state.vibe_show_advanced:
            overrides.update(
                {
                    "session_gap_seconds": int(float(self.state.vibe_session_gap_minutes) * 60),
                    "minimum_pair_similarity": float(self.state.vibe_minimum_similarity),
                    "minimum_group_cohesion": float(self.state.vibe_minimum_cohesion),
                    "maximum_group_size": int(self.state.vibe_maximum_group_size),
                    "batch_size": int(self.state.vibe_batch_size),
                }
            )
        return preset_config(preset, **overrides)

    @staticmethod
    def _format_vibe_item_detail(path: Path, group: Any) -> str:
        parts = ["Vibe group member"]
        if group.metadata.get("duplicate_subgroup_count"):
            parts.append("Near-duplicate subgroups available")
        parts.append(path.suffix.lower().lstrip(".").upper() or "Image")
        return " • ".join(parts)

    @staticmethod
    def _format_vibe_time_range(
        start_timestamp: float | None,
        end_timestamp: float | None,
    ) -> str | None:
        if start_timestamp is None and end_timestamp is None:
            return None
        if start_timestamp is None:
            return f"Ends {PhotoCleanerViewModel._format_timestamp(end_timestamp)}"
        if end_timestamp is None or abs(end_timestamp - start_timestamp) < 1.0:
            return PhotoCleanerViewModel._format_timestamp(start_timestamp)
        start_text = PhotoCleanerViewModel._format_timestamp(start_timestamp)
        end_text = PhotoCleanerViewModel._format_timestamp(end_timestamp)
        return f"{start_text} -> {end_text}"

    @staticmethod
    def _format_timestamp(timestamp: float | None) -> str:
        if timestamp is None:
            return "Unknown time"
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def _format_elapsed(seconds: int) -> str:
        minutes, remainder = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{remainder:02d}"
        return f"{minutes:02d}:{remainder:02d}"
