from __future__ import annotations

import os
import json
from pathlib import Path
import statistics
import sys
import time
from functools import cache

from PySide6.QtCore import Qt, QTimer
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication

from app_paths import qml_path

from .bridge import QtPhotoCleanerBridge
from .photo_model import PhotoListModel
from .thumbnails import ThumbnailService
from .project_controller import ProjectController
from .processing_controller import ProjectProcessingController
from .inspector_controller import PhotoInspectorController
from .previews import PreviewService
from .face_previews import FacePreviewService
from .people_controller import PeopleController
from .duplicate_controller import DuplicateReviewController
from .favorites_controller import FavoritesController
from .trash_controller import TrashController
from .export_controller import ExportController


_VALID_START_PAGES = {"loading", "auth", "home", "library", "gallery"}


@cache
def configure_quick_controls_style() -> None:
    """Select a customizable default once, before any Controls QML is loaded.

    Explicit QT_QUICK_CONTROLS_STYLE overrides remain available (e.g. Fusion).
    Native platform styles do not support Room 36's customized controls.
    """
    QQuickStyle.setStyle(os.environ.get("QT_QUICK_CONTROLS_STYLE", "").strip() or "Basic")


def _start_page_from_environment() -> str:
    requested_page = os.environ.get("ROOM36_START_PAGE", "home").strip().lower()
    if requested_page in _VALID_START_PAGES:
        return requested_page
    return "home"


def _home_preview_data_from_environment() -> bool:
    return os.environ.get("ROOM36_HOME_PREVIEW_DATA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def run_app() -> int:
    configure_quick_controls_style()
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv)

    # Keep controls, dialogs, and window chrome light regardless of the OS theme.
    application.styleHints().setColorScheme(Qt.ColorScheme.Light)

    bridge = QtPhotoCleanerBridge(parent=application)
    thumbnails = ThumbnailService(parent=application)
    photo_model = PhotoListModel(thumbnails, parent=application)
    processing = ProjectProcessingController(thumbnails, parent=application)
    face_previews = FacePreviewService(parent=application)
    people = PeopleController(processing, face_previews, thumbnails, parent=application)
    duplicates = DuplicateReviewController(processing, thumbnails, parent=application)
    previews = PreviewService(parent=application)
    inspector = PhotoInspectorController(processing.blurModel, previews, parent=application)
    processing.sessionChanged.connect(inspector.close)
    projects = ProjectController(photo_model, parent=application, processing=processing)
    favorites = FavoritesController(projects, photo_model, thumbnails, parent=application)
    trash = TrashController(projects, photo_model, processing, duplicates, favorites,
                            inspector, thumbnails, parent=application)
    exports = ExportController(projects, photo_model, people, processing, parent=application)
    from .project_watcher import ProjectFilesystemWatcher
    watcher = ProjectFilesystemWatcher(projects, parent=application)
    application.aboutToQuit.connect(watcher.shutdown)
    application.aboutToQuit.connect(projects.shutdown)
    application.aboutToQuit.connect(inspector.close)
    application.aboutToQuit.connect(processing.shutdown)
    application.aboutToQuit.connect(people.shutdown)
    application.aboutToQuit.connect(duplicates.shutdown)
    application.aboutToQuit.connect(favorites.shutdown)
    application.aboutToQuit.connect(trash.shutdown)
    application.aboutToQuit.connect(exports.shutdown)
    application.aboutToQuit.connect(face_previews.shutdown)
    application.aboutToQuit.connect(previews.shutdown)
    application.aboutToQuit.connect(photo_model.close)
    application.aboutToQuit.connect(thumbnails.shutdown)

    engine = QQmlApplicationEngine()
    engine.setInitialProperties(
        {
            "appBridge": bridge,
            "photoModel": photo_model,
            "projectController": projects,
            "processingController": processing,
            "peopleController": people,
            "duplicateController": duplicates,
            "favoritesController": favorites,
            "trashController": trash,
            "exportController": exports,
            "inspectorController": inspector,
            "startPage": _start_page_from_environment(),
            "useHomePreviewData": _home_preview_data_from_environment(),
        }
    )

    app_qml_path = qml_path("App.qml")
    try:
        engine.load(app_qml_path.as_uri())
        if not engine.rootObjects():
            raise RuntimeError(f"Failed to load QML root object from {app_qml_path}")
        # Opt-in benchmark hook: run the actual Qt discovery/processing lifecycle
        # with isolated application data supplied by the benchmark launcher.
        benchmark_folder = os.environ.get("ROOM36_BENCHMARK_PROJECT")
        if benchmark_folder:
            heartbeat_path = os.environ.get("ROOM36_BENCHMARK_UI_METRICS")
            if heartbeat_path:
                intervals_ms = []
                heartbeat = QTimer(application)
                heartbeat.setInterval(50)
                heartbeat.setTimerType(Qt.TimerType.PreciseTimer)
                last_tick = time.perf_counter_ns()

                def sample_ui_delay():
                    nonlocal last_tick
                    now = time.perf_counter_ns()
                    if processing.overallState in {"QUEUED", "RUNNING"} and len(intervals_ms) < 30_000:
                        intervals_ms.append(max(0.0, (now - last_tick) / 1e6 - 50.0))
                    last_tick = now

                def save_ui_delays():
                    if not intervals_ms:
                        payload = {"sample_count": 0, "note": "No processing heartbeat samples"}
                    else:
                        ordered = sorted(intervals_ms)
                        payload = {"sample_count": len(ordered),
                            "median_delay_ms": statistics.median(ordered),
                            "p95_delay_ms": ordered[int(.95 * (len(ordered) - 1))],
                            "p99_delay_ms": ordered[int(.99 * (len(ordered) - 1))],
                            "max_delay_ms": ordered[-1],
                            "over_50ms": sum(value > 50 for value in ordered),
                            "over_100ms": sum(value > 100 for value in ordered),
                            "over_250ms": sum(value > 250 for value in ordered),
                            "timer_interval_ms": 50}
                    output = Path(heartbeat_path)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                heartbeat.timeout.connect(sample_ui_delay)
                heartbeat.start()
                application.aboutToQuit.connect(save_ui_delays)
            def finish_benchmark():
                if projects.busy:
                    return
                state = processing.overallState
                if state in {"COMPLETE", "FAILED", "CANCELLED"}:
                    application.exit(0 if state == "COMPLETE" else 2)
            processing.changed.connect(finish_benchmark)
            QTimer.singleShot(0, lambda: projects.openFolder(benchmark_folder))
            QTimer.singleShot(1_200_000, lambda: application.exit(3))
        return application.exec()
    finally:
        watcher.shutdown()
        projects.shutdown()
        inspector.close()
        processing.shutdown()
        people.shutdown()
        duplicates.shutdown()
        favorites.shutdown()
        trash.shutdown()
        exports.shutdown()
        face_previews.shutdown()
        previews.shutdown()
        photo_model.close()
        thumbnails.shutdown()
