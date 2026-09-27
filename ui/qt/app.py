from __future__ import annotations

import os
import sys

from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

from app_paths import qml_path

from .bridge import QtPhotoCleanerBridge


_VALID_START_PAGES = {"loading", "auth", "home", "gallery"}


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
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv)

    bridge = QtPhotoCleanerBridge(parent=application)

    engine = QQmlApplicationEngine()
    engine.setInitialProperties(
        {
            "appBridge": bridge,
            "startPage": _start_page_from_environment(),
            "useHomePreviewData": _home_preview_data_from_environment(),
        }
    )

    app_qml_path = qml_path("App.qml")
    engine.load(app_qml_path.as_uri())

    if not engine.rootObjects():
        raise RuntimeError(f"Failed to load QML root object from {app_qml_path}")

    return application.exec()
