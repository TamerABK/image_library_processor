from __future__ import annotations

import os
import sys
from functools import cache

from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication

from app_paths import qml_path

from .bridge import QtPhotoCleanerBridge


_VALID_START_PAGES = {"loading", "auth", "home", "gallery"}


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
