from __future__ import annotations

import sys

from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

from app_paths import qml_path

from .bridge import QtPhotoCleanerBridge


def run_app() -> int:
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv)

    bridge = QtPhotoCleanerBridge(parent=application)

    engine = QQmlApplicationEngine()
    engine.setInitialProperties({"appBridge": bridge})

    app_qml_path = qml_path("App.qml")
    engine.load(app_qml_path.as_uri())

    if not engine.rootObjects():
        raise RuntimeError(f"Failed to load QML root object from {app_qml_path}")

    return application.exec()
