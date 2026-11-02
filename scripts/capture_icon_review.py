"""Capture Home and Library layouts; run as `python -m scripts.capture_icon_review`."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_paths import qml_path, qml_root
from ui.qt.app import configure_quick_controls_style


def main() -> None:
    configure_quick_controls_style()
    app = QApplication([])
    output = Path(__file__).resolve().parents[1] / "build" / "icon_review"
    output.mkdir(parents=True, exist_ok=True)
    for page, width, height in (("home", 1440, 900), ("home", 960, 640),
                                ("library", 1440, 900), ("library", 960, 640)):
        engine = QQmlApplicationEngine()
        engine.addImportPath(str(qml_root()))
        engine.setInitialProperties({"startPage": page, "useHomePreviewData": True})
        engine.load(QUrl.fromLocalFile(str(qml_path("App.qml"))))
        if not engine.rootObjects():
            raise RuntimeError("Room36 QML root did not load")
        window = engine.rootObjects()[0]
        window.setWidth(width)
        window.setHeight(height)
        QTest.qWait(250)
        app.processEvents()
        image = window.grabWindow()
        path = output / f"{page}_{width}x{height}.png"
        if image.isNull() or not image.save(str(path)):
            raise RuntimeError(f"Could not capture {path}")
        print(path)
        window.close()
        engine.deleteLater()


if __name__ == "__main__":
    main()
