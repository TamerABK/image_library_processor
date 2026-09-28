import base64
from io import BytesIO
import os
from pathlib import Path
import time
import unittest

from PIL import Image
from PySide6.QtCore import QObject, QEvent, QtMsgType, qInstallMessageHandler
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem
from PySide6.QtWidgets import QApplication

from app_paths import qml_path
from ui.models import ResultItem
from ui.qt.app import configure_quick_controls_style
from ui.qt.photo_model import PhotoListModel
from ui.qt.thumbnails import ThumbnailService
from ui.test_qt_thumbnails import wait_until

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class PhotoGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])
        cls.sources = []
        for color in ("red", "green", "blue"):
            output = BytesIO()
            Image.new("RGB", (8, 4), color).save(output, format="PNG")
            cls.sources.append("data:image/png;base64," + base64.b64encode(output.getvalue()).decode())

    def setUp(self):
        self.warnings = []
        def capture(kind, context, message):
            if kind in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
                self.warnings.append(message)
        self.previous_handler = qInstallMessageHandler(capture)
        self.service = ThumbnailService(decoder=self.source_for, cache_entries=16, cache_bytes=16384)
        self.model = PhotoListModel(self.service)
        start = time.perf_counter()
        self.model.replace_items(ResultItem(Path(f"synthetic/{i:05}.jpg"), "", "") for i in range(6000))
        self.population_ms = (time.perf_counter() - start) * 1000
        self.engine = QQmlApplicationEngine()
        self.engine.warnings.connect(lambda errors: self.warnings.extend(error.toString() for error in errors))
        self.engine.setInitialProperties({"startPage": "library", "photoModel": self.model})
        self.engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(self.engine.rootObjects(), self.warnings)
        self.window = self.engine.rootObjects()[0]
        wait_until(lambda: bool(self.delegates()))

    def tearDown(self):
        self.window.close()
        self.engine.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.model.close()
        self.service.shutdown()
        qInstallMessageHandler(self.previous_handler)
        self.assertEqual([], self.warnings)

    def source_for(self, path, edge=384):
        return self.sources[int(Path(path).stem) % len(self.sources)]

    def items(self, name):
        found = []
        def visit(item):
            if item.objectName() == name:
                found.append(item)
            for child in item.childItems():
                visit(child)
        visit(self.window.contentItem())
        return found

    def delegates(self):
        return self.items("photoGridDelegate")

    def grid(self):
        return self.items("photoGrid")[0]

    def test_large_grid_virtualization_selection_and_identity(self):
        grid = self.grid()
        first = self.delegates()[0]
        identity = first.property("photoId")
        first.findChild(QObject, "gridPhotoCard").clicked.emit()
        self.assertTrue(first.property("selected"))
        wait_until(lambda: first.property("thumbnailState") == "ready")
        self.assertEqual(self.source_for(identity), first.property("thumbnailSource"))
        initial_ids = {item.property("photoId") for item in self.delegates()}
        counts = [len(initial_ids)]
        for fraction in (0.3, 0.7, 1.0):
            grid.setProperty("contentY", max(0, grid.property("contentHeight") - grid.height()) * fraction)
            wait_until(lambda: bool(self.delegates()) and not initial_ids.intersection(
                item.property("photoId") for item in self.delegates()))
            wait_until(lambda: all(item.property("thumbnailState") == "ready" for item in self.delegates()))
            counts.append(len(self.delegates()))
            for delegate in self.delegates():
                self.assertEqual(self.source_for(delegate.property("photoId")), delegate.property("thumbnailSource"))
        grid.setProperty("contentY", 0)
        wait_until(lambda: any(item.property("photoId") == identity for item in self.delegates()))
        returned = next(item for item in self.delegates() if item.property("photoId") == identity)
        self.assertTrue(returned.property("selected"))
        self.assertLess(max(counts), 100)
        self.assertLessEqual(self.service.cache_entries, 16)
        self.assertLessEqual(self.service.cache_bytes, 16384)
        print(f"\nPhase 5 diagnostic: {self.model.rowCount()} rows, {self.population_ms:.1f} ms population, "
              f"delegate samples {counts}, cache {self.service.cache_entries} entries/{self.service.cache_bytes} bytes")

    def test_desktop_sizes_and_view_size_hook(self):
        grid = self.grid()
        for width, height in ((1280, 720), (1440, 900), (1920, 1080)):
            self.window.setWidth(width)
            self.window.setHeight(height)
            self.app.processEvents()
            wait_until(lambda: grid.width() > 0 and bool(self.delegates()))
            self.assertGreater(grid.property("columns"), 0)
            self.assertGreater(grid.property("cellHeight"), 0)
            self.assertLess(len(self.delegates()), 100)
            self.assertLessEqual(grid.property("cellWidth") * grid.property("columns"), grid.width() + 0.01)
        medium = grid.property("columns")
        self.items("libraryViewButton")[0].optionSelected.emit("Large")
        self.assertEqual("Large", grid.property("viewSize"))
        self.assertLess(grid.property("columns"), medium)
        self.items("libraryViewButton")[0].optionSelected.emit("Small")
        self.assertGreater(grid.property("columns"), medium)

    def test_other_routes_release_thumbnails_and_remain_placeholders(self):
        shell = self.items("libraryShell")[0]
        shell.setProperty("currentRoute", "blurry")
        self.app.processEvents()
        self.assertEqual([], self.items("photoGrid"))
        self.assertEqual("blurry", self.items("libraryRoutePlaceholder")[0].property("routeId"))
        self.assertEqual(0, self.service.pending_count)
        self.assertTrue(all(self.model.data(self.model.index(i), self.model.ROLE["thumbnailSource"]) == ""
                            for i in range(self.model.rowCount())))

    def test_sort_refresh_and_model_replacement_keep_delegate_contract(self):
        self.model.sortByFilename(True)
        wait_until(lambda: all(item.property("thumbnailState") == "ready" for item in self.delegates()))
        for item in self.delegates():
            self.assertEqual(self.source_for(item.property("photoId")), item.property("thumbnailSource"))
        self.model.replace_items([ResultItem(Path("synthetic/00003.jpg"), "", "")])
        wait_until(lambda: len(self.delegates()) == 1 and self.delegates()[0].property("thumbnailState") == "ready")
        self.assertEqual(self.sources[0], self.delegates()[0].property("thumbnailSource"))
        self.window.setProperty("photoModel", None)
        self.app.processEvents()
        self.assertEqual(0, self.grid().property("count"))
        self.assertEqual(0, self.service.pending_count)
