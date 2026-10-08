import tempfile
from pathlib import Path
import threading
import unittest

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from app_paths import qml_path
from automatic_scan.state_store import AutomaticStateStore
from ui.phase6_test_support import FixtureRunner
from ui.qt.app import configure_quick_controls_style
from ui.qt.inspector_controller import PhotoInspectorController
from ui.qt.photo_model import PhotoListModel
from ui.qt.previews import PreviewService
from ui.qt.processing_controller import ProjectProcessingController
from ui.qt.project_controller import ProjectController
from ui.qt.thumbnails import ThumbnailService
from ui import test_qml_project_flow as phase5
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until


class BlurFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "photos"
        self.folder.mkdir()
        for name in ("a.jpg", "b.jpg", "c.png"):
            Image.new("RGB", (1200, 600), "gray").save(self.folder / name)
        self.choice = str(self.folder)
        self.gate = threading.Event()
        self.runner = FixtureRunner(self.root, AutomaticStateStore(self.root / "state"))
        self.thumbnails = ThumbnailService()
        self.photos = PhotoListModel(self.thumbnails)
        self.processing = ProjectProcessingController(self.thumbnails,
            cache_path=self.root / "analysis.sqlite3", state_root=self.root / "state", runner=self.runner)
        self.previews = PreviewService()
        self.inspector = PhotoInspectorController(self.processing.blurModel, self.previews)
        self.processing.sessionChanged.connect(self.inspector.close)
        self.controller = ProjectController(self.photos, registry_path=self.root / "projects.json",
            chooser=lambda: self.choice, processing=self.processing)
        self.engine = QQmlApplicationEngine()
        self.engine.setInitialProperties(dict(projectController=self.controller, photoModel=self.photos,
            processingController=self.processing, inspectorController=self.inspector))
        self.engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(self.engine.rootObjects())
        self.window = self.engine.rootObjects()[0]
        self.app.processEvents()

    def tearDown(self):
        self.gate.set()
        self.window.close()
        self.engine.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.controller.shutdown()
        self.inspector.close()
        self.processing.shutdown()
        self.photos.close()
        self.previews.shutdown()
        self.thumbnails.shutdown()
        self.assertFalse(self.processing._worker.is_alive())
        self.temp.cleanup()

    items = phase5.ProjectFlowTests.items
    item = phase5.ProjectFlowTests.item
    click = phase5.ProjectFlowTests.click
    create = phase5.ProjectFlowTests.create
    home = phase5.ProjectFlowTests.home

    def blur(self):
        self.click("libraryNav_blurry")
        wait_until(lambda: self.processing.stage("blur")["state"] == "COMPLETE", timeout=6)
        wait_until(lambda: bool(self.items("gridPhotoCard")))

    def test_pointer_review_full_flow_and_keyboard(self):
        self.create()
        self.assertTrue(self.processing.overallState in ("QUEUED", "RUNNING", "COMPLETE"))
        self.blur()
        self.click("gridPhotoCard")
        self.assertTrue(self.inspector.opened)
        wait_until(lambda: self.inspector.previewState == "ready")
        identity = self.inspector.photo["photoId"]
        for control, state in (("inspectorKeep", "KEEP"), ("inspectorReview", "REVIEW"), ("inspectorRemove", "REMOVE")):
            self.click(control)
            self.assertEqual(state, self.inspector.photo["reviewState"])
            self.assertTrue(Path(identity).is_file())
        self.click("inspectorNext")
        self.assertNotEqual(identity, self.inspector.photo["photoId"])
        self.click("inspectorPrevious")
        self.assertEqual(identity, self.inspector.photo["photoId"])
        QTest.keyClick(self.window, Qt.Key.Key_Right)
        self.assertNotEqual(identity, self.inspector.photo["photoId"])
        QTest.keyClick(self.window, Qt.Key.Key_Left)
        self.assertEqual(identity, self.inspector.photo["photoId"])
        self.click("inspectorBack")
        self.assertFalse(self.inspector.opened)
        self.assertEqual("REMOVE", self.processing.blurModel.photo(identity)["reviewState"])
        self.assertTrue(any("Remove" in card.property("badgeText") for card in self.items("gridPhotoCard")))
        self.click("gridPhotoCard")
        QTest.keyClick(self.window, Qt.Key.Key_Escape)
        self.assertFalse(self.inspector.opened)
        self.assertEqual("blurry", self.item("libraryShell").property("currentRoute"))

    def test_home_navigation_processing_continues_and_reopen_reuses_cache(self):
        self.runner.gate = self.gate
        self.create()
        self.blur()
        self.assertEqual("RUNNING", self.processing.overallState)
        session = self.processing._session
        self.home()
        self.assertEqual(session, self.processing._session)
        self.gate.set()
        wait_until(lambda: self.processing.overallState == "COMPLETE", timeout=6)
        self.click("projectCard_0")
        wait_until(lambda: self.window.property("currentPage") == "library" and not self.controller.busy)
        wait_until(lambda: self.processing.overallState == "COMPLETE", timeout=6)
        self.assertEqual(3, self.processing.diagnostics["blur"]["cache_hits"])
        self.blur()
        self.assertEqual(3, self.item("photoGrid").property("count"))

    def test_project_switch_during_processing_cannot_show_old_candidates(self):
        self.runner.gate = self.gate
        self.create()
        self.blur()
        old = set(self.processing.blurModel.results)
        self.home()
        second = self.root / "second"
        second.mkdir()
        Image.new("RGB", (80, 40), "gray").save(second / "other.jpg")
        self.choice = str(second)
        self.create()
        self.click("libraryNav_blurry")
        self.assertEqual(0, self.processing.blurModel.rowCount())
        self.gate.set()
        wait_until(lambda: self.processing.overallState == "COMPLETE", timeout=6)
        self.assertEqual(1, self.processing.blurModel.rowCount())
        self.assertTrue(old.isdisjoint(self.processing.blurModel.results))

    def test_inspector_responsive_controls_and_aspect_ratio(self):
        self.create()
        self.blur()
        self.click("gridPhotoCard")
        wait_until(lambda: self.inspector.previewState == "ready")
        for width, height in ((1280, 720), (1440, 900), (1920, 1080)):
            self.window.setWidth(width)
            self.window.setHeight(height)
            QTest.qWait(80)
            for name in ("inspectorBack", "inspectorKeep", "inspectorReview", "inspectorRemove", "inspectorPrevious", "inspectorNext"):
                control = self.item(name)
                start = control.mapToScene(QPointF(0, 0))
                end = control.mapToScene(QPointF(control.width(), control.height()))
                self.assertGreaterEqual(start.x(), 0, (width, height, name))
                self.assertGreaterEqual(start.y(), 0, (width, height, name))
                self.assertLessEqual(end.x(), width, (width, height, name))
                self.assertLessEqual(end.y(), height, (width, height, name))
            image = self.item("inspectorImage")
            self.assertAlmostEqual(2, image.property("paintedWidth") / image.property("paintedHeight"), places=2)

    def test_processing_and_no_candidates_empty_states(self):
        self.create()
        self.blur()
        self.processing.blurModel.setFilter("Review")
        self.assertEqual(0, self.item("photoGrid").property("count"))
        self.assertEqual("No blurry photos need review.", self.item("photoGrid").property("emptyText"))
        self.assertEqual("COMPLETE", self.item("blurryPhotosPage").property("blurStage")["state"])
        self.processing.blurModel.setSort("Least problematic first")
        self.click("libraryNav_library")
        self.click("libraryNav_blurry")
        self.assertEqual("Review", self.item("blurFilter").property("currentText"))
        self.assertEqual("Least problematic first", self.item("blurSort").property("currentText"))

    def test_ten_thousand_candidates_stay_virtualized(self):
        self.create()
        wait_until(lambda: self.processing.overallState == "COMPLETE", timeout=6)
        from ui.qt.photo_model import photo_identity
        self.processing.blurModel.set_results([dict(photoId=photo_identity(self.folder / f"{i:05}.jpg"), filename=f"{i:05}.jpg",
            final_score=.1, status="Blurry", reviewState="UNDECIDED", file_size=1, mtime_ns=1) for i in range(10000)])
        self.click("libraryNav_blurry")
        grid = self.item("photoGrid")
        counts = []
        for view, offset in (("Small", 0), ("Large", 500000), ("Medium", 1000000), ("Small", 0)):
            grid.setProperty("viewSize", view)
            grid.setProperty("contentY", min(offset, grid.property("contentHeight") - grid.height()))
            QTest.qWait(50)
            counts.append(len(self.items("photoGridDelegate")))
            self.assertLess(counts[-1], 100)
            self.assertLess(self.thumbnails.pending_count, 256)
        print(f"\nPhase 6 grid diagnostic: 10000 candidates; live delegates {counts}, max {max(counts)}; thumbnail workers 2, pending {self.thumbnails.pending_count}/256")

    def test_pointer_bulk_review_keeps_selection_visible_after_sort(self):
        self.create()
        self.blur()
        self.click("photoSelectionCheckbox")
        self.assertFalse(self.inspector.opened)
        self.assertEqual(1, self.processing.blurModel.selectedCount)
        identity = next(iter(self.processing.blurModel._selection))
        self.click("bulkReview")
        self.assertEqual("REVIEW", self.processing.blurModel.photo(identity)["reviewState"])
        self.processing.blurModel.setSort("Filename Z–A")
        QTest.qWait(80)
        self.assertEqual(1, sum(bool(checkbox.property("checked")) for checkbox in self.items("photoSelectionCheckbox")))
        self.assertTrue(all(Path(key).is_file() for key in self.processing.blurModel.results))
