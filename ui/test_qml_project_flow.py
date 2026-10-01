from pathlib import Path
import tempfile
import unittest

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_paths import qml_path
from ui.qt.app import configure_quick_controls_style
from ui.qt.photo_model import PhotoListModel
from ui.qt.project_controller import ProjectController
from ui.qt.thumbnails import ThumbnailService
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until


class ProjectFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "Real photos"
        self.folder.mkdir()
        for name, color in (("b.jpg", "red"), ("a.png", "blue")):
            Image.new("RGB", (32, 16), color).save(self.folder / name)
        (self.folder / "notes.txt").write_text("unsupported", encoding="utf-8")
        self.choice = str(self.folder)
        self.thumbnails = ThumbnailService()
        self.photos = PhotoListModel(self.thumbnails)
        self.controller = ProjectController(self.photos, registry_path=self.root / "registry.json", chooser=lambda: self.choice)
        self.engine = QQmlApplicationEngine()
        self.engine.setInitialProperties({"projectController": self.controller, "photoModel": self.photos})
        self.engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(self.engine.rootObjects())
        self.window = self.engine.rootObjects()[0]
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.engine.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.controller.shutdown()
        self.photos.close()
        self.thumbnails.shutdown()
        self.temp.cleanup()

    def items(self, name):
        matches = []
        def visit(item):
            if item.objectName() == name:
                matches.append(item)
            for child in item.childItems():
                visit(child)
        visit(self.window.contentItem())
        return matches

    def item(self, name):
        found = self.items(name)
        self.assertTrue(found, name)
        return found[0]

    def click(self, name):
        # Loader/layout changes are polished on the next rendered frame. A single
        # processEvents() can leave old coordinates after the status band closes.
        QTest.qWait(50)
        item = self.item(name)
        position = item.mapToScene(QPointF(item.width() / 2, item.height() / 2))
        QTest.mouseClick(self.window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         position.toPoint())
        self.app.processEvents()

    def create(self):
        self.click("newProjectCard")
        wait_until(lambda: self.window.property("currentPage") == "library" and not self.controller.busy)

    def home(self):
        self.click("libraryHomeButton")
        self.app.processEvents()
        self.assertEqual("home", self.window.property("currentPage"))

    def test_new_project_real_thumbnails_back_and_reopen(self):
        self.assertEqual("home", self.window.property("currentPage"))
        self.create()
        self.assertEqual(1, self.controller.projects.rowCount())
        self.assertEqual(2, self.photos.rowCount())
        self.assertEqual(self.controller.activeProjectId, self.window.property("activeProjectId"))
        self.assertEqual("Real photos", self.item("libraryShell").property("projectName"))
        self.assertEqual(2, self.item("photoGrid").property("count"))
        for i in range(2):
            source_path = self.photos.data(self.photos.index(i), self.photos.ROLE["sourcePath"])
            self.assertTrue(Path(source_path).is_file())
        wait_until(lambda: all(self.photos.data(self.photos.index(i), self.photos.ROLE["thumbnailState"]) == "ready" for i in range(2)))
        self.assertTrue(self.photos.data(self.photos.index(0), self.photos.ROLE["thumbnailSource"]).startswith("data:image/png;base64,"))
        self.item("libraryViewButton").optionSelected.emit("Small")
        self.assertEqual("Small", self.item("photoGrid").property("viewSize"))
        self.item("librarySortButton").optionSelected.emit("Name Z–A")
        self.assertEqual("b.jpg", self.photos.data(self.photos.index(0), self.photos.ROLE["filename"]))
        identity = self.controller.activeProjectId
        self.home()
        wait_until(lambda: bool(self.items("projectCard_0")))
        self.click("projectCard_0")
        wait_until(lambda: self.window.property("currentPage") == "library" and not self.controller.busy)
        self.assertEqual(identity, self.controller.activeProjectId)
        self.assertEqual(2, self.item("photoGrid").property("count"))

    def test_restart_reopens_persisted_project_from_home(self):
        self.create()
        identity = self.controller.activeProjectId
        self.home()
        self.controller.shutdown()
        self.photos.replace_items([], preserve_selection=False)
        self.controller = ProjectController(self.photos, registry_path=self.root / "registry.json",
                                            chooser=lambda: self.choice)
        self.window.setProperty("projectController", self.controller)
        self.app.processEvents()
        self.assertEqual(1, self.controller.projects.rowCount())
        self.assertEqual(0, self.photos.rowCount())
        self.click("projectCard_0")
        wait_until(lambda: self.window.property("currentPage") == "library" and not self.controller.busy)
        self.assertEqual(identity, self.window.property("activeProjectId"))
        self.assertEqual(2, self.item("photoGrid").property("count"))

    def test_unavailable_actions_ignore_mouse_clicks_and_routes_remain_navigable(self):
        events = []
        self.window.subscriptionRequested.connect(lambda: events.append("subscription"))
        self.window.settingsRequested.connect(lambda: events.append("settings"))
        self.window.profileRequested.connect(lambda: events.append("profile"))
        for name in ("homeViewButton", "homeSubscriptionButton", "homeSettingsButton", "homeProfileButton"):
            self.click(name)
            self.assertEqual("home", self.window.property("currentPage"))
        self.assertEqual([], events)
        self.assertFalse(self.item("homeViewButton").property("open"))
        self.create()
        for route in ("blurry", "duplicates", "knownPeople", "library"):
            self.click("libraryNav_" + route)
            self.assertEqual(route, self.item("libraryShell").property("currentRoute"))
            for name in ("librarySortButton", "libraryViewButton"):
                self.assertEqual(route == "library", self.item(name).property("enabled"))
        self.click("librarySidebarHome")
        self.assertEqual("home", self.window.property("currentPage"))

    def test_home_search_sort_and_disabled_actions(self):
        self.create()
        self.home()
        for name in ("homeViewButton", "homeSubscriptionButton", "homeSettingsButton", "homeProfileButton", "projectMenuButton"):
            self.assertFalse(self.item(name).property("enabled"), name)
        for name in ("newProjectCard", "homeSortButton", "homeSearchField"):
            self.assertTrue(self.item(name).property("enabled"), name)
        self.item("homeSearchField").setProperty("text", "REAL")
        self.assertEqual(1, self.controller.projects.rowCount())
        self.item("homeSearchField").setProperty("text", "absent")
        self.assertEqual(0, self.controller.projects.rowCount())
        self.item("homeSearchField").setProperty("text", "")
        self.assertEqual(1, self.controller.projects.rowCount())
        self.item("homeSortButton").optionSelected.emit("Oldest")
        self.assertEqual("Oldest", self.controller.projects.sortOrder)
        signals = []
        self.window.subscriptionRequested.connect(lambda: signals.append("subscription"))
        self.window.settingsRequested.connect(lambda: signals.append("settings"))
        self.window.profileRequested.connect(lambda: signals.append("profile"))
        # Disabled controls cannot be clicked by the user, but their contracts
        # are forwarded correctly for future implementations.
        self.item("homeSubscriptionButton").clicked.emit()
        self.item("homeSettingsButton").clicked.emit()
        self.item("homeProfileButton").clicked.emit()
        self.assertEqual(["subscription", "settings", "profile"], signals)

    def test_cancel_empty_and_missing_folder_error(self):
        self.choice = ""
        self.item("newProjectCard").clicked.emit()
        self.assertEqual("home", self.window.property("currentPage"))
        self.assertEqual(0, self.controller.projects.rowCount())
        self.choice = str(self.root / "missing")
        self.item("newProjectCard").clicked.emit()
        wait_until(lambda: not self.controller.busy)
        self.assertEqual("home", self.window.property("currentPage"))
        self.assertTrue(self.item("projectStatusBanner").property("visible"))
        self.assertIn("missing", self.item("projectStatusText").property("text"))
        empty = self.root / "Empty"
        empty.mkdir()
        self.choice = str(empty)
        self.create()
        self.assertEqual(0, self.item("photoGrid").property("count"))

    def test_switch_projects_through_home_resets_selection(self):
        self.create()
        first_id = self.controller.activeProjectId
        identity = self.photos.data(self.photos.index(0), self.photos.ROLE["photoId"])
        self.photos.setSelected(identity, True)
        self.home()
        self.choice = str(self.root)  # Overlapping folder contains the same photos.
        self.create()
        self.assertNotEqual(first_id, self.controller.activeProjectId)
        self.assertFalse(any(self.photos.data(self.photos.index(i), self.photos.ROLE["selected"]) for i in range(self.photos.rowCount())))
        self.assertEqual(2, self.controller.projects.rowCount())
