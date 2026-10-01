import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QAbstractItemModelTester

from ui.qt.photo_model import PhotoListModel
from ui.qt.project_controller import ProjectController, discover_photos
from ui.qt.project_model import Project, ProjectListModel
from ui.qt.thumbnails import ThumbnailService
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "Holiday"
        self.folder.mkdir()
        Image.new("RGB", (24, 12), "red").save(self.folder / "one.jpg")
        self.registry = self.root / "projects.json"
        self.thumbnails = ThumbnailService()
        self.photos = PhotoListModel(self.thumbnails)
        self.addCleanup(self.thumbnails.shutdown)
        self.addCleanup(self.photos.close)
        self.choice = str(self.folder)
        self.controller = self.make_controller()
        self.tester = QAbstractItemModelTester(self.controller.projects,
            QAbstractItemModelTester.FailureReportingMode.Warning)

    def make_controller(self, **kwargs):
        controller = ProjectController(self.photos, registry_path=self.registry,
                                       chooser=lambda: self.choice, **kwargs)
        self.addCleanup(controller.shutdown)
        return controller

    def open_folder(self, folder=None):
        self.controller.openFolder(str(folder or self.folder))
        self.assertTrue(self.controller.busy)
        wait_until(lambda: not self.controller.busy)

    def test_create_cancel_and_supported_file_discovery(self):
        (self.folder / "notes.txt").write_text("unsupported", encoding="utf-8")
        nested = self.folder / "nested"
        nested.mkdir()
        Image.new("RGB", (10, 10)).save(nested / "two.PNG")
        self.controller.newProject()
        wait_until(lambda: not self.controller.busy)
        self.assertEqual(2, self.photos.rowCount())
        self.assertEqual(1, self.controller.projects.rowCount())
        identity = self.controller.activeProjectId
        self.assertTrue(identity)
        self.assertEqual("Holiday", self.controller.activeProjectName)
        self.choice = ""
        self.controller.newProject()
        self.assertFalse(self.controller.busy)
        self.assertEqual(identity, self.controller.activeProjectId)
        self.assertEqual(2, self.photos.rowCount())
        self.assertEqual("", self.controller.message)

    def test_empty_folder_is_a_valid_project(self):
        empty = self.root / "Empty"
        empty.mkdir()
        (empty / "text.txt").write_text("no photos", encoding="utf-8")
        self.open_folder(empty)
        self.assertEqual(0, self.photos.rowCount())
        self.assertEqual(1, self.controller.projects.rowCount())
        self.assertEqual("", self.controller.message)

    def test_native_chooser_boundary_and_discovery_never_decode_images(self):
        controller = ProjectController(self.photos, registry_path=self.registry)
        self.addCleanup(controller.shutdown)
        gui_thread = threading.get_ident()
        model_threads = []
        self.photos.modelReset.connect(lambda: model_threads.append(threading.get_ident()))
        with patch("ui.qt.project_controller.QFileDialog.getExistingDirectory", return_value=str(self.folder)) as chooser, \
             patch("image_loader.ImageLoader.read_metadata", side_effect=AssertionError("Discovery decoded metadata")), \
             patch("image_loader.ImageLoader.load_pil_for_hashing", side_effect=AssertionError("Discovery decoded pixels")):
            controller.newProject()
            wait_until(lambda: not controller.busy)
        chooser.assert_called_once()
        self.assertEqual(1, self.photos.rowCount())
        self.assertEqual([gui_thread], model_threads)
        for role in ("detail", "badgeText"):
            self.assertEqual("", self.photos.data(self.photos.index(0), self.photos.ROLE[role]))
        self.assertIsNone(self.photos.data(self.photos.index(0), self.photos.ROLE["personId"]))

    def test_persistence_reload_missing_and_reopen(self):
        self.open_folder()
        identity = self.controller.activeProjectId
        data = json.loads(self.registry.read_text())
        self.assertEqual(1, data["version"])
        self.assertEqual({"project_id", "folder", "name", "created_at", "last_opened"}, set(data["projects"][0]))
        reloaded = self.make_controller()
        self.assertIsNotNone(reloaded.projects.get(identity))
        reloaded.openProject(identity)
        wait_until(lambda: not reloaded.busy)
        self.assertEqual(1, self.photos.rowCount())
        moved = self.root / "Moved"
        self.folder.rename(moved)
        reloaded.openProject(identity)
        wait_until(lambda: not reloaded.busy)
        self.assertIn("missing", reloaded.message)
        self.assertEqual(1, reloaded.projects.rowCount())
        # Missing folders don't crash registry loading or silently drop the card.
        self.assertEqual(1, self.make_controller().projects.rowCount())

    def test_cross_project_selection_is_cleared_and_folder_is_not_duplicated(self):
        self.open_folder()
        identity = self.controller.activeProjectId
        photo_id = self.photos.data(self.photos.index(0), self.photos.ROLE["photoId"])
        self.photos.setSelected(photo_id, True)
        # The parent project contains the same file: selection must still reset.
        self.open_folder(self.root)
        self.assertNotEqual(identity, self.controller.activeProjectId)
        self.assertFalse(self.photos.data(self.photos.index(0), self.photos.ROLE["selected"]))
        self.open_folder()
        self.assertEqual(identity, self.controller.activeProjectId)
        self.assertEqual(2, self.controller.projects.rowCount())

    def test_search_and_all_sort_orders(self):
        model = ProjectListModel([Project("a", "a", "Zebra", 10, 10), Project("b", "b", "alpha", 20, 20)])
        role = next(role for role, name in model.roleNames().items() if name == b"projectId")
        def ids():
            return [model.data(model.index(i), role) for i in range(model.rowCount())]
        self.assertEqual(["b", "a"], ids())
        model.setSort("Oldest")
        self.assertEqual(["a", "b"], ids())
        model.setSort("Name")
        self.assertEqual(["b", "a"], ids())
        model.setSearch("ALP")
        self.assertEqual(["b"], ids())
        model.setSearch("")
        self.assertEqual(2, model.rowCount())

    def test_corrupt_registry_and_invalid_records_are_safe(self):
        self.registry.write_text("broken json", encoding="utf-8")
        broken = self.make_controller()
        self.assertEqual(0, broken.projects.rowCount())
        self.assertIn("could not be read", broken.message)
        self.registry.write_text(json.dumps({"version": 1, "projects": [{"bad": True}]}))
        invalid = self.make_controller()
        self.assertEqual(0, invalid.projects.rowCount())
        self.assertIn("skipped", invalid.message)

    def test_access_errors_and_persistence_failure_are_clean(self):
        with patch("ui.qt.project_controller.os.scandir", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                discover_photos(self.folder)
        inaccessible = self.make_controller(discover=lambda folder: (_ for _ in ()).throw(PermissionError()))
        inaccessible.openFolder(str(self.folder))
        wait_until(lambda: not inaccessible.busy)
        self.assertIn("cannot access", inaccessible.message)
        self.assertEqual(0, inaccessible.projects.rowCount())
        with patch.object(Path, "replace", side_effect=PermissionError):
            self.open_folder()
        self.assertEqual(1, self.photos.rowCount())
        self.assertIn("could not be saved", self.controller.message)

    def test_obsolete_discovery_is_ignored_and_off_gui_thread(self):
        gate, started = threading.Event(), threading.Event()
        worker_ids = []
        def discover(folder):
            worker_ids.append(threading.get_ident())
            if Path(folder).name.casefold() == "holiday":
                started.set()
                gate.wait(2)
            return discover_photos(folder)
        controller = self.make_controller(discover=discover)
        controller.openFolder(str(self.folder))
        wait_until(started.is_set)
        empty = self.root / "Other"
        empty.mkdir()
        controller.openFolder(str(empty))
        gate.set()
        wait_until(lambda: not controller.busy)
        self.assertEqual("Other", controller.activeProjectName)
        self.assertEqual(0, self.photos.rowCount())
        self.assertEqual(1, controller.projects.rowCount())
        self.assertTrue(all(identity != threading.get_ident() for identity in worker_ids))

    def test_shutdown_ignores_late_discovery(self):
        gate = threading.Event()
        controller = self.make_controller(discover=lambda folder: (gate.wait(1), discover_photos(folder))[1])
        controller.openFolder(str(self.folder))
        controller.shutdown()
        gate.set()
        self.app.processEvents()
        self.assertEqual(0, controller.projects.rowCount())
