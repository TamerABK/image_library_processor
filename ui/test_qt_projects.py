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
        self.addCleanup(self.close_controller, controller)
        return controller

    def close_controller(self, controller):
        controller.shutdown()
        controller._worker.join(timeout=2)
        self.assertFalse(controller._worker.is_alive(), "Leaked project-discovery worker")

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
        self.assertEqual({"project_id", "folder", "name", "created_at", "last_opened", "photo_count"}, set(data["projects"][0]))
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

    def test_photo_count_is_persisted_in_version_one_registry(self):
        self.open_folder()
        payload = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertEqual(1, payload["version"])
        self.assertEqual(1, payload["projects"][0]["photo_count"])
        self.assertIs(type(payload["projects"][0]["photo_count"]), int)
        empty = self.root / "Empty"
        empty.mkdir()
        self.open_folder(empty)
        payload = json.loads(self.registry.read_text(encoding="utf-8"))
        self.assertEqual([1, 0], [record["photo_count"] for record in payload["projects"]])

    def test_saved_photo_count_survives_restart_and_reopen_refreshes_it(self):
        self.open_folder()
        identity = self.controller.activeProjectId
        self.close_controller(self.controller)
        self.photos.replace_items([], preserve_selection=False)
        with patch("ui.qt.project_controller.find_supported_files", side_effect=AssertionError("Rescanned at startup")):
            reloaded = self.make_controller()
            self.assertEqual(1, reloaded.projects.get(identity).photo_count)
            self.assertEqual(0, self.photos.rowCount())
            self.assertFalse(reloaded.busy)
        Image.new("RGB", (10, 10)).save(self.folder / "new.png")
        reloaded.openProject(identity)
        wait_until(lambda: not reloaded.busy)
        self.assertEqual(2, reloaded.projects.get(identity).photo_count)
        self.assertEqual(2, json.loads(self.registry.read_text())["projects"][0]["photo_count"])

    def test_legacy_registry_without_photo_count_or_last_opened_loads(self):
        self.open_folder()
        payload = json.loads(self.registry.read_text())
        record = payload["projects"][0]
        del record["photo_count"]
        del record["last_opened"]
        self.registry.write_text(json.dumps(payload), encoding="utf-8")
        reloaded = self.make_controller()
        self.assertEqual(1, reloaded.projects.rowCount())
        project = reloaded.projects.get(record["project_id"])
        self.assertEqual(-1, project.photo_count)
        self.assertEqual(0, project.last_opened)
        role = next(role for role, name in reloaded.projects.roleNames().items() if name == b"lastOpened")
        self.assertEqual("", reloaded.projects.data(reloaded.projects.index(0), role))
        self.assertEqual("", reloaded.message)

    def test_invalid_photo_counts_fall_back_without_losing_projects(self):
        counts = [-1, 0, 7, -2, None, True, False, 1.0, 1.5, "7", "invalid", [], {}]
        records = [{"project_id": str(i), "folder": str(self.root / str(i)), "name": str(i),
                    "created_at": 100, "last_opened": 0, "photo_count": count}
                   for i, count in enumerate(counts)]
        self.registry.write_text(json.dumps({"version": 1, "projects": records}), encoding="utf-8")
        reloaded = self.make_controller()
        self.assertEqual(len(counts), reloaded.projects.rowCount())
        for i, count in enumerate(counts):
            with self.subTest(count=count):
                self.assertEqual(count if i < 3 else -1, reloaded.projects.get(str(i)).photo_count)
        self.assertEqual("", reloaded.message)
        self.assertEqual("", reloaded._save_registry())
        saved = json.loads(self.registry.read_text())["projects"]
        self.assertTrue(all(type(record["photo_count"]) is int and record["photo_count"] >= -1 for record in saved))

    def test_last_opened_display_ignores_uninitialized_and_invalid_timestamps(self):
        role = next(role for role, name in ProjectListModel.ROLES.items() if name == b"lastOpened")
        for timestamp in (0, -1, None, "invalid", float("nan"), float("inf"), 1e30, True):
            with self.subTest(timestamp=timestamp):
                model = ProjectListModel([Project("a", "a", "A", 0, timestamp)])
                self.assertEqual("", model.data(model.index(0), role))
        self.open_folder()
        label = self.controller.projects.data(self.controller.projects.index(0), role)
        self.assertRegex(label, r"^Last opened \d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
        self.assertNotIn("1970", label)

    def test_malformed_registry_timestamps_are_rejected(self):
        self.open_folder()
        valid = json.loads(self.registry.read_text())["projects"][0]
        records = [valid]
        for key in ("created_at", "last_opened"):
            for value in (None, "invalid", float("nan"), float("inf"), -1, 1e30, True, [], {}):
                records.append({**valid, "project_id": str(len(records)),
                                "folder": str(self.root / str(len(records))), key: value})
        self.registry.write_text(json.dumps({"version": 1, "projects": records}), encoding="utf-8")
        reloaded = self.make_controller()
        self.assertEqual(1, reloaded.projects.rowCount())
        self.assertEqual(valid["project_id"], reloaded.projects.get(valid["project_id"]).project_id)
        self.assertIn("skipped", reloaded.message)

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
        # Last-opened order deliberately disagrees with creation order.
        model = ProjectListModel([Project("a", "a", "Zebra", 10, 100), Project("b", "b", "alpha", 20, 1),
                                  Project("c", "c", "ALPHA", 20, 2)])
        role = next(role for role, name in model.roleNames().items() if name == b"projectId")
        def ids():
            return [model.data(model.index(i), role) for i in range(model.rowCount())]
        self.assertEqual(["c", "b", "a"], ids())
        model.register(Project("a", "a", "Zebra", 10, 1000))
        self.assertEqual(["c", "b", "a"], ids())
        model.setSort("Oldest")
        self.assertEqual(["a", "b", "c"], ids())
        model.setSort("Name")
        self.assertEqual(["b", "c", "a"], ids())
        model.setSearch("ALP")
        self.assertEqual(["b", "c"], ids())
        model.setSearch("")
        self.assertEqual(3, model.rowCount())

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
        wait_until(lambda: not controller._worker.is_alive())
        self.app.processEvents()
        self.assertEqual(0, controller.projects.rowCount())
