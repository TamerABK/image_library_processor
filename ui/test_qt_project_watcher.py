"""Deterministic project-watch reconciliation tests."""
import os
from pathlib import Path
import tempfile
import threading
import unittest

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from ui.qt.photo_model import PhotoListModel, photo_identity
from ui.qt.project_controller import ProjectController, discover_photos
from ui.qt.project_watcher import ProjectFilesystemWatcher, enumerate_directories
from ui.qt.thumbnails import ThumbnailService
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until


class FakeDirectoryWatcher(QObject):
    directoryChanged = Signal(str)

    def __init__(self):
        super().__init__()
        self.paths = set()
        self.fail = set()

    def addPaths(self, paths):
        failed = set(paths) & self.fail
        self.paths.update(set(paths) - failed)
        return list(failed)

    def removePaths(self, paths):
        self.paths.difference_update(paths)
        return []

    def directories(self):
        return list(self.paths)


class WatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a = self.root / "A"
        self.b = self.root / "B"
        self.a.mkdir(); self.b.mkdir()
        (self.a / "one.jpg").write_bytes(b"one")
        self.thumbnails = ThumbnailService()
        self.photos = PhotoListModel(self.thumbnails)
        self.scans = []
        self.discovery_gate = None
        self.discovery_entered = threading.Event()

        def discover(folder):
            self.scans.append(folder)
            if self.discovery_gate is not None and len(self.scans) == 2:
                self.discovery_entered.set()
                self.discovery_gate.wait(10)
            return discover_photos(folder)

        self.projects = ProjectController(self.photos, registry_path=self.root / "registry.json",
                                          database_path=self.root / "app.sqlite3", discover=discover)
        self.backend = FakeDirectoryWatcher()
        self.watcher = ProjectFilesystemWatcher(self.projects, watcher=self.backend, debounce_ms=25)
        self.projects.openFolder(str(self.a))
        wait_until(lambda: not self.projects.busy and self.projects.activeProjectFolder == photo_identity(self.a), timeout=10)
        wait_until(lambda: photo_identity(self.a) in self.backend.paths, timeout=10)

    def tearDown(self):
        if self.discovery_gate is not None:
            self.discovery_gate.set()
        self.watcher.shutdown()
        self.projects.shutdown()
        self.photos.close()
        self.thumbnails.shutdown()
        self.assertFalse(self.watcher._worker.is_alive())
        self.assertFalse(self.projects._worker.is_alive())
        self.temp.cleanup()

    def changed(self, path=None):
        self.watcher.notifyDirectoryChanged(str(path or self.a))

    def wait_refresh(self, count):
        wait_until(lambda: len(self.scans) >= count and not self.projects.busy, timeout=10)

    def test_external_add_delete_rename_and_modify(self):
        new = self.a / "new.jpg"
        new.write_bytes(b"new")
        self.changed()
        self.wait_refresh(2)
        self.assertIn(photo_identity(new), self.photos.identities())
        renamed = self.a / "renamed.jpg"
        new.rename(renamed)
        self.changed()
        self.wait_refresh(3)
        self.assertNotIn(photo_identity(new), self.photos.identities())
        self.assertIn(photo_identity(renamed), self.photos.identities())
        renamed.write_bytes(b"modified content")
        self.changed()
        self.wait_refresh(4)
        self.assertEqual(renamed.read_bytes(), b"modified content")
        renamed.unlink()
        self.changed()
        self.wait_refresh(5)
        self.assertNotIn(photo_identity(renamed), self.photos.identities())

    def test_burst_is_one_refresh_and_explicit_refresh_absorbs_prior_event(self):
        for _ in range(100):
            self.changed()
        self.wait_refresh(2)
        self.assertEqual(len(self.scans), 2)
        self.changed()
        self.assertTrue(self.projects.refreshActiveProject())
        self.wait_refresh(3)
        self.assertEqual(len(self.scans), 3)

    def test_event_during_discovery_gets_bounded_followup(self):
        self.discovery_gate = threading.Event()
        self.assertTrue(self.projects.refreshActiveProject())
        self.assertTrue(self.discovery_entered.wait(10))
        self.changed()
        self.watcher._reconcile()
        self.assertTrue(self.watcher._dirty)
        self.discovery_gate.set()
        self.wait_refresh(3)
        self.assertEqual(len(self.scans), 3)

    def test_old_project_event_does_not_refresh_new_project(self):
        old_generation = self.watcher._generation
        self.projects.openFolder(str(self.b))
        wait_until(lambda: not self.projects.busy and self.projects.activeProjectFolder == photo_identity(self.b), timeout=10)
        wait_until(lambda: photo_identity(self.b) in self.backend.paths, timeout=10)
        count = len(self.scans)
        self.watcher.notifyDirectoryChanged(str(self.a), generation=old_generation)
        self.watcher._reconcile()
        self.assertEqual(len(self.scans), count)

    def test_queued_old_event_is_ignored_even_when_path_becomes_new_parent(self):
        nested_project = self.a / "Nested"
        nested_project.mkdir()
        old_handler = self.watcher._event_handler
        self.projects.openFolder(str(nested_project))
        wait_until(lambda: not self.projects.busy and
                   self.projects.activeProjectFolder == photo_identity(nested_project), timeout=10)
        wait_until(lambda: photo_identity(self.a) in self.backend.paths, timeout=10)
        count = len(self.scans)
        old_handler(str(self.a))  # A was watched for old project and is now B's parent.
        self.watcher._reconcile()
        self.assertEqual(len(self.scans), count)

    def test_new_nested_directory_is_watched_after_refresh(self):
        nested = self.a / "nested"
        nested.mkdir()
        (nested / "inside.jpg").write_bytes(b"inside")
        self.changed()
        self.wait_refresh(2)
        wait_until(lambda: photo_identity(nested) in self.backend.paths, timeout=10)
        self.assertIn(photo_identity(nested / "inside.jpg"), self.photos.identities())

    def test_missing_root_reports_error_and_parent_watch_detects_return(self):
        self.assertIn(photo_identity(self.root), self.backend.paths)
        relocated = self.root / "moved-away"
        self.a.rename(relocated)
        self.backend.paths.discard(photo_identity(self.a))  # Qt drops a removed directory watch.
        self.changed(self.root)
        self.wait_refresh(2)
        self.assertIn("missing", self.projects.message.lower())
        relocated.rename(self.a)
        self.changed(self.root)
        self.wait_refresh(3)
        wait_until(lambda: photo_identity(self.a) in self.backend.paths, timeout=10)
        self.assertIn(photo_identity(self.a / "one.jpg"), self.photos.identities())

    def test_thousands_of_photos_do_not_add_photo_watches(self):
        class FileEntry:
            def is_dir(self, *, follow_symlinks=False):
                return False

        class Entries:
            def __enter__(self):
                return iter(FileEntry() for _ in range(6000))

            def __exit__(self, *_):
                pass

        from unittest.mock import patch
        with patch("ui.qt.project_watcher.os.scandir", return_value=Entries()):
            paths = enumerate_directories(str(self.a))
        self.assertEqual(paths, {str(self.a), str(self.root)})
        self.assertTrue(all(path in (str(self.a), str(self.root)) for path in paths))

    def test_failed_registration_enables_degraded_poll_and_shutdown_clears_watches(self):
        nested = self.a / "unwatchable"
        nested.mkdir()
        self.backend.fail.add(photo_identity(nested))
        self.changed()
        self.wait_refresh(2)
        wait_until(lambda: self.watcher.degraded, timeout=10)
        self.assertTrue(self.watcher._fallback.isActive())
        count = len(self.scans)
        self.watcher._fallback_refresh()
        self.wait_refresh(count + 1)
        self.watcher.shutdown()
        self.assertFalse(self.backend.paths)
        self.assertFalse(self.watcher._worker.is_alive())

    def test_export_outside_project_does_not_request_reconciliation(self):
        from ui.qt.export_controller import execute_plan, snapshot_plan
        plan = snapshot_plan(str(self.a), 1, "selected", [(str(self.a / "one.jpg"), "")], 0)
        output = execute_plan(plan, str(self.root / "exports"), "copy", threading.Event(), lambda *_: None)
        self.assertEqual(output["exported"], 1)
        self.assertEqual(len(self.scans), 1)
