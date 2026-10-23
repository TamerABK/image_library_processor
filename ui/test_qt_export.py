"""Phase 8D export safety and Qt presentation contracts."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PySide6.QtCore import QObject, Property, QPointF, Qt, QUrl, Signal, Slot
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_paths import qml_path
from ui.models import ResultItem
from ui.qt.app import configure_quick_controls_style
from ui.qt.export_controller import (ExportController, destination_outside_project,
                                     execute_plan, person_folders, safe_component, snapshot_plan)
from ui.qt.favorites_controller import FavoritesController
from ui.qt.favorites_store import FavoritesStore
from ui.qt.face_previews import FacePreviewService
from ui.qt.blur_model import BlurCandidateModel
from ui.qt.duplicate_model import DuplicateGroupModel
from ui.qt.people_model import PeopleListModel
from ui.qt.photo_model import PhotoListModel, photo_identity
from ui.qt.thumbnails import ThumbnailService
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until


class ProjectStub(QObject):
    projectOpening = Signal(str)
    activeProjectChanged = Signal()
    projectOpened = Signal(str, str)

    def __init__(self, folder):
        super().__init__()
        self.folder = str(folder)

    @Property(str, notify=activeProjectChanged)
    def activeProjectFolder(self):
        return self.folder

    def switch(self, folder):
        self.projectOpening.emit(str(folder))
        self.folder = str(folder)
        self.activeProjectChanged.emit()


class ProcessingStub(QObject):
    sessionChanged = Signal()

    def __init__(self, folder):
        super().__init__()
        self.folder = str(folder)

    @Property(str, notify=sessionChanged)
    def projectFolder(self):
        return self.folder


class PeopleStub:
    state = "COMPLETE"

    def __init__(self, groups=()):
        self.groups = list(groups)
        self.selected = self.groups[0] if self.groups else None

    def known_export_groups(self):
        return self.groups

    def selected_known_export_group(self):
        return self.selected


class PeoplePresentationStub(QObject):
    changed = Signal()

    def __init__(self, backend, thumbnails):
        super().__init__()
        self.backend = backend
        self.previews = FacePreviewService()
        self.known = PeopleListModel(self.previews, "known")
        self.detail = PhotoListModel(thumbnails)
        self.selected = {}
        self.known.set_rows([dict(key=key, name=name, personId=int(key),
                                  faceCount=len(paths), photoCount=len(paths),
                                  representativePhotoId=paths[0], representativeBBox=[1, 1, 10, 10],
                                  paths=list(paths), ignored=False)
                             for key, name, paths in backend.groups], revision="test")

    @Property(QObject, constant=True)
    def knownModel(self):
        return self.known

    @Property(QObject, constant=True)
    def detailPhotoModel(self):
        return self.detail

    @Property(str, notify=changed)
    def state(self):
        return "COMPLETE"

    @Property(int, notify=changed)
    def knownCount(self):
        return len(self.backend.groups)

    @Property(int, notify=changed)
    def unknownCount(self):
        return 0

    @Property("QVariantMap", notify=changed)
    def selectedKnown(self):
        return self.selected

    @Slot(str)
    def openKnown(self, key):
        group = next(group for group in self.backend.groups if group[0] == key)
        self.backend.selected = group
        self.selected = {"key": key, "name": group[1], "photoCount": len(group[2]),
                         "faceCount": len(group[2]), "previewSource": ""}
        self.detail.replace_items([ResultItem(Path(path), Path(path).name, "") for path in group[2]])
        self.changed.emit()

    def shutdown(self):
        self.known.close()
        self.detail.close()
        self.previews.shutdown()


class BlurPageStub(QObject):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @Property(QObject, constant=True)
    def blurModel(self):
        return self.model

    @Property("QVariantList", constant=True)
    def stages(self):
        return [{"state": "COMPLETE", "error": ""}]

    @Property(str, constant=True)
    def message(self):
        return ""


class DuplicatePageStub(QObject):
    def __init__(self, thumbnails, detail):
        super().__init__()
        self.groups = DuplicateGroupModel(thumbnails)
        self.detail = detail

    @Property(QObject, constant=True)
    def groupModel(self):
        return self.groups

    @Property(QObject, constant=True)
    def detailPhotoModel(self):
        return self.detail

    @Property("QVariantMap", constant=True)
    def selectedGroup(self):
        return {"groupId": "test", "photoCount": self.detail.rowCount(),
                "recommendationState": "ready", "explanation": [], "members": []}

    @Property(str, constant=True)
    def state(self):
        return "COMPLETE"

    @Property(str, constant=True)
    def rankingState(self):
        return "COMPLETE"

    @Property(bool, constant=True)
    def reviewReady(self):
        return False

    @Property(bool, constant=True)
    def busy(self):
        return False

    @Property(str, constant=True)
    def error(self):
        return ""

    def shutdown(self):
        self.groups.close()


class UnknownPageStub(QObject):
    changed = Signal()

    def __init__(self, thumbnails, detail):
        super().__init__()
        self.previews = FacePreviewService()
        self.model = PeopleListModel(self.previews, "unknown")
        self.detail = detail

    @Property(QObject, constant=True)
    def unknownModel(self):
        return self.model

    @Property(QObject, constant=True)
    def detailPhotoModel(self):
        return self.detail

    @Property("QVariantMap", notify=changed)
    def selectedUnknown(self):
        return {"key": "test", "faceCount": 1, "photoCount": self.detail.rowCount(),
                "ignored": False, "paths": self.detail.identities(), "previewSource": ""}

    @Property(str, constant=True)
    def state(self):
        return "COMPLETE"

    @Property(bool, constant=True)
    def busy(self):
        return False

    @Property(str, constant=True)
    def error(self):
        return ""

    @Property("QVariantMap", constant=True)
    def suggestion(self):
        return {}

    @Property("QVariantList", constant=True)
    def knownOptions(self):
        return []

    def shutdown(self):
        self.model.close()
        self.previews.shutdown()


class ExportFilesystemTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.destination = self.root / "exports"
        self.a = self.project / "photo.jpg"
        self.b = self.project / "other.jpg"
        self.a.write_bytes(b"original photo")
        self.b.write_bytes(b"another photo")

    def plan(self, paths=None, mode="selected"):
        return snapshot_plan(str(self.project), 1, mode,
                             [(str(path), "") for path in (paths or [self.a, self.b])], 0)

    def run_plan(self, plan, method="copy", canceled=None):
        return execute_plan(plan, str(self.destination), method,
                            canceled or threading.Event(), lambda *_: None)

    def test_copy_selected_collision_and_source_unchanged(self):
        self.destination.mkdir()
        (self.destination / "photo.jpg").write_bytes(b"existing")
        (self.destination / "photo_1.jpg").write_bytes(b"existing one")
        outcome = self.run_plan(self.plan())
        self.assertEqual((outcome["exported"], outcome["failed"]), (2, 0))
        self.assertEqual((self.destination / "photo_2.jpg").read_bytes(), b"original photo")
        self.assertEqual((self.destination / "photo.jpg").read_bytes(), b"existing")
        self.assertEqual((self.destination / "photo_1.jpg").read_bytes(), b"existing one")
        self.assertEqual(self.a.read_bytes(), b"original photo")

    def test_file_exists_race_retries_without_overwrite(self):
        real_open = os.open
        attempts = []

        def race(path, flags, mode=0o777):
            if Path(path).name.startswith(".room36-export-"):
                return real_open(path, flags, mode)
            attempts.append(Path(path).name)
            if len(attempts) == 1:
                raise FileExistsError(path)
            return real_open(path, flags, mode)

        with patch("ui.qt.export_controller.os.open", side_effect=race):
            outcome = self.run_plan(self.plan([self.a]))
        self.assertEqual(outcome["exported"], 1)
        self.assertEqual(attempts, ["photo.jpg", "photo_1.jpg"])
        self.assertEqual((self.destination / "photo_1.jpg").read_bytes(), b"original photo")

    def test_post_reservation_replacement_is_never_overwritten_or_removed(self):
        real_copy = __import__("shutil").copyfileobj
        external = self.root / "external"
        external.write_bytes(b"another process owns this")

        def replaced(source, output, length=0):
            # Model a rebind after exclusive creation, even on Windows where a
            # normal open handle usually prevents this replacement.
            os.close(output.fileno())
            os.replace(external, self.destination / "photo.jpg")
            return real_copy(source, output, length)

        with patch("ui.qt.export_controller.shutil.copyfileobj", side_effect=replaced):
            outcome = self.run_plan(self.plan([self.a]))
        self.assertEqual((outcome["exported"], outcome["failed"]), (0, 1))
        self.assertEqual((self.destination / "photo.jpg").read_bytes(), b"another process owns this")

    def test_source_replacement_with_same_revision_is_rejected(self):
        plan = self.plan([self.a])
        original = self.a.stat()
        replacement = self.root / "replacement"
        replacement.write_bytes(b"replaced photo")
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        os.replace(replacement, self.a)
        outcome = self.run_plan(plan)
        self.assertEqual(outcome["failed"], 1)
        self.assertFalse((self.destination / "photo.jpg").exists())

    def test_source_modified_during_copy_removes_only_owned_partial_output(self):
        real_copy = __import__("shutil").copyfileobj

        def change_during_copy(source, output, length=0):
            real_copy(source, output, length)
            self.a.write_bytes(b"changed while exporting")

        with patch("ui.qt.export_controller.shutil.copyfileobj", side_effect=change_during_copy):
            outcome = self.run_plan(self.plan([self.a]))
        self.assertEqual((outcome["exported"], outcome["failed"]), (0, 1))
        self.assertFalse((self.destination / "photo.jpg").exists())

    def test_hardlink_source_replacement_during_link_is_rejected(self):
        plan = self.plan([self.a])
        original = self.a.stat()
        real_link = os.link

        def replace_then_link(source, target):
            replacement = self.root / "replacement"
            replacement.write_bytes(b"replaced photo")
            os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
            os.replace(replacement, source)
            real_link(source, target)

        with patch("ui.qt.export_controller.os.link", side_effect=replace_then_link):
            outcome = self.run_plan(plan, "hardlink")
        self.assertEqual(outcome["failed"], 1)
        self.assertFalse((self.destination / "photo.jpg").exists())

    def test_person_folder_cannot_reenter_project(self):
        plan = snapshot_plan(str(self.project), 1, "all_people", [(str(self.a), "project", "1")], 1)
        outcome = execute_plan(plan, str(self.root), "copy", threading.Event(), lambda *_: None)
        self.assertEqual((outcome["exported"], outcome["failed"]), (0, 1))
        self.assertEqual(set(self.project.iterdir()), {self.a, self.b})

    def test_resolved_person_folder_escape_is_rejected(self):
        from ui.qt import export_controller as module
        plan = snapshot_plan(str(self.project), 1, "all_people", [(str(self.a), "Alice", "1")], 1)
        real_realpath = os.path.realpath

        def escaped(path, *args, **kwargs):
            if Path(path) == self.destination / "Alice":
                return str(self.project)
            return real_realpath(path, *args, **kwargs)

        with patch.object(module.os.path, "realpath", side_effect=escaped):
            outcome = self.run_plan(plan)
        self.assertEqual(outcome["failed"], 1)
        self.assertFalse((self.destination / "Alice").exists())

    def test_hard_link_shares_file_identity_when_supported(self):
        outcome = self.run_plan(self.plan([self.a]), "hardlink")
        if outcome["failed"]:
            self.skipTest("Hard links are unavailable on this filesystem")
        self.assertTrue(os.path.samefile(self.a, self.destination / "photo.jpg"))

    def test_hard_link_failure_never_falls_back_to_copy(self):
        with patch("ui.qt.export_controller.os.link", side_effect=OSError("link unavailable")), \
                patch("ui.qt.export_controller.shutil.copy2") as copied:
            outcome = self.run_plan(self.plan([self.a]), "hardlink")
        self.assertEqual(outcome["failed"], 1)
        copied.assert_not_called()

    def test_missing_changed_and_partial_failure_continue(self):
        third = self.project / "third.jpg"
        third.write_bytes(b"third")
        plan = self.plan([self.a, self.b, third])
        self.a.unlink()
        self.b.write_bytes(b"changed")
        outcome = self.run_plan(plan)
        self.assertEqual((outcome["exported"], outcome["failed"]), (1, 2))
        self.assertFalse((self.destination / "photo.jpg").exists())
        self.assertFalse((self.destination / "other.jpg").exists())
        self.assertEqual((self.destination / "third.jpg").read_bytes(), b"third")

    def test_copy_permission_failure_continues(self):
        third = self.project / "third.jpg"
        third.write_bytes(b"third")
        real_copy = __import__("shutil").copyfileobj

        def fail_middle(source, destination, length=0):
            if Path(source.name).name == "other.jpg":
                raise PermissionError("not writable")
            return real_copy(source, destination, length)

        with patch("ui.qt.export_controller.shutil.copyfileobj", side_effect=fail_middle):
            outcome = self.run_plan(self.plan([self.a, self.b, third]))
        self.assertEqual((outcome["exported"], outcome["failed"]), (2, 1))
        self.assertFalse((self.destination / "other.jpg").exists())

    def test_destination_inside_project_and_equal_root_blocked(self):
        for destination in (self.project, self.project / "exports"):
            self.assertFalse(destination_outside_project(str(destination), str(self.project)))
            with patch("ui.qt.export_controller.shutil.copyfileobj") as copied:
                with self.assertRaisesRegex(ValueError, "outside the current project"):
                    execute_plan(self.plan([self.a]), str(destination), "copy",
                                 threading.Event(), lambda *_: None)
            copied.assert_not_called()

    def test_person_names_and_overlapping_membership(self):
        names = [("1", "A/B", (str(self.a),)), ("2", "CON", (str(self.b),)),
                 ("3", "Jane.", (str(self.a),)), ("4", "John:Smith", (str(self.b),)),
                 ("5", "", (str(self.a),)), ("6", "A\\B", (str(self.a),)),
                 ("7", "../../OtherFolder", (str(self.a),))]
        folders = person_folders(names)
        self.assertEqual(len({folder.casefold() for folder in folders.values()}), len(names))
        self.assertEqual(folders["1"], "A_B")
        self.assertEqual(folders["6"], "A_B (2)")
        self.assertEqual(safe_component("CON"), "CON_")
        self.assertEqual(safe_component("CON.txt"), "CON_.txt")
        self.assertEqual(safe_component("Jane."), "Jane")
        self.assertTrue(all("/" not in value and "\\" not in value and value not in (".", "..")
                            for value in folders.values()))

    def test_cancellation_stops_before_next_file(self):
        canceled = threading.Event()
        seen = []

        def progress(done, total):
            seen.append(done)
            canceled.set()

        outcome = execute_plan(self.plan(), str(self.destination), "copy", canceled, progress)
        self.assertEqual((outcome["exported"], outcome["canceled"]), (1, True))
        self.assertEqual(seen, [1])
        self.assertFalse((self.destination / "other.jpg").exists())

    def test_six_thousand_lightweight_plan_placements(self):
        requested = [(str(self.a), f"Person {i}") for i in range(6000)]
        plan = snapshot_plan(str(self.project), 1, "all_people", requested, 6000)
        self.assertEqual((len(plan.placements), plan.unique_count), (6000, 1))
        self.assertEqual(plan.total_bytes, self.a.stat().st_size)


class ExportControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "project"
        self.folder.mkdir()
        self.photos_paths = [self.folder / "a.jpg", self.folder / "b.jpg", self.folder / "shared.jpg"]
        for path in self.photos_paths:
            path.write_bytes(path.name.encode())
        self.thumbnails = ThumbnailService()
        self.master = PhotoListModel(self.thumbnails)
        self.master.replace_items([ResultItem(path, path.name, "") for path in self.photos_paths])
        self.projects = ProjectStub(self.folder)
        self.processing = ProcessingStub(self.folder)
        self.people = PeopleStub([("1", "Alice", (str(self.photos_paths[0]), str(self.photos_paths[2]))),
                                  ("2", "Bob", (str(self.photos_paths[1]), str(self.photos_paths[2])))])
        self.destination = self.root / "out"
        self.exporter = ExportController(self.projects, self.master, self.people, self.processing,
                                         chooser=lambda: str(self.destination))

    def tearDown(self):
        self.exporter.shutdown()
        self.master.close()
        self.thumbnails.shutdown()
        self.assertFalse(self.exporter._worker.is_alive())
        self.temp.cleanup()

    def wait_plan(self):
        wait_until(lambda: self.exporter.prepared)

    def wait_done(self):
        wait_until(lambda: not self.exporter.busy, timeout=10)

    def test_selected_copy_keeps_selection_and_never_rescans(self):
        selected = photo_identity(self.photos_paths[0])
        self.master.setSelected(selected, True)
        self.assertEqual(self.master.selectedCount, 1)
        self.assertTrue(self.exporter.prepareSelected(self.master))
        self.wait_plan()
        self.assertTrue(self.exporter.chooseDestination())
        self.assertTrue(self.exporter.execute())
        self.wait_done()
        self.assertEqual(self.exporter.progressDone, 1)
        self.assertIn("Exported 1 photo", self.exporter.summary)
        self.assertEqual(self.master.selectedCount, 1)
        self.assertEqual((self.destination / "a.jpg").read_bytes(), b"a.jpg")
        self.assertTrue(self.photos_paths[0].exists())

    def test_one_person_and_all_people_ignore_search_filter(self):
        self.assertTrue(self.exporter.preparePerson())
        self.wait_plan()
        self.assertEqual(self.exporter.planInfo["placements"], 2)
        self.exporter._plan = None
        self.assertTrue(self.exporter.prepareAllPeople())
        self.wait_plan()
        self.assertEqual((self.exporter.planInfo["peopleCount"],
                          self.exporter.planInfo["uniquePhotos"],
                          self.exporter.planInfo["placements"]), (2, 3, 4))
        self.exporter.setDestination(str(self.destination))
        self.assertTrue(self.exporter.execute())
        self.wait_done()
        self.assertEqual(self.exporter.progressDone, 4)
        self.assertEqual((self.destination / "Alice" / "shared.jpg").read_bytes(), b"shared.jpg")
        self.assertEqual((self.destination / "Bob" / "shared.jpg").read_bytes(), b"shared.jpg")

    def test_project_switch_discards_old_plan_and_result(self):
        self.master.setSelected(photo_identity(self.photos_paths[0]), True)
        self.assertTrue(self.exporter.prepareSelected(self.master))
        self.wait_plan()
        self.exporter.setDestination(str(self.destination))
        entered = threading.Event()
        release = threading.Event()
        from ui.qt import export_controller as module
        original = module._export_one

        def held(item, directory, method):
            entered.set()
            release.wait(5)
            return original(item, directory, method)

        with patch("ui.qt.export_controller._export_one", side_effect=held):
            self.assertTrue(self.exporter.execute())
            self.assertTrue(entered.wait(3))
            second = self.root / "other-project"
            second.mkdir()
            self.projects.switch(second)
            self.master.replace_items([], preserve_selection=False)
            self.assertFalse(self.exporter.prepared)
            self.assertEqual(self.exporter.summary, "")
            release.set()
            self.wait_done()
        self.assertEqual(self.exporter.summary, "")
        self.assertEqual(self.master.rowCount(), 0)
        self.assertTrue((self.destination / "a.jpg").exists())

    def test_cancel_after_first_output_keeps_result_and_worker_reusable(self):
        for path in self.photos_paths[:2]:
            self.master.setSelected(photo_identity(path), True)
        self.assertTrue(self.exporter.prepareSelected(self.master))
        self.wait_plan()
        self.exporter.setDestination(str(self.destination))
        entered = threading.Event()
        release = threading.Event()
        from ui.qt import export_controller as module
        original = module._export_one
        calls = []

        def held(item, directory, method):
            calls.append(item.source)
            entered.set()
            release.wait(5)
            return original(item, directory, method)

        with patch("ui.qt.export_controller._export_one", side_effect=held):
            self.assertTrue(self.exporter.execute())
            self.assertTrue(entered.wait(3))
            self.exporter.cancel()
            release.set()
            self.wait_done()
        self.assertEqual(len(calls), 1)
        self.assertIn("Export canceled", self.exporter.summary)
        self.assertEqual(self.exporter.progressDone, 1)
        self.assertTrue((self.destination / Path(calls[0]).name).exists())
        self.assertTrue(self.exporter.prepareSelected(self.master))
        self.wait_plan()
        self.exporter.setDestination(str(self.destination))
        self.assertTrue(self.exporter.execute())
        self.wait_done()
        self.assertEqual(self.exporter.progressDone, 2)

    def test_partial_result_reports_success_and_failure(self):
        for path in self.photos_paths[:2]:
            self.master.setSelected(photo_identity(path), True)
        self.assertTrue(self.exporter.prepareSelected(self.master))
        self.wait_plan()
        self.exporter.setDestination(str(self.destination))
        real_copy = __import__("shutil").copyfileobj

        def fail_second(source, target, length=0):
            if Path(source.name).name == "b.jpg":
                raise PermissionError("not writable")
            return real_copy(source, target, length)

        with patch("ui.qt.export_controller.shutil.copyfileobj", side_effect=fail_second):
            self.assertTrue(self.exporter.execute())
            self.wait_done()
        self.assertIn("Exported 1 photo", self.exporter.summary)
        self.assertIn("1 could not be exported", self.exporter.summary)
        self.assertEqual(len(self.exporter.failures), 1)

    def test_active_export_configuration_cannot_change(self):
        self.master.setSelected(photo_identity(self.photos_paths[0]), True)
        self.assertTrue(self.exporter.prepareSelected(self.master))
        self.wait_plan()
        self.assertTrue(self.exporter.setDestination(str(self.destination)))
        entered, release = threading.Event(), threading.Event()
        from ui.qt import export_controller as module
        original = module._export_one

        def held(item, directory, method):
            entered.set()
            release.wait(5)
            return original(item, directory, method)

        with patch("ui.qt.export_controller._export_one", side_effect=held):
            self.assertTrue(self.exporter.execute())
            self.assertTrue(entered.wait(3))
            self.assertFalse(self.exporter.setDestination(str(self.root / "different")))
            self.assertFalse(self.exporter.chooseDestination())
            self.assertFalse(self.exporter.setMethod("hardlink"))
            release.set()
            self.wait_done()
        self.assertIn(str(self.destination).lower(), self.exporter.summary.lower())
        self.assertIn("Exported 1 photo", self.exporter.summary)

    def test_only_current_model_members_and_safe_destination(self):
        other = PhotoListModel(self.thumbnails)
        stale = self.root / "stale.jpg"
        stale.write_bytes(b"not in project")
        other.replace_items([ResultItem(stale, stale.name, "")])
        other.setSelected(photo_identity(stale), True)
        self.assertFalse(self.exporter.prepareSelected(other))
        self.master.setSelected(photo_identity(self.photos_paths[0]), True)
        self.assertTrue(self.exporter.prepareSelected(self.master))
        self.wait_plan()
        self.assertFalse(self.exporter.setDestination(str(self.folder / "exports")))
        self.assertFalse(self.exporter.setDestination(str(self.folder)))
        self.assertFalse(self.exporter.execute())
        self.assertEqual(self.exporter._tasks.maxsize, 1)
        self.assertEqual(self.exporter._results.maxsize, 32)
        self.assertEqual(self.thumbnails.pending_count, 0)
        other.close()

    def click_qml(self, window, item):
        self.app.processEvents()
        point = item.mapToScene(QPointF(item.width() / 2, item.height() / 2))
        QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         point.toPoint())
        self.app.processEvents()

    @staticmethod
    def find_item(window, name):
        def visit(item):
            if item.objectName() == name:
                return item
            for child in item.childItems():
                found = visit(child)
                if found is not None:
                    return found
            return None
        return visit(window.contentItem()) or window.findChild(QObject, name)

    def load_isolated_page(self, page, **objects):
        engine = QQmlApplicationEngine()
        for name, value in objects.items():
            engine.rootContext().setContextProperty(name, value)
        source_property = {"BlurryPhotosPage": "processing: processingObject",
                           "NearDuplicatesPage": "duplicates: duplicatesObject",
                           "UnknownPeoplePage": "people: peopleObject"}[page]
        source = ("import QtQuick\nimport QtQuick.Window\nimport \"pages\" as Pages\n"
                  "Window { width: 1280; height: 720; visible: true; "
                  f"Pages.{page} {{ anchors.fill: parent; exporter: exportObject; "
                  f"{source_property} }} }}")
        engine.loadData(source.encode(), QUrl.fromLocalFile(str(qml_path("App.qml"))))
        self.assertTrue(engine.rootObjects())
        return engine, engine.rootObjects()[0]

    def test_qml_blur_duplicate_and_unknown_detail_selected_actions(self):
        source = photo_identity(self.photos_paths[0])
        blur = BlurCandidateModel(self.thumbnails)
        blur.set_results([dict(photoId=source, filename="a.jpg", status="Blurry", final_score=0.1)])
        blur.setSelected(source, True)
        self.assertEqual(blur.selectedCount, 1)
        detail = PhotoListModel(self.thumbnails)
        detail.replace_items([ResultItem(self.photos_paths[0], "a.jpg", "")])
        detail.setSelected(source, True)
        duplicate = DuplicatePageStub(self.thumbnails, detail)
        unknown = UnknownPageStub(self.thumbnails, detail)
        processing_page = BlurPageStub(blur)
        cases = [("BlurryPhotosPage", "blurExportSelected"),
                 ("NearDuplicatesPage", "duplicateExportSelected"),
                 ("UnknownPeoplePage", "peopleExportSelected")]
        try:
            for page, action in cases:
                engine, window = self.load_isolated_page(page, exportObject=self.exporter,
                    processingObject=processing_page, duplicatesObject=duplicate, peopleObject=unknown)
                try:
                    button = self.find_item(window, action)
                    self.assertIsNotNone(button, page)
                    self.assertTrue(button.property("enabled"), page)
                    self.click_qml(window, button)
                    self.wait_plan()
                    self.assertEqual(self.exporter.planInfo["placements"], 1)
                    self.exporter._plan = None
                    self.exporter.changed.emit()
                finally:
                    window.close()
                    engine.deleteLater()
                    self.app.processEvents()
        finally:
            duplicate.shutdown()
            unknown.shutdown()
            detail.close()
            blur.close()

    def test_qml_progress_cancel_and_partial_result_are_visible(self):
        for path in self.photos_paths[:2]:
            self.master.setSelected(photo_identity(path), True)
        engine = QQmlApplicationEngine()
        engine.setInitialProperties(dict(startPage="library", photoModel=self.master,
                                         exportController=self.exporter))
        entered, release = threading.Event(), threading.Event()
        from ui.qt import export_controller as module
        original = module._export_one

        def held(item, directory, method):
            entered.set()
            release.wait(5)
            return original(item, directory, method)

        try:
            engine.load(qml_path("App.qml").as_uri())
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            self.click_qml(window, self.find_item(window, "libraryExportSelected"))
            self.wait_plan()
            self.exporter.setDestination(str(self.destination))
            with patch("ui.qt.export_controller._export_one", side_effect=held):
                self.click_qml(window, self.find_item(window, "exportConfirm"))
                self.assertTrue(entered.wait(3))
                self.app.processEvents()
                self.assertTrue(self.find_item(window, "exportProgress").property("visible"))
                cancel = self.find_item(window, "exportCancel")
                self.assertEqual(cancel.property("text"), "Cancel export")
                self.click_qml(window, cancel)
                release.set()
                self.wait_done()
            self.assertIn("Export canceled", self.find_item(window, "exportSummary").property("text"))
            window.close()
        finally:
            release.set()
            engine.deleteLater()

    def test_qml_partial_failure_reports_success_and_failure(self):
        for path in self.photos_paths[:2]:
            self.master.setSelected(photo_identity(path), True)
        engine = QQmlApplicationEngine()
        engine.setInitialProperties(dict(startPage="library", photoModel=self.master,
                                         exportController=self.exporter))
        real_copy = __import__("shutil").copyfileobj

        def fail_second(source, output, length=0):
            if Path(source.name).name == "b.jpg":
                raise PermissionError("not writable")
            return real_copy(source, output, length)

        try:
            engine.load(qml_path("App.qml").as_uri())
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            self.click_qml(window, self.find_item(window, "libraryExportSelected"))
            self.wait_plan()
            self.exporter.setDestination(str(self.destination))
            with patch("ui.qt.export_controller.shutil.copyfileobj", side_effect=fail_second):
                self.click_qml(window, self.find_item(window, "exportConfirm"))
                self.wait_done()
            summary = self.find_item(window, "exportSummary")
            self.assertTrue(summary.property("visible"))
            self.assertIn("Exported 1 photo", summary.property("text"))
            self.assertIn("1 could not be exported", summary.property("text"))
            window.close()
        finally:
            engine.deleteLater()

    def test_qml_library_action_opens_shared_export_dialog(self):
        self.master.setSelected(photo_identity(self.photos_paths[0]), True)
        engine = QQmlApplicationEngine()
        engine.setInitialProperties(dict(startPage="library", photoModel=self.master,
                                         exportController=self.exporter))
        try:
            engine.load(qml_path("App.qml").as_uri())
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            button = self.find_item(window, "libraryExportSelected")
            self.assertIsNotNone(button)
            self.assertTrue(button.property("enabled"))
            self.click_qml(window, button)
            wait_until(lambda: self.exporter.prepared)
            dialog = self.find_item(window, "exportDialog")
            self.assertTrue(dialog.property("visible"))
            self.assertIsNotNone(self.find_item(window, "exportChooseDestination"))
            self.assertIsNotNone(self.find_item(window, "exportMethod"))
            method = self.find_item(window, "exportMethod")
            self.click_qml(window, method)
            QTest.qWait(30)  # Popup delegates are created on the next render frame.
            option = self.find_item(window, "roundedComboOption_1")
            self.assertIsNotNone(option)
            self.click_qml(window, option)
            self.assertEqual(self.exporter.method, "hardlink")
            self.exporter.setMethod("copy")
            self.exporter.chooseDestination()
            confirm = self.find_item(window, "exportConfirm")
            self.assertTrue(confirm.property("enabled"))
            self.click_qml(window, confirm)
            self.wait_done()
            self.assertTrue((self.destination / "a.jpg").exists())
            window.close()
        finally:
            engine.deleteLater()

    def test_qml_favorites_export_selected(self):
        database_path = self.root / "favorites.sqlite3"
        favorite = photo_identity(self.photos_paths[1])
        FavoritesStore(database_path).set_favorite(self.folder, favorite, True)
        favorites = FavoritesController(self.projects, self.master, self.thumbnails,
                                        database_path=database_path)
        engine = QQmlApplicationEngine()
        engine.setInitialProperties(dict(startPage="library", photoModel=self.master,
                                         favoritesController=favorites, exportController=self.exporter))
        try:
            wait_until(lambda: favorites.loaded)
            favorites.photoModel.setSelected(favorite, True)
            engine.load(qml_path("App.qml").as_uri())
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            self.click_qml(window, self.find_item(window, "libraryNav_favorites"))
            button = self.find_item(window, "favoritesExportSelected")
            self.assertTrue(button.property("enabled"))
            self.click_qml(window, button)
            self.wait_plan()
            self.assertEqual(self.exporter.planInfo["uniquePhotos"], 1)
            self.assertEqual(self.exporter._plan.placements[0].source, favorite)
            window.close()
        finally:
            engine.deleteLater()
            favorites.shutdown()

    def test_qml_known_people_all_and_person_actions(self):
        presentation = PeoplePresentationStub(self.people, self.thumbnails)
        engine = QQmlApplicationEngine()
        engine.setInitialProperties(dict(startPage="library", photoModel=self.master,
                                         peopleController=presentation, exportController=self.exporter))
        try:
            engine.load(qml_path("App.qml").as_uri())
            self.assertTrue(engine.rootObjects())
            window = engine.rootObjects()[0]
            self.click_qml(window, self.find_item(window, "libraryNav_knownPeople"))
            all_button = self.find_item(window, "knownExportAll")
            self.assertTrue(all_button.property("enabled"))
            self.click_qml(window, all_button)
            self.wait_plan()
            self.assertEqual(self.exporter.planInfo["placements"], 4)
            dialog = self.find_item(window, "exportDialog")
            dialog.close()
            self.exporter._plan = None
            self.exporter.changed.emit()
            self.click_qml(window, self.find_item(window, "personCardOpen"))
            self.assertTrue(presentation.selectedKnown)
            person_button = self.find_item(window, "knownExportPerson")
            self.assertTrue(person_button.property("enabled"))
            self.click_qml(window, person_button)
            self.wait_plan()
            self.assertEqual(self.exporter.planInfo["placements"], 2)
            window.close()
        finally:
            engine.deleteLater()
            presentation.shutdown()
