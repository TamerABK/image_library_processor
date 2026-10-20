"""Phase 8C removal intent, OS Trash, and project isolation contracts."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from PySide6.QtCore import QObject, Property, QPointF, Qt, Signal, Slot
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_paths import qml_path
from automatic_scan.state_store import AutomaticStateStore, DeletionTransactionRecord, PersistedImageDecision
from ui.qt.app import configure_quick_controls_style
from ui.qt.blur_model import BlurCandidateModel
from ui.qt.favorites_controller import FavoritesController
from ui.qt.inspector_controller import PhotoInspectorController
from ui.qt.photo_model import PhotoListModel, photo_identity
from ui.qt.previews import PreviewService
from ui.qt.project_controller import ProjectController
from ui.qt.thumbnails import ThumbnailService
from ui.qt.trash_controller import TrashController
from ui.qt.trash_data import project_trash_rows
from ui.qt.trash_model import TrashPhotoModel
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until


class FakeProcessing(QObject):
    sessionChanged = Signal()

    def __init__(self, blur):
        super().__init__()
        self.blurModel = blur


class FakeDuplicates(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.state = "COMPLETE"
        self.rows = []
        self.reloads = 0

    def review_rows(self):
        return self.rows

    def reload_reviews(self):
        self.reloads += 1
        self.rows = [dict(row, intents={}) for row in self.rows]
        self.changed.emit()


class LargeTrashFixture(QObject):
    def __init__(self, model):
        super().__init__()
        self._model = model

    @Property(QObject, constant=True)
    def photoModel(self):
        return self._model

    @Property(bool, constant=True)
    def loaded(self):
        return True

    @Property(bool, constant=True)
    def busy(self):
        return False

    @Property(int, constant=True)
    def count(self):
        return self._model.rowCount()

    @Property(int, constant=True)
    def readyCount(self):
        return self._model.rowCount()

    @Property(int, constant=True)
    def attentionCount(self):
        return 0

    @Property(str, constant=True)
    def error(self):
        return ""

    @Property(str, constant=True)
    def summary(self):
        return ""

    @Slot(str, result=bool)
    def isExecutable(self, _path):
        return True


class TrashDataTests(unittest.TestCase):
    def test_stale_review_save_does_not_erase_new_deletion_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "project"
            folder.mkdir()
            store = AutomaticStateStore(temporary)
            stale = store.load(folder)
            store.save_deletion_transaction(folder, DeletionTransactionRecord("2026-01-01T00:00:00Z", ["a"], ["b"]))
            store.save(stale)
            transactions = store.load(folder).deletion_transactions
            self.assertEqual(len(transactions), 1)
            self.assertEqual(transactions[0].paths, ["a"])
            self.assertEqual(transactions[0].failed_paths, ["b"])

    def test_aggregation_conflicts_and_recommendations_are_not_intents(self):
        a, b, c = "C:/a.jpg", "C:/b.jpg", "C:/c.jpg"
        decisions = {
            a: PersistedImageDecision(a, 10, 1, True, False, False, None, review_state="REMOVE"),
            b: PersistedImageDecision(b, 10, 1, False, False, True, None, review_state="KEEP"),
            c: PersistedImageDecision(c, 10, 1, False, False, False, None, review_state="REVIEW"),
        }
        groups = [dict(groupId="g", signature="valid", intents={a: "REMOVE", b: "REMOVE", c: "REMOVE"})]
        revisions = {path: (10, 1) for path in (a, b, c)}
        rows = {row["photoId"]: row for row in project_trash_rows((a, b, c), decisions, groups, {a}, revisions)}
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[a]["removalSources"], ["Blur", "Near duplicates"])
        self.assertFalse(rows[a]["executable"])
        self.assertTrue(rows[a]["favorite"])
        self.assertIn("Blur says Keep", rows[b]["conflictReason"])
        self.assertIn("Blur says Review", rows[c]["conflictReason"])
        self.assertEqual(project_trash_rows((a,), {}, [dict(groupId="g", signature="valid", intents={})],
                                            set(), revisions), [])

    def test_large_candidate_model_selects_only_ready_without_image_requests(self):
        app = QApplication.instance() or QApplication([])
        thumbnails = ThumbnailService()
        model = TrashPhotoModel(thumbnails)
        try:
            rows = [dict(photoId=f"C:/photos/{index:05}.jpg", filename=f"{index:05}.jpg",
                         removalSources=["Blur"], conflictReason="" if index % 2 else "Favorite",
                         executable=bool(index % 2), favorite=not bool(index % 2),
                         fileSize=10, badgeText="Blur") for index in range(6000)]
            model.set_rows(rows)
            self.assertEqual(model.rowCount(), 6000)
            model.selectAllReady()
            self.assertEqual(model.selectedCount, 3000)
            self.assertFalse(model.setSelected(rows[0]["photoId"], True))
            self.assertEqual(len(model._requests), 0)
            model.setFilter("Needs attention")
            self.assertEqual(model.rowCount(), 3000)
        finally:
            model.close()
            thumbnails.shutdown()

    def test_large_trash_route_keeps_delegates_and_thumbnails_bounded(self):
        configure_quick_controls_style()
        app = QApplication.instance() or QApplication([])
        capture_qt_warnings(self)
        thumbnails = ThumbnailService(decoder=lambda *_: "")
        model = TrashPhotoModel(thumbnails)
        rows = [dict(photoId=f"C:/photos/{index:05}.jpg", filename=f"{index:05}.jpg",
                     removalSources=["Blur"], conflictReason="", executable=True, favorite=False,
                     fileSize=10, badgeText="Blur") for index in range(6000)]
        model.set_rows(rows)
        fixture = LargeTrashFixture(model)
        engine = QQmlApplicationEngine()
        engine.setInitialProperties(dict(startPage="library", trashController=fixture))
        engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(engine.rootObjects())
        window = engine.rootObjects()[0]
        try:
            shell = window.findChild(QObject, "libraryShell")
            shell.showRoute("trash")
            for width, height in ((1280, 720), (1440, 900), (1920, 1080)):
                window.setWidth(width)
                window.setHeight(height)
                QTest.qWait(80)
                delegates, found = [], {}
                def visit(item):
                    if item.objectName() == "photoGridDelegate":
                        delegates.append(item)
                    if item.objectName() in ("trashGrid", "trashDecisionList"):
                        found[item.objectName()] = item
                    for child in item.childItems():
                        visit(child)
                visit(window.contentItem())
                self.assertGreater(len(delegates), 0)
                self.assertLess(len(delegates), 100)
                grid, detail = found["trashGrid"], found["trashDecisionList"]
                grid_left = grid.mapToScene(QPointF(0, 0)).x()
                detail_left = detail.mapToScene(QPointF(0, 0)).x()
                self.assertLessEqual(grid_left + grid.width(), detail_left + 1)
                self.assertLessEqual(len(thumbnails._pending), thumbnails.max_pending)
        finally:
            window.close()
            engine.deleteLater()
            app.processEvents()
            model.close()
            thumbnails.shutdown()


class TrashFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / "project-a", self.root / "project-b"
        self.a.mkdir(); self.b.mkdir()
        for folder in (self.a, self.b):
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                Image.new("RGB", (32, 24), "blue").save(folder / name)
        self.db_path = self.root / "room36_app.sqlite3"
        self.store = AutomaticStateStore(self.root)
        self.thumbnails = ThumbnailService()
        self.master = PhotoListModel(self.thumbnails)
        self.projects = ProjectController(self.master, registry_path=self.root / "projects.json",
                                          database_path=self.db_path)
        self.favorites = FavoritesController(self.projects, self.master, self.thumbnails,
                                             database_path=self.db_path)
        self.blur = BlurCandidateModel(self.thumbnails)
        self.processing = FakeProcessing(self.blur)
        self.duplicates = FakeDuplicates()
        self.previews = PreviewService()
        self.inspector = PhotoInspectorController(self.blur, self.previews)
        self.trash = TrashController(self.projects, self.master, self.processing, self.duplicates,
                                     self.favorites, self.inspector, self.thumbnails, state_root=self.root)
        self.projects.openFolder(str(self.a))
        wait_until(lambda: self.projects.activeProjectFolder == photo_identity(self.a)
                   and self.favorites.loaded and self.trash.loaded)

    def tearDown(self):
        self.trash.shutdown()
        self.favorites.shutdown()
        self.projects.shutdown()
        self.inspector.close()
        self.blur.close()
        self.master.close()
        self.previews.shutdown()
        self.thumbnails.shutdown()
        self.assertFalse(self.trash._worker.is_alive())
        self.temp.cleanup()

    def mark_blur(self, path, state="REMOVE"):
        path = photo_identity(path)
        stat = Path(path).stat()
        folder = Path(path).parent
        saved = self.store.load(folder)
        saved.image_decisions[path] = PersistedImageDecision(path, stat.st_size, stat.st_mtime_ns,
            state == "REMOVE", False, state == "KEEP", None, review_state=state)
        self.store.save(saved)
        self.blur.results[path] = dict(photoId=path, file_size=stat.st_size,
                                       mtime_ns=stat.st_mtime_ns, reviewState=state)
        self.blur.reviewChanged.emit(path)
        if state == "REMOVE":
            wait_until(lambda: self.trash.count >= 1)

    def mark_duplicate(self, path, intent="REMOVE"):
        path = photo_identity(path)
        stat = Path(path).stat()
        self.blur.results[path] = dict(photoId=path, file_size=stat.st_size,
                                       mtime_ns=stat.st_mtime_ns, reviewState="UNDECIDED")
        self.store.save_duplicate_review(self.a, "group", "signature", [photo_identity(self.a / "a.jpg")],
                                         {path: intent}, True)
        self.duplicates.rows = [dict(groupId="group", signature="signature", intents={path: intent})]
        self.duplicates.changed.emit()
        wait_until(lambda: self.trash.count >= 1)

    def test_blur_restore_and_unrelated_fields_preserved(self):
        path = photo_identity(self.a / "a.jpg")
        self.mark_blur(path)
        saved = self.store.load(self.a)
        saved.image_decisions[path].duplicate_group_id = "old-group"
        saved.image_decisions[path].is_duplicate_keeper = False
        self.store.save(saved)
        self.assertEqual(self.trash.count, 1)
        old_generation = self.projects._generation
        self.assertTrue(self.trash.restorePhoto(path))
        wait_until(lambda: not self.trash.busy and self.trash.count == 0)
        row = self.store.load(self.a).image_decisions[path]
        self.assertEqual(row.review_state, "UNDECIDED")
        self.assertFalse(row.selected_for_deletion)
        self.assertEqual(row.duplicate_group_id, "old-group")
        self.assertFalse(row.is_duplicate_keeper)
        self.assertTrue(Path(path).exists())
        self.assertEqual(self.projects._generation, old_generation)

    def test_duplicate_restore_preserves_keeper_and_group_state(self):
        path = photo_identity(self.a / "b.jpg")
        self.mark_duplicate(path)
        self.assertTrue(self.trash.restorePhoto(path))
        wait_until(lambda: not self.trash.busy and self.trash.count == 0)
        review = self.store.duplicate_reviews(self.a)["group"]
        self.assertEqual(review["keepers"], [photo_identity(self.a / "a.jpg")])
        self.assertEqual(review["intents"], {})
        self.assertFalse(review["reviewed"])
        self.assertTrue(Path(path).exists())

    def test_two_removal_sources_restore_together(self):
        path = photo_identity(self.a / "b.jpg")
        self.mark_blur(path)
        self.mark_duplicate(path)
        wait_until(lambda: len(self.trash.row(path).get("removalSources", [])) == 2)
        self.assertEqual(self.trash.count, 1)
        self.assertTrue(self.trash.restorePhoto(path))
        wait_until(lambda: not self.trash.busy and self.trash.count == 0)
        self.assertEqual(self.store.load(self.a).image_decisions[path].review_state, "UNDECIDED")
        self.assertEqual(self.store.duplicate_reviews(self.a)["group"]["intents"], {})
        self.assertTrue(Path(path).exists())

    def test_keep_and_review_conflicts_block_execution(self):
        for decision in ("KEEP", "REVIEW"):
            with self.subTest(decision=decision):
                path = photo_identity(self.a / "b.jpg")
                self.mark_blur(path, decision)
                self.mark_duplicate(path)
                wait_until(lambda: self.trash.count == 1 and not self.trash.isExecutable(path))
                self.assertIn("Decision conflict", self.trash.row(path)["conflictReason"])
                self.trash.photoModel.selectAllReady()
                self.assertEqual(self.trash.photoModel.selectedCount, 0)
                self.assertEqual(self.trash.prepareSelection(), {})
                self.duplicates.rows = []
                self.duplicates.changed.emit()
                wait_until(lambda: self.trash.count == 0)

    def test_favorite_conflict_blocks_until_unfavorite_and_project_switch(self):
        path = photo_identity(self.a / "a.jpg")
        self.mark_blur(path)
        self.assertTrue(self.favorites.setFavorite(path, True))
        wait_until(lambda: self.favorites.isFavorite(path) and not self.trash.isExecutable(path))
        self.assertEqual(self.trash.attentionCount, 1)
        self.assertEqual(self.trash.prepareSelection(), {})
        self.assertTrue(self.favorites.isFavorite(path))
        self.assertTrue(self.favorites.setFavorite(path, False))
        wait_until(lambda: self.trash.isExecutable(path))
        self.projects.openFolder(str(self.b))
        wait_until(lambda: self.projects.activeProjectFolder == photo_identity(self.b)
                   and self.trash.loaded)
        self.assertEqual(self.trash.count, 0)

    def test_removal_read_failure_is_local_and_blocks_execution(self):
        path = photo_identity(self.a / "a.jpg")
        self.mark_blur(path)
        with patch.object(AutomaticStateStore, "load", side_effect=sqlite3.OperationalError("read only")), patch(
                "ui.qt.trash_controller.log"):
            self.trash._schedule_refresh()
            wait_until(lambda: not self.trash.loaded and bool(self.trash.error))
        self.assertEqual(self.trash.count, -1)
        self.assertEqual(self.trash.photoModel.rowCount(), 0)
        self.assertFalse(self.trash.isExecutable(path))
        self.assertEqual(self.master.rowCount(), 3)

    def test_restore_write_failure_keeps_candidate_and_file(self):
        path = photo_identity(self.a / "a.jpg")
        self.mark_blur(path)
        with patch.object(AutomaticStateStore, "clear_removal_intents",
                          side_effect=sqlite3.OperationalError("read only")), patch(
                "ui.qt.trash_controller.log"):
            self.assertTrue(self.trash.restorePhoto(path))
            wait_until(lambda: not self.trash.busy and bool(self.trash.error))
        self.assertEqual(self.trash.count, 1)
        self.assertTrue(Path(path).exists())
        self.assertEqual(self.store.load(self.a).image_decisions[path].review_state, "REMOVE")

    def test_confirmation_revision_partial_failure_and_history(self):
        a, b = photo_identity(self.a / "a.jpg"), photo_identity(self.a / "b.jpg")
        self.mark_blur(a); self.mark_blur(b)
        wait_until(lambda: self.trash.count == 2)
        self.trash.photoModel.selectAllReady()
        self.assertEqual(self.trash.prepareSelection()["count"], 2)
        moved = []
        def fake_trash(path):
            moved.append(path)
            if path == b:
                raise OSError("denied")
            Path(path).rename(self.root / "moved.jpg")
        with patch("ui.qt.trash_controller.send2trash", side_effect=fake_trash), patch(
                "ui.qt.trash_controller.log"):
            self.assertTrue(self.trash.executeSelection())
            wait_until(lambda: not self.trash.busy and len(self.store.load(self.a).deletion_transactions) == 1)
        self.assertEqual(moved, [a, b])
        transaction = self.store.load(self.a).deletion_transactions[0]
        self.assertEqual(transaction.paths, [a])
        self.assertEqual(transaction.failed_paths, [b])
        wait_until(lambda: self.projects.activeProjectFolder == photo_identity(self.a)
                   and a not in self.master.identities())
        self.assertTrue(Path(b).exists())

    def test_changed_file_is_not_moved(self):
        path = photo_identity(self.a / "a.jpg")
        self.mark_blur(path)
        self.trash.photoModel.selectAllReady()
        self.assertEqual(self.trash.prepareSelection()["count"], 1)
        Path(path).write_bytes(b"changed")
        with patch("ui.qt.trash_controller.send2trash") as move:
            self.assertTrue(self.trash.executeSelection())
            wait_until(lambda: not self.trash.busy and len(self.store.load(self.a).deletion_transactions) == 1)
        move.assert_not_called()
        self.assertEqual(self.store.load(self.a).deletion_transactions[0].failed_paths, [path])
        wait_until(lambda: not self.trash.isExecutable(path))
        self.assertIn("Photo changed", self.trash.row(path)["conflictReason"])

    def test_missing_file_is_failed_and_excluded_after_refresh(self):
        path = photo_identity(self.a / "a.jpg")
        self.mark_blur(path)
        self.trash.photoModel.selectAllReady()
        self.trash.prepareSelection()
        Path(path).rename(self.root / "externally-moved.jpg")
        with patch("ui.qt.trash_controller.send2trash") as move:
            self.assertTrue(self.trash.executeSelection())
            wait_until(lambda: self.trash.loaded and self.trash.count == 0)
        move.assert_not_called()
        self.assertEqual(self.store.load(self.a).deletion_transactions[0].failed_paths, [path])

    def test_history_failure_does_not_retry_move_and_reports_separately(self):
        path = photo_identity(self.a / "a.jpg")
        self.mark_blur(path)
        self.trash.photoModel.selectAllReady()
        self.assertEqual(self.trash.prepareSelection()["count"], 1)
        moved = []
        def fake_trash(value):
            moved.append(value)
            Path(value).rename(self.root / "moved.jpg")
        with patch("ui.qt.trash_controller.send2trash", side_effect=fake_trash), patch(
                "ui.qt.trash_controller.log"), patch.object(
                AutomaticStateStore, "save_deletion_transaction",
                side_effect=sqlite3.OperationalError("read only")):
            self.assertTrue(self.trash.executeSelection())
            wait_until(lambda: path not in self.master.identities() and self.trash.loaded)
        self.assertEqual(moved, [path])
        self.assertIn("could not save the deletion history", self.trash.error)
        self.assertIn("1 photos moved", self.trash.summary)

    def test_project_switch_during_move_keeps_result_in_original_project(self):
        path = photo_identity(self.a / "a.jpg")
        self.mark_blur(path)
        self.trash.photoModel.selectAllReady()
        self.trash.prepareSelection()
        entered, release = threading.Event(), threading.Event()
        def fake_trash(value):
            entered.set()
            release.wait(3)
            Path(value).rename(self.root / "moved.jpg")
        try:
            with patch("ui.qt.trash_controller.send2trash", side_effect=fake_trash):
                self.assertTrue(self.trash.executeSelection())
                self.assertTrue(entered.wait(2))
                self.projects.openFolder(str(self.b))
                wait_until(lambda: self.projects.activeProjectFolder == photo_identity(self.b))
                release.set()
                wait_until(lambda: self.trash.loaded and
                           len(self.store.load(self.a).deletion_transactions) == 1)
                self.assertEqual(self.trash.count, 0)
                self.assertEqual(self.projects.activeProjectFolder, photo_identity(self.b))
        finally:
            release.set()

    def test_qml_route_and_confirmation_controls(self):
        path = photo_identity(self.a / "a.jpg")
        self.mark_blur(path)
        engine = QQmlApplicationEngine()
        engine.setInitialProperties(dict(startPage="library", projectController=self.projects,
            photoModel=self.master, favoritesController=self.favorites,
            inspectorController=self.inspector, trashController=self.trash))
        engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(engine.rootObjects())
        window = engine.rootObjects()[0]
        try:
            def item(name, within=None):
                def find(candidate):
                    if candidate.objectName() == name:
                        return candidate
                    for child in candidate.childItems():
                        found = find(child)
                        if found is not None:
                            return found
                    return None
                found = find(within or window.contentItem())
                self.assertIsNotNone(found, name)
                return found
            def click(name, within=None):
                QTest.qWait(50)
                target = item(name, within)
                point = target.mapToScene(QPointF(target.width() / 2, target.height() / 2)).toPoint()
                QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
                self.app.processEvents()
            click("libraryNav_trash")
            self.app.processEvents()
            self.assertIsNotNone(item("trashPage"))
            self.assertEqual(self.trash.count, 1)
            activations = []
            item("trashGrid").photoActivated.connect(activations.append)
            click("photoCardImage", item("trashGrid"))
            self.assertEqual(activations, [path])
            self.assertTrue(self.inspector.opened)
            click("inspectorBack")
            self.assertFalse(self.inspector.opened)
            click("trashSelectAllReady")
            self.assertEqual(self.trash.photoModel.selectedCount, 1)
            click("trashMoveSelected")
            dialog = window.findChild(QObject, "trashConfirmation")
            self.assertIsNotNone(dialog)
            self.assertTrue(dialog.property("visible"))
            dialog.close()
            self.assertFalse(dialog.property("visible"))
            QTest.qWait(300)  # Allow the modal overlay's exit transition to finish.
            self.assertEqual(self.trash.prepareSelection()["count"], 1)
            click("trashFilter")
            click("toolbarDropdownOption_Needs attention")
            self.assertEqual(self.trash.photoModel.rowCount(), 0)
            click("trashFilter")
            click("toolbarDropdownOption_All")
            self.assertEqual(self.trash.photoModel.rowCount(), 1)
            self.assertFalse(item("trashFilter").property("open"))
            click("trashRestorePhoto")
            wait_until(lambda: self.trash.count == 0 and not self.trash.busy)
            self.assertTrue(Path(path).exists())
        finally:
            window.close()
            engine.deleteLater()
