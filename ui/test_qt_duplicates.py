"""Phase 8A review uses published central results and never scans on navigation."""
from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_paths import qml_path
from automatic_scan.duplicate_selection import _group_id
from automatic_scan.state_store import AutomaticStateStore, PersistedImageDecision
from ui.qt.app import configure_quick_controls_style
from ui.qt.duplicate_controller import DuplicateReviewController
from ui.qt import duplicate_controller as duplicate_module
from ui.qt.duplicate_data import group_revision, project_duplicate_rows
from ui.qt.duplicate_model import DuplicateGroupModel
from ui.qt.inspector_controller import PhotoInspectorController
from ui.qt.photo_model import PhotoListModel
from ui.qt.previews import PreviewService
from ui.qt.processing_controller import ProjectProcessingController
from ui.qt.thumbnails import ThumbnailService
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until
from ui import test_qml_project_flow as phase5


class DuplicateDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_pending_ranking_enrichment_identity_and_revision(self):
        paths = ["a.jpg", "b.jpg", "c.jpg"]
        revisions = {path: (100, 1) for path in paths}
        group_id = _group_id(paths)
        pending = project_duplicate_rows([paths], None, revisions, {})[0]
        self.assertEqual((pending["groupId"], pending["photoCount"],
                          pending["recommendationState"]), (group_id, 3, "pending"))
        ranking = [dict(group_id=group_id, image_paths=list(reversed(paths)),
                        keeper_paths=[paths[1], paths[2]], excluded_paths=[paths[0]],
                        ranking={paths[1]: dict(explanation=["Closer to the duplicate group center"])})]
        ready = project_duplicate_rows([paths], ranking, revisions, {})[0]
        self.assertEqual(ready["currentKeepers"], ["b.jpg", "c.jpg"])
        self.assertEqual(ready["representativePhotoId"], "b.jpg")
        self.assertEqual(ready["explanation"], ["Closer to the duplicate group center"])
        saved = {group_id: dict(signature=ready["signature"], keepers=["a.jpg", "c.jpg"],
                                intents={"b.jpg": "REMOVE"}, reviewed=True)}
        restored = project_duplicate_rows([paths], ranking, revisions, saved)[0]
        self.assertTrue(restored["reviewed"])
        self.assertEqual(restored["currentKeepers"], ["a.jpg", "c.jpg"])
        changed = dict(revisions, **{"a.jpg": (101, 2)})
        invalidated = project_duplicate_rows([paths], ranking, changed, saved)[0]
        self.assertFalse(invalidated["reviewed"])
        self.assertEqual(invalidated["currentKeepers"], ["b.jpg", "c.jpg"])
        self.assertNotEqual(group_revision(paths, revisions), group_revision(paths, changed))

    def test_model_filter_sort_and_large_list_without_eager_thumbnails(self):
        service = ThumbnailService(decoder=lambda *_: "")
        model = DuplicateGroupModel(service)
        try:
            rows = [dict(groupId=f"g{i:04}", photoCount=2 + i % 4,
                         representativePhotoId=f"unused{i}.jpg", reviewed=(i % 2 == 0),
                         reviewState="Reviewed" if i % 2 == 0 else "Needs review",
                         recommendationState="ready", recommendedKeeperCount=1,
                         removeCount=0, signature="same")
                    for i in range(2000)]
            model.set_rows(rows)
            self.assertEqual((model.totalCount, model.visibleCount, model.needsReviewCount),
                             (2000, 1000, 1000))
            self.assertEqual(service.pending_count, 0)
            model.setFilter("Reviewed")
            self.assertEqual(model.visibleCount, 1000)
            model.setSort("Smallest groups")
            self.assertEqual(model.data(model.index(0), model.ROLE["photoCount"]), 2)
            model.set_rows([])
            self.assertEqual((model.totalCount, model.visibleCount), (0, 0))
        finally:
            model.close()
            service.shutdown()


class DuplicateFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_quick_controls_style()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "project-a"
        self.folder.mkdir()
        self.paths = [self.folder / name for name in ("a.jpg", "b.jpg", "c.jpg")]
        for path in self.paths:
            Image.new("RGB", (80, 60), "gray").save(path)
        self.group_id = _group_id([str(path) for path in self.paths])
        self.state_root = self.root / "state"
        self.store = AutomaticStateStore(self.state_root)
        self.thumbnails = ThumbnailService()
        self.photos = PhotoListModel(self.thumbnails)
        self.processing = ProjectProcessingController(self.thumbnails, state_root=self.state_root)
        self.duplicates = DuplicateReviewController(self.processing, self.thumbnails,
                                                     state_root=self.state_root)
        self.previews = PreviewService()
        self.inspector = PhotoInspectorController(self.processing.blurModel, self.previews)
        self.processing.sessionChanged.connect(self.inspector.close)
        self.engine = QQmlApplicationEngine()
        self.engine.setInitialProperties(dict(startPage="library", photoModel=self.photos,
            processingController=self.processing, duplicateController=self.duplicates,
            inspectorController=self.inspector))
        self.engine.load(qml_path("App.qml").as_uri())
        self.assertTrue(self.engine.rootObjects())
        self.window = self.engine.rootObjects()[0]
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.engine.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.inspector.close()
        self.duplicates.shutdown()
        self.processing.shutdown()
        self.photos.close()
        self.previews.shutdown()
        self.thumbnails.shutdown()
        self.assertFalse(self.duplicates._worker.is_alive())
        self.temp.cleanup()

    items = phase5.ProjectFlowTests.items
    item = phase5.ProjectFlowTests.item
    click = phase5.ProjectFlowTests.click

    def click_object(self, item):
        QTest.qWait(50)
        position = item.mapToScene(QPointF(item.width() / 2, item.height() / 2))
        QTest.mouseClick(self.window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         position.toPoint())
        self.app.processEvents()

    def publish(self, *, ranked=True, folder=None):
        self.processing.invalidate()
        self.processing._folder = str(folder or self.folder)
        rows = []
        for path in self.paths:
            stat = path.stat()
            rows.append(dict(photoId=str(path), filename=path.name,
                             file_size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                             status="Sharp", final_score=0.8, reviewState="UNDECIDED"))
        self.processing.blurModel.set_results(rows)
        self.processing._stages["duplicates"] = dict(state="COMPLETE", done=0, total=0, error="")
        self.processing.results["duplicates"] = [[str(path) for path in self.paths]]
        if ranked:
            self.processing._stages["ranking"] = dict(state="COMPLETE", done=0, total=0, error="")
            self.processing.results["ranking"] = [dict(group_id=self.group_id,
                image_paths=[str(path) for path in self.paths],
                keeper_paths=[str(self.paths[0])],
                excluded_paths=[str(path) for path in self.paths[1:]],
                ranking={str(self.paths[0]): dict(explanation=["Sharper important faces"])})]
        self.processing.changed.emit()
        self.app.processEvents()
        self.assertEqual(self.duplicates.state, "COMPLETE")

    def test_qml_route_group_inspector_keeper_confirm_filter_and_sort(self):
        self.publish()
        session = self.processing.sessionId
        self.assertTrue(self.processing._tasks.empty())
        self.assertEqual(self.item("libraryNav_duplicates").property("count"), 1)
        self.click("libraryNav_duplicates")
        self.assertEqual(self.processing.sessionId, session)
        self.assertTrue(self.processing._tasks.empty())
        self.assertTrue(self.items("nearDuplicatesPage"))
        self.assertEqual(self.item("duplicateGroupGrid").property("count"), 1)
        self.click("duplicateGroupOpen")
        self.assertEqual(self.duplicates.selectedGroup["groupId"], self.group_id)
        self.assertEqual(self.item("duplicateDetailGrid").property("count"), 3)
        self.click("gridPhotoCard")
        self.assertTrue(self.inspector.opened)
        self.click("inspectorBack")
        self.assertFalse(self.inspector.opened)
        self.assertEqual(self.duplicates.selectedGroup["groupId"], self.group_id)
        self.click_object(self.items("duplicateSetKeeper")[1])
        wait_until(lambda: not self.duplicates.busy)
        self.assertEqual(self.duplicates.selectedGroup["currentKeepers"], [str(self.paths[1])])
        self.click("duplicateConfirm")
        wait_until(lambda: not self.duplicates.busy)
        self.assertTrue(self.duplicates.selectedGroup["reviewed"])
        self.assertEqual(self.duplicates.needsReviewCount, 0)
        self.assertEqual(self.item("libraryNav_duplicates").property("count"), 0)
        self.click("duplicateBack")
        self.click("duplicateFilter")
        self.click("toolbarDropdownOption_Reviewed")
        self.assertEqual(self.duplicates.groupModel.groupFilter, "Reviewed")
        self.click("librarySortButton")
        self.click("toolbarDropdownOption_Smallest groups")
        self.assertEqual(self.duplicates.groupModel.sortOrder, "Smallest groups")
        self.assertEqual(self.item("duplicateGroupGrid").property("count"), 1)
        self.assertEqual(self.processing.sessionId, session)

    def test_manual_review_persists_and_restore_does_not_delete_or_change_blur(self):
        self.publish()
        self.duplicates.openGroup(self.group_id)
        self.assertTrue(self.duplicates.setKeeper(str(self.paths[1])))
        wait_until(lambda: not self.duplicates.busy)
        self.assertTrue(self.duplicates.confirmGroup())
        wait_until(lambda: not self.duplicates.busy)
        saved = self.store.duplicate_reviews(self.folder)[self.group_id]
        self.assertEqual(saved["keepers"], [str(self.paths[1])])
        self.assertEqual(saved["intents"][str(self.paths[0])], "REMOVE")
        self.assertTrue(saved["reviewed"])
        reopened_reviewed = DuplicateReviewController(self.processing, self.thumbnails,
                                                      state_root=self.state_root)
        try:
            self.assertTrue(reopened_reviewed.groupModel.get_row(self.group_id)["reviewed"])
            self.assertEqual(reopened_reviewed.needsReviewCount, 0)
        finally:
            reopened_reviewed.shutdown()
        self.assertTrue(all(path.exists() for path in self.paths))
        self.assertTrue(self.duplicates.setIntent(str(self.paths[0]), "UNDECIDED"))
        wait_until(lambda: not self.duplicates.busy)
        self.assertEqual(self.duplicates.selectedGroup["intents"].get(str(self.paths[0])), None)
        self.assertFalse(self.duplicates.selectedGroup["reviewed"])
        self.assertEqual(self.processing.blurModel.results[str(self.paths[0])]["reviewState"], "UNDECIDED")
        self.publish()
        self.assertEqual(self.duplicates.groupModel.get_row(self.group_id)["currentKeepers"],
                         [str(self.paths[1])])
        self.assertFalse(self.duplicates.groupModel.get_row(self.group_id)["reviewed"])
        reopened = DuplicateReviewController(self.processing, self.thumbnails,
                                             state_root=self.state_root)
        try:
            self.assertEqual(reopened.groupModel.get_row(self.group_id)["currentKeepers"],
                             [str(self.paths[1])])
            self.assertFalse(reopened.groupModel.get_row(self.group_id)["reviewed"])
        finally:
            reopened.shutdown()

    def test_explicit_blur_keep_blocks_bulk_remove_without_overwrite(self):
        self.publish()
        keep = self.paths[2]
        stat = keep.stat()
        state = self.store.load(self.folder)
        state.image_decisions[str(keep)] = PersistedImageDecision(str(keep), stat.st_size,
            stat.st_mtime_ns, False, False, True, "keep", review_state="KEEP")
        self.store.save(state)
        self.processing.blurModel.results[str(keep)]["reviewState"] = "KEEP"
        self.duplicates.openGroup(self.group_id)
        self.assertFalse(self.duplicates.confirmGroup())
        self.assertIn("conflicting", self.duplicates.error)
        self.assertEqual(self.store.load(self.folder).image_decisions[str(keep)].review_state, "KEEP")
        self.assertEqual(self.store.duplicate_reviews(self.folder), {})

    def test_retry_and_project_switch_discard_old_review_result(self):
        self.publish()
        self.duplicates.openGroup(self.group_id)
        old_session = self.processing.sessionId
        old_cancellation = self.duplicates._canceled
        self.processing._overall = "FAILED"
        self.processing.retry()
        self.assertEqual(self.processing.sessionId, old_session + 1)
        self.assertTrue(old_cancellation.is_set())
        self.assertEqual(self.duplicates.selectedGroup, {})
        self.assertEqual(self.duplicates.groupModel.totalCount, 0)
        self.duplicates._results.put((old_session, str(self.folder), self.group_id, "stale",
                                      [str(self.paths[1])], {}, True, ""))
        self.duplicates._drain()
        self.assertEqual(self.duplicates.groupModel.totalCount, 0)
        other = self.root / "project-b"
        other.mkdir()
        self.publish(folder=other)
        self.assertEqual(self.duplicates.needsReviewCount, 1)
        self.assertFalse(self.duplicates.groupModel.get_row(self.group_id)["reviewed"])

    def test_changed_photo_invalidates_saved_review(self):
        self.publish()
        self.duplicates.openGroup(self.group_id)
        self.assertTrue(self.duplicates.confirmGroup())
        wait_until(lambda: not self.duplicates.busy)
        Image.new("RGB", (90, 60), "blue").save(self.paths[0])
        self.publish()
        self.assertFalse(self.duplicates.groupModel.get_row(self.group_id)["reviewed"])

    def test_changed_photo_during_save_reports_safe_error(self):
        self.publish()
        self.duplicates.openGroup(self.group_id)
        with patch.object(AutomaticStateStore, "save_duplicate_review") as save:
            Image.new("RGB", (90, 60), "blue").save(self.paths[0])
            self.assertTrue(self.duplicates.confirmGroup())
            wait_until(lambda: not self.duplicates.busy)
            save.assert_not_called()
        self.assertIn("changed", self.duplicates.error)
        self.assertEqual(self.store.duplicate_reviews(self.folder), {})

    def test_unwritable_database_reports_safe_error_without_mutating_group(self):
        self.publish()
        self.duplicates.openGroup(self.group_id)
        with patch.object(AutomaticStateStore, "save_duplicate_review", side_effect=OSError("disk")):
            with patch.object(duplicate_module.log, "exception") as logged:
                self.assertTrue(self.duplicates.confirmGroup())
                wait_until(lambda: not self.duplicates.busy)
                logged.assert_called_once()
        self.assertIn("could not be saved", self.duplicates.error)
        self.assertFalse(self.duplicates.selectedGroup["reviewed"])
        self.assertEqual(self.store.duplicate_reviews(self.folder), {})

    def test_inflight_project_a_write_cannot_populate_project_b(self):
        self.publish()
        self.duplicates.openGroup(self.group_id)
        entered, release = threading.Event(), threading.Event()
        original = AutomaticStateStore.save_duplicate_review
        def delayed(store, *args):
            entered.set()
            release.wait(2)
            return original(store, *args)
        with patch.object(AutomaticStateStore, "save_duplicate_review", delayed):
            self.assertTrue(self.duplicates.confirmGroup())
            wait_until(entered.is_set)
            other = self.root / "project-b"
            other.mkdir()
            self.publish(folder=other)
            release.set()
            wait_until(lambda: self.duplicates._results.qsize() > 0)
            self.duplicates._drain()
            self.assertEqual(self.duplicates.needsReviewCount, 1)
            self.assertFalse(self.duplicates.groupModel.get_row(self.group_id)["reviewed"])
            self.assertEqual(self.store.duplicate_reviews(other), {})

    def test_qml_remove_restore_and_keeper_count_of_two(self):
        self.publish()
        self.processing.results["ranking"] = [dict(group_id=self.group_id,
            image_paths=[str(path) for path in self.paths],
            keeper_paths=[str(self.paths[0]), str(self.paths[1])],
            excluded_paths=[str(self.paths[2])], ranking={})]
        self.processing.changed.emit()
        self.click("libraryNav_duplicates")
        self.click("duplicateGroupOpen")
        self.assertEqual(len(self.duplicates.selectedGroup["currentKeepers"]), 2)
        self.click_object(self.items("duplicateToggleRemove")[2])
        wait_until(lambda: not self.duplicates.busy)
        self.assertEqual(self.duplicates.selectedGroup["intents"][str(self.paths[2])], "REMOVE")
        self.click_object(self.items("duplicateToggleRemove")[2])
        wait_until(lambda: not self.duplicates.busy)
        self.assertNotIn(str(self.paths[2]), self.duplicates.selectedGroup["intents"])
        self.assertTrue(self.duplicates.setKeeper(str(self.paths[2])))
        wait_until(lambda: not self.duplicates.busy)
        self.assertEqual(len(self.duplicates.selectedGroup["currentKeepers"]), 2)
        self.assertIn(str(self.paths[2]), self.duplicates.selectedGroup["currentKeepers"])

    def test_processing_empty_failure_and_pending_recommendation_states(self):
        self.click("libraryNav_duplicates")
        self.assertIn("Finding near duplicates", self.item("duplicateStateText").property("text"))
        self.publish(ranked=False)
        self.assertIn("Choosing recommended keepers", self.item("duplicateStateText").property("text"))
        self.assertEqual(self.duplicates.groupModel.get_row(self.group_id)["recommendationState"], "pending")
        self.processing._stages["ranking"] = dict(state="FAILED", done=0, total=0, error="No ranking")
        self.processing.changed.emit()
        self.assertEqual(self.duplicates.groupModel.get_row(self.group_id)["recommendationState"], "unavailable")
        self.assertIn("unavailable", self.item("duplicateStateText").property("text"))
        self.processing._stages["duplicates"] = dict(state="FAILED", done=0, total=0,
            error="Duplicate analysis exceeded its safe candidate limit.")
        self.processing.changed.emit()
        self.assertIn("safe candidate limit", self.item("duplicateStateText").property("text"))
        self.assertEqual(self.duplicates.groupModel.totalCount, 0)
        self.processing._stages["duplicates"] = dict(state="COMPLETE", done=0, total=0, error="")
        self.processing.results["duplicates"] = []
        self.processing.changed.emit()
        self.assertIn("No near-duplicate groups", self.item("duplicateStateText").property("text"))

    def test_large_qml_grid_keeps_delegates_and_preview_queue_bounded(self):
        self.publish()
        rows = [dict(groupId=f"g{i:04}", photoCount=2, representativePhotoId=str(self.paths[0]),
                     reviewed=False, reviewState="Needs review", recommendationState="pending",
                     recommendedKeeperCount=0, removeCount=0, signature=str(i)) for i in range(2000)]
        self.duplicates.groupModel.set_rows(rows)
        self.click("libraryNav_duplicates")
        self.app.processEvents()
        self.assertLess(len(self.items("duplicateGroupDelegate")), 100)
        self.assertLessEqual(self.thumbnails.pending_count, self.thumbnails.max_pending)
        for width, height in ((1280, 720), (1440, 900), (1920, 1080)):
            self.window.resize(width, height)
            self.app.processEvents()
            self.assertLess(len(self.items("duplicateGroupDelegate")), 100)
            grid = self.item("duplicateGroupGrid")
            card = self.item("duplicateGroupCard")
            self.assertLessEqual(card.width(), grid.property("cellWidth"))
            self.assertLessEqual(card.height(), grid.property("cellHeight"))

    def test_detail_photo_grid_and_actions_fit_supported_sizes(self):
        self.publish()
        self.click("libraryNav_duplicates")
        self.click("duplicateGroupOpen")
        for width, height in ((1280, 720), (1440, 900), (1920, 1080)):
            self.window.resize(width, height)
            self.app.processEvents()
            page = self.item("nearDuplicatesPage")
            photos = self.item("duplicateDetailGrid")
            actions = self.item("duplicateMemberList")
            self.assertGreaterEqual(actions.x(), photos.x() + photos.width() - 1)
            self.assertLessEqual(actions.x() + actions.width(), page.width() + 1)
