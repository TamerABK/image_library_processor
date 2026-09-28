from pathlib import Path
import threading
import unittest

from PySide6.QtCore import QModelIndex, QPersistentModelIndex
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QAbstractItemModelTester

from ui.models import ResultItem
from ui.qt.photo_model import PhotoListModel, photo_identity
from ui.qt.thumbnails import ThumbnailService
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until


class PhotoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.service = ThumbnailService(decoder=lambda path, edge: "thumb:" + path)
        self.model = PhotoListModel(self.service)
        self.addCleanup(self.service.shutdown)
        self.addCleanup(self.model.close)
        self.tester = QAbstractItemModelTester(self.model, QAbstractItemModelTester.FailureReportingMode.Warning)
        self.items = [ResultItem(Path("b.jpg"), "B", "detail", badge_text="badge", person_id=7),
                      ResultItem(Path("a.jpg"), "A", "")]
        self.model.replace_items(self.items)
        self.changes = []
        self.resets = []
        self.model.dataChanged.connect(lambda first, last, roles: self.changes.append((first.row(), last.row(), roles)))
        self.model.modelReset.connect(lambda: self.resets.append(True))

    def value(self, row, name):
        return self.model.data(self.model.index(row), self.model.ROLE[name])

    def test_roles_identity_and_invalid_items(self):
        self.assertEqual(2, self.model.rowCount())
        self.assertEqual(0, self.model.rowCount(self.model.index(0)))
        self.assertEqual(set(PhotoListModel.ROLE_NAMES), {bytes(name).decode() for name in self.model.roleNames().values()})
        self.assertEqual(photo_identity("b.jpg"), self.value(0, "photoId"))
        self.assertEqual("badge", self.value(0, "badgeText"))
        self.assertEqual(7, self.value(0, "personId"))
        self.assertIsNone(self.model.data(QModelIndex()))
        self.assertFalse(self.model.setSelected("absent", True))
        self.assertFalse(self.model.removePhoto("absent"))
        self.assertEqual(0, self.model.acquireThumbnail("absent", 64))

    def test_selection_and_sort_preserve_identity_without_reset(self):
        identity = self.value(0, "photoId")
        persistent = QPersistentModelIndex(self.model.index(0))
        self.model.setSelected(identity, True)
        self.assertEqual([(0, 0, [self.model.ROLE["selected"]])], self.changes)
        self.model.setSelected(identity, True)
        self.assertEqual(1, len(self.changes))
        self.model.sortByFilename()
        self.assertEqual(1, persistent.row())
        self.assertTrue(self.value(1, "selected"))
        self.assertEqual(identity, self.value(1, "photoId"))
        self.assertEqual([], self.resets)
        self.model.replace_items(reversed(self.items))
        self.assertTrue(self.value(1, "selected"))

    def test_insert_remove_and_explicit_refresh_notifications(self):
        inserts, removes = [], []
        self.model.rowsInserted.connect(lambda parent, first, last: inserts.append((first, last)))
        self.model.rowsRemoved.connect(lambda parent, first, last: removes.append((first, last)))
        self.model.append_items([self.items[0], ResultItem(Path("c.jpg"), "C", "")])
        self.assertEqual([(2, 2)], inserts)
        self.assertTrue(self.model.removePhoto(photo_identity("a.jpg")))
        self.assertEqual([(1, 1)], removes)
        self.assertEqual([], self.resets)
        self.model.replace_items([self.items[0], self.items[0]])
        self.assertEqual(1, self.model.rowCount())
        self.assertEqual([True], self.resets)

    def test_thumbnail_completion_changes_only_one_row_and_release_drops_source(self):
        owner = threading.get_ident()
        notifications = []
        self.model.dataChanged.connect(lambda *_: notifications.append(threading.get_ident()))
        identity = self.value(0, "photoId")
        lease = self.model.acquireThumbnail(identity, 64)
        wait_until(lambda: self.value(0, "thumbnailState") == "ready")
        self.assertEqual("thumb:" + identity, self.value(0, "thumbnailSource"))
        self.assertEqual("idle", self.value(1, "thumbnailState"))
        self.assertTrue(all(first == last == 0 for first, last, roles in self.changes))
        self.assertEqual({owner}, set(notifications))
        self.assertEqual([], self.resets)
        self.model.releaseThumbnail(lease)
        self.assertEqual("", self.value(0, "thumbnailSource"))

    def test_refresh_and_removal_reject_late_results(self):
        identity = self.value(0, "photoId")
        self.model.acquireThumbnail(identity, 64)
        self.model.replace_items(self.items)
        # First request ticket is canceled. Even forced late delivery is ignored.
        self.service.completed.emit(1, "stale", "ready")
        self.assertEqual("", self.value(0, "thumbnailSource"))
        self.model.acquireThumbnail(identity, 64)
        self.model.removePhoto(identity)
        self.service.completed.emit(2, "stale", "ready")
        self.assertEqual("", self.value(0, "thumbnailSource"))

    def test_destroying_model_cancels_orphaned_requests(self):
        from shiboken6 import delete
        other = PhotoListModel(self.service)
        other.replace_items(self.items)
        other.acquireThumbnail(photo_identity("b.jpg"), 64)
        self.assertEqual(1, self.service.pending_count)
        delete(other)
        self.assertEqual(0, self.service.pending_count)

    def test_worker_cannot_mutate_model(self):
        errors = []
        def mutate():
            try:
                self.model.setSelected(photo_identity("b.jpg"), True)
            except RuntimeError as error:
                errors.append(str(error))
        worker = threading.Thread(target=mutate)
        worker.start()
        worker.join()
        self.assertEqual(1, len(errors))
        self.assertFalse(self.value(0, "selected"))
