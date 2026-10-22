from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app_paths import qml_path
from automatic_scan.state_store import AutomaticStateStore
from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_detector.face_database_sqlite import SQLiteFaceDatabase
from face_processing.cache import FaceScanCache
from face_processing.models import EmbeddedFace
from ui.qt.app import configure_quick_controls_style
from ui.qt.face_previews import FacePreviewService, decode_face_preview
from ui.qt.people_model import PeopleListModel
from ui.qt.inspector_controller import PhotoInspectorController
from ui.qt.people_controller import PeopleController, validate_person_name
from ui.qt import people_controller as people_module
from ui.qt.people_data import project_people_rows, stable_cluster_id
from ui.qt.photo_model import PhotoListModel
from ui.qt.previews import PreviewService
from ui.qt.processing_controller import ProjectProcessingController
from ui.qt.project_controller import ProjectController
from ui.qt.thumbnails import ThumbnailService
from ui.phase6_test_support import FixtureRunner
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until
from ui import test_qml_project_flow as phase5


def vector(index):
    value = np.zeros(512, dtype=np.float32)
    value[index] = 1
    return value


class PeopleDataTests(unittest.TestCase):
    def test_stable_cluster_key_ignores_transient_label_and_face_order(self):
        faces = [dict(path="C:/project/a.jpg", bbox=[1, 2, 20, 30]),
                 dict(path="C:/project/b.jpg", bbox=[3, 4, 15, 25])]
        first = dict(id=2, faces=faces)
        second = dict(id=200, faces=list(reversed(faces)))
        self.assertEqual(stable_cluster_id(first), stable_cluster_id(second))
        self.assertNotEqual(stable_cluster_id(first), stable_cluster_id(
            dict(id=2, faces=[faces[0]])))

    def test_known_grouping_representative_counts_and_sort(self):
        payload = {"known": [dict(path="a.jpg", person_id=2, bbox=[0, 0, 20, 20]),
                             dict(path="a.jpg", person_id=2, bbox=[2, 2, 30, 30]),
                             dict(path="b.jpg", person_id=2, bbox=[1, 1, 25, 25]),
                             dict(path="c.jpg", person_id=1, bbox=[0, 0, 10, 10])],
                   "unknown": []}
        known, _ = project_people_rows(payload, {1: "Amy", 2: "Zoë"}, set())
        self.assertEqual([row["name"] for row in known], ["Amy", "Zoë"])
        self.assertEqual((known[1]["faceCount"], known[1]["photoCount"]), (3, 2))
        self.assertEqual(known[1]["representativeBBox"], [2, 2, 30, 30])

    def test_unknown_counts_representative_and_ignored_flag(self):
        faces = [dict(path="a.jpg", bbox=[1, 2, 20, 30]),
                 dict(path="a.jpg", bbox=[35, 2, 20, 30]),
                 dict(path="b.jpg", bbox=[1, 2, 20, 30])]
        cluster = dict(id=88, faces=faces, representative="b.jpg",
                       representative_bbox=[1, 2, 20, 30])
        key = stable_cluster_id(cluster)
        _, unknown = project_people_rows({"known": [], "unknown": [cluster]}, {}, {key})
        self.assertEqual((unknown[0]["faceCount"], unknown[0]["photoCount"]), (3, 2))
        self.assertEqual(unknown[0]["representativePhotoId"], "b.jpg")
        self.assertTrue(unknown[0]["ignored"])

    def test_name_validation_accepts_unicode_and_rejects_invalid(self):
        self.assertEqual(validate_person_name("  Zoë 李  "), "Zoë 李")
        for value in (" ", "a\nb", "a\u0085b", "x" * 101):
            with self.assertRaises(ValueError):
                validate_person_name(value)

    def test_atomic_database_assignment_reuses_casefolded_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = SQLiteFaceDatabase(Path(temporary) / "faces.sqlite3", CosineEmbeddingSimilarity())
            first = database.assign_embeddings([vector(0)], name="Élodie")
            second = database.assign_embeddings([vector(1)], name="élodie")
            self.assertEqual(first.id, second.id)
            self.assertEqual(len(database._embeddings()), 2)
            self.assertEqual(len(database.list_people()), 1)
            with self.assertRaises(ValueError):
                database.assign_embeddings([vector(2)], person_id=999)
            self.assertEqual(len(database._embeddings()), 2)
            canceled = threading.Event()
            canceled.set()
            with self.assertRaises(ValueError):
                database.assign_embeddings([vector(3)], name="Canceled", canceled=canceled)
            self.assertEqual(len(database.list_people()), 1)
            self.assertEqual(len(database._embeddings()), 2)


class FacePreviewTests(unittest.TestCase):
    def test_crop_padding_clamping_orientation_and_bad_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "face.jpg"
            image = Image.new("RGB", (120, 80), "red")
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            request = json.dumps([str(path), [70, 10, 80, 50]])
            source = decode_face_preview(request, 64)
            self.assertTrue(source.startswith("data:image/png;base64,"))
            crop = Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1])))
            self.assertLessEqual(max(crop.size), 64)
            self.assertGreater(min(crop.size), 0)
            self.assertLess(crop.width / crop.height, 0.5)  # EXIF-rotated coordinate space.
            self.assertEqual(decode_face_preview(json.dumps([str(path) + "-missing", [1, 1, 2, 2]]), 64), "")
            corrupt = Path(temporary) / "bad.jpg"
            corrupt.write_bytes(b"invalid")
            self.assertEqual(decode_face_preview(json.dumps([str(corrupt), [1, 1, 2, 2]]), 64), "")

    def test_bounded_cache_queue_and_cancellation(self):
        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "photo.jpg"
            Image.new("RGB", (120, 80), "red").save(path)
            service = FacePreviewService()
            try:
                self.assertEqual(service.max_pending, 64)
                tickets = [service.request_face(str(path), [10, 10, 30, 30], 128, str(i))
                           for i in range(80)]
                self.assertLessEqual(sum(bool(ticket) for ticket in tickets), 64)
                for ticket in tickets:
                    if ticket:
                        service.cancel(ticket)
                wait_until(lambda: service.pending_count == 0)
                self.assertLessEqual(service.cache_entries, 96)
            finally:
                service.shutdown()
            self.assertEqual(service.workers_alive, 0)


class PeopleModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.previews = FacePreviewService()
        self.addCleanup(self.previews.shutdown)

    def test_known_search_sort_and_project_reset(self):
        model = PeopleListModel(self.previews, "known")
        rows = [dict(key="1", name="Zoë", personId=1, faceCount=8, photoCount=2,
                     representativePhotoId="z.jpg", representativeBBox=[1, 1, 10, 10]),
                dict(key="2", name="Ana", personId=2, faceCount=2, photoCount=5,
                     representativePhotoId="a.jpg", representativeBBox=[1, 1, 10, 10])]
        model.set_rows(rows, revision="project-a")
        self.assertEqual(model.data(model.index(0), model.ROLE["name"]), "Ana")
        model.setSort("Most faces")
        self.assertEqual(model.data(model.index(0), model.ROLE["name"]), "Zoë")
        model.setSearch("AN")
        self.assertEqual(model.visibleCount, 1)
        self.assertEqual(model.data(model.index(0), model.ROLE["name"]), "Ana")
        model.set_rows([], revision="project-b")
        self.assertEqual((model.totalCount, model.visibleCount), (0, 0))
        model.close()

    def test_unknown_filter_sort_and_stale_preview_completion(self):
        model = PeopleListModel(self.previews, "unknown")
        rows = [dict(key="small", clusterId="small", name="", faceCount=1, photoCount=1,
                     representativePhotoId="missing.jpg", representativeBBox=[1, 1, 10, 10], ignored=False),
                dict(key="large", clusterId="large", name="", faceCount=5, photoCount=3,
                     representativePhotoId="missing2.jpg", representativeBBox=[1, 1, 10, 10], ignored=True)]
        model.set_rows(rows, revision="project-a")
        self.assertEqual(model.needsReviewCount, 1)
        self.assertEqual(model.visibleCount, 1)
        lease = model.acquirePreview("small", 128)
        ticket = model.get_row("small")["ticket"]
        self.assertGreater(lease, 0)
        model.set_rows(rows, revision="project-b")
        model._completed(ticket, "old-image", "ready")
        self.assertEqual(model.get_row("small")["previewSource"], "")
        model.setFilter("All")
        self.assertEqual(model.data(model.index(0), model.ROLE["key"]), "large")
        model.setSort("Smallest clusters")
        self.assertEqual(model.data(model.index(0), model.ROLE["key"]), "small")
        self.assertEqual(model.data(model.index(0), model.ROLE["clusterId"]), "small")
        model.setFilter("Ignored")
        self.assertEqual(model.visibleCount, 1)
        model.close()


class CentralPipelinePeopleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_project_discovery_publishes_people_without_route_scan(self):
        capture_qt_warnings(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "photos"
            folder.mkdir()
            Image.new("RGB", (80, 60), "gray").save(folder / "photo.jpg")
            store = AutomaticStateStore(root / "state")
            runner = FixtureRunner(root, store)
            thumbnails = ThumbnailService()
            face_previews = FacePreviewService()
            photos = PhotoListModel(thumbnails)
            processing = ProjectProcessingController(thumbnails,
                cache_path=root / "analysis.sqlite3", state_root=root / "state", runner=runner)
            people = PeopleController(processing, face_previews, thumbnails,
                cache_path=root / "analysis.sqlite3", face_database_path=root / "faces.sqlite3",
                state_root=root / "state")
            projects = ProjectController(photos, registry_path=root / "projects.json", processing=processing)
            try:
                projects.openFolder(str(folder))
                wait_until(lambda: not projects.busy)
                wait_until(lambda: people.state == "COMPLETE", timeout=6)
                self.assertEqual((people.knownCount, people.unknownCount), (0, 0))
                self.assertEqual(len(runner.sessions), 1)
                people.knownModel.setSearch("Any name")
                people.unknownModel.setFilter("All")
                self.assertEqual(len(runner.sessions), 1)
            finally:
                projects.shutdown()
                people.shutdown()
                processing.shutdown()
                photos.close()
                face_previews.shutdown()
                thumbnails.shutdown()


class PeopleFlowTests(unittest.TestCase):
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
        self.known_photo = self.folder / "known.jpg"
        self.unknown_photo = self.folder / "unknown.jpg"
        for path in (self.known_photo, self.unknown_photo):
            Image.new("RGB", (120, 120), "gray").save(path)
        self.cache_path = self.root / "analysis.sqlite3"
        self.db_path = self.root / "faces.sqlite3"
        self.state_root = self.root / "state"
        self.database = SQLiteFaceDatabase(self.db_path, CosineEmbeddingSimilarity())
        self.alex = self.database.assign_embeddings([vector(0)], name="Alex")
        self.cache = FaceScanCache(self.cache_path)
        self.cache_face(self.unknown_photo, vector(1), (20, 20, 50, 50))
        self.thumbnails = ThumbnailService()
        self.photos = PhotoListModel(self.thumbnails)
        self.processing = ProjectProcessingController(self.thumbnails,
            cache_path=self.cache_path, state_root=self.state_root)
        self.face_previews = FacePreviewService()
        self.previews = PreviewService()
        self.people = PeopleController(self.processing, self.face_previews, self.thumbnails,
            cache_path=self.cache_path, face_database_path=self.db_path, state_root=self.state_root)
        self.inspector = PhotoInspectorController(self.processing.blurModel, self.previews)
        self.processing.sessionChanged.connect(self.inspector.close)
        self.engine = QQmlApplicationEngine()
        self.engine.setInitialProperties(dict(startPage="library", photoModel=self.photos,
            processingController=self.processing, peopleController=self.people,
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
        self.people.shutdown()
        self.processing.shutdown()
        self.photos.close()
        self.face_previews.shutdown()
        self.previews.shutdown()
        self.thumbnails.shutdown()
        self.assertFalse(self.people._worker.is_alive())
        self.assertFalse(self.processing._worker.is_alive())
        self.temp.cleanup()

    items = phase5.ProjectFlowTests.items
    item = phase5.ProjectFlowTests.item
    click = phase5.ProjectFlowTests.click

    def cache_face(self, path, embedding, bbox):
        stat = path.stat()
        face = EmbeddedFace(bbox, .95, np.zeros((5, 2), dtype=np.float32),
                            embedding, path)
        self.cache.put(path, stat.st_size, stat.st_mtime_ns, [face],
                       coverage=FaceScanCache.COVERAGE_ALL_FACES)

    def payload(self):
        face = dict(path=str(self.unknown_photo), bbox=[20, 20, 50, 50])
        return {"known": [dict(path=str(self.known_photo), person_id=self.alex.id,
                               bbox=[10, 10, 30, 30])],
                "unknown": [dict(id=7, paths=[str(self.unknown_photo)],
                                 representative=str(self.unknown_photo),
                                 representative_bbox=face["bbox"], face_count=1, faces=[face])]}

    def publish(self, payload=None, folder=None):
        self.processing.invalidate()
        self.processing._folder = str(folder or self.folder)
        self.processing._stages["people"] = dict(state="COMPLETE", done=0, total=0, error="")
        self.processing.results["people"] = payload if payload is not None else self.payload()
        self.processing.changed.emit()
        self.app.processEvents()
        self.assertEqual(self.people.state, "COMPLETE")

    def test_qml_known_unknown_routes_detail_inspector_and_counts(self):
        self.publish()
        self.assertEqual((self.people.knownCount, self.people.unknownCount), (1, 1))
        self.assertEqual(self.item("libraryNav_knownPeople").property("count"), 1)
        self.assertEqual(self.item("libraryNav_unknownPeople").property("count"), 1)
        self.click("libraryNav_knownPeople")
        self.assertTrue(self.items("knownPeoplePage"))
        wait_until(lambda: bool(self.items("personCardOpen")))
        wait_until(lambda: self.people.knownModel.get_row(str(self.alex.id))["previewState"] == "ready")
        self.click("personCardOpen")
        self.assertEqual(self.people.selectedKnown["name"], "Alex")
        self.assertEqual(self.item("photoGrid").property("count"), 1)
        self.click("gridPhotoCard")
        self.assertTrue(self.inspector.opened)
        self.click("inspectorBack")
        self.assertFalse(self.inspector.opened)
        self.assertEqual(self.people.selectedKnown["name"], "Alex")
        self.click("peopleDetailBack")
        self.click("libraryNav_unknownPeople")
        self.assertTrue(self.items("unknownPeoplePage"))
        self.click("personCardOpen")
        self.assertTrue(self.people.selectedUnknown["key"])
        self.assertEqual(self.item("photoGrid").property("count"), 1)

    def test_topbar_controls_and_unknown_card_spacing(self):
        self.publish()
        self.click("libraryNav_knownPeople")
        self.click("librarySortButton")
        self.click("toolbarDropdownOption_Most faces")
        self.assertEqual(self.people.knownModel.sortOrder, "Most faces")
        self.click("libraryViewButton")
        self.click("toolbarDropdownOption_Large")
        self.assertEqual(self.item("peopleGrid").property("viewSize"), "Large")
        self.click("personCardOpen")
        self.assertEqual(self.item("photoGrid").property("viewSize"), "Large")
        self.click("libraryNav_unknownPeople")
        self.click("librarySortButton")
        self.click("toolbarDropdownOption_Smallest clusters")
        self.assertEqual(self.people.unknownModel.sortOrder, "Smallest clusters")
        self.click("libraryViewButton")
        self.click("toolbarDropdownOption_Small")
        grid = self.item("peopleGrid")
        self.assertEqual(grid.property("viewSize"), "Small")
        self.assertEqual(self.item("peopleCard").property("title"), "")
        self.assertEqual(self.item("peopleCard").property("subtitle"), "1 photo · 1 face")
        self.assertEqual(self.item("personCardName").property("normalFill").name(), "#d7f205")
        self.assertEqual(self.item("personCardIgnore").property("normalFill").name(), "#d7f205")
        for name in ("personCardName", "personCardIgnore"):
            button = self.item(name)
            self.assertTrue(button.property("compact"))
            self.assertTrue(button.property("vignette"))
            label = button.findChild(QObject, "pillCompactLabel")
            self.assertIsNotNone(label)
            center = label.mapToItem(button, QPointF(label.width() / 2, label.height() / 2))
            self.assertAlmostEqual(center.x(), button.width() / 2, delta=0.5)
            self.assertAlmostEqual(center.y(), button.height() / 2, delta=0.5)
            self.assertEqual(button.property("background").opacity(), 1)
            self.assertGreaterEqual(button.parentItem().height() - button.y() - button.height(), 9)
        self.assertLessEqual(grid.property("cellWidth") * grid.property("columns"),
                             grid.width() - grid.property("scrollbarGutter") + 0.1)
        self.assertEqual(self.item("peopleGridScrollBar").width(), 8)

    def test_name_new_and_existing_persist_without_pipeline_restart(self):
        self.publish()
        old_signature = self.database.cache_signature()
        self.click("libraryNav_unknownPeople")
        self.click("personCardOpen")
        self.click("unknownDetailName")
        self.item("unknownPersonName").setProperty("text", "Zoë")
        self.click("unknownConfirmName")
        wait_until(lambda: self.people.knownCount == 2 and self.people.unknownCount == 0)
        self.assertEqual(self.item("libraryNav_knownPeople").property("count"), 2)
        self.assertEqual(self.item("libraryNav_unknownPeople").property("count"), 0)
        reopened = SQLiteFaceDatabase(self.db_path, CosineEmbeddingSimilarity())
        self.assertEqual(len(reopened.list_people_names()), 2)
        self.assertNotEqual(old_signature, reopened.cache_signature())
        self.assertEqual(len(reopened._embeddings()), 2)
        self.assertEqual(self.processing.results["people"]["unknown"], [])
        self.assertEqual(self.processing._tasks.qsize(), 0)

    def test_assign_existing_via_qml_and_case_insensitive_new_name_guard(self):
        self.publish()
        old_signature = self.database.cache_signature()
        self.click("libraryNav_unknownPeople")
        self.click("personCardOpen")
        self.assertFalse(self.people.nameNew("alex"))
        self.assertIn("already exists", self.people.error)
        self.click("peopleDetailBack")
        self.click("personCardName")
        self.click("unknownNamingMode")
        self.click("roundedComboOption_1")
        self.assertEqual(self.item("unknownNamingMode").property("currentIndex"), 1)
        self.click("unknownExistingPerson")
        self.click("roundedComboOption_0")
        self.click("unknownConfirmName")
        wait_until(lambda: self.people.unknownCount == 0)
        self.assertEqual(self.people.knownCount, 1)
        reopened = SQLiteFaceDatabase(self.db_path, CosineEmbeddingSimilarity())
        self.assertEqual(len(reopened.list_people()), 1)
        self.assertEqual(len(reopened._embeddings()), 2)
        self.assertNotEqual(old_signature, reopened.cache_signature())
        self.assertEqual(self.processing._tasks.qsize(), 0)

    def test_ignore_restart_restore_and_project_isolation(self):
        self.publish()
        key = self.people.unknownModel._all[0]["key"]
        self.assertTrue(self.people.setIgnored(key, True))
        wait_until(lambda: self.people.unknownCount == 0)
        reopened_store = AutomaticStateStore(self.state_root)
        self.assertIn(key, reopened_store.ignored_clusters(self.folder))
        self.publish()
        self.assertEqual(self.people.unknownCount, 0)
        other = self.root / "project-b"
        other.mkdir()
        self.assertNotIn(key, reopened_store.ignored_clusters(other))
        self.people.unknownModel.setFilter("Ignored")
        self.assertEqual(self.people.unknownModel.visibleCount, 1)
        self.assertTrue(self.people.setIgnored(key, False))
        wait_until(lambda: self.people.unknownCount == 1)
        self.assertNotIn(key, AutomaticStateStore(self.state_root).ignored_clusters(self.folder))

    def test_project_switch_clears_ui_and_discards_stale_result(self):
        self.publish()
        old = self.processing.sessionId
        self.processing.invalidate()
        self.assertEqual((self.people.knownCount, self.people.unknownCount), (-1, -1))
        self.assertEqual(self.people.knownModel.totalCount, 0)
        self.assertFalse(self.people.selectedUnknown)
        second = self.root / "project-b"
        second.mkdir()
        self.processing._folder = str(second)
        self.processing._stages["people"] = dict(state="COMPLETE", done=0, total=0, error="")
        self.processing.results["people"] = {"known": [], "unknown": []}
        self.processing.changed.emit()
        self.people._results.put(dict(session=old, folder=str(self.folder), key="old",
                                      operation="new", error="", person_id=42, name="Old"))
        self.people._drain()
        self.assertEqual((self.people.knownCount, self.people.unknownCount), (0, 0))
        self.assertEqual(self.people._folder, str(second))

    def test_retry_advances_people_generation_and_drops_old_review_result(self):
        self.publish()
        key = self.people.unknownModel._all[0]["key"]
        self.people.openUnknown(key)
        old_session = self.processing.sessionId
        old_cancellation = self.people._canceled
        self.processing._stages["people"] = dict(state="FAILED", done=0, total=0,
            error="People matching is unavailable.")
        self.processing._overall = "FAILED"
        self.processing.changed.emit()
        self.assertEqual(self.people.state, "FAILED")
        self.people._results.put(dict(session=old_session, folder=str(self.folder), key=key,
            operation="new", error="", person_id=99, name="Stale result"))
        self.people._pending = 1
        observed = []
        self.processing.sessionChanged.connect(lambda: observed.append(
            (self.processing.sessionId, self.processing.stage("people")["state"])))

        self.processing.retry()

        self.assertEqual(self.processing.sessionId, old_session + 1)
        self.assertEqual(observed, [(old_session + 1, "FAILED")])
        self.assertEqual(self.people._session, self.processing.sessionId)
        self.assertTrue(old_cancellation.is_set())
        self.assertEqual(self.people._pending, 0)
        self.assertEqual(self.people.state, "QUEUED")
        self.assertEqual((self.people.knownModel.totalCount,
                          self.people.unknownModel.totalCount), (0, 0))
        self.assertFalse(self.people.selectedKnown)
        self.assertFalse(self.people.selectedUnknown)
        self.assertEqual(self.people.detailPhotoModel.rowCount(), 0)
        self.people._drain()
        self.assertEqual((self.people.knownModel.totalCount,
                          self.people.unknownModel.totalCount), (0, 0))
        self.assertEqual(self.people._pending, 0)
        # The retry's processing session is not needed for this generation test.
        self.processing.invalidate()

    def test_known_suggestion_uses_recognition_threshold(self):
        similarity = CosineEmbeddingSimilarity()
        self.assertEqual(similarity.default_threshold, 0.60)
        self.publish()
        key = self.people.unknownModel._all[0]["key"]
        for score, expected in ((0.59, {}), (0.61, {"name": "Alex"})):
            with self.subTest(score=score):
                embedding = vector(0) * score + vector(1) * np.sqrt(1 - score * score)
                self.cache_face(self.unknown_photo, embedding, (20, 20, 50, 50))
                self.people.openUnknown(key)
                wait_until(lambda: self.people._pending == 0)
                for field, value in expected.items():
                    self.assertEqual(self.people.suggestion.get(field), value)
                if not expected:
                    self.assertEqual(self.people.suggestion, {})
                self.people.back()

    def test_inflight_name_is_canceled_on_project_switch(self):
        self.publish()
        self.people.openUnknown(self.people.unknownModel._all[0]["key"])
        entered, release = threading.Event(), threading.Event()
        original = people_module._cached_cluster_faces
        def blocked(cache, refs, canceled):
            entered.set()
            release.wait(2)
            return original(cache, refs, canceled)
        before = self.database.cache_signature()
        with patch("ui.qt.people_controller._cached_cluster_faces", side_effect=blocked):
            self.assertTrue(self.people.nameNew("Old project"))
            wait_until(entered.is_set)
            self.processing.invalidate()
            second = self.root / "project-b"
            second.mkdir()
            self.processing._folder = str(second)
            self.processing._stages["people"] = dict(state="COMPLETE", done=0, total=0, error="")
            self.processing.results["people"] = {"known": [], "unknown": []}
            self.processing.changed.emit()
            release.set()
            QTest.qWait(100)
        self.assertEqual((self.people.knownCount, self.people.unknownCount), (0, 0))
        self.assertEqual(SQLiteFaceDatabase(self.db_path, CosineEmbeddingSimilarity()).cache_signature(), before)

    def test_changed_photo_or_database_error_preserves_unknown_review(self):
        self.publish()
        key = self.people.unknownModel._all[0]["key"]
        self.people.openUnknown(key)
        before = self.database.cache_signature()
        with self.unknown_photo.open("ab") as photo:
            photo.write(b"revision")
        self.assertTrue(self.people.nameNew("Stale name"))
        wait_until(lambda: not self.people.busy)
        self.assertIn("unavailable", self.people.error)
        self.assertEqual(self.people.selectedUnknown["key"], key)
        self.assertEqual(self.people.unknownCount, 1)
        self.assertEqual(SQLiteFaceDatabase(self.db_path, CosineEmbeddingSimilarity()).cache_signature(), before)

        self.cache_face(self.unknown_photo, vector(1), (20, 20, 50, 50))
        with patch("ui.qt.people_controller.SQLiteFaceDatabase.assign_embeddings",
                   side_effect=sqlite3.OperationalError("unwritable")):
            self.assertTrue(self.people.nameNew("Database error"))
            wait_until(lambda: not self.people.busy)
        self.assertIn("could not be saved", self.people.error)
        self.assertEqual(self.people.unknownCount, 1)

    def test_processing_failure_empty_and_large_virtualized_grids(self):
        self.publish({"known": [], "unknown": []})
        self.click("libraryNav_knownPeople")
        self.assertEqual(self.people.knownCount, 0)
        self.assertIn("No known people", self.item("knownPeopleStateText").property("text"))
        self.click("libraryNav_unknownPeople")
        self.assertEqual(self.people.unknownCount, 0)
        self.assertIn("No unknown people need review", self.item("unknownPeopleStateText").property("text"))
        self.processing.invalidate()
        self.assertEqual(self.people.state, "NOT_STARTED")
        self.assertIn("Processing people", self.item("unknownPeopleStateText").property("text"))
        self.processing._stages["people"] = dict(state="FAILED", done=0, total=0,
            error="People matching has too many unknown faces to group safely. Other analysis remains available.")
        self.processing.changed.emit()
        self.assertEqual(self.people.state, "FAILED")
        self.assertIn("too many unknown faces", self.people.stageError)
        self.assertIn("too many unknown faces", self.item("unknownPeopleStateText").property("text"))

        # Lightweight summaries exercise GridView without allocating photos or embeddings.
        known = [dict(key=str(i), name=f"Person {i:04}", personId=i,
                      faceCount=1, photoCount=1, representativePhotoId="missing.jpg",
                      representativeBBox=[1, 1, 10, 10]) for i in range(1000)]
        unknown = [dict(key=f"cluster-{i}", name="", faceCount=1, photoCount=1,
                        representativePhotoId="missing.jpg", representativeBBox=[1, 1, 10, 10],
                        ignored=False) for i in range(2000)]
        self.people._state = "COMPLETE"
        self.people.knownModel.set_rows(known, revision="large")
        self.people.unknownModel.set_rows(unknown, revision="large")
        self.people.changed.emit()
        self.click("libraryNav_knownPeople")
        for width, height in ((1280, 720), (1440, 900), (1920, 1080)):
            self.window.setWidth(width)
            self.window.setHeight(height)
            QTest.qWait(60)
            self.assertEqual(self.item("peopleGrid").property("count"), 1000)
            self.assertLess(len(self.items("peopleGridDelegate")), 100)
            self.assertLessEqual(self.face_previews.pending_count, 64)
        self.click("libraryNav_unknownPeople")
        QTest.qWait(60)
        self.assertEqual(self.item("peopleGrid").property("count"), 2000)
        self.assertLess(len(self.items("peopleGridDelegate")), 100)

    def test_qml_search_sort_filter_and_card_action(self):
        zoe = self.database.assign_embeddings([vector(2)], name="Zoë")
        payload = self.payload()
        payload["known"].extend([
            dict(path=str(self.unknown_photo), person_id=zoe.id, bbox=[0, 0, 20, 20]),
            dict(path=str(self.known_photo), person_id=zoe.id, bbox=[5, 5, 20, 20])])
        self.publish(payload)
        self.click("libraryNav_knownPeople")
        self.assertEqual(self.item("peopleGrid").property("count"), 2)
        self.click("knownPeopleSort")
        self.click("toolbarDropdownOption_Most faces")
        self.assertEqual(self.people.knownModel.sortOrder, "Most faces")
        self.click("knownPeopleSearch")
        for key in (Qt.Key.Key_A, Qt.Key.Key_L, Qt.Key.Key_E, Qt.Key.Key_X):
            QTest.keyClick(self.window, key)
        wait_until(lambda: self.people.knownModel.visibleCount == 1)
        self.click("libraryNav_unknownPeople")
        self.click("personCardIgnore")
        wait_until(lambda: self.people.unknownCount == 0)
        self.click("unknownPeopleFilter")
        self.click("toolbarDropdownOption_Ignored")
        self.assertEqual(self.people.unknownModel.visibleCount, 1)
        self.click("unknownPeopleSort")
        self.click("toolbarDropdownOption_Smallest clusters")
        self.assertEqual(self.people.unknownModel.sortOrder, "Smallest clusters")


if __name__ == "__main__":
    unittest.main()
