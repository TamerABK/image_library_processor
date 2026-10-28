from __future__ import annotations

import base64
from contextlib import closing
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
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
from ui.qt.face_previews import FacePreviewService, decode_face_preview, face_crop_bounds, compose_face_preview
from image_loader import ImageLoader
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
    def test_rename_and_delete_person_refresh_other_database_caches_and_keep_photos(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "faces.sqlite3"
            photo = Path(temporary) / "photo.jpg"
            Image.new("RGB", (16, 16), "blue").save(photo)
            writer = SQLiteFaceDatabase(path, CosineEmbeddingSimilarity())
            alex = writer.assign_embeddings([vector(0)], name="Alex")
            zoe = writer.assign_embeddings([vector(1)], name="Zoë")
            reader = SQLiteFaceDatabase(path, CosineEmbeddingSimilarity())
            self.assertEqual(reader.get_person(alex.id).name, "Alex")
            self.assertEqual(reader.find_nearest_embedding(vector(0)).person_id, alex.id)
            writer.rename_person(alex.id, "Alexandra")
            self.assertEqual(reader.get_person(alex.id).name, "Alexandra")
            with self.assertRaisesRegex(ValueError, "already exists"):
                writer.rename_person(alex.id, "ZOË")
            canceled = threading.Event()
            canceled.set()
            with self.assertRaisesRegex(ValueError, "canceled"):
                writer.delete_person(alex.id, canceled=canceled)
            self.assertEqual(reader.get_person(alex.id).name, "Alexandra")
            writer.delete_person(alex.id)
            self.assertIsNone(reader.get_person(alex.id))
            self.assertEqual(reader.find_nearest_embedding(vector(1)).person_id, zoe.id)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE person_id=?", (alex.id,)).fetchone()[0], 0)
            self.assertTrue(photo.is_file())

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
                 dict(path="b.jpg", bbox=[1, 2, 35, 40])]
        cluster = dict(id=88, faces=faces, representative="a.jpg",
                       representative_bbox=[1, 2, 20, 30])
        key = stable_cluster_id(cluster)
        _, unknown = project_people_rows({"known": [], "unknown": [cluster]}, {}, {key})
        self.assertEqual((unknown[0]["faceCount"], unknown[0]["photoCount"]), (3, 2))
        self.assertEqual(unknown[0]["representativePhotoId"], "b.jpg")
        self.assertEqual(unknown[0]["representativeBBox"], [1, 2, 35, 40])
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
            self.assertAlmostEqual(crop.width / crop.height, 1.55, delta=0.05)
            self.assertEqual(decode_face_preview(json.dumps([str(path) + "-missing", [1, 1, 2, 2]]), 64), "")
            corrupt = Path(temporary) / "bad.jpg"
            corrupt.write_bytes(b"invalid")
            self.assertEqual(decode_face_preview(json.dumps([str(corrupt), [1, 1, 2, 2]]), 64), "")

    def test_detection_space_boxes_frame_multiple_faces_and_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, size, faces in (
                ("landscape", (600, 360), [(10, 30, 65, 75), (400, 190, 90, 100)]),
                ("portrait", (360, 600), [(230, 20, 65, 75), (15, 460, 55, 60)]),
                ("small", (600, 360), [(300, 120, 22, 25)]),
                ("narrow", (178, 218), [(51, 56, 88, 132)]),
            ):
                path = Path(temporary) / (name + ".png")
                image = Image.new("RGB", size, "black")
                draw = ImageDraw.Draw(image)
                for x, y, width, height in faces:
                    draw.rectangle((x, y, x + width - 1, y + height - 1), fill="red")
                image.save(path)
                for bbox in faces:
                    source = decode_face_preview(json.dumps([str(path), bbox]), 192)
                    crop = Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1])))
                    self.assertAlmostEqual(crop.width / crop.height, 1.55, delta=0.05)
                    self.assertGreater(np.asarray(crop)[:, :, 0].max(), 100)
                    bounds = face_crop_bounds(size, bbox)
                    self.assertLessEqual(bounds[0], bbox[0])
                    self.assertGreaterEqual(bounds[2], bbox[0] + bbox[2])
                    self.assertLessEqual(bounds[1], bbox[1])
                    self.assertGreaterEqual(bounds[3], bbox[1] + bbox[3])

    def test_edge_face_is_centered_with_dark_blurred_surroundings(self):
        image = Image.new("RGB", (240, 160), "white")
        draw = ImageDraw.Draw(image)
        for x in range(0, 240, 8):
            draw.rectangle((x, 0, x + 3, 159), fill="gray")
        draw.rectangle((2, 45, 53, 104), fill="red")
        preview = compose_face_preview(image, (2, 45, 52, 60), 192)
        self.assertIsNotNone(preview)
        pixels = np.asarray(preview)
        center = pixels[preview.height // 2, preview.width // 2]
        corner = pixels[5, preview.width - 6]
        self.assertGreater(int(center[0]), 160)
        self.assertLess(int(center[1]), 100)
        self.assertLess(int(corner.mean()), 90)
        self.assertAlmostEqual(preview.width / preview.height, 1.55, delta=0.05)

    def test_reduced_jpeg_and_rotated_exif_use_detector_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            for rotated in (False, True):
                path = Path(temporary) / ("rotated.jpg" if rotated else "large.jpg")
                image = Image.new("RGB", (4200, 2800), "black")
                ImageDraw.Draw(image).rectangle((2500, 900, 3100, 1500), fill="red")
                exif = Image.Exif()
                if rotated:
                    exif[274] = 6
                image.save(path, exif=exif, quality=95)
                detection = ImageLoader().load_for_detection(path)
                reduced = ImageLoader().load_pil_for_hashing(path, max_dimension=2048)
                self.assertLess(max(reduced.size), max(detection.shape[:2]))
                # Find the colored region in the actual oriented detector image.
                red = (detection[:, :, 2] > 130) & (detection[:, :, 1] < 60)
                ys, xs = np.where(red)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1)]
                source = decode_face_preview(json.dumps([str(path), bbox]), 192)
                crop = Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1])))
                self.assertGreater(int(np.asarray(crop)[crop.height // 2, crop.width // 2, 0]), 100)

    def test_preview_uses_visible_alternate_from_cluster(self):
        with tempfile.TemporaryDirectory() as temporary:
            edge = Path(temporary) / "edge.png"
            center = Path(temporary) / "center.png"
            Image.new("RGB", (200, 150), "red").save(edge)
            Image.new("RGB", (200, 150), "green").save(center)
            request = json.dumps({"candidates": [
                [str(edge), [2, 25, 90, 100]],
                [str(center), [60, 40, 70, 70]],
            ]})
            source = decode_face_preview(request, 128)
            preview = Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1])))
            red, green, _ = preview.getpixel((preview.width // 2, preview.height // 2))
            self.assertGreater(green, red)

    def test_cached_quality_prefers_clear_face_over_larger_dark_face(self):
        with tempfile.TemporaryDirectory() as temporary:
            dark = Path(temporary) / "dark.png"
            clear = Path(temporary) / "clear.png"
            Image.new("RGB", (240, 160), (8, 8, 8)).save(dark)
            Image.new("RGB", (240, 160), (170, 200, 90)).save(clear)
            database = Path(temporary) / "quality.sqlite3"
            with closing(sqlite3.connect(database)) as db:
                db.execute("CREATE TABLE face_scan_cache_faces (entry_path TEXT, bbox_x INTEGER, "
                           "bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER, "
                           "selection_score REAL, exposure_ranking_score REAL)")
                db.executemany("INSERT INTO face_scan_cache_faces VALUES (?,?,?,?,?,?,?)", [
                    (str(dark), 60, 30, 120, 100, 0.35, 0.07),
                    (str(clear), 80, 45, 80, 70, 0.89, 0.93),
                ])
                db.commit()
            source = decode_face_preview(json.dumps({"quality_cache": str(database),
                "candidates": [[str(dark), [60, 30, 120, 100]],
                               [str(clear), [80, 45, 80, 70]]]}), 128)
            preview = Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1])))
            self.assertGreater(preview.getpixel((preview.width // 2, preview.height // 2))[1], 150)

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
        self.click("photoCardFooter")
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

    def test_known_person_rename_and_delete_preserve_photographs(self):
        self.publish()
        session = self.processing.sessionId
        self.click("libraryNav_knownPeople")
        self.click("personCardOpen")
        self.click("knownManagePerson")
        self.assertTrue(self.window.findChild(QObject, "knownManageDialog").property("visible"))
        self.window.findChild(QObject, "knownPersonName").setProperty("text", "")
        self.click("knownSaveName")
        self.assertEqual(self.people.selectedKnown["name"], "Alex")
        self.assertTrue(self.window.findChild(QObject, "knownManageDialog").property("visible"))
        self.window.findChild(QObject, "knownPersonName").setProperty("text", "Alexandra")
        self.click("knownSaveName")
        wait_until(lambda: self.people.selectedKnown.get("name") == "Alexandra")
        self.assertEqual(self.database.get_person(self.alex.id).name, "Alexandra")
        self.assertEqual(self.processing.sessionId, session)
        self.assertTrue(self.known_photo.is_file())
        self.click("knownManagePerson")
        self.click("knownDeletePerson")
        self.assertTrue(self.window.findChild(QObject, "knownDeleteDialog").property("visible"))
        self.click("knownConfirmDelete")
        wait_until(lambda: self.people.knownCount == 0)
        self.assertIsNone(self.database.get_person(self.alex.id))
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE person_id=?", (self.alex.id,)).fetchone()[0], 0)
        self.assertTrue(self.known_photo.is_file())
        self.assertEqual(self.processing.sessionId, session)

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

    def test_people_preview_fills_shortened_card_frame(self):
        self.publish()
        for route in ("libraryNav_knownPeople", "libraryNav_unknownPeople"):
            self.click(route)
            wait_until(lambda: bool(self.items("peopleCardPreviewImage")))
            grid = self.item("peopleGrid")
            card = self.item("peopleCard")
            frame = self.item("peopleCardPreviewFrame")
            preview = self.item("peopleCardPreviewImage")
            self.assertTrue(frame.property("fillsPreview"))
            self.assertTrue(frame.property("clip"))
            self.assertAlmostEqual(preview.width(), frame.width(), delta=0.1)
            self.assertAlmostEqual(preview.height(), frame.height(), delta=0.1)
            self.assertLess(frame.height(), card.height() - 68)
            self.assertAlmostEqual(grid.property("cellHeight"),
                                   max(180, min(270, grid.property("cellWidth") * 0.86)), delta=0.1)

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
