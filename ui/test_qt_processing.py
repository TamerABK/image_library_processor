import base64
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import time
import unittest

from PIL import Image
from PySide6.QtCore import QThread
from PySide6.QtTest import QAbstractItemModelTester
from PySide6.QtWidgets import QApplication
from automatic_scan.state_store import AutomaticStateStore, PersistedImageDecision
from ui.models import ResultItem
from ui.phase6_test_support import FixtureRunner
from ui.qt.blur_model import BlurCandidateModel
from ui.qt.inspector_controller import PhotoInspectorController
from ui.qt.photo_model import PhotoListModel, photo_identity
from ui.qt.previews import PreviewService
from ui.qt.processing_controller import ProjectProcessingController
from ui.qt.project_controller import ProjectController
from ui.qt.thumbnails import ThumbnailService
from ui.test_qt_thumbnails import capture_qt_warnings, wait_until


class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "photos"
        self.folder.mkdir()
        for name in ("a.jpg", "b.png", "c.jpg"):
            Image.new("RGB", (640, 320), "gray").save(self.folder / name)
        (self.folder / "notes.txt").write_text("ignored", encoding="utf-8")
        self.gate = threading.Event()
        self.store = AutomaticStateStore(self.root / "state")
        self.runner = FixtureRunner(self.root, self.store)
        self.thumbnails = ThumbnailService()
        self.photos = PhotoListModel(self.thumbnails)
        self.processing = ProjectProcessingController(self.thumbnails,
            cache_path=self.root / "analysis.sqlite3", state_root=self.root / "state", runner=self.runner)
        self.projects = ProjectController(self.photos, registry_path=self.root / "projects.json",
            processing=self.processing)
        self.previews = PreviewService()
        self.inspector = PhotoInspectorController(self.processing.blurModel, self.previews)
        self.processing.sessionChanged.connect(self.inspector.close)
        self.tester = QAbstractItemModelTester(self.processing.blurModel,
            QAbstractItemModelTester.FailureReportingMode.Warning)

    def tearDown(self):
        self.gate.set()
        self.projects.shutdown()
        self.inspector.close()
        self.processing.shutdown()
        self.photos.close()
        self.previews.shutdown()
        self.thumbnails.shutdown()
        self.assertFalse(self.processing._worker.is_alive())
        self.assertEqual(0, self.previews.workers_alive)
        self.temp.cleanup()

    def open(self, folder=None):
        self.projects.openFolder(str(folder or self.folder))
        wait_until(lambda: not self.projects.busy)

    def finish(self):
        wait_until(lambda: self.processing.overallState in ("COMPLETE", "FAILED"), timeout=6)

    def test_automatic_start_without_feature_page(self):
        threads = []
        self.processing.blurModel.modelReset.connect(lambda: threads.append(QThread.currentThread()))
        self.open()
        self.finish()
        self.assertEqual(1, len(self.runner.sessions))
        self.assertEqual(3, self.photos.rowCount())
        self.assertEqual("COMPLETE", self.processing.overallState)
        self.assertEqual(3, self.processing.routeCounts["blurry"])
        self.assertTrue(all(t == self.processing.thread() for t in threads))
        self.assertTrue(all(v["state"] == "COMPLETE" for v in self.processing.stages))

    def test_blur_review_preserves_unrelated_persisted_photo_fields(self):
        self.open()
        self.finish()
        identity = self.processing.blurModel.identities()[0]
        state = self.store.load(self.folder)
        result = self.processing.blurModel.results[identity]
        state.image_decisions[identity] = PersistedImageDecision(
            identity, result["file_size"], result["mtime_ns"], False, False, False, None,
            duplicate_group_id="existing-group", is_duplicate_keeper=False)
        self.store.save(state)
        self.assertTrue(self.processing.setReview(identity, "REMOVE"))
        saved = self.store.load(self.folder).image_decisions[identity]
        self.assertEqual(saved.review_state, "REMOVE")
        self.assertEqual(saved.duplicate_group_id, "existing-group")
        self.assertFalse(saved.is_duplicate_keeper)

    def test_progressive_blur_before_face_completion(self):
        self.runner.gate = self.gate
        self.open()
        wait_until(lambda: self.processing.stage("blur")["state"] == "COMPLETE", timeout=10)
        self.assertEqual(3, self.photos.rowCount())
        self.assertEqual(3, self.processing.blurModel.rowCount())
        self.assertEqual("RUNNING", self.processing.overallState)
        self.assertNotEqual("COMPLETE", self.processing.stage("faces")["state"])
        # GUI continues dispatching model selection while native work is blocked.
        identity = self.processing.blurModel.identities()[0]
        self.processing.blurModel.setSelected(identity, True)
        self.assertEqual(1, self.processing.blurModel.selectedCount)
        self.gate.set()
        self.finish()

    def test_stale_session_and_project_review_isolation(self):
        self.runner.gate = self.gate
        self.open()
        wait_until(lambda: self.processing.blurModel.rowCount() == 3)
        previous = self.processing._session
        old_identity = self.processing.blurModel.identities()[0]
        self.assertTrue(self.processing.setReview(old_identity, "KEEP"))
        second = self.root / "second"
        second.mkdir()
        Image.new("RGB", (80, 40), "gray").save(second / "second.jpg")
        self.open(second)
        self.assertEqual(0, self.processing.blurModel.rowCount())
        self.gate.set()
        self.finish()
        self.processing._results.put((previous, "blur", "COMPLETE", 0, 0, "", [{"photoId": old_identity}], {}))
        self.processing._drain()
        self.assertEqual(1, self.processing.blurModel.rowCount())
        current = self.processing.blurModel.identities()[0]
        self.assertEqual("UNDECIDED", self.processing.blurModel.photo(current)["reviewState"])
        self.assertFalse(self.processing.setReview(old_identity, "REMOVE"))
        self.assertNotIn(old_identity, self.processing.blurModel.results)

    def test_review_restart_sort_filter_bulk_and_no_delete(self):
        self.open()
        self.finish()
        model = self.processing.blurModel
        identity = model.identities()[0]
        for intent in ("KEEP", "REVIEW", "REMOVE"):
            self.assertTrue(self.processing.setReview(identity, intent))
            self.assertTrue(Path(identity).is_file())
        model.setSort("Filename Z–A")
        model.setFilter("Review")
        self.assertEqual(0, model.rowCount())
        model.setFilter("All candidates")
        self.assertEqual("REMOVE", model.photo(identity)["reviewState"])
        for other in model.identities():
            if other != identity:
                model.setSelected(other, True)
        self.processing.reviewSelected("KEEP")
        other = next(key for key in model.identities() if key != identity)
        self.assertTrue(self.processing.setReview(other, "REVIEW"))
        decisions = {key: model.photo(key)["reviewState"] for key in model.identities()}
        self.processing.shutdown()
        restarted = ProjectProcessingController(self.thumbnails,
            cache_path=self.root / "analysis.sqlite3", state_root=self.root / "state", runner=self.runner)
        try:
            project = self.projects.projects.get(self.projects.activeProjectId)
            restarted.start(project, [ResultItem(p, p.name, "") for p in self.folder.glob("*.jpg")] +
                            [ResultItem(self.folder / "b.png", "b.png", "")])
            wait_until(lambda: restarted.overallState == "COMPLETE", timeout=6)
            self.assertEqual("REMOVE", restarted.blurModel.photo(identity)["reviewState"])
            self.assertEqual(decisions, {key: restarted.blurModel.photo(key)["reviewState"] for key in decisions})
            self.assertEqual(3, restarted.diagnostics["blur"]["cache_hits"])
        finally:
            restarted.shutdown()

    def test_candidate_sort_filter_and_count(self):
        model = self.processing.blurModel
        rows = [dict(photoId=photo_identity(self.folder / name), filename=name, final_score=score,
            status=status, reviewState="UNDECIDED", file_size=1, mtime_ns=1)
            for name, score, status in (("z.jpg", .1, "Blurry"), ("a.jpg", .6, "Review"), ("sharp.jpg", .9, "Sharp"))]
        model.set_results(rows, new_project=True)
        self.assertEqual(2, model.candidateCount)
        self.assertEqual("z.jpg", model.photo(model.identities()[0])["filename"])
        model.setSort("Least problematic first")
        self.assertEqual("a.jpg", model.photo(model.identities()[0])["filename"])
        model.setSort("Filename Z–A")
        self.assertEqual("z.jpg", model.photo(model.identities()[0])["filename"])
        model.setSort("Filename A–Z")
        self.assertEqual("a.jpg", model.photo(model.identities()[0])["filename"])
        model.setFilter("Blurry")
        self.assertEqual(1, model.rowCount())
        self.assertEqual(2, model.candidateCount)  # Sidebar count is all candidates.
        model.setFilter("Review")
        self.assertEqual("Review", model.data(model.index(0), model.ROLE["classification"]))

    def test_preview_larger_than_grid_orientation_and_bounds(self):
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (2400, 1200), "gray").save(self.folder / "a.jpg", exif=exif)
        self.open()
        self.finish()
        identity = photo_identity(self.folder / "a.jpg")
        self.inspector.open(identity)
        wait_until(lambda: self.inspector.previewState == "ready")
        preview = Image.open(BytesIO(base64.b64decode(self.inspector.previewSource.split(",")[1])))
        self.assertGreater(max(preview.size), 512)
        self.assertLessEqual(max(preview.size), 1600)
        self.assertLess(preview.width, preview.height)
        self.assertLessEqual(self.previews.pending_count, 3)
        self.assertLessEqual(self.previews.cache_entries, 3)
        self.assertLessEqual(self.previews.cache_bytes, 24 * 1024 * 1024)
        self.inspector.next()
        self.inspector.previous()
        wait_until(lambda: self.inspector.previewState == "ready")
        self.assertGreater(self.previews.cache_hits, 0)

    def test_missing_or_corrupt_preview_is_safe(self):
        self.open()
        self.finish()
        identities = self.processing.blurModel.identities()
        Path(identities[0]).unlink()  # Fixture simulates an externally removed source.
        Path(identities[1]).write_bytes(b"corrupt")
        for identity in identities[:2]:
            self.inspector.open(identity)
            wait_until(lambda: self.inspector.previewState == "error")
        self.assertTrue(Path(identities[2]).is_file())

    def test_shutdown_discards_native_completion(self):
        self.runner.gate = self.gate
        self.open()
        wait_until(lambda: self.processing.blurModel.rowCount() == 3)
        snapshot = dict(self.processing.results)
        self.processing.shutdown()
        self.gate.set()
        wait_until(lambda: not self.processing._worker.is_alive())
        self.processing._drain()
        self.assertEqual(snapshot, self.processing.results)

    def test_retry_resumes_central_pipeline_after_failure(self):
        class FailOnce:
            def __init__(inner):
                inner.calls = 0
            def process_project(inner, folder, paths, **kwargs):
                inner.calls += 1
                if inner.calls == 1:
                    kwargs["emit"]("blur", "FAILED", 0, 0, "Fixture stage failed", None, {})
                else:
                    self.runner.process_project(folder, paths, **kwargs)
        failing = FailOnce()
        processing = ProjectProcessingController(self.thumbnails, runner=failing,
            cache_path=self.root / "analysis.sqlite3", state_root=self.root / "state")
        try:
            self.open()
            self.finish()
            project = self.projects.projects.get(self.projects.activeProjectId)
            processing.start(project, [ResultItem(self.folder / "a.jpg", "a.jpg", "")])
            wait_until(lambda: processing.overallState == "FAILED")
            processing.retry()
            wait_until(lambda: processing.overallState == "COMPLETE", timeout=6)
            self.assertEqual(2, failing.calls)
            self.assertEqual(1, processing.blurModel.candidateCount)
        finally:
            processing.shutdown()

    def test_ten_thousand_lightweight_identities_are_bounded(self):
        model = self.processing.blurModel
        started = time.perf_counter()
        model.set_results([dict(photoId=photo_identity(self.folder / f"{i:05}.jpg"), filename=f"{i:05}.jpg",
            final_score=(i % 60) / 100, status="Blurry" if i % 60 < 45 else "Review",
            file_size=1, mtime_ns=1, reviewState="UNDECIDED") for i in range(10000)], new_project=True)
        elapsed = (time.perf_counter() - started) * 1000
        self.assertEqual(10000, model.rowCount())
        self.assertEqual(0, self.thumbnails.pending_count)
        self.assertEqual(1, self.processing._tasks.maxsize)
        self.assertEqual(16, self.processing._results.maxsize)
        print(f"\nPhase 6 diagnostic: 10000 identities populated in {elapsed:.1f} ms; processing workers 1, request queue 1, result queue 16; preview cache 3 entries/24 MiB, edge 1600")
