"""One project-scoped scheduler. Workers publish plain data through bounded queues."""
from queue import Empty, Full, Queue
import threading

from PySide6.QtCore import QObject, Property, QThread, QTimer, Signal, Slot
from automatic_scan.orchestrator import AutomaticScanOrchestrator
from automatic_scan.project_pipeline import STAGES
from automatic_scan.state_store import AutomaticStateStore, PersistedImageDecision
from automatic_scan.scan_logging import get_scan_logger
from scan_controls import CancellationToken
from .blur_model import BlurCandidateModel
from .photo_model import photo_identity


def _processing_worker(tasks, results, stop, runner, store, cache_path):
    while not stop.is_set():
        try:
            session, folder, paths, token = tasks.get(timeout=0.05)
        except Empty:
            continue
        def emit(stage, state, done, total, error, data, stats):
            if stop.is_set():
                token.cancel()
                return
            value = (session, stage, str(state), done, total, error, data, stats)
            # Backpressure on transitions, coalesce progress. No unbounded backlog.
            if str(state) == "RUNNING" and data is None and results.full():
                return
            while not stop.is_set() and not token.is_canceled:
                try:
                    results.put(value, timeout=0.05)
                    return
                except Full:
                    continue
        try:
            get_scan_logger().info("processing_session_started session=%s folder=%s photo_count=%d",
                                   session, folder, len(paths))
            orchestrator = runner or AutomaticScanOrchestrator(state_store=store)
            orchestrator.process_project(folder, paths, emit=emit,
                cancellation_token=token, cache_path=cache_path)
            get_scan_logger().info("processing_session_finished session=%s folder=%s", session, folder)
        except Exception:
            get_scan_logger().exception("processing_worker_failed session=%s folder=%s photo_count=%d",
                                        session, folder, len(paths))
            for stage in STAGES:
                emit(stage, "FAILED", 0, 0, "Processing could not start. Check that application data is writable.", None, {})
        emit("", "FINISHED", 0, 0, "", None, {})


class ProjectProcessingController(QObject):
    changed = Signal()
    sessionChanged = Signal()
    LABELS = dict(blur="Image quality", duplicates="Near duplicates", faces="Face detection",
        face_quality="Face quality", embeddings="Face embeddings", people="People matching", ranking="Representatives")

    def __init__(self, thumbnails, parent=None, *, cache_path=None, state_root=None, runner=None):
        super().__init__(parent)
        self._blur = BlurCandidateModel(thumbnails, self)
        self._blur.changed.connect(self.changed)
        self._store = AutomaticStateStore(state_root)
        self._cache_path = cache_path
        self._session = 0
        self._project_id = ""
        self._folder = ""
        self._paths = []
        self._closed = False
        self._token = CancellationToken()
        self._stages = {}
        self._overall = "NOT_STARTED"
        self._message = ""
        self.results = {}
        self.diagnostics = {}
        self._tasks, self._results = Queue(maxsize=1), Queue(maxsize=16)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=_processing_worker,
            args=(self._tasks, self._results, self._stop, runner, self._store, cache_path),
            name="room36-project-processing", daemon=True)
        self._worker.start()
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._drain)
        self._lifecycle = {"token": self._token}
        stop, lifecycle = self._stop, self._lifecycle
        self.destroyed.connect(lambda *_: (stop.set(), lifecycle["token"].cancel()))

    @Property(QObject, constant=True)
    def blurModel(self):
        return self._blur

    @Property(str, notify=changed)
    def overallState(self):
        return self._overall

    @Property(str, notify=changed)
    def currentStage(self):
        return next((k for k, v in self._stages.items() if v["state"] == "RUNNING"), "")

    @Property(str, notify=changed)
    def message(self):
        return self._message

    @Property(str, notify=changed)
    def projectFolder(self):
        return self._folder

    @Property(int, notify=sessionChanged)
    def sessionId(self):
        return self._session

    @Property("QVariantList", notify=changed)
    def stages(self):
        return [dict(stage=k, label=self.LABELS[k], **v) for k, v in self._stages.items()]

    @Property("QVariantMap", notify=changed)
    def routeCounts(self):
        return {"blurry": self._blur.candidateCount}

    @Slot(str, result="QVariantMap")
    def stage(self, name):
        return self._stages.get(name, dict(state="NOT_STARTED", done=0, total=0, error=""))

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Project processing mutations require the owning thread")

    def invalidate(self):
        self._assert_owner()
        self._token.cancel()
        self._session += 1
        self._token = CancellationToken()
        self._lifecycle["token"] = self._token
        self._blur.set_results([], new_project=True)
        self.results, self.diagnostics = {}, {}
        self._stages = {k: dict(state="NOT_STARTED", done=0, total=0, error="") for k in STAGES}
        self._overall, self._message = "NOT_STARTED", ""
        self._timer.stop()
        self.sessionChanged.emit()
        self.changed.emit()

    def start(self, project, items):
        self._assert_owner()
        if self._closed:
            return
        self.invalidate()
        self._project_id, self._folder = project.project_id, project.folder
        self._paths = [photo_identity(item.path) for item in items]
        self._schedule()

    def _schedule(self):
        for value in self._stages.values():
            value.update(state="QUEUED", done=0, total=0, error="")
        self._overall = "QUEUED"
        try:
            self._tasks.get_nowait()
        except Empty:
            pass
        self._tasks.put_nowait((self._session, self._folder, list(self._paths), self._token))
        self._timer.start()
        self.changed.emit()

    @Slot()
    def retry(self):
        self._assert_owner()
        if self._closed or not self._folder or self._overall in ("QUEUED", "RUNNING"):
            return
        # All stages resume through the same cache-first lifecycle.
        self._token.cancel()
        self._session += 1
        self._token = CancellationToken()
        self._lifecycle["token"] = self._token
        self._schedule()

    @Slot()
    def _drain(self):
        self._assert_owner()
        for _ in range(16):
            try:
                session, stage, state, done, total, error, data, stats = self._results.get_nowait()
            except Empty:
                break
            if self._closed or session != self._session:
                continue
            if state == "FINISHED":
                values = [v["state"] for v in self._stages.values()]
                self._overall = "FAILED" if "FAILED" in values else "CANCELLED" if "CANCELLED" in values else "COMPLETE"
                self._timer.stop()
            else:
                self._stages[stage] = dict(state=state, done=done, total=total, error=error)
                self._overall = "RUNNING"
                if stats:
                    self.diagnostics[stage] = stats
                if data is not None:
                    self.results[stage] = data
                    if stage == "blur":
                        # A review made while a retry reads cache wins over that
                        # session's earlier snapshot, provided the file is unchanged.
                        for row in data:
                            previous = self._blur.results.get(row["photoId"])
                            if previous and (previous["file_size"], previous["mtime_ns"]) == (row["file_size"], row["mtime_ns"]):
                                row["reviewState"] = previous["reviewState"]
                        self._blur.set_results(data)
            self.changed.emit()

    def replace_people_result(self, payload):
        """Apply a completed metadata review without scheduling processing."""
        self._assert_owner()
        if self._stages.get("people", {}).get("state") != "COMPLETE":
            return False
        self.results["people"] = payload
        self.changed.emit()
        return True

    @Slot(str, str, result=bool)
    def setReview(self, identity, intent):
        return self._save_reviews([identity], intent)

    def _save_reviews(self, identities, intent):
        self._assert_owner()
        if self._closed or intent not in ("UNDECIDED", "KEEP", "REVIEW", "REMOVE"):
            return False
        rows = [self._blur.results[key] for key in identities if key in self._blur.results]
        if len(rows) != len(identities) or not rows:
            return False
        try:
            state = self._store.load(self._folder)
            # Persist intent only; this controller has no deletion execution API.
            for row in rows:
                identity = row["photoId"]
                state.image_decisions[identity] = PersistedImageDecision(identity, row["file_size"], row["mtime_ns"],
                    intent == "REMOVE", False, intent == "KEEP", "keep" if intent == "KEEP" else None,
                    review_state=intent)
            self._store.save(state)
        except (OSError, ValueError, KeyError, TypeError):
            self._message = "This review decision could not be saved. Check application data permissions."
            self.changed.emit()
            return False
        self._message = ""
        for identity in identities:
            self._blur.apply_review(identity, intent)
        self.changed.emit()
        return True

    @Slot(str)
    def reviewSelected(self, intent):
        self._save_reviews(list(self._blur._selection), intent)

    @Slot()
    def shutdown(self):
        self._assert_owner()
        if self._closed:
            return
        self._closed = True
        self._token.cancel()
        self._stop.set()
        self._session += 1
        self._timer.stop()
        self._worker.join(timeout=0.5)
        self._blur.close()
