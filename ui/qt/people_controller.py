"""Project-scoped presentation and metadata review of Phase 6 People results."""
from __future__ import annotations

from pathlib import Path
from queue import Empty, Full, Queue
import sqlite3
import threading
import unicodedata

from PySide6.QtCore import QObject, Property, QThread, QTimer, Signal, Slot

from app_paths import app_data_path
from automatic_scan.scan_logging import get_scan_logger
from automatic_scan.state_store import AutomaticStateStore
from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_detector.face_database_sqlite import SQLiteFaceDatabase
from face_processing.cache import FaceScanCache
from ui.models import ResultItem
from .people_data import project_people_rows, stable_cluster_id
from .people_model import PeopleListModel
from .photo_model import PhotoListModel


class PeopleReviewError(ValueError):
    pass


def validate_person_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise PeopleReviewError("Enter a name for this person.")
    if len(name) > 100:
        raise PeopleReviewError("Use a name of 100 characters or fewer.")
    if any(unicodedata.category(character) == "Cc" for character in name):
        raise PeopleReviewError("Names cannot contain control characters.")
    return name


def _cached_cluster_faces(cache, face_refs, canceled):
    by_path = {}
    used = {}
    selected = []
    for face_ref in face_refs:
        if canceled.is_set():
            raise PeopleReviewError("This People review was canceled.")
        path = Path(face_ref["path"])
        if path not in by_path:
            try:
                stat = path.stat()
            except OSError as exc:
                raise PeopleReviewError("A cluster photo changed or is unavailable. Reopen the project.") from exc
            faces = cache.get(path, stat.st_size, stat.st_mtime_ns,
                              include_unknown_faces=True, require_analysis=False)
            if faces is None:
                raise PeopleReviewError("Cached face embeddings are unavailable. Reopen the project.")
            by_path[path] = faces
            used[path] = set()
        bbox = tuple(int(value) for value in face_ref["bbox"])
        match = next((i for i, face in enumerate(by_path[path])
                      if i not in used[path] and tuple(face.bbox) == bbox), None)
        if match is None:
            raise PeopleReviewError("A cluster face changed. Reopen the project before naming it.")
        used[path].add(match)
        selected.append(by_path[path][match])
    if not selected:
        raise PeopleReviewError("This cluster has no cached face embeddings.")
    return selected


def _review_worker(tasks, results, stop, cache_path, database_path, state_root):
    while not stop.is_set():
        try:
            task = tasks.get(timeout=0.05)
        except Empty:
            continue
        session, folder, key, operation, value, faces, canceled = task
        result = {"session": session, "folder": folder, "key": key,
                  "operation": operation, "error": ""}
        try:
            if canceled.is_set():
                continue
            if operation == "ignore":
                if canceled.is_set():
                    continue
                AutomaticStateStore(state_root).set_cluster_ignored(folder, key, bool(value))
                result["ignored"] = bool(value)
            else:
                selected = _cached_cluster_faces(FaceScanCache(cache_path), faces, canceled)
                similarity = CosineEmbeddingSimilarity()
                database = SQLiteFaceDatabase(database_path, similarity)
                if operation == "suggest":
                    match = database.find_nearest_embedding(selected[0].embedding)
                    person = database.get_person(match.person_id) if match else None
                    result["suggestion"] = ({"name": person.name, "personId": person.id,
                                              "score": round(match.score, 3)}
                                             if person and match.score >= similarity.default_threshold else {})
                else:
                    if canceled.is_set():
                        continue
                    if operation == "new" and any(
                            person.name.casefold() == value.casefold()
                            for person in database.list_people()):
                        raise PeopleReviewError(
                            "That person already exists. Choose Assign to existing person.")
                    person = database.assign_embeddings(
                        (face.embedding for face in selected),
                        name=value if operation == "new" else None,
                        person_id=value if operation == "existing" else None,
                        canceled=canceled)
                    result["person_id"] = person.id
                    result["name"] = person.name
        except PeopleReviewError as exc:
            result["error"] = str(exc)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            if canceled.is_set():
                continue
            get_scan_logger().exception("people_review_failed folder=%s operation=%s cluster=%s",
                                        folder, operation, key)
            result["error"] = "People review could not be saved. Check application data and reopen the project."
        if not stop.is_set():
            while not stop.is_set():
                try:
                    results.put(result, timeout=0.05)
                    break
                except Full:
                    continue


class PeopleController(QObject):
    changed = Signal()

    def __init__(self, processing, face_previews, thumbnails, parent=None, *,
                 cache_path=None, face_database_path=None, state_root=None):
        super().__init__(parent)
        self._processing = processing
        self._cache_path = cache_path if cache_path is not None else processing._cache_path
        self._database_path = face_database_path or app_data_path("face_embeddings.sqlite3")
        self._store = AutomaticStateStore(state_root)
        self._database = SQLiteFaceDatabase(self._database_path, CosineEmbeddingSimilarity())
        self._known = PeopleListModel(face_previews, "known", self)
        self._unknown = PeopleListModel(face_previews, "unknown", self)
        self._detail_photos = PhotoListModel(thumbnails, self)
        self._state = "NOT_STARTED"
        self._stage_error = ""
        self._error = ""
        self._busy = False
        self._busy_operation = ""
        self._suggestion = {}
        self._known_options = []
        self._selected_known = ""
        self._selected_unknown = ""
        self._detail_lease = 0
        self._detail_preview_model = None
        self._folder = ""
        self._session = processing.sessionId
        self._payload = None
        self._ignored = set()
        self._canceled = threading.Event()
        self._tasks, self._results = Queue(maxsize=2), Queue(maxsize=2)
        self._pending = 0
        self._stop = threading.Event()
        self._worker = threading.Thread(target=_review_worker,
            args=(self._tasks, self._results, self._stop, self._cache_path,
                  self._database_path, state_root), name="room36-people-review", daemon=True)
        self._worker.start()
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._drain)
        processing.sessionChanged.connect(self._reset_session)
        processing.changed.connect(self._sync)
        self._known.dataChanged.connect(lambda *_: self.changed.emit())
        self._unknown.dataChanged.connect(lambda *_: self.changed.emit())
        self._sync()

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("People presentation mutations require the owning Qt thread")

    @Property(QObject, constant=True)
    def knownModel(self):
        return self._known

    @Property(QObject, constant=True)
    def unknownModel(self):
        return self._unknown

    @Property(QObject, constant=True)
    def detailPhotoModel(self):
        return self._detail_photos

    @Property(str, notify=changed)
    def state(self):
        return self._state

    @Property(str, notify=changed)
    def stageError(self):
        return self._stage_error

    @Property(str, notify=changed)
    def error(self):
        return self._error

    @Property(bool, notify=changed)
    def busy(self):
        return self._busy

    @Property(str, notify=changed)
    def busyMessage(self):
        return "Saving review…" if self._busy_operation == "ignore" else "Saving person…"

    @Property("QVariantMap", notify=changed)
    def suggestion(self):
        return dict(self._suggestion)

    @Property("QVariantList", notify=changed)
    def knownOptions(self):
        return list(self._known_options)

    @Property("QVariantMap", notify=changed)
    def selectedKnown(self):
        return self._public_row(self._known.get_row(self._selected_known))

    @Property("QVariantMap", notify=changed)
    def selectedUnknown(self):
        return self._public_row(self._unknown.get_row(self._selected_unknown))

    @Property(int, notify=changed)
    def knownCount(self):
        return self._known.totalCount if self._state == "COMPLETE" else -1

    @Property(int, notify=changed)
    def unknownCount(self):
        return self._unknown.needsReviewCount if self._state == "COMPLETE" else -1

    @staticmethod
    def _public_row(row):
        return {key: value for key, value in row.items()
                if key not in ("faces", "paths", "ticket", "edge")} if row else {}

    @Slot()
    def _reset_session(self):
        self._assert_owner()
        self._canceled.set()
        self._release_detail_preview()
        self._canceled = threading.Event()
        self._session = self._processing.sessionId
        self._folder = ""
        self._payload = None
        self._state = "NOT_STARTED"
        self._stage_error = self._error = ""
        self._busy = False
        self._busy_operation = ""
        self._suggestion = {}
        self._known_options = []
        self._ignored = set()
        self._selected_known = self._selected_unknown = ""
        self._pending = 0
        self._known.set_rows([], revision=str(self._session))
        self._unknown.set_rows([], revision=str(self._session))
        self._detail_photos.replace_items([], preserve_selection=False)
        self.changed.emit()

    @Slot()
    def _sync(self):
        self._assert_owner()
        stage = self._processing.stage("people")
        state = stage["state"]
        folder = self._processing.projectFolder
        if folder and folder != self._folder:
            self._folder = folder
        if state != "COMPLETE":
            if self._payload is not None:
                self._payload = None
                self._known.set_rows([], revision=str(self._session))
                self._unknown.set_rows([], revision=str(self._session))
                self._detail_photos.replace_items([], preserve_selection=False)
                self._selected_known = self._selected_unknown = ""
            if state != self._state or stage.get("error", "") != self._stage_error:
                self._state, self._stage_error = state, stage.get("error", "")
                self.changed.emit()
            return
        payload = self._processing.results.get("people")
        if payload is None or (payload is self._payload and self._state == "COMPLETE"):
            return
        try:
            people = self._database.list_people()
            names = {person.id: person.name for person in people}
            self._known_options = [dict(personId=person.id, name=person.name) for person in people]
            self._ignored = self._store.ignored_clusters(self._folder)
            known, unknown = project_people_rows(payload, names, self._ignored)
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError):
            get_scan_logger().exception("people_presentation_failed folder=%s", self._folder)
            self._state = "FAILED"
            self._stage_error = "People results could not be opened. Check application data permissions."
            self.changed.emit()
            return
        self._payload = payload
        self._state, self._stage_error = "COMPLETE", ""
        revision = f"{self._session}:{len(known)}:{len(unknown)}"
        self._release_detail_preview()
        self._known.set_rows(known, revision=revision)
        self._unknown.set_rows(unknown, revision=revision)
        if self._selected_known and self._known.get_row(self._selected_known) is None:
            self._selected_known = ""
        if self._selected_unknown and self._unknown.get_row(self._selected_unknown) is None:
            self._selected_unknown = ""
        if self._selected_known:
            self._detail_preview_model = self._known
            self._detail_lease = self._known.acquirePreview(self._selected_known, 256)
        elif self._selected_unknown:
            self._detail_preview_model = self._unknown
            self._detail_lease = self._unknown.acquirePreview(self._selected_unknown, 256)
        self._set_detail()
        self.changed.emit()

    def _set_detail(self):
        row = (self._known.get_row(self._selected_known) or
               self._unknown.get_row(self._selected_unknown))
        items = [ResultItem(Path(path), Path(path).name, "People photo")
                 for path in row["paths"]] if row else []
        self._detail_photos.replace_items(items, preserve_selection=False)

    def _release_detail_preview(self):
        if self._detail_lease and self._detail_preview_model is not None:
            self._detail_preview_model.releasePreview(self._detail_lease)
        self._detail_lease = 0
        self._detail_preview_model = None

    @Slot(str)
    def openKnown(self, key):
        self._assert_owner()
        if self._state != "COMPLETE" or self._known.get_row(key) is None:
            return
        self._release_detail_preview()
        self._selected_known, self._selected_unknown = key, ""
        self._detail_preview_model = self._known
        self._detail_lease = self._known.acquirePreview(key, 256)
        self._error = ""
        self._set_detail()
        self.changed.emit()

    @Slot(str)
    def openUnknown(self, key):
        self._assert_owner()
        if self._state != "COMPLETE" or self._unknown.get_row(key) is None:
            return
        self._release_detail_preview()
        self._selected_unknown, self._selected_known = key, ""
        self._detail_preview_model = self._unknown
        self._detail_lease = self._unknown.acquirePreview(key, 256)
        self._suggestion, self._error = {}, ""
        self._set_detail()
        self.changed.emit()
        self._submit(key, "suggest", None, busy=False)

    @Slot()
    def back(self):
        self._assert_owner()
        self._release_detail_preview()
        self._selected_known = self._selected_unknown = ""
        self._suggestion, self._error = {}, ""
        self._detail_photos.replace_items([], preserve_selection=False)
        self.changed.emit()

    def _submit(self, key, operation, value, *, busy=True):
        row = self._unknown.get_row(key)
        if self._state != "COMPLETE" or row is None or (busy and self._busy):
            return False
        task = (self._session, self._folder, key, operation, value,
                list(row["faces"]), self._canceled)
        try:
            self._tasks.put_nowait(task)
        except Full:
            if busy:
                self._error = "People review is busy. Try again in a moment."
                self.changed.emit()
            return False
        self._pending += 1
        if busy:
            self._busy = True
            self._busy_operation = operation
            self._error = ""
            self._timer.start()
            self.changed.emit()
        else:
            self._timer.start()
        return True

    @Slot(str, result=bool)
    def nameNew(self, value):
        self._assert_owner()
        try:
            name = validate_person_name(value)
        except PeopleReviewError as exc:
            self._error = str(exc)
            self.changed.emit()
            return False
        if any(option["name"].casefold() == name.casefold() for option in self._known_options):
            self._error = "That person already exists. Choose Assign to existing person."
            self.changed.emit()
            return False
        return self._submit(self._selected_unknown, "new", name)

    @Slot(int, result=bool)
    def assignExisting(self, person_id):
        self._assert_owner()
        if not any(option["personId"] == person_id for option in self._known_options):
            self._error = "Select an existing person."
            self.changed.emit()
            return False
        return self._submit(self._selected_unknown, "existing", person_id)

    @Slot(str, bool, result=bool)
    def setIgnored(self, key, ignored):
        self._assert_owner()
        return self._submit(key, "ignore", ignored)

    @Slot()
    def retryProcessing(self):
        self._assert_owner()
        self._processing.retry()

    @Slot()
    def _drain(self):
        self._assert_owner()
        for _ in range(2):
            try:
                result = self._results.get_nowait()
            except Empty:
                break
            if result["session"] != self._session or result["folder"] != self._folder:
                continue
            self._pending = max(0, self._pending - 1)
            operation, key = result["operation"], result["key"]
            if operation != "suggest":
                self._busy = False
                self._busy_operation = ""
            if result["error"]:
                if operation != "suggest":
                    self._error = result["error"]
                self.changed.emit()
                continue
            if operation == "suggest":
                if key == self._selected_unknown:
                    self._suggestion = result["suggestion"]
                    self.changed.emit()
            elif operation == "ignore":
                if result["ignored"]:
                    self._ignored.add(key)
                else:
                    self._ignored.discard(key)
                _, unknown = project_people_rows(self._payload,
                    {option["personId"]: option["name"] for option in self._known_options}, self._ignored)
                self._release_detail_preview()
                self._unknown.set_rows(unknown, revision=str(self._session))
                if self._selected_unknown:
                    self._detail_preview_model = self._unknown
                    self._detail_lease = self._unknown.acquirePreview(self._selected_unknown, 256)
                self._error = ""
                self.changed.emit()
            else:
                cluster = next((item for item in self._payload["unknown"]
                                if stable_cluster_id(item) == key), None)
                if cluster is None:
                    self._error = "This cluster changed. Reopen the project."
                    self.changed.emit()
                    continue
                payload = dict(self._payload)
                payload["unknown"] = [item for item in self._payload["unknown"]
                                      if stable_cluster_id(item) != key]
                payload["known"] = list(self._payload["known"]) + [
                    dict(path=face["path"], bbox=face["bbox"], person_id=result["person_id"])
                    for face in cluster["faces"]]
                self._database = SQLiteFaceDatabase(self._database_path, CosineEmbeddingSimilarity())
                self._release_detail_preview()
                self._selected_unknown = ""
                self._suggestion = {}
                self._error = ""
                self._processing.replace_people_result(payload)
        if self._pending == 0 and not self._busy:
            self._timer.stop()

    @Slot()
    def shutdown(self):
        self._assert_owner()
        self._canceled.set()
        self._release_detail_preview()
        self._stop.set()
        self._timer.stop()
        self._worker.join(timeout=0.5)
        self._known.close()
        self._unknown.close()
        self._detail_photos.close()
