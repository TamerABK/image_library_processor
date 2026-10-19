"""Project-scoped review of central duplicate and ranking stage results."""
from __future__ import annotations

import logging
from pathlib import Path
from queue import Empty, Full, Queue
import sqlite3
import threading

from PySide6.QtCore import QObject, Property, QThread, QTimer, Signal, Slot

from automatic_scan.state_store import AutomaticStateStore
from ui.models import ResultItem
from .duplicate_data import group_revision, project_duplicate_rows
from .duplicate_model import DuplicateGroupModel
from .photo_model import PhotoListModel


log = logging.getLogger(__name__)


def _review_worker(tasks, results, stop, state_root):
    while not stop.is_set():
        try:
            task = tasks.get(timeout=0.05)
        except Empty:
            continue
        session, folder, group_id, signature, keepers, intents, reviewed, revisions, canceled = task
        if canceled.is_set():
            continue
        error = ""
        try:
            for path, expected in revisions.items():
                stat = Path(path).stat()
                if (stat.st_size, stat.st_mtime_ns) != expected:
                    raise ValueError("A group photo changed. Reopen the project before reviewing it.")
            if group_revision(list(revisions), revisions) != signature:
                raise ValueError("This duplicate group changed. Reopen the project before reviewing it.")
            if canceled.is_set():
                continue
            AutomaticStateStore(state_root).save_duplicate_review(
                folder, group_id, signature, keepers, intents, reviewed)
        except (OSError, sqlite3.Error):
            log.exception("duplicate_review_save_failed folder=%s group=%s", folder, group_id)
            error = "Duplicate review could not be saved. Check application data permissions."
        except ValueError as exc:
            error = str(exc)
        result = (session, folder, group_id, signature, keepers, intents, reviewed, error)
        while not stop.is_set():
            try:
                results.put(result, timeout=0.05)
                break
            except Full:
                continue


class DuplicateReviewController(QObject):
    changed = Signal()

    def __init__(self, processing, thumbnails, parent=None, *, state_root=None):
        super().__init__(parent)
        self._processing = processing
        self._store = AutomaticStateStore(state_root)
        self._groups = DuplicateGroupModel(thumbnails, self)
        self._detail = PhotoListModel(thumbnails, self)
        self._folder = ""
        self._session = processing.sessionId
        self._duplicates_payload = None
        self._ranking_payload = None
        self._persisted = {}
        self._selected = ""
        self._state = "NOT_STARTED"
        self._ranking_state = "NOT_STARTED"
        self._stage_error = ""
        self._error = ""
        self._busy = False
        self._canceled = threading.Event()
        self._tasks, self._results = Queue(maxsize=2), Queue(maxsize=2)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=_review_worker,
            args=(self._tasks, self._results, self._stop, state_root),
            name="room36-duplicate-review", daemon=True)
        self._worker.start()
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._drain)
        processing.sessionChanged.connect(self._reset_session)
        processing.changed.connect(self._sync)
        self._groups.changed.connect(self.changed)
        self._sync()

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Duplicate review mutations require the owning Qt thread")

    @Property(QObject, constant=True)
    def groupModel(self):
        return self._groups

    @Property(QObject, constant=True)
    def detailPhotoModel(self):
        return self._detail

    @Property(str, notify=changed)
    def state(self):
        return self._state

    @Property(str, notify=changed)
    def rankingState(self):
        return self._ranking_state

    @Property(str, notify=changed)
    def stageError(self):
        return self._stage_error

    @Property(str, notify=changed)
    def error(self):
        return self._error

    @Property(bool, notify=changed)
    def busy(self):
        return self._busy

    @Property(int, notify=changed)
    def needsReviewCount(self):
        return self._groups.needsReviewCount if self._state == "COMPLETE" else -1

    @Property("QVariantMap", notify=changed)
    def selectedGroup(self):
        row = self._groups.get_row(self._selected)
        return {key: value for key, value in row.items()
                if key not in ("paths", "ticket", "edge", "previewSource") } if row else {}

    @Property("QVariantList", notify=changed)
    def detailMembers(self):
        row = self._groups.get_row(self._selected)
        if not row:
            return []
        return [dict(photoId=path, filename=Path(path).name,
                     recommended=path in row["recommendedKeepers"],
                     keeper=path in row["currentKeepers"],
                     intent=row["intents"].get(path, "UNDECIDED"),
                     explanation=row["explanation"] if path == row["representativePhotoId"] else [])
                for path in row["paths"]]

    @Slot()
    def _reset_session(self):
        self._assert_owner()
        self._canceled.set()
        self._canceled = threading.Event()
        self._session = self._processing.sessionId
        self._folder = ""
        self._duplicates_payload = self._ranking_payload = None
        self._persisted = {}
        self._selected = ""
        self._state = self._ranking_state = "NOT_STARTED"
        self._stage_error = self._error = ""
        self._busy = False
        self._groups.set_rows([])
        self._detail.replace_items([], preserve_selection=False)
        self.changed.emit()

    @Slot()
    def _sync(self):
        self._assert_owner()
        folder = self._processing.projectFolder
        if folder and folder != self._folder:
            self._folder = folder
            try:
                self._persisted = self._store.duplicate_reviews(folder)
            except (OSError, sqlite3.Error, ValueError, TypeError):
                log.exception("duplicate_review_load_failed folder=%s", folder)
                self._persisted = {}
                self._error = "Saved duplicate reviews could not be loaded. Check application data permissions."
        stage = self._processing.stage("duplicates")
        ranking_stage = self._processing.stage("ranking")
        state = stage["state"]
        ranking_state = ranking_stage["state"]
        error = stage.get("error", "")
        duplicates = self._processing.results.get("duplicates") if state == "COMPLETE" else None
        ranking = self._processing.results.get("ranking") if ranking_state == "COMPLETE" else None
        changed = (duplicates is not self._duplicates_payload or ranking is not self._ranking_payload or
                   state != self._state or ranking_state != self._ranking_state or error != self._stage_error)
        if not changed:
            return
        self._state, self._ranking_state, self._stage_error = state, ranking_state, error
        self._duplicates_payload, self._ranking_payload = duplicates, ranking
        self._rebuild()

    def _rebuild(self):
        revisions = {path: (row["file_size"], row["mtime_ns"])
                     for path, row in self._processing.blurModel.results.items()}
        rows = project_duplicate_rows(self._duplicates_payload, self._ranking_payload,
                                       revisions, self._persisted,
                                       ranking_failed=self._ranking_state in ("FAILED", "CANCELLED") or
                                           (self._ranking_state == "COMPLETE" and not self._ranking_payload))
        self._groups.set_rows(rows)
        if self._selected and self._groups.get_row(self._selected) is None:
            self._selected = ""
        self._refresh_detail()
        self.changed.emit()

    def _refresh_detail(self):
        row = self._groups.get_row(self._selected)
        items = []
        if row:
            for path in row["paths"]:
                intent = row["intents"].get(path, "UNDECIDED")
                badge = ("Remove" if intent == "REMOVE" else "Keep" if intent == "KEEP" else
                         "Keeper" if path in row["currentKeepers"] else
                         "Recommended" if path in row["recommendedKeepers"] else "Alternative")
                items.append(ResultItem(Path(path), Path(path).name, "Duplicate group photo",
                                        badge_text=badge))
        self._detail.replace_items(items, preserve_selection=False)

    @Slot(str)
    def openGroup(self, group_id):
        self._assert_owner()
        if self._groups.get_row(group_id):
            self._selected = group_id
            self._refresh_detail()
            self.changed.emit()

    @Slot()
    def back(self):
        self._assert_owner()
        self._selected = ""
        self._detail.replace_items([], preserve_selection=False)
        self.changed.emit()

    def _submit(self, keepers, intents, reviewed):
        self._assert_owner()
        row = self._groups.get_row(self._selected)
        if self._busy or not row or self._state != "COMPLETE" or not row["signature"]:
            return False
        if row["recommendationState"] != "ready":
            self._error = "Keeper recommendation is still pending."
            self.changed.emit()
            return False
        if len(keepers) != len(row["recommendedKeepers"]) or not set(keepers).issubset(row["paths"]):
            return False
        for path, intent in intents.items():
            previous = self._processing.blurModel.results.get(path, {}).get("reviewState", "UNDECIDED")
            if (intent == "REMOVE" and previous in ("KEEP", "REVIEW")) or (
                    intent == "KEEP" and previous == "REMOVE"):
                self._error = ("This photo has a conflicting image-quality review decision. "
                               "Resolve that choice before confirming the duplicate group.")
                self.changed.emit()
                return False
        revisions = {path: (self._processing.blurModel.results[path]["file_size"],
                            self._processing.blurModel.results[path]["mtime_ns"])
                     for path in row["paths"]}
        task = (self._session, self._folder, row["groupId"], row["signature"],
                list(keepers), dict(intents), reviewed, revisions, self._canceled)
        try:
            self._tasks.put_nowait(task)
        except Full:
            self._error = "A duplicate review is already being saved."
            self.changed.emit()
            return False
        self._busy = True
        self._error = ""
        self._timer.start()
        self.changed.emit()
        return True

    @Slot(str, result=bool)
    def setKeeper(self, path):
        row = self._groups.get_row(self._selected)
        if not row or path not in row["paths"]:
            return False
        keepers = list(row["currentKeepers"])
        if path in keepers:
            return True
        if not keepers:
            return False
        keepers[-1] = path
        intents = dict(row["intents"])
        if row["reviewed"]:
            intents = {member: "KEEP" if member in keepers else "REMOVE" for member in row["paths"]}
        else:
            intents.pop(path, None)
        return self._submit(keepers, intents, row["reviewed"])

    @Slot(str, str, result=bool)
    def setIntent(self, path, intent):
        row = self._groups.get_row(self._selected)
        if not row or path not in row["paths"] or intent not in ("KEEP", "REMOVE", "UNDECIDED"):
            return False
        if path in row["currentKeepers"] and intent == "REMOVE":
            self._error = "A current keeper cannot be marked for removal. Choose another keeper first."
            self.changed.emit()
            return False
        intents = dict(row["intents"])
        if intent == "UNDECIDED":
            intents.pop(path, None)
        else:
            intents[path] = intent
        return self._submit(row["currentKeepers"], intents, False)

    @Slot(result=bool)
    def confirmGroup(self):
        row = self._groups.get_row(self._selected)
        if not row:
            return False
        if any(row["intents"].get(path) == "KEEP" for path in row["paths"]
               if path not in row["currentKeepers"]):
            self._error = "An alternative is marked Keep. Restore it before confirming removal of alternatives."
            self.changed.emit()
            return False
        intents = {path: "KEEP" if path in row["currentKeepers"] else "REMOVE"
                   for path in row["paths"]}
        return self._submit(row["currentKeepers"], intents, True)

    @Slot()
    def retryProcessing(self):
        self._processing.retry()

    @Slot()
    def _drain(self):
        self._assert_owner()
        while True:
            try:
                session, folder, group_id, signature, keepers, intents, reviewed, error = self._results.get_nowait()
            except Empty:
                break
            if session != self._session or folder != self._folder:
                continue
            self._busy = False
            row = self._groups.get_row(group_id)
            if row is None or row["signature"] != signature:
                continue
            if error:
                self._error = error
            else:
                self._error = ""
                self._persisted[group_id] = dict(signature=signature, keepers=keepers,
                                                  intents=intents, reviewed=reviewed)
                self._rebuild()
            self.changed.emit()
        if not self._busy:
            self._timer.stop()

    @Slot()
    def shutdown(self):
        self._assert_owner()
        self._canceled.set()
        self._stop.set()
        self._timer.stop()
        self._worker.join(timeout=0.5)
        self._groups.close()
        self._detail.close()
