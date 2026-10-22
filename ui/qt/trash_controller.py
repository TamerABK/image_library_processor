"""Project-scoped final review and explicitly confirmed OS Trash moves."""
from __future__ import annotations

import logging
from pathlib import Path
from queue import Empty, Full, Queue
import sqlite3
import threading

from PySide6.QtCore import QObject, Property, QThread, QTimer, Signal, Slot
from send2trash import send2trash

from automatic_scan.state_store import AutomaticStateStore, FolderAutomaticState
from .photo_model import photo_identity
from .trash_data import project_trash_rows
from .trash_model import TrashPhotoModel


log = logging.getLogger(__name__)


def _trash_worker(tasks, results, stop, state_root):
    while True:
        try:
            generation, folder, action, payload = tasks.get(timeout=0.05)
        except Empty:
            if stop.is_set():
                return
            continue
        output = {"projectionGeneration": payload} if action == "load" else {}
        try:
            store = AutomaticStateStore(state_root)
        except (OSError, sqlite3.Error):
            log.exception("trash_store_open_failed folder=%s", folder)
            if action == "execute":
                output.update(succeeded=[], failed=[row["photoId"] for row in payload],
                              reasons={row["photoId"]: "Removal history is unavailable. No photos were moved."
                                       for row in payload})
            else:
                output["error"] = "Removal decisions are unavailable. Check application data permissions."
            while not stop.is_set():
                try:
                    results.put((generation, folder, action, output), timeout=0.05)
                    break
                except Full:
                    continue
            continue
        if action == "load":
            try:
                output["decisions"] = store.load(folder).image_decisions
            except (OSError, sqlite3.Error, ValueError, TypeError):
                log.exception("trash_decisions_load_failed folder=%s", folder)
                output["error"] = "Removal decisions could not be loaded. Check application data permissions."
        elif action == "restore":
            try:
                path, revision, blur_revision, groups = payload
                current = Path(path).stat()
                if revision is None or (current.st_size, current.st_mtime_ns) != tuple(revision):
                    raise ValueError("This photo changed. Reopen the project before restoring its decision.")
                output["cleared"] = store.clear_removal_intents(
                    folder, path, blur_revision=blur_revision, duplicate_groups=groups)
                output["path"] = path
            except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
                log.exception("trash_restore_failed folder=%s", folder)
                output["error"] = str(exc) if isinstance(exc, ValueError) else (
                    "Removal decision could not be restored. Check application data permissions.")
        elif action == "execute":
            succeeded, failed, reasons = [], [], {}
            for row in payload:
                path, expected = row["photoId"], row["revision"]
                try:
                    current = Path(path).stat()
                    if not Path(path).is_file():
                        raise FileNotFoundError(path)
                    if (current.st_size, current.st_mtime_ns) != tuple(expected):
                        reasons[path] = "Photo changed since review. Review it again before removal."
                        failed.append(path)
                        continue
                    send2trash(path)
                    succeeded.append(path)
                except FileNotFoundError:
                    reasons[path] = "File no longer exists."
                    failed.append(path)
                except Exception:
                    log.exception("trash_move_failed folder=%s path=%s", folder, path)
                    reasons[path] = "This photo could not be moved to Trash. Check file permissions."
                    failed.append(path)
            output.update(succeeded=succeeded, failed=failed, reasons=reasons)
            try:
                state = FolderAutomaticState(folder=folder)
                store.record_deletion_transaction(state, paths=succeeded, failed_paths=failed)
                store.save_deletion_transaction(folder, state.deletion_transactions[-1])
            except (OSError, sqlite3.Error):
                log.exception("trash_history_save_failed folder=%s", folder)
                output["historyError"] = (
                    "Photos were moved to Trash, but Room 36 could not save the deletion history."
                    if succeeded else "Room 36 could not save the deletion attempt history.")
        result = (generation, folder, action, output)
        while not stop.is_set():
            try:
                results.put(result, timeout=0.05)
                break
            except Full:
                continue


class TrashController(QObject):
    changed = Signal()

    def __init__(self, projects, master_photos, processing, duplicates, favorites, inspector,
                 thumbnails, parent=None, *, state_root=None):
        super().__init__(parent)
        self._projects, self._master, self._processing = projects, master_photos, processing
        self._duplicates, self._favorites, self._inspector = duplicates, favorites, inspector
        self._model = TrashPhotoModel(thumbnails, self)
        self._folder = ""
        self._generation = 0
        self._projection_generation = 0
        self._projection_ready = False
        self._loaded = False
        self._busy = False
        self._error = ""
        self._summary = ""
        self._plan = None
        self._pending_refresh = False
        self._refresh_dirty = False
        self._refresh_after_delete = ""
        self._preserve_error_after_delete = False
        self._failed_reasons = {}
        self._pending_overlap_paths = set()
        self._closed = False
        self._tasks, self._results = Queue(maxsize=4), Queue(maxsize=4)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=_trash_worker,
            args=(self._tasks, self._results, self._stop, state_root),
            name="room36-trash", daemon=True)
        self._worker.start()
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._drain)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(30)
        self._refresh_timer.timeout.connect(self._load)
        projects.projectOpening.connect(self._project_opening)
        projects.projectOpened.connect(self._project_opened)
        processing.blurModel.reviewChanged.connect(self._schedule_refresh)
        processing.sessionChanged.connect(self._schedule_refresh)
        duplicates.changed.connect(self._schedule_refresh)
        favorites.changed.connect(self._schedule_refresh)
        master_photos.modelReset.connect(self._schedule_refresh)
        if projects.activeProjectFolder:
            self._project_opened()

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Trash model mutations require the owning Qt thread")

    @Property(QObject, constant=True)
    def photoModel(self):
        return self._model

    @Property(int, notify=changed)
    def count(self):
        return len(self._model.all_rows()) if self._loaded else -1

    @Property(int, notify=changed)
    def readyCount(self):
        return sum(row["executable"] for row in self._model.all_rows())

    @Property(int, notify=changed)
    def attentionCount(self):
        return sum(not row["executable"] for row in self._model.all_rows())

    @Property(bool, notify=changed)
    def busy(self):
        return self._busy

    @Property(str, notify=changed)
    def error(self):
        return self._error

    @Property(str, notify=changed)
    def summary(self):
        return self._summary

    @Property(bool, notify=changed)
    def loaded(self):
        return self._loaded

    @Property(bool, notify=changed)
    def projectionReady(self):
        return self._destructive_ready()

    @Slot(str)
    def _project_opening(self, folder):
        self._assert_owner()
        same_refresh = self._refresh_after_delete == photo_identity(folder)
        self._refresh_after_delete = ""
        self._generation += 1
        self._projection_generation += 1
        self._projection_ready = False
        self._folder = ""
        self._loaded = False
        self._plan = None
        self._busy = False
        if not same_refresh:
            self._error = ""
            self._summary = ""
            self._preserve_error_after_delete = False
            self._failed_reasons = {}
        self._refresh_timer.stop()
        self._pending_refresh = False
        self._refresh_dirty = False
        self._model.set_rows([])
        self.changed.emit()

    @Slot(str, str)
    def _project_opened(self, *_):
        self._assert_owner()
        folder = self._projects.activeProjectFolder
        if folder and not self._closed:
            self._folder = photo_identity(folder)
            self._schedule_refresh()
            self._refresh_active_overlap()

    @Slot()
    def _schedule_refresh(self, *_):
        self._assert_owner()
        self._projection_generation += 1
        self._projection_ready = False
        self._plan = None
        self._block_stale_rows()
        if self._folder and not self._closed and not self._busy:
            if self._pending_refresh:
                self._refresh_dirty = True
            self._refresh_timer.start()

    def _refresh_active_overlap(self):
        if not self._pending_overlap_paths or self._projects.busy:
            return
        active = self._projects.activeProjectFolder
        if not active:
            return
        overlapping = self._pending_overlap_paths.intersection(self._master.identities())
        self._pending_overlap_paths.clear()
        if overlapping:
            self._projects.openFolder(active)

    def _review_ready(self):
        return bool(self._favorites.loaded and self._favorites.pendingCount == 0
                    and self._duplicates.state == "COMPLETE"
                    and self._duplicates.reviewReady)

    def _block_stale_rows(self):
        """Invalidate every prior executable row and destructive selection synchronously."""
        rows = self._model.all_rows()
        reason = "Review choices changed. Refreshing removal status."
        blocked = []
        for row in rows:
            row = dict(row)
            if row["executable"]:
                row["conflictReason"] = (row["conflictReason"] + " " + reason).strip()
                row["executable"] = False
                row["badgeText"] = "Needs attention"
            blocked.append(row)
        self._model.set_rows(blocked)
        self.changed.emit()

    def _load(self):
        if not self._folder or self._busy:
            return
        if self._pending_refresh:
            return
        try:
            self._tasks.put_nowait((self._generation, self._folder, "load",
                                    self._projection_generation))
        except Full:
            self._error = "Removal decisions are busy. Try again shortly."
            self.changed.emit()
            return
        self._pending_refresh = True
        self._timer.start()

    def _project(self, decisions):
        available = self._master.identities()
        revisions = {path: (row["file_size"], row["mtime_ns"])
                     for path, row in self._processing.blurModel.results.items()}
        favorites = {path for path in available if self._favorites.isFavorite(path)}
        ready = self._review_ready()
        rows = project_trash_rows(available, decisions, self._duplicates.review_rows(), favorites,
                                  revisions, review_ready=ready)
        for row in rows:
            failed_reason = self._failed_reasons.get(row["photoId"])
            if failed_reason:
                row["conflictReason"] = failed_reason
                row["executable"] = False
                row["badgeText"] = "Needs attention"
        self._model.set_rows(rows)
        self._loaded = True
        self._projection_ready = ready
        self.changed.emit()

    def _destructive_ready(self):
        return bool(self._projection_ready and self._loaded and not self._busy
                    and self._folder and self._folder == self._projects.activeProjectFolder
                    and self._review_ready())

    @Slot(str, result="QVariantMap")
    def row(self, path):
        row = self._model.row(path)
        if not row:
            return {}
        return {key: value for key, value in row.items() if key not in ("revision", "duplicateGroups")}

    @Slot(str, result=bool)
    def isExecutable(self, path):
        row = self._model.row(path)
        return bool(self._destructive_ready() and row and row["executable"]
                    and self._master.photo(path) and not self._favorites.isPending(path))

    @Slot(str, result=bool)
    def restorePhoto(self, path):
        self._assert_owner()
        row = self._model.row(path)
        if not self._destructive_ready() or not row:
            return False
        payload = (path, row["revision"], row["revision"] if row["blurRemove"] else None,
                   dict(row["duplicateGroups"]))
        try:
            self._tasks.put_nowait((self._generation, self._folder, "restore", payload))
        except Full:
            self._error = "Another Trash action is still running."
            self.changed.emit()
            return False
        self._busy = True
        self._schedule_refresh()
        self._timer.start()
        self.changed.emit()
        return True

    @Slot(result="QVariantMap")
    def prepareSelection(self):
        self._assert_owner()
        rows = self._model.selected_rows()
        self._plan = None
        if (not self._destructive_ready() or not self._model.selection_is_visible()
                or not rows or len(rows) != self._model.selectedCount
                or any(not self.isExecutable(row["photoId"]) for row in rows)):
            if not self._model.selection_is_visible():
                self._model.clearSelection()
            return {}
        self._plan = (self._generation, self._folder, self._projection_generation,
                      tuple(dict(row) for row in rows))
        blur = sum(row["removalSources"] == ["Blur"] for row in rows)
        duplicates = sum(row["removalSources"] == ["Near duplicates"] for row in rows)
        both = len(rows) - blur - duplicates
        return dict(count=len(rows), totalBytes=sum(row["fileSize"] for row in rows),
                    blur=blur, duplicates=duplicates, both=both)

    @Slot(result=bool)
    def executeSelection(self):
        self._assert_owner()
        plan = self._plan
        self._plan = None
        if (not self._destructive_ready() or not plan or
                plan[:3] != (self._generation, self._folder, self._projection_generation)):
            return False
        rows = plan[3]
        if (not self._model.selection_is_visible() or
                {row["photoId"] for row in rows} !=
                {row["photoId"] for row in self._model.selected_rows()} or
                any(not self.isExecutable(row["photoId"]) or
                    self._model.row(row["photoId"]) != row for row in rows)):
            self._error = "Review choices changed. Select the photos again before moving them."
            self.changed.emit()
            return False
        try:
            self._tasks.put_nowait((self._generation, self._folder, "execute", rows))
        except Full:
            self._error = "Another Trash action is still running."
            self.changed.emit()
            return False
        self._inspector.close()
        self._busy = True
        self._error = ""
        self._timer.start()
        self.changed.emit()
        return True

    @Slot()
    def _drain(self):
        self._assert_owner()
        while True:
            try:
                generation, folder, action, output = self._results.get_nowait()
            except Empty:
                break
            if generation != self._generation or folder != self._folder or self._closed:
                if action == "execute" and not self._closed and output.get("succeeded"):
                    self._pending_overlap_paths.update(output["succeeded"])
                    self._refresh_active_overlap()
                continue
            if action == "load":
                self._pending_refresh = False
                if output["projectionGeneration"] != self._projection_generation:
                    self._refresh_dirty = False
                    self._refresh_timer.start()
                    continue
                if output.get("error"):
                    self._error = output["error"]
                    self._loaded = False
                    self._projection_ready = False
                    self._model.set_rows([])
                else:
                    if not self._preserve_error_after_delete:
                        self._error = ""
                    self._project(output["decisions"])
                if self._refresh_dirty:
                    self._refresh_dirty = False
                    self._refresh_timer.start()
            elif action == "restore":
                self._busy = False
                self._error = output.get("error", "")
                if not self._error:
                    path = output["path"]
                    blur, duplicate = output["cleared"]
                    if blur and path in self._processing.blurModel.results:
                        self._processing.blurModel.apply_review(path, "UNDECIDED")
                    if duplicate:
                        try:
                            self._duplicates.reload_reviews()
                        except (OSError, sqlite3.Error, ValueError, TypeError):
                            log.exception("trash_duplicate_reload_failed folder=%s", folder)
                            self._error = "Duplicate review could not be refreshed. Reopen the project."
                self._schedule_refresh()
            elif action == "execute":
                self._busy = False
                moved, failed = output["succeeded"], output["failed"]
                moved_word = "photo" if len(moved) == 1 else "photos"
                failed_word = "photo" if len(failed) == 1 else "photos"
                self._summary = (f"{len(moved)} {moved_word} moved to Trash. "
                                 f"{len(failed)} {failed_word} could not be moved.")
                self._error = output.get("historyError", "")
                self._failed_reasons.update({path: reason for path, reason in output["reasons"].items()
                                              if reason.startswith("Photo changed")})
                if failed and not self._error:
                    self._error = next(iter(output["reasons"].values()), "Some photos could not be moved to Trash.")
                self._preserve_error_after_delete = bool(self._error)
                missing = any(reason == "File no longer exists." for reason in output["reasons"].values())
                if moved or missing:
                    self._refresh_after_delete = folder
                    self._projects.openFolder(folder)  # One discovery invalidates processing and all feature models.
                    if moved:
                        self._master.removePhotos(moved)  # Hide moved paths while discovery is pending.
                else:
                    self._schedule_refresh()
            self.changed.emit()
        if not self._pending_refresh and not self._busy:
            self._timer.stop()

    @Slot()
    def shutdown(self):
        self._assert_owner()
        self._closed = True
        self._generation += 1
        self._refresh_timer.stop()
        self._timer.stop()
        self._stop.set()
        self._worker.join()  # A confirmed physical batch must finish before exit.
        self._model.close()
