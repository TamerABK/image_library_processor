"""Coalesced directory notifications for the active project's existing refresh path."""
from __future__ import annotations

import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading

from PySide6.QtCore import QFileSystemWatcher, QObject, Property, QThread, QTimer, Signal, Slot

from automatic_scan.scan_logging import get_scan_logger
from .photo_model import photo_identity


WATCH_DEBOUNCE_MS = 900
WATCH_FALLBACK_MS = 30000


def enumerate_directories(folder: str) -> set[str]:
    """Enumerate paths only; never decode images or register watches off-thread."""
    found = set()
    pending = [folder]
    while pending:
        path = pending.pop()
        if path in found:
            continue
        try:
            with os.scandir(path) as entries:
                found.add(path)
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue
    parent = str(Path(folder).parent)
    if parent != folder and Path(parent).is_dir():
        found.add(parent)
    return found


def _enumeration_worker(tasks, results, stop, enumerate_fn):
    while not stop.is_set():
        try:
            generation, request_id, folder = tasks.get(timeout=0.05)
        except Empty:
            continue
        try:
            paths = enumerate_fn(folder)
        except Exception:
            get_scan_logger().exception("project_watch_enumeration_failed folder=%s", folder)
            paths = set()
        if stop.is_set():
            break
        try:
            results.put_nowait((generation, request_id, folder, paths))
        except Full:
            try:
                results.get_nowait()
            except Empty:
                pass
            results.put_nowait((generation, request_id, folder, paths))


class ProjectFilesystemWatcher(QObject):
    changed = Signal()

    def __init__(self, projects, parent=None, *, watcher=None, enumerate_fn=enumerate_directories,
                 debounce_ms=WATCH_DEBOUNCE_MS):
        super().__init__(parent)
        self._projects = projects
        self._watcher = watcher or QFileSystemWatcher(self)
        self._folder = photo_identity(projects.activeProjectFolder) if projects.activeProjectFolder else ""
        self._generation = 0
        self._event_handler = None
        self._connect_events()
        self._request_id = 0
        self._watched = set()
        self._failed = set()
        self._dirty = False
        self._closed = False
        self._tasks, self._results = Queue(maxsize=1), Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=_enumeration_worker,
            args=(self._tasks, self._results, self._stop, enumerate_fn),
            name="room36-project-watch-enumeration", daemon=True)
        self._worker.start()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._reconcile)
        self._result_timer = QTimer(self)
        self._result_timer.setInterval(30)
        self._result_timer.timeout.connect(self._drain)
        self._fallback = QTimer(self)
        self._fallback.setInterval(WATCH_FALLBACK_MS)
        self._fallback.timeout.connect(self._fallback_refresh)
        projects.projectOpening.connect(self._project_opening)
        projects.projectOpened.connect(self._project_opened)
        projects.busyChanged.connect(self._discovery_state_changed)
        if self._folder:
            self._request_enumeration()

    @Property(bool, notify=changed)
    def degraded(self):
        return bool(self._failed)

    @Property(int, notify=changed)
    def watchedDirectoryCount(self):
        return len(self._watched)

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Project watcher changes require the owning Qt thread")

    def _connect_events(self):
        if self._event_handler is not None:
            self._watcher.directoryChanged.disconnect(self._event_handler)
        generation = self._generation
        self._event_handler = lambda path: self.notifyDirectoryChanged(path, generation)
        self._watcher.directoryChanged.connect(self._event_handler)

    @Slot(str)
    def _project_opening(self, folder):
        self._assert_owner()
        normalized = photo_identity(folder)
        if normalized == self._folder:
            # An explicit refresh (notably after Room36 Trash) already covers
            # notifications received before its discovery snapshot starts.
            self._dirty = False
            self._debounce.stop()
            return
        self._generation += 1
        self._connect_events()
        self._folder = normalized
        self._dirty = False
        self._debounce.stop()
        if self._watched:
            self._watcher.removePaths(list(self._watched))
        self._watched.clear()
        self._failed.clear()
        self._fallback.stop()
        self._result_timer.stop()
        self.changed.emit()

    @Slot(str, str)
    def _project_opened(self, *_):
        if self._closed:
            return
        self._request_enumeration()
        if self._dirty:
            self._debounce.start()

    @Slot()
    def _discovery_state_changed(self):
        if not self._projects.busy:
            QTimer.singleShot(0, self._after_discovery)

    def _after_discovery(self):
        if self._closed or self._projects.busy:
            return
        active = self._projects.activeProjectFolder
        if active and photo_identity(active) != self._folder:
            self._project_opening(active)
            self._request_enumeration()
        if self._dirty:
            self._debounce.start()

    def _request_enumeration(self):
        if not self._folder or self._closed:
            return
        self._request_id += 1
        task = (self._generation, self._request_id, self._folder)
        try:
            self._tasks.put_nowait(task)
        except Full:
            try:
                self._tasks.get_nowait()
            except Empty:
                pass
            self._tasks.put_nowait(task)
        self._result_timer.start()

    @Slot()
    def _drain(self):
        self._assert_owner()
        try:
            generation, request_id, folder, paths = self._results.get_nowait()
        except Empty:
            return
        if (self._closed or generation != self._generation or folder != self._folder or
                request_id != self._request_id):
            return
        # Qt silently drops a directory watch when that directory disappears.
        # Re-read registrations so a recreated root or renamed child is added.
        self._watched = {photo_identity(path) for path in self._watcher.directories()}
        wanted = {photo_identity(path) for path in paths}
        removed = self._watched - wanted
        if removed:
            self._watcher.removePaths(list(removed))
        self._watched.difference_update(removed)
        additions = wanted - self._watched
        failed = set(self._watcher.addPaths(list(additions))) if additions else set()
        self._watched.update(additions - failed)
        self._failed = failed
        if failed:
            get_scan_logger().warning("project_watch_registration_failed count=%d folder=%s", len(failed), folder)
            self._fallback.start()
        else:
            self._fallback.stop()
        self.changed.emit()
        self._result_timer.stop()

    @Slot(str)
    def directoryChanged(self, path):
        self.notifyDirectoryChanged(path)

    def notifyDirectoryChanged(self, path, generation=None):
        """Deterministic event seam for tests; stale project events are ignored."""
        self._assert_owner()
        if self._closed or (generation is not None and generation != self._generation):
            return
        if photo_identity(path) not in self._watched:
            return
        self._dirty = True
        self._debounce.start()

    @Slot()
    def _reconcile(self):
        if self._closed or not self._dirty:
            return
        if self._projects.busy:
            return  # _discovery_state_changed schedules one follow-up.
        self._dirty = False
        self._projects.refreshActiveProject()

    @Slot()
    def _fallback_refresh(self):
        if not self._closed and self._failed and self._folder:
            self._dirty = True
            self._reconcile()

    @Slot()
    def shutdown(self):
        self._assert_owner()
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._watcher.directoryChanged.disconnect(self._event_handler)
        self._debounce.stop()
        self._result_timer.stop()
        self._fallback.stop()
        if self._watched:
            self._watcher.removePaths(list(self._watched))
        self._watched.clear()
        self._stop.set()
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            raise RuntimeError("Project watch enumeration worker did not stop")
