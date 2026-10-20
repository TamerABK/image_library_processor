"""One project-scoped Favorites owner; no discovery or analysis work."""
from __future__ import annotations

import logging
from pathlib import Path
from queue import Empty, Full, Queue
import sqlite3
import threading

from PySide6.QtCore import QObject, Property, QThread, QTimer, Signal, Slot

from ui.models import ResultItem
from .favorites_store import FavoritesStore
from .photo_model import PhotoListModel, photo_identity


log = logging.getLogger(__name__)
_SAVE_ERROR = "Favorite could not be saved. Check application data permissions."
_LOAD_ERROR = "Favorites could not be loaded. Check application data permissions."


def _favorites_worker(tasks, results, stop, database_path):
    while not stop.is_set():
        try:
            generation, folder, operation, path, favorite = tasks.get(timeout=0.05)
        except Empty:
            continue
        payload, error = None, ""
        try:
            store = FavoritesStore(database_path)
            if operation == "load":
                payload = store.favorite_paths(folder)
            else:
                store.set_favorite(folder, path, favorite)
        except (OSError, sqlite3.Error):
            log.exception("favorite_%s_failed folder=%s path=%s", operation, folder, path)
            error = _LOAD_ERROR if operation == "load" else _SAVE_ERROR
        result = (generation, folder, operation, path, favorite, payload, error)
        while not stop.is_set():
            try:
                results.put(result, timeout=0.05)
                break
            except Full:
                continue


class FavoritesController(QObject):
    changed = Signal()
    favoriteChanged = Signal(str, bool)
    pendingChanged = Signal(str, bool)

    def __init__(self, projects, master_photos, thumbnails, parent=None, *, database_path=None):
        super().__init__(parent)
        self._projects, self._master = projects, master_photos
        self._model = PhotoListModel(thumbnails, self)
        self._folder = ""
        self._generation = 0
        self._loaded = False
        self._descending = False
        self._favorites: set[str] = set()
        self._available: set[str] = set()
        self._pending: set[str] = set()
        self._inflight = 0
        self._error = ""
        self._closed = False
        self._tasks, self._results = Queue(maxsize=64), Queue(maxsize=64)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=_favorites_worker,
            args=(self._tasks, self._results, self._stop, database_path),
            name="room36-favorites", daemon=True)
        self._worker.start()
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._drain)
        projects.projectOpening.connect(self._project_opening)
        projects.projectOpened.connect(self._project_opened)
        master_photos.modelReset.connect(self._master_reset)
        if projects.activeProjectFolder:
            self._project_opened()

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Favorite models must be changed on their Qt owner thread")

    @Property(QObject, constant=True)
    def photoModel(self):
        return self._model

    @Property(int, notify=changed)
    def count(self):
        return self._model.rowCount() if self._loaded else -1

    @Property(str, notify=changed)
    def error(self):
        return self._error

    @Property(bool, notify=changed)
    def loaded(self):
        return self._loaded

    def _item(self, path):
        photo = self._master.photo(path)
        return ResultItem(Path(path), photo.get("title", Path(path).name), photo.get("detail", ""),
                          badge_text=photo.get("badgeText", ""))

    def _refresh_model(self):
        self._available = set(self._master.identities())
        paths = sorted(self._favorites & self._available,
                       key=lambda value: (Path(value).name.casefold(), value))
        self._model.replace_items([self._item(path) for path in paths], preserve_selection=False)
        if self._descending:
            self._model.sortByFilename(True)
        self.changed.emit()

    @Slot(bool)
    def sortByFilename(self, descending=False):
        self._assert_owner()
        self._descending = descending
        self._model.sortByFilename(descending)

    @Slot(str)
    def _project_opening(self, _folder):
        self._assert_owner()
        self._generation += 1
        self._folder = ""
        self._loaded = False
        self._favorites.clear()
        self._available.clear()
        self._pending.clear()
        self._error = ""
        self._model.replace_items([], preserve_selection=False)
        self.changed.emit()

    @Slot(str, str)
    def _project_opened(self, *_):
        self._assert_owner()
        folder = self._projects.activeProjectFolder
        if not folder or self._closed:
            return
        self._folder = photo_identity(folder)
        self._available = set(self._master.identities())
        try:
            self._tasks.put_nowait((self._generation, self._folder, "load", "", False))
        except Full:
            self._error = _LOAD_ERROR
            self.changed.emit()
            return
        self._inflight += 1
        self._timer.start()

    @Slot()
    def _master_reset(self):
        self._assert_owner()
        if self._folder and self._loaded:
            self._refresh_model()

    @Slot(str, result=bool)
    def isFavorite(self, path):
        return photo_identity(path) in self._favorites if path else False

    @Slot(str, result=bool)
    def isPending(self, path):
        return photo_identity(path) in self._pending if path else False

    @Slot(str, bool, result=bool)
    def setFavorite(self, path, favorite):
        self._assert_owner()
        if self._closed or not self._loaded or not self._folder or not path:
            return False
        path = photo_identity(path)
        if path not in self._available or path in self._pending:
            return False
        if (path in self._favorites) == favorite:
            return True
        try:
            self._tasks.put_nowait((self._generation, self._folder, "write", path, favorite))
        except Full:
            self._error = _SAVE_ERROR
            self.changed.emit()
            return False
        self._inflight += 1
        self._pending.add(path)
        self._error = ""
        self.pendingChanged.emit(path, True)
        self.changed.emit()
        self._timer.start()
        return True

    @Slot(str, result=bool)
    def toggle(self, path):
        return self.setFavorite(path, not self.isFavorite(path))

    @Slot()
    def _drain(self):
        self._assert_owner()
        while True:
            try:
                generation, folder, operation, path, favorite, payload, error = self._results.get_nowait()
            except Empty:
                break
            self._inflight -= 1
            if generation != self._generation or folder != self._folder or self._closed:
                continue
            if operation == "load":
                self._loaded = not bool(error)
                self._error = error
                self._favorites = payload or set()
                self._refresh_model()
                continue
            self._pending.discard(path)
            self.pendingChanged.emit(path, False)
            if error:
                self._error = error
            else:
                self._error = ""
                if favorite:
                    self._favorites.add(path)
                    if path in self._available:
                        self._model.append_items([self._item(path)])
                        self._model.sortByFilename(self._descending)
                else:
                    self._favorites.discard(path)
                    self._model.removePhoto(path)
                self.favoriteChanged.emit(path, favorite)
            self.changed.emit()
        if self._inflight == 0:
            self._timer.stop()

    @Slot()
    def shutdown(self):
        self._assert_owner()
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._stop.set()
        self._timer.stop()
        self._worker.join(timeout=0.5)
        self._model.close()
