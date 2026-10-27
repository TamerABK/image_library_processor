"""Persistent Home covers, selected and decoded away from the Qt thread."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from queue import Empty, Queue
import sqlite3
import threading

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from app_state_database import AppStateDatabase
from image_file_utils import find_supported_files
from image_loader import ImageLoader


def _cover_for_project(database_path, project_id, folder, preferred_path=""):
    database = AppStateDatabase(database_path)
    cache_dir = Path(database_path).parent / "project_covers"
    try:
        with database.connect() as db:
            row = db.execute("SELECT source_path, file_size, mtime_ns, cache_path "
                             "FROM project_covers WHERE project_id=?", (project_id,)).fetchone()
        if row:
            path = Path(row["source_path"])
            try:
                stat = path.stat()
                if stat.st_size == row["file_size"] and stat.st_mtime_ns == row["mtime_ns"]:
                    cached = Path(row["cache_path"])
                    if cached.is_file():
                        return cached.as_uri()
                    preferred_path = str(path)
            except OSError:
                pass
        root = Path(folder)
        if not root.is_dir():
            return ""
        extensions = ImageLoader().supported_extensions()
        def candidates():
            if preferred_path:
                yield Path(preferred_path)
            # Enumeration is only needed for old projects or a missing/corrupt
            # selected source. Reopening a healthy cover checks one stat.
            for candidate in sorted(find_supported_files(root, extensions),
                                    key=lambda p: str(p).casefold()):
                if str(candidate) != preferred_path:
                    yield candidate
        loader = ImageLoader(max_decode_dimension=512)
        for path in candidates():
            try:
                stat = path.stat()
                image = loader.load_pil_for_hashing(path, max_dimension=512)
                if image is None:
                    continue
                cache_dir.mkdir(parents=True, exist_ok=True)
                revision = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
                key = sha256(f"{project_id}|{revision}".encode()).hexdigest()[:24]
                target = cache_dir / f"{key}.png"
                if not target.is_file():
                    output = BytesIO()
                    image.save(output, format="PNG")
                    temporary = cache_dir / f"{key}.{threading.get_ident()}.tmp"
                    temporary.write_bytes(output.getvalue())
                    os.replace(temporary, target)
                with database.connect() as db:
                    db.execute("INSERT INTO project_covers VALUES (?,?,?,?,?) ON CONFLICT(project_id) "
                               "DO UPDATE SET source_path=excluded.source_path, file_size=excluded.file_size, "
                               "mtime_ns=excluded.mtime_ns, cache_path=excluded.cache_path",
                               (project_id, str(path), stat.st_size, stat.st_mtime_ns, str(target)))
                if row and row["cache_path"] != str(target):
                    Path(row["cache_path"]).unlink(missing_ok=True)
                return target.as_uri()
            except (OSError, ValueError):
                continue
    except (OSError, ValueError, sqlite3.DatabaseError):
        pass
    return ""


def _worker(tasks, results, stop, database_path):
    while not stop.is_set():
        try:
            job = tasks.get(timeout=0.05)
        except Empty:
            continue
        ticket, project_id, folder, preferred = job
        try:
            source = _cover_for_project(database_path, project_id, folder, preferred)
        except Exception:
            source = ""
        if not stop.is_set():
            results.put((ticket, source))


class ProjectCoverService(QObject):
    completed = Signal(int, str)

    def __init__(self, database_path, parent=None, *, workers=2):
        super().__init__(parent)
        self._database_path = str(database_path)
        self._tasks = Queue()
        self._results = Queue()
        self._stop = threading.Event()
        self._ticket = 0
        self._pending = 0
        self._threads = [threading.Thread(target=_worker, args=(self._tasks, self._results,
            self._stop, self._database_path), daemon=True, name=f"room36-cover-{i}")
            for i in range(max(1, workers))]
        for thread in self._threads:
            thread.start()
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._drain)
        self.destroyed.connect(lambda *_: self._stop.set())

    def request(self, project_id, folder, preferred_path=""):
        if self._stop.is_set():
            return 0
        self._ticket += 1
        self._pending += 1
        self._tasks.put((self._ticket, project_id, folder, preferred_path))
        self._timer.start()
        return self._ticket

    @Slot()
    def _drain(self):
        while True:
            try:
                ticket, source = self._results.get_nowait()
            except Empty:
                break
            self._pending -= 1
            self.completed.emit(ticket, source)
        if self._pending == 0:
            self._timer.stop()

    @Slot()
    def shutdown(self):
        self._stop.set()
        self._timer.stop()
        for thread in self._threads:
            thread.join(timeout=0.5)
