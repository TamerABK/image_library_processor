"""Folder browsing sessions, separate from analysis and thumbnail processing."""
from dataclasses import replace
from datetime import datetime
import json
import math
import os
import stat
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from uuid import uuid4

from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from app_paths import app_data_path
from image_file_utils import find_supported_files
from image_loader import default_image_loader
from ui.models import ResultItem
from .photo_model import photo_identity
from .project_model import Project, ProjectListModel


def discover_photos(folder):
    """Worker boundary: validate/enumerate without reading image metadata/pixels."""
    root = Path(folder)
    if not stat.S_ISDIR(root.stat().st_mode):
        raise FileNotFoundError(folder)
    with os.scandir(root):  # rglob can suppress access errors: check root explicitly.
        pass
    paths = find_supported_files(root, default_image_loader.supported_extensions())
    return [ResultItem(path, path.name, "") for path in sorted(paths, key=lambda path: str(path).casefold())]


def _discovery_worker(tasks, results, stop, discover):
    while not stop.is_set():
        try:
            generation, project = tasks.get(timeout=0.05)
        except Empty:
            continue
        try:
            items = discover(project.folder)
            error = ""
        except FileNotFoundError:
            items, error = [], "This project folder is missing or was moved. Choose its current folder with New Project."
        except PermissionError:
            items, error = [], "Room 36 cannot access this folder. Check its permissions and try again."
        except Exception:
            items, error = [], "This folder could not be opened. Check that it is available and try again."
        if stop.is_set():
            return
        try:
            results.put_nowait((generation, project, items, error))
        except Full:
            try:
                results.get_nowait()  # Only the latest requested project can apply.
            except Empty:
                pass
            results.put_nowait((generation, project, items, error))


class ProjectController(QObject):
    busyChanged = Signal()
    messageChanged = Signal()
    activeProjectChanged = Signal()
    projectOpened = Signal(str, str)

    def __init__(self, photo_model, parent=None, *, registry_path=None, chooser=None,
                 discover=discover_photos):
        super().__init__(parent)
        self._photos = photo_model
        self._registry = Path(registry_path) if registry_path is not None else app_data_path("room36_projects.json")
        self._chooser = chooser or (lambda: QFileDialog.getExistingDirectory(None, "Choose a project folder", str(Path.home())))
        self._message = ""
        self._busy = False
        self._closed = False
        self._active_id = ""
        self._active_name = ""
        self._generation = 0
        self._projects = ProjectListModel(self._load_registry(), self)
        self._tasks, self._results = Queue(maxsize=1), Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=_discovery_worker,
            args=(self._tasks, self._results, self._stop, discover), daemon=True, name="room36-project-discovery")
        self._worker.start()
        stop = self._stop
        self.destroyed.connect(lambda *_: stop.set())
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._drain)

    @Property(QObject, constant=True)
    def projects(self):
        return self._projects

    @Property(bool, notify=busyChanged)
    def busy(self):
        return self._busy

    @Property(str, notify=messageChanged)
    def message(self):
        return self._message

    @Property(str, notify=activeProjectChanged)
    def activeProjectId(self):
        return self._active_id

    @Property(str, notify=activeProjectChanged)
    def activeProjectName(self):
        return self._active_name

    def _set_message(self, text):
        if text != self._message:
            self._message = text
            self.messageChanged.emit()

    def _set_busy(self, value):
        if value != self._busy:
            self._busy = value
            self.busyChanged.emit()

    def _load_registry(self):
        try:
            payload = json.loads(self._registry.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("projects"), list):
                raise ValueError("Invalid registry")
        except FileNotFoundError:
            return []
        except (OSError, ValueError):
            self._message = "Saved projects could not be read. You can create a new project."
            return []
        projects, ids, folders = [], set(), set()
        for record in payload["projects"]:
            try:
                if not isinstance(record, dict) or not all(isinstance(record.get(key), str) and record[key].strip()
                        for key in ("project_id", "folder", "name")):
                    raise ValueError("Invalid project")
                created, opened = float(record["created_at"]), float(record["last_opened"])
                if not all(math.isfinite(value) and 0 <= value <= 253402214400 for value in (created, opened)):
                    raise ValueError("Invalid timestamp")
                datetime.fromtimestamp(opened)  # Validate platform-supported display range.
                if not Path(record["folder"]).is_absolute():
                    raise ValueError("Project folder must be absolute")
                folder = photo_identity(record["folder"])
                if record["project_id"] in ids or folder in folders:
                    continue
                projects.append(Project(record["project_id"], folder, record["name"], created, opened))
                ids.add(record["project_id"])
                folders.add(folder)
            except (KeyError, TypeError, ValueError, OSError, OverflowError):
                self._message = "Some saved projects could not be read and were skipped."
        return projects

    def _save_registry(self):
        temporary = self._registry.with_suffix(".json.tmp")
        try:
            self._registry.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps({"version": 1, "projects": self._projects.metadata()}, indent=2), encoding="utf-8")
            temporary.replace(self._registry)
            return ""
        except OSError:
            return "Project opened, but its details could not be saved. It may be unavailable after restart."

    @Slot()
    def newProject(self):
        if self._closed:
            return
        folder = self._chooser()
        if folder:
            self.openFolder(folder)

    @Slot(str)
    def openFolder(self, folder):
        if self._closed or not folder.strip():
            return
        path = photo_identity(Path(folder).expanduser())
        project = self._projects.for_folder(path)
        if project is None:
            project = Project(uuid4().hex, path, Path(folder).name or folder, time.time(), 0)
        self._start(project)

    @Slot(str)
    def openProject(self, project_id):
        if self._closed:
            return
        project = self._projects.get(project_id)
        if project is None:
            self._set_message("This project is no longer available.")
            return
        self._start(project)

    def _start(self, project):
        self._generation += 1
        self._set_message("Opening " + project.name + "…")
        self._set_busy(True)
        try:
            self._tasks.get_nowait()
        except Empty:
            pass
        self._tasks.put_nowait((self._generation, project))
        self._timer.start()

    @Slot()
    def _drain(self):
        try:
            generation, project, items, error = self._results.get_nowait()
        except Empty:
            return
        if generation != self._generation or self._closed:
            return
        self._timer.stop()
        self._set_busy(False)
        if error:
            self._set_message(error)
            return
        project = replace(project, last_opened=time.time(), photo_count=len(items))
        # Selection is session state: never carry it into a different folder.
        self._photos.replace_items(items, preserve_selection=False)
        self._projects.register(project)
        self._active_id, self._active_name = project.project_id, project.name
        self.activeProjectChanged.emit()
        self._set_message(self._save_registry())
        self.projectOpened.emit(project.project_id, project.name)

    @Slot()
    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        self._stop.set()
        self._timer.stop()
        self._worker.join(timeout=0.2)
