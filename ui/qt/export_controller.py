"""Project-scoped, non-destructive Qt export with one bounded filesystem worker."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import shutil
import stat
import threading

from PySide6.QtCore import QObject, Property, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from automatic_scan.scan_logging import get_scan_logger
from .photo_model import PhotoListModel, photo_identity


MAX_COLLISION_ATTEMPTS = 10000
EXPORT_SHUTDOWN_TIMEOUT_SECONDS = 15.0
_INVALID_COMPONENT = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAME = re.compile(r'^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$', re.IGNORECASE)


def safe_component(value: str, fallback: str = "Unnamed person") -> str:
    component = _INVALID_COMPONENT.sub("_", str(value)).strip().rstrip(" .")
    if not component or component in (".", ".."):
        component = fallback
    if _RESERVED_NAME.fullmatch(component.split(".", 1)[0]):
        stem, separator, extension = component.partition(".")
        component = stem + "_" + (separator + extension if separator else "")
    return component


def person_folders(groups):
    """Map distinct person IDs to deterministic, collision-free safe folders."""
    used = set()
    folders = {}
    for key, name, _ in sorted(groups, key=lambda group: str(group[0])):
        base = safe_component(name)
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base} ({suffix})"
            suffix += 1
        used.add(candidate.casefold())
        folders[key] = candidate
    return folders


def destination_outside_project(destination: str, project_folder: str) -> bool:
    if not destination or not project_folder:
        return False
    # Resolve only these two roots, never every project photo on the GUI thread.
    project = os.path.normcase(os.path.realpath(project_folder))
    target = os.path.normcase(os.path.realpath(destination))
    try:
        return os.path.commonpath((project, target)) != project
    except ValueError:  # Different Windows volumes.
        return True


@dataclass(frozen=True, slots=True)
class ExportPlacement:
    source: str
    folder: str
    size: int
    mtime_ns: int
    person_key: str = ""


@dataclass(frozen=True, slots=True)
class ExportPlan:
    project_folder: str
    generation: int
    mode: str
    placements: tuple[ExportPlacement, ...]
    people_count: int
    unique_count: int
    total_bytes: int


def snapshot_plan(project_folder, generation, mode, requested, people_count):
    snapshots = {}
    placements = []
    for request in requested:
        source, folder = request[:2]
        person_key = str(request[2]) if len(request) > 2 else ""
        if source not in snapshots:
            try:
                info = os.stat(source)
                if not stat.S_ISREG(info.st_mode):
                    raise OSError("not a regular file")
                snapshots[source] = (info.st_size, info.st_mtime_ns)
            except OSError:
                snapshots[source] = (-1, -1)
        size, mtime_ns = snapshots[source]
        placements.append(ExportPlacement(source, folder, size, mtime_ns, person_key))
    return ExportPlan(project_folder, generation, mode, tuple(placements), people_count,
                      len(snapshots), sum(size for size, _ in snapshots.values() if size >= 0))


def _next_filename(source: str, attempt: int) -> str:
    name = safe_component(Path(source).name, "Photo")
    stem, extension = os.path.splitext(name)
    return f"{stem}{'_' + str(attempt) if attempt else ''}{extension}"


def _export_one(item: ExportPlacement, directory: Path, method: str) -> str:
    try:
        info = os.stat(item.source)
    except OSError as exc:
        raise ValueError("Photo is missing or unavailable. Select it again before exporting.") from exc
    if (not stat.S_ISREG(info.st_mode) or item.size < 0 or
            (info.st_size, info.st_mtime_ns) != (item.size, item.mtime_ns)):
        raise ValueError("Photo changed since export was prepared. Select it again before exporting.")
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise OSError("Export destination is not a directory.")
    if method == "hardlink" and info.st_dev != directory.stat().st_dev:
        raise OSError("Hard links require the source and destination to be on the same drive.")
    for attempt in range(MAX_COLLISION_ATTEMPTS):
        target = directory / _next_filename(item.source, attempt)
        try:
            if method == "hardlink":
                os.link(item.source, target)
            else:
                # Reserve the name exclusively before copy2; existing files are
                # never opened as copy destinations.
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                os.close(descriptor)
                try:
                    shutil.copy2(item.source, target)
                except Exception:
                    target.unlink(missing_ok=True)  # Only our incomplete output.
                    raise
            return str(target)
        except FileExistsError:
            continue
        except OSError as exc:
            if method == "hardlink" and exc.errno == errno.EXDEV:
                raise OSError("Hard links require the source and destination to be on the same drive.") from exc
            raise
    raise OSError("Too many files already use this photo name at the destination.")


def execute_plan(plan: ExportPlan, destination: str, method: str, canceled, report):
    completed = 0
    failures = []
    for index, item in enumerate(plan.placements):
        if canceled.is_set():
            break
        try:
            if not destination_outside_project(destination, plan.project_folder):
                raise ValueError("Choose a destination outside the current project folder.")
            output_dir = Path(destination) / item.folder if item.folder else Path(destination)
            _export_one(item, output_dir, method)
            completed += 1
        except (OSError, ValueError) as exc:
            get_scan_logger().warning("export_photo_failed photo=%s reason=%s", item.source, exc)
            failures.append((item.source, str(exc)))
        report(index + 1, len(plan.placements))
    return {"exported": completed, "failed": len(failures), "failures": failures,
            "canceled": canceled.is_set(), "done": completed + len(failures)}


def _export_worker(tasks, results, stop):
    def publish(value):
        while not stop.is_set():
            try:
                results.put(value, timeout=0.05)
                return
            except Full:
                if value[0] == "progress":
                    return
                try:
                    results.get_nowait()  # Discard an old progress update.
                except Empty:
                    pass

    while not stop.is_set():
        try:
            kind, generation, args = tasks.get(timeout=0.05)
        except Empty:
            continue
        if kind == "prepare":
            folder, mode, requested, people_count = args
            try:
                result = snapshot_plan(folder, generation, mode, requested, people_count)
                publish(("prepared", generation, result))
            except Exception:
                get_scan_logger().exception("export_prepare_failed folder=%s", folder)
                publish(("error", generation, "Export could not be prepared. Check the selected photos."))
        elif kind == "execute":
            plan, destination, method, canceled = args
            try:
                outcome = execute_plan(plan, destination, method, canceled,
                                       lambda done, total: publish(("progress", generation, (done, total))))
                publish(("finished", generation, outcome))
            except Exception:
                get_scan_logger().exception("export_batch_failed folder=%s", plan.project_folder)
                publish(("error", generation, "Export could not finish. Check the destination and try again."))


class ExportController(QObject):
    changed = Signal()
    planReady = Signal()

    def __init__(self, projects, master_photos, people, processing, parent=None, *, chooser=None):
        super().__init__(parent)
        self._projects, self._master, self._people, self._processing = projects, master_photos, people, processing
        self._chooser = chooser or (lambda: QFileDialog.getExistingDirectory(
            None, "Choose export destination", str(Path.home())))
        self._folder = projects.activeProjectFolder
        self._generation = 0
        self._plan = None
        self._destination = ""
        self._method = "copy"
        self._busy = False
        self._preparing = False
        self._progress_done = self._progress_total = 0
        self._error = self._summary = ""
        self._failures = []
        self._canceled = threading.Event()
        self._tasks, self._results = Queue(maxsize=1), Queue(maxsize=32)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=_export_worker, args=(self._tasks, self._results, self._stop),
                                        name="room36-export", daemon=True)
        self._worker.start()
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._drain)
        projects.projectOpening.connect(self._reset)
        projects.activeProjectChanged.connect(self._reset)
        master_photos.modelReset.connect(self._reset)
        processing.sessionChanged.connect(self._reset)

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Export presentation mutations require the owning Qt thread")

    @Property(bool, notify=changed)
    def busy(self):
        return self._busy

    @Property(bool, notify=changed)
    def preparing(self):
        return self._preparing

    @Property(bool, notify=changed)
    def prepared(self):
        return self._plan is not None

    @Property(int, notify=changed)
    def progressDone(self):
        return self._progress_done

    @Property(int, notify=changed)
    def progressTotal(self):
        return self._progress_total

    @Property(str, notify=changed)
    def error(self):
        return self._error

    @Property(str, notify=changed)
    def summary(self):
        return self._summary

    @Property("QVariantList", notify=changed)
    def failures(self):
        return [f"{Path(path).name}: {reason}" for path, reason in self._failures[:5]]

    @Property(str, notify=changed)
    def destination(self):
        return self._destination

    @Property(str, notify=changed)
    def method(self):
        return self._method

    @Property("QVariantMap", notify=changed)
    def planInfo(self):
        plan = self._plan
        return ({"mode": plan.mode, "peopleCount": plan.people_count,
                 "uniquePhotos": plan.unique_count, "placements": len(plan.placements),
                 "totalBytes": plan.total_bytes} if plan else {})

    @Slot()
    def _reset(self, *_):
        self._assert_owner()
        self._generation += 1
        self._folder = self._projects.activeProjectFolder
        self._plan = None
        self._preparing = False
        self._destination = ""
        self._progress_done = self._progress_total = 0
        self._error = self._summary = ""
        self._failures = []
        # A confirmed old-project batch keeps running, but cannot publish into
        # the newly active project's presentation.
        self.changed.emit()

    def _prepare(self, mode, requested, people_count=0):
        self._assert_owner()
        if self._stop.is_set() or self._busy or self._preparing or not self._folder:
            return False
        available = set(self._master.identities())
        seen = set()
        placements = []
        for source, folder, person_key in requested:
            identity = photo_identity(source)
            pair = (identity, folder)
            if identity in available and pair not in seen:
                seen.add(pair)
                placements.append((identity, folder, person_key))
        if not placements:
            self._error = "Select current project photos to export."
            self.changed.emit()
            return False
        self._plan = None
        self._error = self._summary = ""
        self._failures = []
        self._preparing = True
        try:
            self._tasks.put_nowait(("prepare", self._generation,
                                    (self._folder, mode, tuple(placements), people_count)))
        except Full:
            self._preparing = False
            self._error = "Another export is still finishing. Try again shortly."
            self.changed.emit()
            return False
        self._timer.start()
        self.changed.emit()
        return True

    @Slot(QObject, result=bool)
    def prepareSelected(self, model):
        if not isinstance(model, PhotoListModel) or model is self._master and not self._folder:
            return False
        members = set(model.identities())
        return self._prepare("selected", ((path, "", "") for path in model.selectedIdentities() if path in members))

    @Slot(result=bool)
    def preparePerson(self):
        group = self._people.selected_known_export_group()
        if not group or self._people.state != "COMPLETE" or self._processing.projectFolder != self._folder:
            self._error = "Known people are still processing or unavailable."
            self.changed.emit()
            return False
        return self._prepare("person", ((path, "", group[0]) for path in group[2]), 1)

    @Slot(result=bool)
    def prepareAllPeople(self):
        if self._people.state != "COMPLETE" or self._processing.projectFolder != self._folder:
            self._error = "Known people are still processing or unavailable."
            self.changed.emit()
            return False
        groups = self._people.known_export_groups()
        folders = person_folders(groups)
        return self._prepare("all_people", ((path, folders[key], key) for key, _, paths in groups for path in paths),
                             len(groups))

    @Slot(str, result=bool)
    def setDestination(self, destination):
        self._assert_owner()
        if not destination:
            return False
        target = photo_identity(destination)
        if not destination_outside_project(target, self._folder):
            self._error = "Choose a destination outside the current project folder."
            self.changed.emit()
            return False
        if Path(target).exists() and not Path(target).is_dir():
            self._error = "Choose a directory for the export destination."
            self.changed.emit()
            return False
        self._destination = target
        self._error = ""
        self.changed.emit()
        return True

    @Slot(result=bool)
    def chooseDestination(self):
        self._assert_owner()
        selected = self._chooser()
        return self.setDestination(selected) if selected else False

    @Slot(str, result=bool)
    def setMethod(self, method):
        if method not in ("copy", "hardlink") or self._busy:
            return False
        self._method = method
        self.changed.emit()
        return True

    @Slot(result=bool)
    def execute(self):
        self._assert_owner()
        if self._busy or self._preparing or self._plan is None or self._plan.generation != self._generation:
            return False
        if not self._destination or not destination_outside_project(self._destination, self._folder):
            self._error = "Choose a destination outside the current project folder."
            self.changed.emit()
            return False
        if Path(self._destination).exists() and not Path(self._destination).is_dir():
            self._error = "Choose a directory for the export destination."
            self.changed.emit()
            return False
        self._canceled = threading.Event()
        self._busy = True
        self._progress_done = 0
        self._progress_total = len(self._plan.placements)
        self._error = self._summary = ""
        try:
            self._tasks.put_nowait(("execute", self._generation,
                                    (self._plan, self._destination, self._method, self._canceled)))
        except Full:
            self._busy = False
            self._error = "Another export is still finishing. Try again shortly."
            self.changed.emit()
            return False
        self._timer.start()
        self.changed.emit()
        return True

    @Slot()
    def cancel(self):
        self._canceled.set()
        self.changed.emit()

    @Slot()
    def _drain(self):
        self._assert_owner()
        for _ in range(32):
            try:
                kind, generation, value = self._results.get_nowait()
            except Empty:
                break
            if self._stop.is_set():
                continue
            if generation != self._generation:
                if kind in ("finished", "error"):
                    self._busy = False
                    self.changed.emit()
                continue
            if kind == "prepared":
                self._plan = value
                self._preparing = False
                self._progress_total = len(value.placements)
                self.changed.emit()
                self.planReady.emit()
            elif kind == "progress":
                self._progress_done, self._progress_total = value
                self.changed.emit()
            elif kind == "finished":
                self._busy = False
                self._progress_done = value["done"]
                self._failures = value["failures"]
                count = value["exported"]
                self._summary = (f"Created {count} hard link{'s' if count != 1 else ''}" if self._method == "hardlink"
                                 else f"Exported {count} photo{'s' if count != 1 else ''}")
                self._summary += f" to {self._destination}."
                if value["failed"]:
                    self._summary += f" {value['failed']} could not be exported."
                if value["canceled"]:
                    self._summary += " Export canceled; completed files remain at the destination."
                self._plan = None
                self.changed.emit()
            elif kind == "error":
                self._preparing = self._busy = False
                self._plan = None
                self._error = value
                self.changed.emit()
        if not self._busy and not self._preparing:
            self._timer.stop()

    @Slot()
    def shutdown(self):
        self._assert_owner()
        if self._stop.is_set():
            return
        self._canceled.set()
        self._stop.set()
        self._timer.stop()
        self._worker.join(timeout=EXPORT_SHUTDOWN_TIMEOUT_SECONDS)
        if self._worker.is_alive():
            get_scan_logger().error("export_worker_shutdown_timed_out")
            raise RuntimeError("Export worker did not stop during shutdown")
