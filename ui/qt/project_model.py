"""Small Home project registry view; contains metadata, never photo pixels."""
from dataclasses import dataclass, asdict
from datetime import datetime
import math

from PySide6.QtCore import QAbstractListModel, QModelIndex, Property, Qt, Signal, Slot


@dataclass
class Project:
    project_id: str
    folder: str
    name: str
    created_at: float
    last_opened: float
    photo_count: int = -1

    def metadata(self):
        return asdict(self)


def _last_opened_label(timestamp):
    try:
        if isinstance(timestamp, bool) or not math.isfinite(timestamp) or timestamp <= 0:
            return ""
        return "Last opened " + datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


class ProjectListModel(QAbstractListModel):
    searchChanged = Signal()
    sortChanged = Signal()
    ROLES = {int(Qt.ItemDataRole.UserRole) + i + 1: role.encode() for i, role in enumerate(
        ("projectId", "name", "photoCount", "thumbnailUrl", "lastOpened"))}

    def __init__(self, projects=(), parent=None):
        super().__init__(parent)
        self._projects = list(projects)
        self._visible = []
        self._query = ""
        self._sort = "Newest"
        self._refresh()

    def roleNames(self):
        return self.ROLES

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._visible)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._visible):
            return None
        project = self._visible[index.row()]
        return {"projectId": project.project_id, "name": project.name,
                "photoCount": project.photo_count, "thumbnailUrl": "",
                "lastOpened": _last_opened_label(project.last_opened)
                }.get(self.ROLES.get(role, b"").decode())

    @Property(str, notify=searchChanged)
    def searchQuery(self):
        return self._query

    @Property(str, notify=sortChanged)
    def sortOrder(self):
        return self._sort

    @Slot(str)
    def setSearch(self, query):
        if query != self._query:
            self._query = query
            self._refresh()
            self.searchChanged.emit()

    @Slot(str)
    def setSort(self, order):
        if order in ("Newest", "Oldest", "Name") and order != self._sort:
            self._sort = order
            self._refresh()
            self.sortChanged.emit()

    def _refresh(self):
        self.beginResetModel()
        self._visible = [project for project in self._projects if self._query.casefold() in project.name.casefold()]
        # Newest/Oldest refer to creation, never to reopening a project.
        key = (lambda project: (project.name.casefold(), project.project_id)) if self._sort == "Name" else (
            lambda project: (project.created_at, project.project_id))
        self._visible.sort(key=key, reverse=self._sort == "Newest")
        self.endResetModel()

    def get(self, project_id):
        return next((project for project in self._projects if project.project_id == project_id), None)

    def for_folder(self, folder):
        return next((project for project in self._projects if project.folder == folder), None)

    def register(self, project):
        self._projects = [item for item in self._projects if item.project_id != project.project_id] + [project]
        self._refresh()

    def metadata(self):
        return [project.metadata() for project in self._projects]
