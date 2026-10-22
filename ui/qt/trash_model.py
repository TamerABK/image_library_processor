"""Virtualized Trash photos with batch selection separate from Library selection."""
from pathlib import Path

from PySide6.QtCore import Property, Signal, Slot, Qt

from ui.models import ResultItem
from .photo_model import PhotoListModel, photo_identity


class TrashPhotoModel(PhotoListModel):
    changed = Signal()
    ROLE_NAMES = PhotoListModel.ROLE_NAMES + ("removalSources", "conflictReason", "executable", "favorite", "fileSize")
    ROLES = {int(Qt.ItemDataRole.UserRole) + i + 1: name.encode() for i, name in enumerate(ROLE_NAMES)}
    ROLE = {name.decode(): role for role, name in ROLES.items()}

    def __init__(self, thumbnails, parent=None):
        super().__init__(thumbnails, parent)
        self._all = {}
        self._selection = set()
        self._filter = "All"
        self._descending = False

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and 0 <= index.row() < len(self._rows):
            path = self._rows[index.row()].identity
            name = self.ROLES.get(role, b"").decode()
            if name in ("removalSources", "conflictReason", "executable", "favorite", "fileSize"):
                return self._all.get(path, {}).get(name)
        return super().data(index, role)

    @Property(int, notify=changed)
    def selectedCount(self):
        return len(self._selection)

    @Property(str, notify=changed)
    def candidateFilter(self):
        return self._filter

    def all_rows(self):
        return list(self._all.values())

    def row(self, path):
        return self._all.get(path)

    def selected_rows(self):
        visible = self._visible_paths()
        return [self._all[path] for path in sorted(self._selection) if path in visible]

    def selection_is_visible(self):
        return self._selection <= self._visible_paths()

    def _visible_paths(self):
        return {row.identity for row in self._rows}

    def set_rows(self, rows):
        self._assert_owner()
        self._all = {photo_identity(row["photoId"]): dict(row, photoId=photo_identity(row["photoId"]))
                     for row in rows}
        self._selection = {path for path in self._selection
                           if path in self._all and self._all[path]["executable"]}
        self._rebuild()

    def _rebuild(self):
        rows = [row for row in self._all.values() if self._filter == "All" or
                row["executable"] == (self._filter == "Ready to remove")]
        rows.sort(key=lambda row: (row["filename"].casefold(), row["photoId"]), reverse=self._descending)
        self.replace_items([ResultItem(Path(row["photoId"]), row["filename"],
                                       "Marked for removal", badge_text=row["badgeText"])
                            for row in rows], preserve_selection=False)
        for path in self._selection:
            super().setSelected(path, True)
        self.changed.emit()

    @Slot(str)
    def setFilter(self, value):
        if value in ("All", "Ready to remove", "Needs attention"):
            if value != self._filter:
                self._selection.clear()
            self._filter = value
            self._rebuild()

    @Slot(bool)
    def sortByFilename(self, descending=False):
        self._descending = descending
        self._rebuild()

    @Slot(str, bool, result=bool)
    def setSelected(self, identity, selected):
        if selected and (identity not in self._all or identity not in self._visible_paths()
                         or not self._all[identity]["executable"]):
            return False
        if identity not in self._all:
            return False
        if selected:
            self._selection.add(identity)
        else:
            self._selection.discard(identity)
        super().setSelected(identity, selected)
        self.changed.emit()
        return True

    @Slot()
    def selectAllReady(self):
        self._selection = {path for path in self._visible_paths() if self._all[path]["executable"]}
        self._rebuild()

    @Slot()
    def clearSelection(self):
        self._selection.clear()
        self._rebuild()
