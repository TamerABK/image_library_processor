"""Generic path-identified photo rows. Selection is UI state, not delete intent."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QAbstractListModel, QModelIndex, QThread, Qt, Slot
from shiboken6 import isValid

from ui.models import ResultItem
from .thumbnails import ThumbnailService


def photo_identity(path: str | Path) -> str:
    # Lexical only: no stat/resolve/network filesystem access on the GUI thread.
    return os.path.normcase(os.path.abspath(os.path.normpath(os.fspath(path))))


@dataclass(slots=True)
class _Photo:
    identity: str
    title: str
    detail: str
    badge: str
    person_id: int | None
    revision: str
    selected: bool = False
    source: str = ""
    state: str = "idle"
    ticket: int = 0
    edge: int = 0


class PhotoListModel(QAbstractListModel):
    ROLE_NAMES = ("photoId", "sourcePath", "filename", "title", "detail", "badgeText",
                  "personId", "selected", "thumbnailSource", "thumbnailState")
    ROLES = {int(Qt.ItemDataRole.UserRole) + i + 1: name.encode()
             for i, name in enumerate(ROLE_NAMES)}
    ROLE = {name.decode(): role for role, name in ROLES.items()}

    def __init__(self, thumbnails: ThumbnailService, parent=None):
        super().__init__(parent)
        self._service = thumbnails
        self._rows = []
        self._positions = {}
        self._leases = {}  # opaque lease -> (photo identity, requested edge)
        self._next_lease = 0
        self._requests = {}  # service ticket -> (identity, revision)
        self._namespace = uuid4().hex
        self._generation = 0
        self._closed = False
        thumbnails.completed.connect(self._thumbnail_completed)
        requests = self._requests
        # Cancel orphan work even when deleted by a Qt owner without close().
        def cancel_orphans(*_):
            if isValid(thumbnails):
                for ticket in list(requests):
                    thumbnails.cancel(ticket)
            requests.clear()
        self.destroyed.connect(cancel_orphans)

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("PhotoListModel mutations require the owning thread")

    def roleNames(self):
        return self.ROLES

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row = self._rows[index.row()]
        name = self.ROLES.get(role, b"").decode()
        if role == Qt.ItemDataRole.DisplayRole or name == "filename":
            return Path(row.identity).name
        if name in ("photoId", "sourcePath"):
            return row.identity
        return {"title": row.title, "detail": row.detail, "badgeText": row.badge,
                "personId": row.person_id, "selected": row.selected,
                "thumbnailSource": row.source, "thumbnailState": row.state}.get(name)

    def _reindex(self):
        self._positions = {row.identity: i for i, row in enumerate(self._rows)}

    def _row(self, identity):
        index = self._positions.get(identity)
        return self._rows[index] if index is not None else None

    def _changed(self, row, *roles):
        index = self.index(self._positions[row.identity])
        self.dataChanged.emit(index, index, [self.ROLE[name] for name in roles])

    def _adapt(self, items):
        seen = set()
        rows = []
        for item in items:
            identity = photo_identity(item.path)
            if identity in seen:
                continue  # One photograph can appear in several result groups.
            seen.add(identity)
            rows.append(_Photo(identity, item.title, item.detail, item.badge_text,
                               item.person_id, f"{self._namespace}:{self._generation}"))
        return rows

    def replace_items(self, items, *, preserve_selection=True):
        """Adapter boundary. Caller supplies discovered ResultItems; no scanning.

        Explicit refresh invalidates thumbnail revisions, including cached errors.
        """
        self._assert_owner()
        self._generation += 1
        rows = self._adapt(items)
        selected = {row.identity for row in self._rows if row.selected} if preserve_selection else set()
        for row in rows:
            row.selected = row.identity in selected
        self.beginResetModel()
        self._cancel_all()
        self._rows = rows
        self._reindex()
        self.endResetModel()

    def append_items(self, items):
        self._assert_owner()
        self._generation += 1  # Reinserting a removed path must not reuse old pixels.
        rows = [row for row in self._adapt(items) if row.identity not in self._positions]
        if not rows:
            return
        first = len(self._rows)
        self.beginInsertRows(QModelIndex(), first, first + len(rows) - 1)
        self._rows.extend(rows)
        self._reindex()
        self.endInsertRows()

    @Slot(str, result=bool)
    def removePhoto(self, identity):
        self._assert_owner()
        row = self._row(identity)
        if row is None:
            return False
        index = self._positions[identity]
        self.beginRemoveRows(QModelIndex(), index, index)
        self._cancel_row(row)
        self._leases = {lease: value for lease, value in self._leases.items() if value[0] != identity}
        del self._rows[index]
        self._reindex()
        self.endRemoveRows()
        return True

    @Slot(str, bool, result=bool)
    def setSelected(self, identity, selected):
        self._assert_owner()
        row = self._row(identity)
        if row is None:
            return False
        if row.selected != selected:
            row.selected = selected
            self._changed(row, "selected")
        return True

    @Slot(bool)
    def sortByFilename(self, descending=False):
        self._assert_owner()
        self.layoutAboutToBeChanged.emit()
        old = self.persistentIndexList()
        identities = [self._rows[index.row()].identity for index in old]
        self._rows.sort(key=lambda row: (Path(row.identity).name.casefold(), row.identity), reverse=descending)
        self._reindex()
        self.changePersistentIndexList(old, [self.index(self._positions[key]) for key in identities])
        self.layoutChanged.emit()

    @Slot(str, int, result=int)
    def acquireThumbnail(self, identity, edge=384):
        self._assert_owner()
        row = self._row(identity)
        if self._closed or row is None:
            return 0
        self._next_lease += 1
        lease = self._next_lease
        edge = max(32, min(512, edge))
        self._leases[lease] = (identity, edge)
        if row.edge < edge or row.state == "idle":
            self._cancel_row(row)
            row.edge = edge
            row.ticket = self._service.request(identity, edge, row.revision)
            row.state = "loading" if row.ticket else "idle"
            row.source = ""
            if row.ticket:
                self._requests[row.ticket] = (identity, row.revision)
            self._changed(row, "thumbnailSource", "thumbnailState")
        return lease

    @Slot(int)
    def releaseThumbnail(self, lease):
        self._assert_owner()
        value = self._leases.pop(lease, None)
        if value is None or any(item[0] == value[0] for item in self._leases.values()):
            return
        row = self._row(value[0])
        if row is not None:
            self._cancel_row(row)
            row.source, row.state, row.edge = "", "idle", 0
            self._changed(row, "thumbnailSource", "thumbnailState")

    def _cancel_row(self, row):
        if row.ticket:
            self._service.cancel(row.ticket)
            self._requests.pop(row.ticket, None)
            row.ticket = 0

    @Slot(int, str, str)
    def _thumbnail_completed(self, ticket, source, state):
        self._assert_owner()
        expected = self._requests.pop(ticket, None)
        if expected is None:
            return
        row = self._row(expected[0])
        if row is None or row.revision != expected[1] or row.ticket != ticket:
            return
        row.ticket = 0
        row.source, row.state = source, state
        self._changed(row, "thumbnailSource", "thumbnailState")

    def _cancel_all(self):
        for ticket in list(self._requests):
            self._service.cancel(ticket)
        self._requests.clear()
        self._leases.clear()

    @Slot()
    def close(self):
        self._assert_owner()
        self._closed = True
        self._cancel_all()
