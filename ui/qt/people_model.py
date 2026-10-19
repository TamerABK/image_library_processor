"""Virtualized summaries of known people or unknown face clusters."""
from __future__ import annotations

from PySide6.QtCore import QAbstractListModel, QModelIndex, Property, QThread, Qt, Signal, Slot


class PeopleListModel(QAbstractListModel):
    changed = Signal()
    ROLE_NAMES = ("key", "clusterId", "personId", "name", "faceCount", "photoCount",
                  "representativePhotoId", "representativeBBox", "ignored",
                  "previewSource", "previewState")
    ROLES = {int(Qt.ItemDataRole.UserRole) + i + 1: name.encode()
             for i, name in enumerate(ROLE_NAMES)}
    ROLE = {name.decode(): role for role, name in ROLES.items()}

    def __init__(self, previews, kind: str, parent=None):
        super().__init__(parent)
        self._previews = previews
        self._kind = kind
        self._all = []
        self._all_by_key = {}
        self._rows = []
        self._positions = {}
        self._leases = {}
        self._tickets = {}
        self._next_lease = 0
        self._revision = "0"
        self._sort = "Name A–Z" if kind == "known" else "Largest clusters"
        self._search = ""
        self._filter = "Needs review"
        previews.completed.connect(self._completed)

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("People model mutations require the owning Qt thread")

    def roleNames(self):
        return self.ROLES

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row = self._rows[index.row()]
        name = self.ROLES.get(role, b"").decode()
        if role == Qt.ItemDataRole.DisplayRole:
            return row.get("name") or row["key"]
        return row.get(name)

    @Property(str, notify=changed)
    def sortOrder(self):
        return self._sort

    @Property(str, notify=changed)
    def searchQuery(self):
        return self._search

    @Property(str, notify=changed)
    def clusterFilter(self):
        return self._filter

    @Property(int, notify=changed)
    def totalCount(self):
        return len(self._all)

    @Property(int, notify=changed)
    def visibleCount(self):
        return len(self._rows)

    @Property(int, notify=changed)
    def needsReviewCount(self):
        return sum(not row.get("ignored", False) for row in self._all)

    def get_row(self, key):
        return self._all_by_key.get(key)

    def set_rows(self, rows, *, revision: str):
        self._assert_owner()
        self.beginResetModel()
        self._cancel_all()
        self._all = [dict(row, previewSource="", previewState="idle", ticket=0, edge=0)
                     for row in rows]
        self._all_by_key = {row["key"]: row for row in self._all}
        self._revision = revision
        self._refresh_rows()
        self.endResetModel()
        self.changed.emit()

    def _refresh_rows(self):
        if self._kind == "known":
            rows = [row for row in self._all if self._search.casefold() in row["name"].casefold()]
            if self._sort == "Most photos":
                rows.sort(key=lambda row: (-row["photoCount"], row["name"].casefold(), row["key"]))
            elif self._sort == "Most faces":
                rows.sort(key=lambda row: (-row["faceCount"], row["name"].casefold(), row["key"]))
            else:
                rows.sort(key=lambda row: (row["name"].casefold(), row["key"]))
        else:
            rows = [row for row in self._all if self._filter == "All" or
                    bool(row["ignored"]) == (self._filter == "Ignored")]
            rows.sort(key=lambda row: ((row["faceCount"] if self._sort == "Smallest clusters"
                                         else -row["faceCount"]), row["key"]))
        self._rows = rows
        self._positions = {row["key"]: index for index, row in enumerate(rows)}

    def _refilter(self):
        self._assert_owner()
        self.beginResetModel()
        self._cancel_all()
        self._refresh_rows()
        self.endResetModel()
        self.changed.emit()

    @Slot(str)
    def setSort(self, order):
        allowed = ("Name A–Z", "Most photos", "Most faces") if self._kind == "known" else (
            "Largest clusters", "Smallest clusters")
        if order in allowed and order != self._sort:
            self._sort = order
            self._refilter()

    @Slot(str)
    def setSearch(self, query):
        if self._kind == "known" and query != self._search:
            self._search = query
            self._refilter()

    @Slot(str)
    def setFilter(self, value):
        if self._kind == "unknown" and value in ("Needs review", "Ignored", "All") and value != self._filter:
            self._filter = value
            self._refilter()

    @Slot(str, int, result=int)
    def acquirePreview(self, key, edge=192):
        self._assert_owner()
        index = self._positions.get(key)
        if index is None:
            return 0
        row = self._rows[index]
        self._next_lease += 1
        lease = self._next_lease
        self._leases[lease] = key
        edge = max(48, min(256, int(edge)))
        if row["edge"] < edge or row["previewState"] == "idle":
            self._cancel_row(row)
            row["edge"] = edge
            ticket = self._previews.request_face(row["representativePhotoId"],
                row["representativeBBox"], edge, self._revision)
            row["ticket"] = ticket
            row["previewState"] = "loading" if ticket else "error"
            row["previewSource"] = ""
            if ticket:
                self._tickets[ticket] = (key, self._revision)
            self._changed(row)
        return lease

    @Slot(int)
    def releasePreview(self, lease):
        self._assert_owner()
        key = self._leases.pop(lease, None)
        if key is None or key in self._leases.values():
            return
        row = self.get_row(key)
        if row is not None:
            self._cancel_row(row)
            row["previewSource"], row["previewState"], row["edge"] = "", "idle", 0
            self._changed(row)

    def _changed(self, row):
        index = self._positions.get(row["key"])
        if index is not None:
            qindex = self.index(index)
            self.dataChanged.emit(qindex, qindex,
                                  [self.ROLE["previewSource"], self.ROLE["previewState"]])

    @Slot(int, str, str)
    def _completed(self, ticket, source, state):
        self._assert_owner()
        expected = self._tickets.pop(ticket, None)
        if expected is None or expected[1] != self._revision:
            return
        row = self.get_row(expected[0])
        if row is None or row["ticket"] != ticket:
            return
        row["ticket"] = 0
        row["previewSource"], row["previewState"] = source, state
        self._changed(row)

    def _cancel_row(self, row):
        ticket = row.get("ticket", 0)
        if ticket:
            self._previews.cancel(ticket)
            self._tickets.pop(ticket, None)
            row["ticket"] = 0

    def _cancel_all(self):
        for ticket in list(self._tickets):
            self._previews.cancel(ticket)
        self._tickets.clear()
        self._leases.clear()

    @Slot()
    def close(self):
        self._assert_owner()
        self._cancel_all()
