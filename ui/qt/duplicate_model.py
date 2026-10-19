"""Virtualized, lease-based duplicate-group summaries."""
from __future__ import annotations

from PySide6.QtCore import QAbstractListModel, QModelIndex, Property, QThread, Qt, Signal, Slot


class DuplicateGroupModel(QAbstractListModel):
    changed = Signal()
    ROLE_NAMES = ("groupId", "photoCount", "representativePhotoId", "recommendedKeeperCount", "reviewed",
                  "hasConflict", "reviewState", "recommendationState", "removeCount", "previewSource")
    ROLES = {int(Qt.ItemDataRole.UserRole) + i + 1: name.encode()
             for i, name in enumerate(ROLE_NAMES)}
    ROLE = {name.decode(): role for role, name in ROLES.items()}

    def __init__(self, thumbnails, parent=None):
        super().__init__(parent)
        self._thumbnails = thumbnails
        self._all = []
        self._by_id = {}
        self._rows = []
        self._filter = "Needs review"
        self._sort = "Largest groups"
        self._leases = {}
        self._tickets = {}
        self._next_lease = 0
        self._generation = 0
        thumbnails.completed.connect(self._completed)

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Duplicate model mutations require the owning Qt thread")

    def roleNames(self):
        return self.ROLES

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        row = self._rows[index.row()]
        return row["groupId"] if role == Qt.ItemDataRole.DisplayRole else row.get(
            self.ROLES.get(role, b"").decode())

    @Property(str, notify=changed)
    def groupFilter(self):
        return self._filter

    @Property(str, notify=changed)
    def sortOrder(self):
        return self._sort

    @Property(int, notify=changed)
    def totalCount(self):
        return len(self._all)

    @Property(int, notify=changed)
    def visibleCount(self):
        return len(self._rows)

    @Property(int, notify=changed)
    def needsReviewCount(self):
        return sum(not row["reviewed"] for row in self._all)

    def get_row(self, group_id):
        return self._by_id.get(group_id)

    def contains_photo(self, path):
        return any(path in row["paths"] for row in self._all)

    def _refilter(self):
        self._rows = [row for row in self._all if self._filter == "All" or
                      row["reviewed"] == (self._filter == "Reviewed")]
        self._rows.sort(key=lambda row: ((row["photoCount"] if self._sort == "Smallest groups"
                                          else -row["photoCount"]), row["groupId"]))

    def set_rows(self, rows):
        self._assert_owner()
        self.beginResetModel()
        self._cancel_all()
        self._generation += 1
        self._all = [dict(row, previewSource="", ticket=0, edge=0) for row in rows]
        self._by_id = {row["groupId"]: row for row in self._all}
        self._refilter()
        self.endResetModel()
        self.changed.emit()

    @Slot(str)
    def setFilter(self, value):
        self._assert_owner()
        if value in ("Needs review", "Reviewed", "All") and value != self._filter:
            self._filter = value
            self.set_rows(self._all)

    @Slot(str)
    def setSort(self, value):
        self._assert_owner()
        if value in ("Largest groups", "Smallest groups") and value != self._sort:
            self._sort = value
            self.set_rows(self._all)

    @Slot(str, int, result=int)
    def acquirePreview(self, group_id, edge=256):
        self._assert_owner()
        row = self._by_id.get(group_id)
        if row is None or row not in self._rows:
            return 0
        self._next_lease += 1
        lease = self._next_lease
        self._leases[lease] = group_id
        edge = max(48, min(384, int(edge)))
        if row["edge"] < edge or not row["previewSource"]:
            self._cancel_row(row)
            ticket = self._thumbnails.request(row["representativePhotoId"], edge,
                f"duplicates:{self._generation}:{row['signature']}")
            row["ticket"], row["edge"] = ticket, edge
            if ticket:
                self._tickets[ticket] = (group_id, self._generation)
        return lease

    @Slot(int)
    def releasePreview(self, lease):
        self._assert_owner()
        group_id = self._leases.pop(lease, None)
        if group_id is None or group_id in self._leases.values():
            return
        row = self._by_id.get(group_id)
        if row:
            self._cancel_row(row)
            row["previewSource"], row["edge"] = "", 0
            self._changed(row)

    def _changed(self, row):
        try:
            index = self._rows.index(row)
        except ValueError:
            return
        qindex = self.index(index)
        self.dataChanged.emit(qindex, qindex, [self.ROLE["previewSource"]])

    @Slot(int, str, str)
    def _completed(self, ticket, source, state):
        self._assert_owner()
        expected = self._tickets.pop(ticket, None)
        if expected is None or expected[1] != self._generation:
            return
        row = self._by_id.get(expected[0])
        if row is None or row["ticket"] != ticket:
            return
        row["ticket"] = 0
        row["previewSource"] = source if state == "ready" else ""
        self._changed(row)

    def _cancel_row(self, row):
        ticket = row.get("ticket", 0)
        if ticket:
            self._thumbnails.cancel(ticket)
            self._tickets.pop(ticket, None)
            row["ticket"] = 0

    def _cancel_all(self):
        for ticket in list(self._tickets):
            self._thumbnails.cancel(ticket)
        self._tickets.clear()
        self._leases.clear()

    @Slot()
    def close(self):
        self._assert_owner()
        self._cancel_all()
