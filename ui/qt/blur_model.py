"""Lightweight blur feature view; the master Library model stays independent."""
from pathlib import Path
from PySide6.QtCore import Property, Signal, Slot, Qt
from ui.models import ResultItem
from .photo_model import PhotoListModel


class BlurCandidateModel(PhotoListModel):
    changed = Signal()
    reviewChanged = Signal(str)
    ROLE_NAMES = PhotoListModel.ROLE_NAMES + ("qualityScore", "classification", "reviewState")
    ROLES = {int(Qt.ItemDataRole.UserRole) + i + 1: name.encode() for i, name in enumerate(ROLE_NAMES)}
    ROLE = {name.decode(): role for role, name in ROLES.items()}

    def __init__(self, thumbnails, parent=None):
        super().__init__(thumbnails, parent)
        self.results = {}
        self._filter = "All candidates"
        self._sort = "Most problematic first"
        self._selection = set()
        self._candidate_count = 0

    @Property(int, notify=changed)
    def candidateCount(self):
        return self._candidate_count

    @Property(str, notify=changed)
    def sortOrder(self):
        return self._sort

    @Property(str, notify=changed)
    def classificationFilter(self):
        return self._filter

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and 0 <= index.row() < len(self._rows):
            row = self._rows[index.row()]
            result = self.results.get(row.identity, {})
            name = self.ROLES.get(role, b"").decode()
            if name in ("qualityScore", "classification", "reviewState"):
                return {"qualityScore": result.get("final_score"), "classification": result.get("status"),
                        "reviewState": result.get("reviewState", "UNDECIDED")}[name]
            if name == "badgeText":
                intent = result.get("reviewState", "UNDECIDED")
                return result.get("status", "") + (" · " + intent.title() if intent != "UNDECIDED" else "")
        return super().data(index, role)

    def set_results(self, results, *, new_project=False):
        self._assert_owner()
        if new_project:
            self._selection.clear()
            self._filter, self._sort = "All candidates", "Most problematic first"
        self.results = {r["photoId"]: dict(r) for r in results}
        self._candidate_count = sum(r["status"] in ("Blurry", "Review") for r in self.results.values())
        self._rebuild()

    def _rebuild(self):
        candidates = [r for r in self.results.values() if r["status"] in ("Blurry", "Review")
                      and (self._filter == "All candidates" or r["status"] == self._filter)]
        if self._sort.startswith("Filename"):
            key = lambda r: (r["filename"].casefold(), r["photoId"])
        else:
            key = lambda r: (r["final_score"], r["filename"].casefold(), r["photoId"])
        candidates.sort(key=key, reverse=self._sort in ("Least problematic first", "Filename Z–A"))
        self.replace_items([ResultItem(Path(r["photoId"]), r["filename"], r["status"]) for r in candidates],
                           preserve_selection=False)
        for row in self._rows:
            if row.identity in self._selection:
                super().setSelected(row.identity, True)
        self.changed.emit()

    @Slot(str)
    def setFilter(self, value):
        self._assert_owner()
        if value in ("All candidates", "Blurry", "Review"):
            self._filter = value
            self._rebuild()

    @Slot(str)
    def setSort(self, value):
        self._assert_owner()
        if value in ("Most problematic first", "Least problematic first", "Filename A–Z", "Filename Z–A"):
            self._sort = value
            self._rebuild()

    @Slot(str, bool, result=bool)
    def setSelected(self, identity, value):
        result = super().setSelected(identity, value)
        if result:
            if value:
                self._selection.add(identity)
            else:
                self._selection.discard(identity)
            self.changed.emit()
        return result

    def apply_review(self, identity, intent):
        self._assert_owner()
        if identity not in self.results:
            return
        self.results[identity]["reviewState"] = intent
        row = self._row(identity)
        if row:
            self._changed(row, "reviewState", "badgeText")
        self.reviewChanged.emit(identity)

    @Slot(str, result="QVariantMap")
    def photo(self, identity):
        result = self.results.get(identity)
        if result is None:
            return {}
        return {**result, "qualityScore": result["final_score"],
                "classification": "Needs review" if result["status"] == "Review" else result["status"]}

    def identities(self):
        return [row.identity for row in self._rows]

    @Slot(result="QVariantList")
    def selectedIdentities(self):
        return [row.identity for row in self._rows if row.identity in self._selection]

    @Slot()
    def clearSelection(self):
        self._selection.clear()
        super().clearSelection()
        self.changed.emit()
