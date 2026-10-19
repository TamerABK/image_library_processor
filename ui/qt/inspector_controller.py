"""Reusable identity-based inspector and bounded current/neighbor preview window."""
from PySide6.QtCore import QObject, Property, QThread, Signal, Slot


class PhotoInspectorController(QObject):
    changed = Signal()

    def __init__(self, model, previews, parent=None):
        super().__init__(parent)
        self._default_model = model
        self._model, self._previews = model, previews
        self._order, self._tickets = [], {}
        self._position = -1
        self._source, self._state = "", "idle"
        previews.completed.connect(self._completed)
        model.reviewChanged.connect(lambda *_: self.changed.emit())

    @Property(bool, notify=changed)
    def opened(self):
        return self._position >= 0

    @Property("QVariantMap", notify=changed)
    def photo(self):
        return self._model.photo(self._order[self._position]) if self.opened else {}

    @Property(str, notify=changed)
    def previewSource(self):
        return self._source

    @Property(str, notify=changed)
    def previewState(self):
        return self._state

    @Property(bool, notify=changed)
    def hasPrevious(self):
        return self._position > 0

    @Property(bool, notify=changed)
    def hasNext(self):
        return self.opened and self._position < len(self._order) - 1

    @Slot(str)
    def open(self, identity):
        self._assert_owner()
        self._model = self._default_model
        self._open(identity)

    @Slot(QObject, str)
    def openFromModel(self, model, identity):
        self._assert_owner()
        self._model = model
        self._open(identity)

    def _open(self, identity):
        order = self._model.identities()
        if identity not in order:
            return
        self._order, self._position = order, order.index(identity)
        self._load()

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("Inspector mutations require the owning thread")

    def _cancel(self):
        for ticket in self._tickets:
            self._previews.cancel(ticket)
        self._tickets.clear()

    def _load(self):
        self._cancel()
        self._source, self._state = "", "loading"
        current = self._order[self._position]
        # Request current first, then just its two neighbors.
        indices = [self._position] + [i for i in (self._position - 1, self._position + 1) if 0 <= i < len(self._order)]
        for i in indices:
            identity = self._order[i]
            photo = self._model.photo(identity)
            revision = photo.get("revision", f"{photo.get('file_size', 0)}:{photo.get('mtime_ns', 0)}")
            ticket = self._previews.request(identity, 1600, revision)
            if ticket:
                self._tickets[ticket] = identity
            elif identity == current:
                self._state = "error"
        self.changed.emit()

    @Slot(int, str, str)
    def _completed(self, ticket, source, state):
        self._assert_owner()
        identity = self._tickets.pop(ticket, None)
        if self.opened and identity == self._order[self._position]:
            self._source, self._state = source, state
            self.changed.emit()

    @Slot()
    def previous(self):
        self._assert_owner()
        if self.hasPrevious:
            self._position -= 1
            self._load()

    @Slot()
    def next(self):
        self._assert_owner()
        if self.hasNext:
            self._position += 1
            self._load()

    @Slot()
    def close(self):
        self._assert_owner()
        self._cancel()
        self._order, self._position = [], -1
        self._model = self._default_model
        self._source, self._state = "", "idle"
        self.changed.emit()
