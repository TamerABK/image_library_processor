"""Bounded, GUI-owned thumbnail scheduler; workers never touch Qt objects."""
from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
import threading
import time

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from image_loader import ImageLoader


def decode_thumbnail(path: str, edge: int) -> str:
    """Runs only on workers. Reuse reduced JPEG/RAW/EXIF handling, not Tk objects."""
    if not Path(path).is_file():
        return ""
    image = ImageLoader(max_decode_dimension=edge).load_pil_for_hashing(
        Path(path), max_dimension=edge
    )
    if image is None:
        return ""
    with BytesIO() as output:
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


@dataclass
class _Job:
    key: tuple[str, int, str]
    tickets: set[int] = field(default_factory=set)
    canceled: threading.Event = field(default_factory=threading.Event)


def _worker(tasks, results, stop, decoder):
    # Only plain Python queues/events/strings cross the thread boundary.
    while not stop.is_set():
        try:
            job = tasks.get(timeout=0.05)
        except Empty:
            continue
        source = ""
        if not job.canceled.is_set() and not stop.is_set():
            try:
                source = decoder(job.key[0], job.key[1])
            except Exception:
                source = ""
        if not stop.is_set():
            results.put((job, source))


class ThumbnailService(QObject):
    """Call request/cancel/shutdown only on the owning (GUI) thread.

    `revision` is caller-owned: change it when file content may have changed.
    completed(ticket, source, state) is always delivered asynchronously on GUI.
    """
    completed = Signal(int, str, str)

    def __init__(self, parent=None, *, workers=2, max_pending=256,
                 cache_bytes=32 * 1024 * 1024, cache_entries=128,
                 decoder=decode_thumbnail):
        super().__init__(parent)
        self.max_pending = max(1, max_pending)
        self.cache_limit = max(0, cache_bytes)
        self.entry_limit = max(0, cache_entries)
        self._cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_hits = 0
        self.decode_count = 0
        self._next_ticket = 0
        self._jobs = {}
        self._pending = OrderedDict()
        self._tickets = {}
        self._deliveries = {}
        self._running = 0
        self._closed = False
        self._stop = threading.Event()
        self._tasks = Queue(maxsize=max(1, workers))
        self._results = Queue(maxsize=max(1, workers))
        self._threads = [threading.Thread(target=_worker, args=(
            self._tasks, self._results, self._stop, decoder), daemon=True,
            name=f"room36-thumbnail-{i}") for i in range(max(1, workers))]
        for thread in self._threads:
            thread.start()
        self._timer = QTimer(self)
        self._timer.setInterval(10)
        self._timer.timeout.connect(self._pump)
        # Also stop plain workers if a Qt owner destroys the service unexpectedly.
        stop = self._stop
        self.destroyed.connect(lambda *_: stop.set())

    def _assert_owner(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError("ThumbnailService must be called on its owning thread")

    @property
    def cache_entries(self):
        return len(self._cache)

    @property
    def pending_count(self):
        return len(self._tickets) + len(self._deliveries)

    @property
    def workers_alive(self):
        return sum(thread.is_alive() for thread in self._threads)

    def request(self, path: str, edge: int = 384, revision: str = "0") -> int:
        self._assert_owner()
        if self._closed or self.pending_count >= self.max_pending:
            return 0  # Caller may retry on a later acquisition; never queue forever.
        self._timer.start()
        self._next_ticket += 1
        ticket = self._next_ticket
        key = (path, max(32, min(512, int(edge))), str(revision))
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            self._deliveries[ticket] = self._cache[key]
        else:
            job = self._jobs.get(key)
            if job is None:
                job = _Job(key)
                self._jobs[key] = job
                self._pending[key] = job
            job.tickets.add(ticket)
            self._tickets[ticket] = job
        return ticket

    def cancel(self, ticket: int):
        self._assert_owner()
        self._deliveries.pop(ticket, None)
        job = self._tickets.pop(ticket, None)
        if job is not None:
            job.tickets.discard(ticket)
            if not job.tickets:
                job.canceled.set()
                self._pending.pop(job.key, None)
                if self._jobs.get(job.key) is job:
                    del self._jobs[job.key]

    def _cache_result(self, key, source):
        cost = max(1, len(source))  # ASCII data URL bytes, with a count bound too.
        if cost > self.cache_limit or not self.entry_limit:
            return
        old = self._cache.pop(key, None)
        if old is not None:
            self.cache_bytes -= max(1, len(old))
        self._cache[key] = source
        self.cache_bytes += cost
        while self.cache_bytes > self.cache_limit or len(self._cache) > self.entry_limit:
            _, evicted = self._cache.popitem(last=False)
            self.cache_bytes -= max(1, len(evicted))

    @Slot()
    def _pump(self):
        if self._closed:
            return
        while True:
            try:
                job, source = self._results.get_nowait()
            except Empty:
                break
            self._running -= 1
            if self._jobs.get(job.key) is job:
                del self._jobs[job.key]
            if not job.canceled.is_set():
                self._cache_result(job.key, source)
                for ticket in job.tickets:
                    self._tickets.pop(ticket, None)
                    self._deliveries[ticket] = source
        for ticket in list(self._deliveries):
            if ticket in self._deliveries:  # A callback can cancel another ticket.
                source = self._deliveries.pop(ticket)
                self.completed.emit(ticket, source, "ready" if source else "error")
        while self._pending and self._running < len(self._threads):
            _, job = self._pending.popitem(last=False)
            self._running += 1
            self.decode_count += 1
            self._tasks.put_nowait(job)
        if not self._running and not self._deliveries:
            self._timer.stop()

    @Slot()
    def shutdown(self, timeout: float = 0.5):
        """Cancel queued work; wait at most timeout for in-flight native decodes.

        A codec cannot be interrupted safely. Remaining daemon workers hold no Qt
        objects and cannot keep the process alive or publish results after stop.
        """
        self._assert_owner()
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._stop.set()
        for job in self._jobs.values():
            job.canceled.set()
        self._jobs.clear()
        self._pending.clear()
        self._tickets.clear()
        self._deliveries.clear()
        self._cache.clear()
        self.cache_bytes = 0
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
