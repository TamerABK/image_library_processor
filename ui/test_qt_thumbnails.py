import base64
import os
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import time
import unittest

from PIL import Image
from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from ui.qt.thumbnails import ThumbnailService

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def capture_qt_warnings(test):
    messages = []
    def capture(kind, context, message):
        if kind in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            messages.append(message)
    previous = qInstallMessageHandler(capture)
    test.addCleanup(lambda: qInstallMessageHandler(previous))
    test.addCleanup(lambda: test.assertEqual([], messages))


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)  # Yield Python's GIL to decoder/import work as well as Qt.
    if not predicate():
        raise AssertionError("Timed out waiting for asynchronous result")


class ThumbnailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        capture_qt_warnings(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "photo.jpg"
        Image.new("RGB", (120, 60), "red").save(self.path)
        self.events = []

    def service(self, **kwargs):
        service = ThumbnailService(**kwargs)
        service.completed.connect(lambda *args: self.events.append(args))
        self.addCleanup(service.shutdown)
        return service

    def test_async_dimensions_orientation_and_cache_hit(self):
        service = self.service()
        ticket = service.request(str(self.path), 48)
        self.assertEqual([], self.events)
        wait_until(lambda: len(self.events) == 1)
        self.assertEqual(ticket, self.events[0][0])
        self.assertEqual("ready", self.events[0][2])
        image = Image.open(BytesIO(base64.b64decode(self.events[0][1].split(",")[1])))
        self.assertLessEqual(max(image.size), 48)
        self.assertEqual(2, image.width / image.height)
        service.request(str(self.path), 48)
        wait_until(lambda: len(self.events) == 2)
        self.assertEqual(1, service.decode_count)
        self.assertEqual(1, service.cache_hits)
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (120, 60), "blue").save(self.path, exif=exif)
        service.request(str(self.path), 48, "changed")
        wait_until(lambda: len(self.events) == 3)
        rotated = Image.open(BytesIO(base64.b64decode(self.events[-1][1].split(",")[1])))
        self.assertLessEqual(max(rotated.size), 48)
        self.assertEqual(2, rotated.height / rotated.width)

    def test_coalescing_and_worker_thread(self):
        gate = threading.Event()
        thread_ids = []
        def decode(path, edge):
            thread_ids.append(threading.get_ident())
            gate.wait(1)
            return "thumbnail"
        service = self.service(decoder=decode)
        one = service.request("same", 64)
        two = service.request("same", 64)
        wait_until(lambda: bool(thread_ids))
        gate.set()
        wait_until(lambda: len(self.events) == 2)
        self.assertEqual({one, two}, {event[0] for event in self.events})
        self.assertEqual(1, service.decode_count)
        self.assertNotEqual(threading.get_ident(), thread_ids[0])

    def test_lru_byte_and_entry_bounds(self):
        service = self.service(cache_bytes=8, cache_entries=2, decoder=lambda path, edge: path * 4)
        for path in ("a", "b", "a", "c", "b"):
            count = len(self.events)
            service.request(path)
            wait_until(lambda: len(self.events) > count)
            self.assertLessEqual(service.cache_bytes, 8)
            self.assertLessEqual(service.cache_entries, 2)
        self.assertEqual(1, service.cache_hits)
        self.assertEqual(4, service.decode_count)

    def test_missing_corrupt_and_removed_files_are_cached_failures(self):
        service = self.service()
        corrupt = Path(self.temp.name) / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        self.path.unlink()
        for path in (self.path, corrupt, Path(self.temp.name) / "unsupported.xyz"):
            count = len(self.events)
            service.request(str(path))
            wait_until(lambda: len(self.events) > count)
            self.assertEqual(("", "error"), self.events[-1][1:])
        count = len(self.events)
        service.request(str(self.path))
        wait_until(lambda: len(self.events) > count)
        self.assertEqual(1, service.cache_hits)

    def test_cancel_stale_and_bounded_queue(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        service = self.service(workers=1, max_pending=2, decoder=lambda path, edge: (gate.wait(1), path)[1])
        stale = service.request("old")
        queued = service.request("queued")
        self.assertEqual(0, service.request("overflow"))
        wait_until(lambda: service.decode_count == 1)
        service.cancel(stale)
        service.cancel(queued)
        current = service.request("old")
        gate.set()
        wait_until(lambda: len(self.events) == 1)
        self.assertEqual(current, self.events[0][0])
        self.assertEqual(2, service.decode_count)
        self.assertEqual(0, service.pending_count)

    def test_shutdown_cancels_work_and_does_not_publish(self):
        gate = threading.Event()
        service = self.service(workers=1, decoder=lambda path, edge: (gate.wait(1), path)[1])
        service.request("running")
        service.request("queued")
        wait_until(lambda: service.decode_count == 1)
        started = time.monotonic()
        service.shutdown(timeout=0.01)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(0, service.request("after shutdown"))
        gate.set()
        wait_until(lambda: service.workers_alive == 0)
        self.assertEqual([], self.events)
        self.assertEqual(0, service.pending_count)

    def test_oversized_cache_entry_is_delivered_but_not_retained(self):
        service = self.service(cache_bytes=4, decoder=lambda path, edge: "too large")
        service.request("large")
        wait_until(lambda: bool(self.events))
        self.assertEqual("ready", self.events[0][2])
        self.assertEqual(0, service.cache_entries)
        self.assertEqual(0, service.cache_bytes)
