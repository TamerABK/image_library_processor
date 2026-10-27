from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from image_cache_batch import CacheBatchSession, CacheBatchWriteError
from image_cache_storage import SharedImageCacheDatabase


class CacheBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "cache.sqlite3"
        self.storage = SharedImageCacheDatabase(self.path)
        self.storage.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def session(self, size=256):
        return CacheBatchSession(self.path, self.storage._open_connection, batch_size=size)

    def _write(self, value):
        with self.storage.connect() as db:
            db.execute("INSERT INTO image_entries(path,file_size,mtime_ns) VALUES (?,?,?) "
                       "ON CONFLICT(path) DO UPDATE SET file_size=excluded.file_size",
                       (value, 1, 1))

    def _rows(self):
        with self.storage.connect() as db:
            return [row[0] for row in db.execute("SELECT path FROM image_entries ORDER BY path")]

    def test_partial_batch_flushes_on_close_and_survives_restart(self):
        with self.session() as session:
            for value in ("a", "b", "c"):
                self._write(value)
            self.assertEqual(session.commits, 0)
            self.assertEqual(session.max_pending, 3)
        self.assertEqual(session.commits, 1)
        self.assertEqual(self._rows(), ["a", "b", "c"])

    def test_full_batch_and_duplicate_upsert_preserve_order(self):
        with self.session(32) as session:
            for index in range(32):
                self._write(str(index))
            self.assertEqual(session.commits, 1)
            self._write("0")
        self.assertEqual(session.commits, 2)
        self.assertEqual(len(self._rows()), 32)

    def test_256_record_batch_and_partial_tail(self):
        with self.session(256) as session:
            for index in range(259):
                self._write(f"image-{index:03d}")
            self.assertEqual(session.commits, 1)
            self.assertEqual(session.records_committed, 256)
        self.assertEqual(session.commits, 2)
        self.assertEqual(session.records_committed, 259)
        self.assertEqual(len(self._rows()), 259)

    def test_read_after_write_flushes_and_other_connection_sees_commit(self):
        with self.session() as session:
            self._write("a")
            self.assertEqual(self._rows(), ["a"])
            self.assertEqual(session.commits, 1)
            observed = []
            thread = threading.Thread(target=lambda: observed.extend(self._rows()))
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(observed, ["a"])

    def test_failed_group_is_discarded_and_failed_batch_rolls_back(self):
        with self.session() as session:
            with self.assertRaises(ValueError):
                with self.storage.connect() as db:
                    db.execute("INSERT INTO image_entries(path,file_size,mtime_ns) VALUES (?,?,?)",
                               ("discard", 1, 1))
                    raise ValueError("abort this group")
            self._write("good")
            with self.storage.connect() as db:
                db.execute("INSERT INTO missing_table VALUES (1)")
            with self.assertRaises(CacheBatchWriteError):
                session.flush()
            self.assertEqual(session.rollbacks, 1)
        self.assertEqual(self._rows(), [])
        with self.session() as later:
            self._write("retry")
        self.assertEqual(self._rows(), ["retry"])

    def test_connection_rejects_other_thread(self):
        with self.session() as session:
            result = []
            thread = threading.Thread(target=lambda: result.append(
                self._attempt_cross_thread_connection(session)))
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, [True])

    def test_locked_database_rolls_back_and_does_not_acknowledge_write(self):
        with self.session() as session:
            session._connection.execute("PRAGMA busy_timeout=1")
            blocker = sqlite3.connect(self.path)
            try:
                blocker.execute("BEGIN IMMEDIATE")
                self._write("uncommitted")
                with self.assertRaises(CacheBatchWriteError):
                    session.flush()
                self.assertEqual(session.commits, 0)
                self.assertEqual(session.rollbacks, 1)
            finally:
                blocker.rollback()
                blocker.close()
        self.assertEqual(self._rows(), [])

    @staticmethod
    def _attempt_cross_thread_connection(session):
        try:
            session.connection()
        except RuntimeError:
            return True
        return False


if __name__ == "__main__":
    unittest.main()
