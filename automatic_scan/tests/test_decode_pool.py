from __future__ import annotations

import os
import json
from dataclasses import replace
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from automatic_scan.decode_pool import (BoundedDecodePool,
    _available_logical_processors, decode_memory_budget_from_environment,
    decode_workers_from_environment)
from automatic_scan.models import DiscoveryRecord
from automatic_scan.config import AutomaticScanConfig, BlurPolicy
from automatic_scan.orchestrator import AutomaticScanOrchestrator
from automatic_scan.project_pipeline import process_project
from automatic_scan.services import AutomaticScanServices
from automatic_scan.state_store import AutomaticStateStore
from image_loader import ImageLoader
from scan_controls import CancellationToken, ScanCancelledError


class DecodePoolTests(unittest.TestCase):
    def _records(self, folder: Path, count=8):
        records = []
        for index in range(count):
            path = folder / f"{index}.jpg"
            pixels = np.zeros((96, 128, 3), dtype=np.uint8)
            pixels[:, :, 0] = index * 19
            pixels[:, :, 1] = np.arange(128, dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            stat = path.stat()
            records.append(DiscoveryRecord(path, stat.st_size, stat.st_mtime_ns,
                ".jpg", "landscape", stat.st_mtime, 128, 96))
        return records

    def test_adaptive_worker_count_and_hard_maximum(self):
        self.assertGreaterEqual(_available_logical_processors(), 1)
        with (patch("automatic_scan.decode_pool._available_logical_processors", return_value=4),
              patch.dict(os.environ, {}, clear=False)):
            os.environ.pop("ROOM36_DECODE_WORKERS", None)
            self.assertEqual(decode_workers_from_environment(), 2)
        with patch.dict(os.environ, {"ROOM36_DECODE_WORKERS": "4"}):
            self.assertEqual(decode_workers_from_environment(), 4)
        with patch("automatic_scan.decode_pool._available_logical_processors", return_value=1):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ROOM36_DECODE_WORKERS", None)
                self.assertEqual(decode_workers_from_environment(), 0)
        with patch("automatic_scan.decode_pool._available_logical_processors", return_value=2):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ROOM36_DECODE_WORKERS", None)
                self.assertEqual(decode_workers_from_environment(), 0)
        with patch("automatic_scan.decode_pool._available_logical_processors", return_value=12):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ROOM36_DECODE_WORKERS", None)
                self.assertEqual(decode_workers_from_environment(), 4)
        with self.assertRaises(ValueError):
            BoundedDecodePool(workers=5, memory_budget_bytes=1024)

    def test_ordered_decodes_match_loader_and_close_threads(self):
        with tempfile.TemporaryDirectory() as folder:
            records = self._records(Path(folder))
            loader = ImageLoader()
            token = CancellationToken()
            pool = BoundedDecodePool(workers=4, memory_budget_bytes=2_000_000)
            try:
                values = list(pool.ordered(records, loader, max_dimension=64,
                                           grayscale=False, token=token))
                self.assertEqual([record.path for record, _ in values],
                                 [record.path for record in records])
                for record, outcome in values:
                    self.assertIsNone(outcome.error)
                    baseline = loader.load_for_scan(record.path, max_dimension=64)
                    self.assertTrue(np.array_equal(outcome.image, baseline))
                    with loader.use_prefetched(record.path, outcome.image, max_dimension=64):
                        self.assertIs(loader.load_for_scan(record.path, max_dimension=64), outcome.image)
                self.assertLessEqual(pool.maximum_pending, 4)
                self.assertLessEqual(pool.maximum_reserved_bytes, pool.memory_budget_bytes)
                self.assertEqual(pool.last_max_pending, pool.maximum_pending)
            finally:
                pool.close()
            self.assertFalse(any(t.name.startswith("room36-decode") for t in threading.enumerate()))

    def test_cancel_stops_submission_and_releases_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            records = self._records(Path(folder), 12)
            token = CancellationToken()
            pool = BoundedDecodePool(workers=2, memory_budget_bytes=2_000_000)
            try:
                iterator = pool.ordered(records, ImageLoader(), max_dimension=None,
                                        grayscale=False, token=token)
                next(iterator)
                token.cancel()
                with self.assertRaises(ScanCancelledError):
                    next(iterator)
                iterator.close()
                self.assertLessEqual(pool.submitted, 2)
            finally:
                pool.close()
            self.assertFalse(any(t.name.startswith("room36-decode") for t in threading.enumerate()))

    def test_closing_iterator_does_not_submit_another_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            records = self._records(Path(folder), 12)
            pool = BoundedDecodePool(workers=2, memory_budget_bytes=2_000_000)
            try:
                iterator = pool.ordered(records, ImageLoader(), max_dimension=64,
                    grayscale=False, token=CancellationToken())
                next(iterator)
                submitted = pool.submitted
                iterator.close()
                self.assertEqual(pool.submitted, submitted)
            finally:
                pool.close()

    def test_memory_configuration_is_bounded(self):
        with patch.dict(os.environ, {"ROOM36_DECODE_MEMORY_MIB": "100000"}):
            self.assertEqual(decode_memory_budget_from_environment(), 2048 * 1024 * 1024)

    def test_low_available_memory_reduces_prefetch_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            records = [replace(record, width=6000, height=4000)
                for record in self._records(Path(folder), 5)]
            pool = BoundedDecodePool(workers=4, memory_budget_bytes=512 * 1024 * 1024)
            try:
                with patch("automatic_scan.decode_pool._available_memory_bytes",
                        return_value=512 * 1024 * 1024):
                    self.assertEqual(len(list(pool.ordered(records, ImageLoader(),
                        max_dimension=None, grayscale=False,
                        token=CancellationToken()))), len(records))
                self.assertEqual(pool.maximum_pending, 1)
                self.assertEqual(pool.last_max_pending, 1)
            finally:
                pool.close()

    def test_full_resolution_grayscale_and_corrupt_jpeg_match_sequential(self):
        with tempfile.TemporaryDirectory() as folder:
            records = self._records(Path(folder), 3)
            bad = Path(folder) / "bad.jpg"
            bad.write_bytes(b"not a JPEG")
            stat = bad.stat()
            records.append(DiscoveryRecord(bad, stat.st_size, stat.st_mtime_ns,
                ".jpg", None, stat.st_mtime, None, None))
            loader = ImageLoader()
            pool = BoundedDecodePool(workers=4, memory_budget_bytes=1_000_000)
            try:
                for grayscale in (False, True):
                    for record, outcome in pool.ordered(records, loader,
                            max_dimension=None, grayscale=grayscale,
                            token=CancellationToken()):
                        self.assertIsNone(outcome.error)
                        baseline = loader.load_for_scan(record.path, grayscale=grayscale)
                        if baseline is None:
                            self.assertIsNone(outcome.image)
                        else:
                            self.assertTrue(np.array_equal(outcome.image, baseline))
                self.assertLessEqual(pool.maximum_reserved_bytes, pool.memory_budget_bytes)
            finally:
                pool.close()

    def test_one_worker_obeys_same_order_and_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            records = self._records(Path(folder), 8)
            pool = BoundedDecodePool(workers=1, memory_budget_bytes=1_000_000)
            try:
                actual = [record.path for record, _ in pool.ordered(records, ImageLoader(),
                    max_dimension=1000, grayscale=False, token=CancellationToken())]
                self.assertEqual(actual, [record.path for record in records])
                self.assertEqual(pool.maximum_pending, 1)
            finally:
                pool.close()

    def test_unavailable_worker_falls_back_to_sequential_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            records = self._records(Path(folder), 2)
            loader = ImageLoader()
            pool = BoundedDecodePool(workers=2, memory_budget_bytes=1_000_000)
            try:
                with patch.object(pool._executor, "submit", side_effect=RuntimeError("thread unavailable")):
                    outcomes = list(pool.ordered(records, loader, max_dimension=64,
                        grayscale=False, token=CancellationToken()))
                self.assertTrue(pool._submission_unavailable)
                for record, outcome in outcomes:
                    self.assertTrue(np.array_equal(outcome.image,
                        loader.load_for_scan(record.path, max_dimension=64)))
            finally:
                pool.close()

    def test_real_blur_and_duplicate_results_match_sequential(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = self._records(root, 3)

            def run(workers: int):
                state = root / f"state-{workers}"
                config = AutomaticScanConfig(blur_policy=BlurPolicy.KEEP_ALL)
                services = AutomaticScanServices(config, lazy_models=True,
                    cache_path=state / "analysis.sqlite3",
                    face_database_path=state / "faces.sqlite3")
                events = {}
                with patch.dict(os.environ, {"ROOM36_DECODE_WORKERS": str(workers)}):
                    process_project(AutomaticScanOrchestrator(
                        state_store=AutomaticStateStore(state / "reviews")),
                        str(root), [record.path for record in records],
                        emit=lambda stage, status, _done, _total, _error, data, _stats:
                            events.__setitem__(stage, data) if str(status) == "COMPLETE" else None,
                        cancellation_token=CancellationToken(),
                        cache_path=state / "analysis.sqlite3", config=config,
                        services=services, benchmark_stop_after="duplicates")
                return {key: json.dumps(events[key], sort_keys=True, default=str)
                        for key in ("blur", "duplicates")}

            self.assertEqual(run(0), run(4))


if __name__ == "__main__":
    unittest.main()
