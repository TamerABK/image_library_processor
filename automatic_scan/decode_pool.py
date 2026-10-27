"""One bounded JPEG prefetch pool shared by the central project stages.

Only JPEG cache misses are submitted. Ordered consumption preserves the
existing per-photo computation and cache-write order. The budget reserves
compressed input plus twice the expected decoded array size, including the
array held by the active consumer. It is a conservative scheduling estimate,
not an OS-level memory limit.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import ctypes
import os
import threading
import time
from pathlib import Path

from image_loader import ImageLoader
from scan_controls import CancellationToken
from .perf import active_profiler, get_active_profiler
from .profiling import profile_count, profile_scope


MAX_DECODE_WORKERS = 4
DEFAULT_MEMORY_MIB = 512


def _available_logical_processors() -> int:
    logical = os.cpu_count() or 1
    if os.name == "nt":
        try:
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
            kernel.GetProcessAffinityMask.restype = ctypes.c_int
            if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),
                    ctypes.byref(process_mask), ctypes.byref(system_mask)):
                return min(logical, process_mask.value.bit_count())
        except (AttributeError, OSError):
            pass
    elif hasattr(os, "sched_getaffinity"):
        try:
            return min(logical, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return logical


def _available_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            from ctypes import wintypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        "total_physical", "available_physical", "total_pagefile",
                        "available_pagefile", "total_virtual", "available_virtual",
                        "available_extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available_physical)
        except (AttributeError, OSError):
            return None
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:  # Optional sampler must not stop a scan.
        return None


def decode_workers_from_environment() -> int:
    logical = _available_logical_processors()
    default = (0 if logical <= 2 else
               min(MAX_DECODE_WORKERS, max(1, logical - 2)))
    try:
        configured = int(os.environ.get("ROOM36_DECODE_WORKERS", str(default)))
    except ValueError:
        return default
    return configured if 0 <= configured <= MAX_DECODE_WORKERS else default


def decode_memory_budget_from_environment() -> int:
    try:
        mib = int(os.environ.get("ROOM36_DECODE_MEMORY_MIB", str(DEFAULT_MEMORY_MIB)))
    except ValueError:
        mib = DEFAULT_MEMORY_MIB
    return max(64, min(2048, mib)) * 1024 * 1024


@dataclass(frozen=True)
class Decoded:
    image: object | None
    error: Exception | None = None


class BoundedDecodePool:
    def __init__(self, *, workers: int, memory_budget_bytes: int):
        if not 1 <= workers <= MAX_DECODE_WORKERS:
            raise ValueError("decode worker count must be 1 through 4")
        self.workers = workers
        self.memory_budget_bytes = max(1, memory_budget_bytes)
        self._executor = ThreadPoolExecutor(max_workers=workers,
                                            thread_name_prefix="room36-decode")
        self._profiler = get_active_profiler()
        self.maximum_pending = 0
        self.maximum_reserved_bytes = 0
        self.last_max_pending = 0
        self.last_max_reserved_bytes = 0
        self.wait_ns = 0
        self.submitted = 0
        self.task_ns = 0
        self.queue_wait_ns = 0
        self._stats_lock = threading.Lock()
        self._closed = False
        self._submission_unavailable = False
        self._effective_budget_bytes = self.memory_budget_bytes
        self._last_pressure_check_ns = 0

    def _current_budget(self) -> int:
        now = time.perf_counter_ns()
        if now - self._last_pressure_check_ns >= 500_000_000:
            self._last_pressure_check_ns = now
            try:
                available = _available_memory_bytes()
                # Lower lookahead as other applications consume RAM. At least
                # one photo is always allowed so the pipeline can progress.
                self._effective_budget_bytes = (min(self.memory_budget_bytes,
                    max(64 * 1024 * 1024, available // 8)) if available is not None
                    else self.memory_budget_bytes)
            except Exception:  # Resource sampling is advisory.
                self._effective_budget_bytes = self.memory_budget_bytes
        return self._effective_budget_bytes

    @staticmethod
    def _reservation(record, max_dimension: int | None, grayscale: bool,
                     memory_budget: int) -> int:
        width, height = record.width, record.height
        if not width or not height:
            return memory_budget  # Unknown dimensions: one image at a time.
        factor = 1
        largest = max(width, height)
        if max_dimension is not None:
            factor = (8 if largest > max_dimension * 4 else
                      4 if largest > max_dimension * 2 else
                      2 if largest > max_dimension else 1)
        pixels = ((width + factor - 1) // factor) * ((height + factor - 1) // factor)
        expected = pixels * (1 if grayscale else 3)
        return min(memory_budget, max(1, record.file_size + 2 * expected))

    def _decode(self, loader: ImageLoader, record, max_dimension: int | None,
                grayscale: bool, token: CancellationToken, queued_ns: int) -> Decoded:
        with active_profiler(self._profiler):
            queue_wait = time.perf_counter_ns() - queued_ns
            with self._stats_lock:
                self.queue_wait_ns += queue_wait
            profile_count("decode.pool.queue_wait_ns", queue_wait)
            if token.is_canceled:
                return Decoded(None)
            started = time.perf_counter_ns()
            try:
                with profile_scope("decode.pool.task", detail="deep", units=1,
                                   width=record.width or 0, height=record.height or 0):
                    image = loader.load_for_scan(record.path, max_dimension=max_dimension,
                                                 grayscale=grayscale)
                return Decoded(image)
            except Exception as exc:
                return Decoded(None, exc)
            finally:
                with self._stats_lock:
                    self.task_ns += time.perf_counter_ns() - started

    def ordered(self, records: list, loader: ImageLoader, *,
                max_dimension: int | None, grayscale: bool,
                token: CancellationToken):
        """Yield (record, Decoded or None) in input order with bounded lookahead.

        None means a non-JPEG record, which keeps its unchanged sequential
        loader path. Closing this generator cancels remaining queued work.
        """
        pending = deque()
        index = 0
        reserved = 0
        local_max_pending = 0
        local_max_reserved = 0

        def fill():
            nonlocal index, reserved, local_max_pending, local_max_reserved
            if token.is_canceled:
                return
            while index < len(records) and len(pending) < self.workers:
                record = records[index]
                jpeg = record.extension in {".jpg", ".jpeg"}
                cost = (self._reservation(record, max_dimension, grayscale,
                                          self.memory_budget_bytes) if jpeg else 0)
                if jpeg and pending and reserved + cost > self._current_budget():
                    break
                index += 1
                queued_ns = time.perf_counter_ns()
                future = None
                if jpeg:
                    if not self._submission_unavailable:
                        try:
                            future = self._executor.submit(self._decode, loader, record,
                                max_dimension, grayscale, token, queued_ns)
                        except (OSError, RuntimeError):
                            self._submission_unavailable = True
                    if future is None:
                        # A thread could not be started. The central processing
                        # worker can still decode this photo sequentially.
                        future = Future()
                        future.set_result(self._decode(loader, record, max_dimension,
                            grayscale, token, queued_ns))
                pending.append((record, future, cost))
                reserved += cost
                if future is not None:
                    self.submitted += 1
                self.maximum_pending = max(self.maximum_pending, len(pending))
                self.maximum_reserved_bytes = max(self.maximum_reserved_bytes, reserved)
                local_max_pending = max(local_max_pending, len(pending))
                local_max_reserved = max(local_max_reserved, reserved)
        try:
            fill()
            while pending:
                token.raise_if_canceled()
                record, future, cost = pending.popleft()
                outcome = None
                if future is not None:
                    started = time.perf_counter_ns()
                    with profile_scope("decode.pool.consumer_wait", detail="deep"):
                        outcome = future.result()
                    self.wait_ns += time.perf_counter_ns() - started
                try:
                    yield record, outcome
                finally:
                    reserved -= cost
                # Do not submit replacement work when close() injects
                # GeneratorExit at the yield point.
                fill()
        finally:
            self.last_max_pending = local_max_pending
            self.last_max_reserved_bytes = local_max_reserved
            for _record, future, _cost in pending:
                if future is not None:
                    future.cancel()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)
