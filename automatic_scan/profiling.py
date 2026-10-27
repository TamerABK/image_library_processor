"""Opt-in, bounded profiling for the progressive project pipeline.

No paths or person identifiers are written to reports.  The legacy scan's
AutomaticScanProfiler hooks remain available through this subclass.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .perf import AutomaticScanProfiler, get_active_profiler


@dataclass
class _Timing:
    calls: int = 0
    inclusive_ns: int = 0
    exclusive_ns: int = 0
    minimum_ns: int = 2**63 - 1
    maximum_ns: int = 0
    samples_ns: deque[int] = field(default_factory=lambda: deque(maxlen=4096))

    def add(self, inclusive: int, exclusive: int) -> None:
        self.calls += 1
        self.inclusive_ns += inclusive
        self.exclusive_ns += exclusive
        self.minimum_ns = min(self.minimum_ns, inclusive)
        self.maximum_ns = max(self.maximum_ns, inclusive)
        self.samples_ns.append(inclusive)


class HierarchicalProfiler(AutomaticScanProfiler):
    """Per-thread nesting, bounded trace and duration samples.

    Exclusive time subtracts child spans on the same thread. Cross-thread
    spans may overlap; summed inclusive time is not project wall time.
    """

    def __init__(self, *, detail: str = "normal", max_events: int = 50000):
        super().__init__()
        if detail not in {"stages", "normal", "deep"}:
            raise ValueError("detail must be stages, normal or deep")
        self.detail = detail
        self.max_events = max_events
        self.started_ns = time.perf_counter_ns()
        self.finished_ns: int | None = None
        self._local = threading.local()
        self._timings: dict[str, _Timing] = defaultdict(_Timing)
        # Keep the whole-run stage timeline even when high-frequency leaf spans
        # exceed the trace budget. Recent leaf spans are more useful than a
        # trace that ends halfway through processing.
        self._stage_capacity = 0 if max_events < 2 else min(max_events // 2, max(128, max_events // 10))
        self._events: deque[dict] = deque(maxlen=max(0, max_events - self._stage_capacity))
        self._stage_events: deque[dict] = deque(maxlen=self._stage_capacity)
        self._dropped_events = 0
        self._counters: dict[str, int] = defaultdict(int)

    @contextlib.contextmanager
    def scope(self, name: str, *, detail: str = "normal", units: int = 0,
              batch_size: int = 0, width: int = 0, height: int = 0,
              face_count: int = 0):
        if {"stages": 0, "normal": 1, "deep": 2}[detail] > {
            "stages": 0, "normal": 1, "deep": 2}[self.detail]:
            yield
            return
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        start = time.perf_counter_ns()
        frame = [start, 0]
        stack.append(frame)
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            stack.pop()
            inclusive = max(0, end - start)
            exclusive = max(0, inclusive - frame[1])
            if stack:
                stack[-1][1] += inclusive
            with self._lock:
                self._timings[name].add(inclusive, exclusive)
                if units:
                    self._counters[name + ".units"] += units
                if face_count:
                    self._counters["faces_seen"] += face_count
                is_stage = detail == "stages" or name.startswith("stage.")
                target = self._stage_events if is_stage and self._stage_capacity else self._events
                if target.maxlen:
                    if len(target) == target.maxlen:
                        self._dropped_events += 1
                    target.append({"name": name, "ph": "X", "ts": (start - self.started_ns) / 1000,
                        "dur": inclusive / 1000, "pid": os.getpid(), "tid": threading.get_ident(),
                        "args": {"worker": threading.current_thread().name, "units": units,
                                 "batch_size": batch_size, "width": width, "height": height,
                                 "face_count": face_count}})
                else:
                    self._dropped_events += 1

    def add_count(self, name: str, count: int = 1) -> None:
        with self._lock:
            self._counters[name] += count

    def finish(self) -> None:
        if self.finished_ns is None:
            self.finished_ns = time.perf_counter_ns()

    @staticmethod
    def _percentile(values: list[int], fraction: float) -> float:
        if not values:
            return 0.0
        position = (len(values) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return (values[low] + (values[high] - values[low]) * (position - low)) / 1e9

    def report(self, *, image_count: int = 0, face_count: int | None = None) -> dict:
        self.finish()
        wall = (self.finished_ns - self.started_ns) / 1e9
        with self._lock:
            timings = {}
            for name, timing in sorted(self._timings.items()):
                samples = sorted(timing.samples_ns)
                timings[name] = {"calls": timing.calls, "inclusive_seconds": timing.inclusive_ns / 1e9,
                    "exclusive_seconds": timing.exclusive_ns / 1e9,
                    "average_seconds": timing.inclusive_ns / timing.calls / 1e9,
                    "median_seconds": self._percentile(samples, .5),
                    "p95_seconds": self._percentile(samples, .95),
                    "p99_seconds": self._percentile(samples, .99),
                    "min_seconds": timing.minimum_ns / 1e9,
                    "max_seconds": timing.maximum_ns / 1e9,
                    "sample_count": len(samples)}
            io = self.image_io_metrics()
            io.pop("by_path", None)  # Existing legacy API retains it; shared reports do not.
            counters = dict(self._counters)
            committed_groups = counters.get("cache.batch.records_committed", 0)
            batch_commits = counters.get("cache.batch.commits", 0)
            batch_timing = timings.get("cache.batch.commit", {})
            cache_batch = {"enabled": bool(counters.get("cache.connection_reuses", 0)),
                "connection_reuses": counters.get("cache.connection_reuses", 0),
                "write_groups": counters.get("cache.write_groups_queued", 0),
                "commits": batch_commits,
                "rollbacks": counters.get("cache.batch.rollbacks", 0),
                "average_batch_occupancy": committed_groups / batch_commits if batch_commits else 0,
                "maximum_batch_occupancy": counters.get("cache.batch.max_pending", 0),
                "total_buffer_dwell_seconds": counters.get("cache.batch.buffer_wait_ns", 0) / 1e9,
                "maximum_buffer_dwell_seconds": counters.get("cache.batch.max_buffer_wait_ns", 0) / 1e9,
                "commit_p95_seconds": batch_timing.get("p95_seconds"),
                "wal_peak_bytes": counters.get("cache.batch.wal_peak_bytes", 0)}
            return {"schema": 1, "wall_seconds": wall, "image_count": image_count,
                "face_count": face_count, "images_per_second": image_count / wall if wall else 0,
                "faces_per_second": face_count / wall if wall and face_count is not None else None,
                "timings": timings, "counters": counters, "cache_batch": cache_batch,
                "trace_dropped_events": self._dropped_events,
                "duration_sample_limit_per_scope": 4096,
                "image_io": io, "models": self.model_metrics(),
                "database": self.database_metrics(), "queues": self.queue_metrics(),
                "startup_breakdown": dict(self.startup_breakdown),
                "model_phase_breakdown": dict(self.model_phase_breakdown)}

    def trace(self) -> dict:
        with self._lock:
            events = sorted((*self._stage_events, *self._events), key=lambda event: event["ts"])
            return {"traceEvents": events, "displayTimeUnit": "ms"}


_NOOP = contextlib.nullcontext()


def profile_scope(name: str, **kwargs):
    """Near-zero disabled cost: one contextvar read and a shared no-op context."""
    profiler = get_active_profiler()
    return profiler.scope(name, **kwargs) if isinstance(profiler, HierarchicalProfiler) else _NOOP


def profile_count(name: str, count: int = 1) -> None:
    profiler = get_active_profiler()
    if isinstance(profiler, HierarchicalProfiler):
        profiler.add_count(name, count)


@contextlib.contextmanager
def profile_lock_wait(name: str, lock):
    """Time lock acquisition separately from work performed while holding it."""
    with profile_scope(name, detail="deep"):
        lock.acquire()
    try:
        yield
    finally:
        lock.release()
