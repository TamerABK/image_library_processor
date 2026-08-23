from __future__ import annotations

import contextlib
import contextvars
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


_ACTIVE_PROFILER: contextvars.ContextVar["AutomaticScanProfiler | None"] = contextvars.ContextVar(
    "automatic_scan_profiler",
    default=None,
)


def get_active_profiler() -> "AutomaticScanProfiler | None":
    return _ACTIVE_PROFILER.get()


@contextlib.contextmanager
def active_profiler(profiler: "AutomaticScanProfiler | None") -> Iterator[None]:
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield
    finally:
        _ACTIVE_PROFILER.reset(token)


@dataclass(slots=True)
class _ModelStats:
    calls: int = 0
    items: int = 0
    total_batch_size: int = 0
    average_batch_size: float = 0.0
    max_batch_size: int = 0
    preprocess_seconds: float = 0.0
    inference_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    session_creation_seconds: float = 0.0
    warmup_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    provider: str | None = None


@dataclass(slots=True)
class _QueueStats:
    wait_seconds: float = 0.0
    max_depth: int = 0
    observations: int = 0
    producer_starvation_count: int = 0


@dataclass(slots=True)
class _DatabaseStats:
    query_count: int = 0
    write_count: int = 0
    commit_count: int = 0
    query_seconds: float = 0.0
    write_seconds: float = 0.0
    commit_seconds: float = 0.0
    connect_count: int = 0
    connect_seconds: float = 0.0


@dataclass(slots=True)
class _ImagePathStats:
    opens: int = 0
    full_decodes: int = 0
    bounded_decodes: int = 0
    header_reads: int = 0
    metadata_reads: int = 0
    exif_reads: int = 0
    grayscale_conversions: int = 0
    color_conversions: int = 0
    normalizations: int = 0
    resizes: int = 0


@dataclass(slots=True)
class AutomaticScanProfiler:
    startup_breakdown: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    discovery_breakdown: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    image_io_breakdown: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    model_phase_breakdown: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _models: dict[str, _ModelStats] = field(default_factory=lambda: defaultdict(_ModelStats))
    _queues: dict[str, _QueueStats] = field(default_factory=lambda: defaultdict(_QueueStats))
    _database: _DatabaseStats = field(default_factory=_DatabaseStats)
    _paths: dict[str, _ImagePathStats] = field(default_factory=lambda: defaultdict(_ImagePathStats))
    _lock: threading.RLock = field(default_factory=threading.RLock)
    face_crop_count: int = 0
    face_alignment_count: int = 0
    full_decodes: int = 0
    bounded_decodes: int = 0
    header_reads: int = 0
    metadata_reads: int = 0
    exif_reads: int = 0
    disk_reads: int = 0
    decode_seconds: float = 0.0
    resize_seconds: float = 0.0
    grayscale_conversion_seconds: float = 0.0
    color_conversion_seconds: float = 0.0
    normalization_seconds: float = 0.0

    def record_startup(self, key: str, seconds: float) -> None:
        with self._lock:
            self.startup_breakdown[key] += seconds

    def record_discovery(self, key: str, seconds: float) -> None:
        with self._lock:
            self.discovery_breakdown[key] += seconds

    def record_path_open(self, path: str | Path) -> None:
        normalized = str(Path(path))
        with self._lock:
            self.disk_reads += 1
            self._paths[normalized].opens += 1

    def record_header_read(self, path: str | Path, seconds: float) -> None:
        normalized = str(Path(path))
        with self._lock:
            self.header_reads += 1
            self.image_io_breakdown["header_read_seconds"] += seconds
            self._paths[normalized].header_reads += 1

    def record_metadata_read(
        self,
        path: str | Path,
        seconds: float,
        *,
        exif: bool,
    ) -> None:
        normalized = str(Path(path))
        with self._lock:
            self.metadata_reads += 1
            self.image_io_breakdown["metadata_read_seconds"] += seconds
            self._paths[normalized].metadata_reads += 1
            if exif:
                self.exif_reads += 1
                self.image_io_breakdown["exif_read_seconds"] += seconds
                self._paths[normalized].exif_reads += 1

    def record_decode(
        self,
        path: str | Path,
        seconds: float,
        *,
        bounded: bool,
    ) -> None:
        normalized = str(Path(path))
        with self._lock:
            self.decode_seconds += seconds
            self.image_io_breakdown["decode_seconds"] += seconds
            if bounded:
                self.bounded_decodes += 1
                self._paths[normalized].bounded_decodes += 1
            else:
                self.full_decodes += 1
                self._paths[normalized].full_decodes += 1

    def record_resize(self, path: str | Path | None, seconds: float) -> None:
        normalized = None if path is None else str(Path(path))
        with self._lock:
            self.resize_seconds += seconds
            self.image_io_breakdown["resize_seconds"] += seconds
            if normalized is not None:
                self._paths[normalized].resizes += 1

    def record_grayscale_conversion(self, path: str | Path | None, seconds: float) -> None:
        normalized = None if path is None else str(Path(path))
        with self._lock:
            self.grayscale_conversion_seconds += seconds
            self.image_io_breakdown["grayscale_conversion_seconds"] += seconds
            if normalized is not None:
                self._paths[normalized].grayscale_conversions += 1

    def record_color_conversion(self, path: str | Path | None, seconds: float) -> None:
        normalized = None if path is None else str(Path(path))
        with self._lock:
            self.color_conversion_seconds += seconds
            self.image_io_breakdown["color_conversion_seconds"] += seconds
            if normalized is not None:
                self._paths[normalized].color_conversions += 1

    def record_normalization(self, path: str | Path | None, seconds: float) -> None:
        normalized = None if path is None else str(Path(path))
        with self._lock:
            self.normalization_seconds += seconds
            self.image_io_breakdown["normalization_seconds"] += seconds
            if normalized is not None:
                self._paths[normalized].normalizations += 1

    def record_face_crop(self, *, aligned: bool = False, count: int = 1) -> None:
        with self._lock:
            self.face_crop_count += count
            if aligned:
                self.face_alignment_count += count

    def record_model_session_creation(
        self,
        model_name: str,
        seconds: float,
        *,
        provider: str | None = None,
    ) -> None:
        with self._lock:
            stats = self._models[model_name]
            stats.session_creation_seconds += seconds
            if provider is not None:
                stats.provider = provider

    def record_model_warmup(self, model_name: str, seconds: float) -> None:
        with self._lock:
            self._models[model_name].warmup_seconds += seconds

    def record_model_call(
        self,
        model_name: str,
        *,
        batch_size: int,
        preprocess_seconds: float = 0.0,
        inference_seconds: float = 0.0,
        postprocess_seconds: float = 0.0,
        provider: str | None = None,
    ) -> None:
        with self._lock:
            stats = self._models[model_name]
            stats.calls += 1
            stats.items += batch_size
            stats.total_batch_size += batch_size
            stats.max_batch_size = max(stats.max_batch_size, batch_size)
            stats.preprocess_seconds += preprocess_seconds
            stats.inference_seconds += inference_seconds
            stats.postprocess_seconds += postprocess_seconds
            stats.average_batch_size = (
                stats.total_batch_size / stats.calls
                if stats.calls
                else 0.0
            )
            if provider is not None:
                stats.provider = provider

    def record_model_phase(self, key: str, seconds: float) -> None:
        with self._lock:
            self.model_phase_breakdown[key] += seconds

    def record_queue_wait(
        self,
        queue_name: str,
        wait_seconds: float,
        *,
        depth: int | None = None,
        producer_starved: bool = False,
    ) -> None:
        with self._lock:
            stats = self._queues[queue_name]
            stats.wait_seconds += wait_seconds
            stats.observations += 1
            if depth is not None:
                stats.max_depth = max(stats.max_depth, depth)
            if producer_starved:
                stats.producer_starvation_count += 1

    def record_database_connect(self, seconds: float) -> None:
        with self._lock:
            self._database.connect_count += 1
            self._database.connect_seconds += seconds

    def record_sql(self, statement: str, seconds: float) -> None:
        keyword = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ""
        with self._lock:
            if keyword in {"SELECT", "PRAGMA", "WITH"}:
                self._database.query_count += 1
                self._database.query_seconds += seconds
            else:
                self._database.write_count += 1
                self._database.write_seconds += seconds

    def record_commit(self, seconds: float) -> None:
        with self._lock:
            self._database.commit_count += 1
            self._database.commit_seconds += seconds

    def image_io_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "full_decodes": self.full_decodes,
                "bounded_decodes": self.bounded_decodes,
                "header_reads": self.header_reads,
                "metadata_reads": self.metadata_reads,
                "exif_reads": self.exif_reads,
                "disk_reads": self.disk_reads,
                "decode_seconds": round(self.decode_seconds, 6),
                "resize_seconds": round(self.resize_seconds, 6),
                "grayscale_conversion_seconds": round(
                    self.grayscale_conversion_seconds,
                    6,
                ),
                "color_conversion_seconds": round(
                    self.color_conversion_seconds,
                    6,
                ),
                "normalization_seconds": round(self.normalization_seconds, 6),
                "face_crop_count": self.face_crop_count,
                "face_alignment_count": self.face_alignment_count,
                "by_path": {
                    path: {
                        "opens": stats.opens,
                        "full_decodes": stats.full_decodes,
                        "bounded_decodes": stats.bounded_decodes,
                        "header_reads": stats.header_reads,
                        "metadata_reads": stats.metadata_reads,
                        "exif_reads": stats.exif_reads,
                        "grayscale_conversions": stats.grayscale_conversions,
                        "color_conversions": stats.color_conversions,
                        "normalizations": stats.normalizations,
                        "resizes": stats.resizes,
                    }
                    for path, stats in sorted(self._paths.items())
                },
            }

    def model_metrics(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                name: {
                    "calls": stats.calls,
                    "items": stats.items,
                    "average_batch_size": round(stats.average_batch_size, 4),
                    "max_batch_size": stats.max_batch_size,
                    "preprocess_seconds": round(stats.preprocess_seconds, 6),
                    "inference_seconds": round(stats.inference_seconds, 6),
                    "postprocess_seconds": round(stats.postprocess_seconds, 6),
                    "session_creation_seconds": round(stats.session_creation_seconds, 6),
                    "warmup_seconds": round(stats.warmup_seconds, 6),
                    "queue_wait_seconds": round(stats.queue_wait_seconds, 6),
                    "provider": stats.provider,
                }
                for name, stats in sorted(self._models.items())
            }

    def queue_metrics(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                name: {
                    "wait_seconds": round(stats.wait_seconds, 6),
                    "max_depth": stats.max_depth,
                    "observations": stats.observations,
                    "producer_starvation_count": stats.producer_starvation_count,
                }
                for name, stats in sorted(self._queues.items())
            }

    def database_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "query_count": self._database.query_count,
                "write_count": self._database.write_count,
                "commit_count": self._database.commit_count,
                "query_seconds": round(self._database.query_seconds, 6),
                "write_seconds": round(self._database.write_seconds, 6),
                "commit_seconds": round(self._database.commit_seconds, 6),
                "connect_count": self._database.connect_count,
                "connect_seconds": round(self._database.connect_seconds, 6),
            }

    def export(
        self,
        *,
        total_seconds: float,
        startup_seconds: float,
        image_count: int,
        cache_mode: str,
    ) -> dict[str, Any]:
        scan_seconds = max(total_seconds - startup_seconds, 0.0)
        with self._lock:
            return {
                "total_seconds": round(total_seconds, 6),
                "startup_seconds": round(startup_seconds, 6),
                "scan_seconds": round(scan_seconds, 6),
                "images_per_second": round(
                    image_count / total_seconds,
                    6,
                )
                if total_seconds > 0
                else 0.0,
                "cache_mode": cache_mode,
                "image_io": self.image_io_metrics(),
                "models": self.model_metrics(),
                "database": self.database_metrics(),
                "queues": self.queue_metrics(),
                "startup_breakdown": {
                    key: round(value, 6)
                    for key, value in sorted(self.startup_breakdown.items())
                },
                "discovery_breakdown": {
                    key: round(value, 6)
                    for key, value in sorted(self.discovery_breakdown.items())
                },
                "model_phase_breakdown": {
                    key: round(value, 6)
                    for key, value in sorted(self.model_phase_breakdown.items())
                },
            }


class InstrumentedSqliteConnection(sqlite3.Connection):
    def execute(self, sql: str, parameters: Any = (), /):  # type: ignore[override]
        started = time.perf_counter()
        try:
            return super().execute(sql, parameters)
        finally:
            profiler = get_active_profiler()
            if profiler is not None:
                profiler.record_sql(sql, time.perf_counter() - started)

    def executemany(self, sql: str, seq_of_parameters: Any, /):  # type: ignore[override]
        started = time.perf_counter()
        try:
            return super().executemany(sql, seq_of_parameters)
        finally:
            profiler = get_active_profiler()
            if profiler is not None:
                profiler.record_sql(sql, time.perf_counter() - started)

    def executescript(self, sql_script: str, /):  # type: ignore[override]
        started = time.perf_counter()
        try:
            return super().executescript(sql_script)
        finally:
            profiler = get_active_profiler()
            if profiler is not None:
                profiler.record_sql(sql_script, time.perf_counter() - started)

    def commit(self) -> None:  # type: ignore[override]
        started = time.perf_counter()
        try:
            return super().commit()
        finally:
            profiler = get_active_profiler()
            if profiler is not None:
                profiler.record_commit(time.perf_counter() - started)
