"""Bounded process-resource samples collected off the processing thread."""
from __future__ import annotations

import ctypes
import gc
import os
import threading
import time
from ctypes import wintypes

from .profiling_gpu_windows import WindowsGpuProcessSampler
from .profiling_disk_windows import WindowsPhysicalDiskSampler


def _windows_counters() -> dict:
    """Read-only Win32 fallback when optional psutil is absent."""
    if os.name != "nt":
        return {}
    from ctypes import c_size_t, c_ulonglong

    class Memory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD),
            ("peak_working_set", c_size_t), ("working_set", c_size_t),
            ("peak_pagefile", c_size_t), ("pagefile", c_size_t),
            ("peak_nonpaged_pool", c_size_t), ("nonpaged_pool", c_size_t),
            ("pagefile_usage", c_size_t), ("peak_pagefile_usage", c_size_t)]

    class Io(ctypes.Structure):
        _fields_ = [(name, c_ulonglong) for name in (
            "read_operations", "write_operations", "other_operations",
            "read_bytes", "write_bytes", "other_bytes")]

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel32.GetCurrentProcess()
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
    kernel32.GetProcessIoCounters.argtypes = (wintypes.HANDLE, ctypes.c_void_p)
    memory = Memory()
    memory.cb = ctypes.sizeof(Memory)
    io = Io()
    result = {}
    if psapi.GetProcessMemoryInfo(process, ctypes.byref(memory), memory.cb):
        result.update(rss_bytes=int(memory.working_set), peak_rss_bytes=int(memory.peak_working_set),
                      page_faults=int(memory.page_faults))
    if kernel32.GetProcessIoCounters(process, ctypes.byref(io)):
        result.update(read_bytes=int(io.read_bytes), write_bytes=int(io.write_bytes))
    return result


def _windows_core_times() -> list[tuple[int, int]] | None:
    """Kernel/user and idle ticks per logical processor via Windows NT query.

    This is sampling telemetry, not a scheduling profiler. Return unavailable
    rather than a guessed utilization if Windows rejects the query.
    """
    if os.name != "nt":
        return None
    class Processor(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in
                    ("idle", "kernel", "user", "dpc", "interrupt")]
        _fields_ += [("interrupt_count", wintypes.ULONG)]
    capacity = max(os.cpu_count() or 1, 64) * 4
    values = (Processor * capacity)()
    returned = wintypes.ULONG()
    try:
        status = ctypes.windll.ntdll.NtQuerySystemInformation(
            8, ctypes.byref(values), ctypes.sizeof(values), ctypes.byref(returned))
    except (AttributeError, OSError):
        return None
    if status != 0 or returned.value % ctypes.sizeof(Processor):
        return None
    return [(int(v.kernel + v.user), int(v.idle))
            for v in values[:returned.value // ctypes.sizeof(Processor)]]


def _windows_thread_cpu_times() -> dict[str, float] | None:
    """Sample native process threads, including OpenCV/ORT threads, via Win32."""
    if os.name != "nt":
        return None
    class ThreadEntry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                    ("thread_id", wintypes.DWORD), ("owner_pid", wintypes.DWORD),
                    ("base_priority", wintypes.LONG), ("delta_priority", wintypes.LONG),
                    ("flags", wintypes.DWORD)]
    class FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]
    kernel = ctypes.windll.kernel32
    kernel.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Thread32First.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry))
    kernel.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry))
    kernel.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenThread.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.GetThreadTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime), ctypes.POINTER(FileTime), ctypes.POINTER(FileTime))
    snapshot = kernel.CreateToolhelp32Snapshot(0x4, 0)
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return None
    result = {}
    entry = ThreadEntry()
    entry.size = ctypes.sizeof(ThreadEntry)
    try:
        found = kernel.Thread32First(snapshot, ctypes.byref(entry))
        while found:
            if entry.owner_pid == os.getpid():
                handle = kernel.OpenThread(0x0800, False, entry.thread_id)
                if handle:
                    try:
                        created, exited, kernel_time, user_time = (FileTime() for _ in range(4))
                        if kernel.GetThreadTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                                 ctypes.byref(kernel_time), ctypes.byref(user_time)):
                            ticks = ((kernel_time.high << 32) | kernel_time.low) + ((user_time.high << 32) | user_time.low)
                            result[str(entry.thread_id)] = ticks / 10_000_000
                    finally:
                        kernel.CloseHandle(handle)
            found = kernel.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return result


class ResourceSampler:
    """One sampling thread; optional psutil adds per-core/thread/context data."""

    def __init__(self, *, interval: float = 0.25, max_samples: int = 20000,
                 dataset: str | None = None):
        self.interval = max(.05, interval)
        self.max_samples = max_samples
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._origin = time.perf_counter()
        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process()
        except ImportError:
            self._psutil = None
            self._process = None
        self._error: str | None = None
        self._previous_core_times: list[tuple[int, int]] | None = None
        self._gpu = WindowsGpuProcessSampler() if os.name == "nt" else None
        self._disk = WindowsPhysicalDiskSampler(dataset) if os.name == "nt" and dataset else None
        self._gpu_counter_available = bool(self._gpu and self._gpu.available)
        self._sample_index = 0
        self._gc_started: dict[int, int] = {}
        self._gc_lock = threading.Lock()
        self._gc_count = 0
        self._gc_pause_ns = 0
        self._gc_max_pause_ns = 0

    def _on_gc(self, phase: str, info: dict) -> None:
        thread_id = threading.get_ident()
        now = time.perf_counter_ns()
        with self._gc_lock:
            if phase == "start":
                self._gc_started[thread_id] = now
            elif phase == "stop":
                started = self._gc_started.pop(thread_id, None)
                if started is not None:
                    elapsed = max(0, now - started)
                    self._gc_count += 1
                    self._gc_pause_ns += elapsed
                    self._gc_max_pause_ns = max(self._gc_max_pause_ns, elapsed)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Resource sampler already started")
        self._origin = time.perf_counter()
        self._stop.clear()
        gc.callbacks.append(self._on_gc)
        self._thread = threading.Thread(target=self._run, name="room36-profile-resources", daemon=True)
        self._thread.start()

    def _read(self) -> dict:
        sample = {"seconds": time.perf_counter() - self._origin,
                  "process_cpu_seconds": time.process_time(),
                  "python_threads": threading.active_count()}
        if self._process is not None:
            process = self._process
            memory = process.memory_info()
            io = process.io_counters() if hasattr(process, "io_counters") else None
            sample.update(rss_bytes=memory.rss, process_threads=process.num_threads(),
                system_cpu_percent=self._psutil.cpu_percent(interval=None),
                per_core_cpu_percent=self._psutil.cpu_percent(interval=None, percpu=True))
            if hasattr(memory, "num_page_faults"):
                sample["process_total_page_faults"] = memory.num_page_faults
            elif hasattr(memory, "pfaults"):
                sample["process_total_page_faults"] = memory.pfaults
            if io is not None:
                sample.update(read_bytes=io.read_bytes, write_bytes=io.write_bytes)
            switches = process.num_ctx_switches()
            sample.update(voluntary_context_switches=switches.voluntary,
                          involuntary_context_switches=switches.involuntary)
            if self._sample_index % 4 == 0:  # ~1 Hz at the default interval.
                sample["thread_cpu_seconds"] = {str(t.id): t.user_time + t.system_time
                    for t in process.threads()}
        else:
            sample.update(_windows_counters())
            if self._sample_index % 4 == 0:
                threads = _windows_thread_cpu_times()
                if threads is not None:
                    sample["thread_cpu_seconds"] = threads
                    sample["process_threads"] = len(threads)
            current = _windows_core_times()
            previous = self._previous_core_times
            self._previous_core_times = current
            if current and previous and len(current) == len(previous):
                percentages = []
                for (total, idle), (old_total, old_idle) in zip(current, previous):
                    elapsed = total - old_total
                    percentages.append(100 * max(0, elapsed - (idle - old_idle)) / elapsed
                                       if elapsed > 0 else 0.0)
                sample["per_core_cpu_percent"] = percentages
                sample["system_cpu_percent"] = sum(percentages) / len(percentages)
        if self._gpu is not None and self._sample_index % 2 == 0:
            try:
                sample.update(self._gpu.sample())
            except Exception as exc:
                self._gpu.error = type(exc).__name__
        if self._disk is not None:
            try:
                sample.update(self._disk.sample())
            except Exception as exc:
                self._disk.error = type(exc).__name__
        self._sample_index += 1
        return sample

    def _run(self) -> None:
        while not self._stop.is_set() and len(self.samples) < self.max_samples:
            try:
                self.samples.append(self._read())
            except Exception as exc:
                self._error = type(exc).__name__
                break
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Resource sampler did not stop")
        if self._gpu is not None:
            self._gpu.close()
        if self._disk is not None:
            self._disk.close()

    def report(self) -> dict:
        first, last = (self.samples[0], self.samples[-1]) if self.samples else ({}, {})
        cpu_wall = last.get("seconds", 0) - first.get("seconds", 0)
        cpu_used = last.get("process_cpu_seconds", 0) - first.get("process_cpu_seconds", 0)
        gpu_compute = [sample["gpu_process_compute_engine_percent"] for sample in self.samples
                       if "gpu_process_compute_engine_percent" in sample]
        gpu_engine_values: dict[str, list[float]] = {}
        for sample in self.samples:
            for name, value in sample.get("gpu_process_engine_percent_sum", {}).items():
                gpu_engine_values.setdefault(name, []).append(value)
        gpu_dedicated = [sample["gpu_process_dedicated_bytes"] for sample in self.samples
                         if "gpu_process_dedicated_bytes" in sample]
        disk = {}
        for key in ("read_mb_s", "write_mb_s", "read_iops", "write_iops",
                    "read_latency_s", "write_latency_s", "queue_depth",
                    "outstanding_io", "activity_percent"):
            values = [s["physical_disk_" + key] for s in self.samples
                      if "physical_disk_" + key in s]
            disk[key] = {"mean": sum(values) / len(values), "peak": max(values)} if values else None
        disk_integrated = {}
        for kind in ("read", "write"):
            field = "physical_disk_" + kind + "_mb_s"
            disk_integrated[kind + "_bytes_estimated"] = sum(
                sample[field] * 1_000_000 * max(0, sample["seconds"] - prior["seconds"])
                for prior, sample in zip(self.samples, self.samples[1:]) if field in sample)
        return {"sampler": "psutil" if self._psutil else "standard-library/Win32",
            "samples": self.samples, "sample_count": len(self.samples),
            "sample_limit": self.max_samples, "error": self._error,
            "process_cpu_core_equivalents": cpu_used / cpu_wall if cpu_wall > 0 else None,
            "peak_rss_bytes": max((s.get("rss_bytes", 0) for s in self.samples), default=None),
            "disk_read_bytes_delta": max(0, last.get("read_bytes", 0) - first.get("read_bytes", 0))
                if "read_bytes" in first and "read_bytes" in last else None,
            "disk_write_bytes_delta": max(0, last.get("write_bytes", 0) - first.get("write_bytes", 0))
                if "write_bytes" in first and "write_bytes" in last else None,
            "gc_collections": self._gc_count,
            "gc_pause_seconds": self._gc_pause_ns / 1e9,
            "gc_max_pause_seconds": self._gc_max_pause_ns / 1e9,
            "gpu_process_compute_engine_percent_mean":
                sum(gpu_compute) / len(gpu_compute) if gpu_compute else None,
            "gpu_process_compute_engine_percent_peak": max(gpu_compute) if gpu_compute else None,
            "gpu_process_engines": {name: {"mean_percent": sum(values) / len(values),
                "peak_percent": max(values)} for name, values in gpu_engine_values.items()},
            "gpu_process_dedicated_bytes_peak": max(gpu_dedicated) if gpu_dedicated else None,
            "gpu_counter_available": self._gpu_counter_available,
            "gpu_counter_error": self._gpu.error if self._gpu else None,
            "device_gpu_utilization": None,
            "gpu_counter_note": "Process GPU Engine/Memory PDH counters; engine percentages are not whole-device occupancy.",
            "physical_disk": {"dataset_drive": self._disk.drive if self._disk else None,
                              "instance": self._disk.instance if self._disk else None,
                              "error": self._disk.error if self._disk else "unavailable",
                              "system_wide_not_process_attributed": True, "metrics": disk},
            "physical_disk_integrated": disk_integrated,
            "process_hard_page_faults": None,
            "page_fault_note": "Windows process total page faults do not distinguish hard faults; process hard faults need ETW and are unavailable here.",
            "unavailable": [name for name, ok in (
                ("per-thread CPU", any("thread_cpu_seconds" in s for s in self.samples)),
                ("context switches", self._psutil is not None),
                ("per-core CPU", self._psutil is not None or any(
                    "per_core_cpu_percent" in s for s in self.samples)),
                ("process GPU compute-engine utilization", bool(gpu_compute)),
                ("process dedicated GPU memory", bool(gpu_dedicated)),
                ("whole-device GPU utilization", False)) if not ok]}

    def trace_events(self, profiler_start_ns: int) -> list[dict]:
        """Chrome/Perfetto counters aligned with the profiler's monotonic clock."""
        offset_us = (self._origin - profiler_start_ns / 1e9) * 1e6
        return [{"name": "resources", "ph": "C", "ts": offset_us + sample["seconds"] * 1e6,
                 "pid": os.getpid(), "tid": 0,
                 "args": {key: sample[key] for key in (
                     "rss_bytes", "process_cpu_seconds", "system_cpu_percent",
                     "read_bytes", "write_bytes", "process_threads", "python_threads",
                     "gpu_process_peak_engine_percent", "gpu_process_compute_engine_percent",
                     "gpu_process_dedicated_bytes",
                     "physical_disk_read_mb_s", "physical_disk_write_mb_s",
                     "physical_disk_read_iops", "physical_disk_write_iops",
                     "physical_disk_read_latency_s", "physical_disk_write_latency_s",
                     "physical_disk_queue_depth", "physical_disk_outstanding_io",
                     "physical_disk_activity_percent")
                     if key in sample}}
                for sample in self.samples]
