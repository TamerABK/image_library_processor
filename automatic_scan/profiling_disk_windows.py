"""Read-only, system-wide PhysicalDisk PDH samples for a dataset volume.

These counters describe the entire device, not Room36 I/O. A drive-letter
match is required; ambiguous/missing instances are reported unavailable.
"""
from __future__ import annotations

import ctypes
import math
import os
import re
from ctypes import wintypes
from pathlib import Path

from .profiling_gpu_windows import _CounterItem, _FMT_DOUBLE, _MORE_DATA, _VALID


_COUNTERS = {
    "read_mb_s": "Disk Read Bytes/sec",
    "write_mb_s": "Disk Write Bytes/sec",
    "read_iops": "Disk Reads/sec",
    "write_iops": "Disk Writes/sec",
    "read_latency_s": "Avg. Disk sec/Read",
    "write_latency_s": "Avg. Disk sec/Write",
    "queue_depth": "Avg. Disk Queue Length",
    "outstanding_io": "Current Disk Queue Length",
    "activity_percent": "% Disk Time",
}


def _matching_instance(names: list[str], drive: str) -> str | None:
    suffix = drive.upper().rstrip("\\/")
    matched = [name for name in names if re.search(r"(?:^|\s)" + re.escape(suffix) + r"(?:\s|$)", name.upper())]
    return matched[0] if len(matched) == 1 else None


class WindowsPhysicalDiskSampler:
    def __init__(self, dataset: str | Path | None):
        self.available = False
        self.error: str | None = None
        self.instance: str | None = None
        self.drive = Path(dataset).drive if dataset is not None else ""
        self._query = ctypes.c_void_p()
        self._counters: dict[str, ctypes.c_void_p] = {}
        if os.name != "nt" or not self.drive:
            self.error = "Windows dataset drive unavailable"
            return
        try:
            pdh = ctypes.WinDLL("pdh")
            pdh.PdhOpenQueryW.argtypes = (wintypes.LPCWSTR, ctypes.c_size_t,
                                         ctypes.POINTER(ctypes.c_void_p))
            pdh.PdhOpenQueryW.restype = wintypes.DWORD
            pdh.PdhAddEnglishCounterW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR,
                                                  ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p))
            pdh.PdhAddEnglishCounterW.restype = wintypes.DWORD
            pdh.PdhCollectQueryData.argtypes = (ctypes.c_void_p,)
            pdh.PdhCollectQueryData.restype = wintypes.DWORD
            pdh.PdhGetFormattedCounterArrayW.argtypes = (ctypes.c_void_p, wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p)
            pdh.PdhGetFormattedCounterArrayW.restype = wintypes.DWORD
            pdh.PdhCloseQuery.argtypes = (ctypes.c_void_p,)
            pdh.PdhCloseQuery.restype = wintypes.DWORD
            self._pdh = pdh
            status = pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
            if status:
                raise OSError(f"PDH open failed: 0x{status:08x}")
            for key, counter_name in _COUNTERS.items():
                counter = ctypes.c_void_p()
                status = pdh.PdhAddEnglishCounterW(self._query,
                    rf"\PhysicalDisk(*)\{counter_name}", 0, ctypes.byref(counter))
                if status:
                    raise OSError(f"PDH {key} unavailable: 0x{status:08x}")
                self._counters[key] = counter
            self.available = True
        except (AttributeError, OSError) as exc:
            self.error = str(exc)
            self.close()

    def _values(self, counter: ctypes.c_void_p) -> list[tuple[str, float]]:
        size, count = wintypes.DWORD(), wintypes.DWORD()
        status = self._pdh.PdhGetFormattedCounterArrayW(counter, _FMT_DOUBLE,
            ctypes.byref(size), ctypes.byref(count), None)
        if status not in (0, _MORE_DATA) or not size.value:
            return []
        storage = ctypes.create_string_buffer(size.value)
        status = self._pdh.PdhGetFormattedCounterArrayW(counter, _FMT_DOUBLE,
            ctypes.byref(size), ctypes.byref(count), storage)
        if status:
            return []
        items = ctypes.cast(storage, ctypes.POINTER(_CounterItem))
        return [(items[i].name or "", float(items[i].value.value.double_value))
                for i in range(count.value) if items[i].value.status in _VALID
                and math.isfinite(items[i].value.value.double_value)]

    def sample(self) -> dict:
        if not self.available:
            return {}
        status = self._pdh.PdhCollectQueryData(self._query)
        if status:
            self.error = f"PDH collection failed: 0x{status:08x}"
            return {}
        values = {key: self._values(counter) for key, counter in self._counters.items()}
        if self.instance is None:
            self.instance = _matching_instance([name for name, _ in values["read_mb_s"]], self.drive)
            if self.instance is None:
                self.error = "No unique PhysicalDisk instance matches dataset drive"
                return {}
            self.error = None
        result = {}
        for key, items in values.items():
            match = next((value for name, value in items if name == self.instance), None)
            if match is not None:
                result["physical_disk_" + key] = match / 1_000_000 if key.endswith("mb_s") else match
        return result

    def close(self) -> None:
        if self._query.value and hasattr(self, "_pdh"):
            self._pdh.PdhCloseQuery(self._query)
            self._query = ctypes.c_void_p()
        self.available = False
