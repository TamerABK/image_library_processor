"""Read-only Windows PDH GPU process counters for optional profiling.

GPU Engine percentages are per process/engine, not whole-device utilization.
No external process or vendor-specific GPU package is required.
"""
from __future__ import annotations

import ctypes
import math
import os
import re
from ctypes import wintypes


_PID = re.compile(r"^pid_(\d+)_", re.IGNORECASE)
_ENGINE = re.compile(r"_engtype_(.+)$", re.IGNORECASE)
_VALID = {0, 1}  # PDH_CSTATUS_VALID_DATA / PDH_CSTATUS_NEW_DATA
_MORE_DATA = 0x800007D2
_FMT_DOUBLE = 0x00000200


class _ValueUnion(ctypes.Union):
    _fields_ = [("long_value", ctypes.c_long), ("double_value", ctypes.c_double),
                ("large_value", ctypes.c_longlong), ("wide_value", ctypes.c_wchar_p)]


class _FormattedValue(ctypes.Structure):
    _fields_ = [("status", wintypes.DWORD), ("value", _ValueUnion)]


class _CounterItem(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar_p), ("value", _FormattedValue)]


class WindowsGpuProcessSampler:
    def __init__(self):
        self.available = False
        self.error: str | None = None
        self._query = ctypes.c_void_p()
        self._counters = {}
        if os.name != "nt":
            self.error = "Windows PDH unavailable on this OS"
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
            paths = {"engine": r"\GPU Engine(*)\Utilization Percentage",
                     "dedicated": r"\GPU Process Memory(*)\Dedicated Usage",
                     "shared": r"\GPU Process Memory(*)\Shared Usage"}
            for name, path in paths.items():
                counter = ctypes.c_void_p()
                status = pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(counter))
                if status:
                    raise OSError(f"PDH {name} counter unavailable: 0x{status:08x}")
                self._counters[name] = counter
            self.available = True
        except (AttributeError, OSError) as exc:
            self.error = str(exc)
            self.close()

    def _values(self, counter) -> list[tuple[str, float]]:
        size = wintypes.DWORD()
        count = wintypes.DWORD()
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter, _FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), None)
        if status not in (0, _MORE_DATA) or size.value == 0:
            return []
        storage = ctypes.create_string_buffer(size.value)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter, _FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), storage)
        if status:
            return []
        items = ctypes.cast(storage, ctypes.POINTER(_CounterItem))
        result = []
        for index in range(count.value):
            item = items[index]
            value = float(item.value.value.double_value)
            if item.value.status in _VALID and math.isfinite(value) and value >= 0:
                result.append((item.name or "", value))
        return result

    def sample(self) -> dict:
        if not self.available:
            return {}
        status = self._pdh.PdhCollectQueryData(self._query)
        if status:
            self.error = f"PDH collection failed: 0x{status:08x}"
            return {}
        pid = os.getpid()
        result = {}
        engines = {}
        for name, value in self._values(self._counters["engine"]):
            match = _PID.match(name)
            if match is None or int(match.group(1)) != pid:
                continue
            engine = _ENGINE.search(name)
            engine_type = engine.group(1) if engine else "unknown"
            engines[engine_type] = engines.get(engine_type, 0.0) + value
        if engines:
            result["gpu_process_engine_percent_sum"] = engines
            result["gpu_process_peak_engine_percent"] = max(engines.values())
            result["gpu_process_compute_engine_percent"] = sum(
                value for name, value in engines.items() if "compute" in name.casefold())
        for kind in ("dedicated", "shared"):
            values = []
            for name, value in self._values(self._counters[kind]):
                match = _PID.match(name)
                if match is not None and int(match.group(1)) == pid:
                    values.append(value)
            if values:
                result[f"gpu_process_{kind}_bytes"] = int(sum(values))
        return result

    def close(self) -> None:
        if self._query.value and hasattr(self, "_pdh"):
            self._pdh.PdhCloseQuery(self._query)
            self._query = ctypes.c_void_p()
        self.available = False
