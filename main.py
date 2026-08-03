from __future__ import annotations

import ctypes
import os
import sys
from multiprocessing import freeze_support
from pathlib import Path


_WINDOWS_DLL_DIRECTORIES: list[object] = []


def _application_roots() -> list[Path]:
    roots: list[Path] = [Path(__file__).resolve().parent]

    if getattr(sys, "frozen", False):
        frozen_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        exe_root = Path(sys.executable).resolve().parent

        for candidate in (frozen_root, exe_root):
            if candidate not in roots:
                roots.append(candidate)

    return roots


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []

    versioned_site_packages = (
        Path(sys.prefix)
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    windows_site_packages = Path(sys.prefix) / "Lib" / "site-packages"

    for candidate in (versioned_site_packages, windows_site_packages):
        if candidate.exists() and candidate not in roots:
            roots.append(candidate)

    if getattr(sys, "frozen", False):
        frozen_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        if frozen_root not in roots:
            roots.append(frozen_root)
        exe_root = Path(sys.executable).resolve().parent
        if exe_root not in roots:
            roots.append(exe_root)

    return roots


def _preload_linux_nvidia_shared_libraries() -> None:
    if sys.platform != "linux":
        return

    for site_packages in _site_package_roots():
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            continue

        library_names = (
            "libcublasLt.so.13",
            "libcublas.so.13",
            "libnvrtc.so.13",
            "libcudart.so.13",
            "libcudnn.so.9",
            "libcurand.so.10",
            "libcufft.so.12",
        )

        for name in library_names:
            for candidate in nvidia_root.glob(f"**/{name}"):
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                break


def _add_windows_dll_directory(path: Path) -> None:
    if sys.platform != "win32":
        return
    if not path.exists() or not path.is_dir():
        return
    if not hasattr(os, "add_dll_directory"):
        return

    handle = os.add_dll_directory(str(path))
    _WINDOWS_DLL_DIRECTORIES.append(handle)


def _preload_windows_nvidia_shared_libraries() -> None:
    if sys.platform != "win32":
        return

    candidate_directories: list[Path] = []

    for root in _application_roots():
        bundled_gpu_runtime = root / "gpu_runtime_dll"
        if bundled_gpu_runtime.exists() and bundled_gpu_runtime not in candidate_directories:
            candidate_directories.append(bundled_gpu_runtime)

    for root in _site_package_roots():
        for direct_child in (
            root,
            root / "onnxruntime" / "capi",
            root / "nvidia",
        ):
            if direct_child.exists() and direct_child not in candidate_directories:
                candidate_directories.append(direct_child)

        nvidia_root = root / "nvidia"
        if nvidia_root.exists():
            for pattern in ("**/bin", "**/lib"):
                for candidate in nvidia_root.glob(pattern):
                    if candidate not in candidate_directories:
                        candidate_directories.append(candidate)

    for candidate in candidate_directories:
        _add_windows_dll_directory(candidate)

    try:
        import onnxruntime as ort
    except Exception:
        return

    preload_dlls = getattr(ort, "preload_dlls", None)
    if preload_dlls is None:
        return

    preload_dlls(cuda=True, cudnn=True, msvc=True)


def _preload_gpu_runtime_dependencies() -> None:
    _preload_linux_nvidia_shared_libraries()
    _preload_windows_nvidia_shared_libraries()


_preload_gpu_runtime_dependencies()

from ui.app import run_app


if __name__ == "__main__":

    freeze_support()
    run_app()
