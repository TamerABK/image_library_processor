from __future__ import annotations

import ctypes
import logging
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
    if getattr(sys, "frozen", False):
        return _application_roots()

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
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    logging.getLogger(__name__).warning("Optional NVIDIA preload failed.")
                break


def _add_windows_dll_directory(path: Path) -> None:
    if sys.platform != "win32":
        return
    if not path.exists() or not path.is_dir():
        return
    if not hasattr(os, "add_dll_directory"):
        return

    try:
        handle = os.add_dll_directory(str(path))
        _WINDOWS_DLL_DIRECTORIES.append(handle)
    except OSError:
        logging.getLogger(__name__).warning("Optional DLL search directory unavailable.")


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

    # The shared ONNX session factory preloads CUDA only if it is selected.


def _preload_gpu_runtime_dependencies() -> None:
    if os.environ.get("ROOM36_ONNX_PROVIDER", "auto").strip().lower() not in (
        "cpu", "directml", "migraphx"
    ):
        _preload_linux_nvidia_shared_libraries()
    _preload_windows_nvidia_shared_libraries()


def _ui_mode_from_environment() -> str:
    # Qt is the Phase 3 frontend; Tk remains an explicit feature-parity/debug fallback.
    requested_mode = os.environ.get("ROOM36_UI", "qt").strip().lower()
    return "tk" if requested_mode == "tk" else "qt"


def run_app() -> int:
    _preload_gpu_runtime_dependencies()
    if "--diagnose-onnx" in sys.argv:
        from face_detector.onnx_diagnostics import main as diagnose

        return diagnose(sys.argv[1:])
    from face_detector.onnx_runtime import log_startup_diagnostics

    log_startup_diagnostics()
    ui_mode = _ui_mode_from_environment()
    if ui_mode == "qt":
        from ui.qt.app import run_app as run_qt_app

        return run_qt_app()

    from ui.app import run_app as run_tk_app

    run_tk_app()
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(run_app())
