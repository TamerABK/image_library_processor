"""Shared PyInstaller configuration for the three Windows release profiles."""
from __future__ import annotations

import json
import os
from pathlib import Path

from build_support.onnx_packaging import (EXPECTED_PROVIDERS, collect_ort_binaries,
                                           keep_native_for_profile, resolve_build_profile,
                                           validate_native_components)


REQUIRED_MODELS = ("open_closed_eye.onnx", "scrfd_10g_bnkps.onnx",
                   "glintr100.onnx", "sixdrepnet.onnx")
HIDDEN_IMPORTS = ("PySide6.QtQuickControls2",)
EXCLUDES = ("torch", "tensorflow", "onnxruntime.backend", "onnxruntime.datasets",
            "onnxruntime.quantization", "onnxruntime.tools", "onnxruntime.transformers")
DIST_NAMES = {"cpu": "Room36-CPU", "cuda": "Room36-CUDA",
              "directml": "Room36-DirectML"}


def discover_model_datas(project_root: Path) -> list[tuple[str, str]]:
    root = project_root / "onnx_models"
    missing = [name for name in REQUIRED_MODELS if not (root / name).is_file()]
    if missing:
        raise RuntimeError("Required production ONNX models are missing: " + ", ".join(missing))
    return [(str(path), "onnx_models") for path in sorted(root.iterdir())
            if path.is_file() and (path.suffix.lower() in (".onnx", ".json", ".npz") or
                                   (path.name.endswith(".onnx.data") and
                                    (root / path.name[:-5]).is_file()))]


def collect_cuda_runtime_binaries(project_root: Path, profile: str) -> list[tuple[str, str]]:
    if profile != "cuda":
        return []
    cuda_root = Path(os.environ.get("GPU_RUNTIME_DLL_DIR", str(project_root / "gpu_runtime_dll")))
    if not cuda_root.is_absolute():
        cuda_root = project_root / cuda_root
    binaries = [(str(path), "gpu_runtime_dll") for path in sorted(cuda_root.glob("*.dll"))]
    bad = [Path(path).name for path, _ in binaries if not keep_native_for_profile(path, "cuda")]
    if bad:
        raise RuntimeError("CUDA runtime directory contains unsupported DLLs: " + ", ".join(bad))
    return binaries


def filter_analysis_binaries(entries, profile: str):
    """Remove GPU binaries that a third-party PyInstaller hook may reintroduce."""
    return [entry for entry in entries if keep_native_for_profile(entry[0], profile)]


def build_room36_spec(explicit_profile: str | None, namespace: dict):
    """Create a PyInstaller COLLECT from globals supplied by a thin .spec file."""
    project_root = Path(namespace["SPECPATH"]).resolve()
    profile = resolve_build_profile(explicit_profile)
    ort_binaries = collect_ort_binaries(profile)
    cuda_binaries = collect_cuda_runtime_binaries(project_root, profile)
    model_datas = discover_model_datas(project_root)
    for folder in ("qml", "assets"):
        if not (project_root / folder).is_dir():
            raise RuntimeError("Required application resource directory is missing: " + folder)

    manifest_dir = project_root / "build" / ("onnx-profile-" + profile)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "onnx_build_profile.json"
    manifest.write_text(json.dumps({
        "schema": 1, "profile": profile, "expected_provider": EXPECTED_PROVIDERS[profile],
        "native_components": sorted({Path(path).name for path, _ in ort_binaries}),
    }, indent=2) + "\n", encoding="utf-8")

    analysis = namespace["Analysis"](
        [str(project_root / "main.py")], pathex=[str(project_root)],
        binaries=ort_binaries + cuda_binaries,
        datas=model_datas + [(str(project_root / "qml"), "qml"),
                             (str(project_root / "assets"), "assets"),
                             (str(manifest), ".")],
        hiddenimports=list(HIDDEN_IMPORTS),
        hookspath=[str(project_root / "build_support" / "hooks")],
        excludes=list(EXCLUDES), noarchive=False,
    )
    analysis.binaries = filter_analysis_binaries(analysis.binaries, profile)
    # Check the final binary list as well as the wheel's initial native inventory.
    validate_native_components([entry[0] for entry in analysis.binaries], profile)
    pyz = namespace["PYZ"](analysis.pure)
    exe = namespace["EXE"](pyz, analysis.scripts, [], exclude_binaries=True,
                             name="Room36", console=True, debug=False, strip=False, upx=False)
    name = DIST_NAMES[profile] if explicit_profile is not None else "Room36"
    return namespace["COLLECT"](exe, analysis.binaries, analysis.datas,
                                strip=False, upx=False, name=name)
