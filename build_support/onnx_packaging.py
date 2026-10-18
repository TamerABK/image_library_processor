from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path

PROFILES = {"cpu": "onnxruntime", "cuda": "onnxruntime-gpu",
            "directml": "onnxruntime-directml"}


def resolve_build_profile(requested: str | None = None) -> str:
    installed = []
    for profile, distribution in PROFILES.items():
        try:
            importlib.metadata.version(distribution)
            installed.append(profile)
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) != 1:
        raise RuntimeError("Install exactly one ORT distribution in the build environment: "
                           "onnxruntime, onnxruntime-gpu, or onnxruntime-directml.")
    requested = requested or os.environ.get("ROOM36_ONNX_BUILD_PROFILE", "auto")
    if requested == "auto":
        return installed[0]
    if requested not in PROFILES or requested != installed[0]:
        raise RuntimeError("Requested build profile does not match the installed ORT distribution.")
    return requested


def validate_native_components(paths, profile: str, expected_names=()) -> list[str]:
    names = {Path(path).name.lower() for path in paths}
    required = {str(name).lower() for name in expected_names}
    if not any(name.startswith("onnxruntime_pybind11_state") and
               name.endswith((".pyd", ".so")) for name in names):
        required.add("onnxruntime_pybind11_state.pyd")
    if profile == "directml":
        required.update(("directml.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll"))
    missing = sorted(required - names)
    if missing:
        raise RuntimeError("Missing packaged ONNX native components: " + ", ".join(missing))
    return sorted(names)


def collect_ort_binaries(profile: str) -> list[tuple[str, str]]:
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or spec.origin is None:
        raise RuntimeError("ONNX Runtime package is not installed.")
    root = Path(spec.origin).parent
    paths = sorted(path for path in root.rglob("*") if path.is_file() and
                   (path.suffix.lower() in (".dll", ".pyd", ".dylib", ".so") or ".so." in path.name))
    # TensorRT was never a supported Room36 build profile. CUDA is separate.
    paths = [path for path in paths if "tensorrt" not in path.name.lower() and
             (profile == "cuda" or "cuda" not in path.name.lower())]
    validate_native_components(paths, profile)
    return [(str(path), str(Path("onnxruntime") / path.relative_to(root).parent)) for path in paths]
