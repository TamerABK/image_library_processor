from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path

PROFILES = {"cpu": "onnxruntime", "cuda": "onnxruntime-gpu",
            "directml": "onnxruntime-directml"}
EXPECTED_PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
                      "directml": "DmlExecutionProvider"}
CUDA_PREFIXES = ("cudart", "cudnn", "cublas", "cufft", "nvrtc", "nvjitlink",
                 "cusolver", "cusparse", "curand", "cublaslt", "onnxruntime_providers_cuda",
                 "onnxruntime_providers_tensorrt")


def resolve_build_profile(requested: str | None = None) -> str:
    override = os.environ.get("ROOM36_ONNX_BUILD_PROFILE", "auto").lower()
    if override not in (*PROFILES, "auto"):
        raise RuntimeError("Unknown ROOM36_ONNX_BUILD_PROFILE: " + override)
    if requested is not None and requested not in PROFILES:
        raise RuntimeError("Unknown ONNX build profile: " + requested)
    if requested is not None and override != "auto" and override != requested:
        raise RuntimeError("ROOM36_ONNX_BUILD_PROFILE=" + override +
                           " conflicts with explicit " + requested + " spec.")
    installed = []
    for profile, distribution in PROFILES.items():
        try:
            importlib.metadata.version(distribution)
            installed.append(profile)
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) != 1:
        raise RuntimeError("Install exactly one ORT distribution in the build environment: "
                           "onnxruntime, onnxruntime-gpu, or onnxruntime-directml. "
                           "Detected: " + (", ".join(PROFILES[p] for p in installed) or "none"))
    requested = requested or override
    if requested == "auto":
        return installed[0]
    if requested not in PROFILES or requested != installed[0]:
        raise RuntimeError("The " + requested + " spec requires " + PROFILES[requested] +
                           "; installed " + PROFILES[installed[0]] + ".")
    return requested


def validate_native_components(paths, profile: str, expected_names=()) -> list[str]:
    if profile not in PROFILES:
        raise ValueError("Unknown ONNX build profile: " + profile)
    names = {Path(path).name.lower() for path in paths}
    required = {str(name).lower() for name in expected_names}
    if not any(name.startswith("onnxruntime_pybind11_state") and
               name.endswith((".pyd", ".so")) for name in names):
        required.add("onnxruntime_pybind11_state.pyd")
    if profile == "directml":
        required.update(("directml.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll"))
    elif profile == "cuda":
        required.add("onnxruntime_providers_cuda.dll")
    forbidden = sorted(name for name in names if
                       (profile != "cuda" and name.startswith(CUDA_PREFIXES)) or
                       (profile != "directml" and name == "directml.dll") or
                       "onnxruntime_providers_tensorrt" in name)
    if forbidden:
        raise RuntimeError("Unexpected native components for " + profile + ": " +
                           ", ".join(forbidden))
    missing = sorted(required - names)
    if missing:
        raise RuntimeError("Missing packaged ONNX native components: " + ", ".join(missing))
    return sorted(names)


def keep_native_for_profile(name: str, profile: str) -> bool:
    name = Path(name).name.lower()
    return not ((profile != "cuda" and name.startswith(CUDA_PREFIXES)) or
                (profile != "directml" and name == "directml.dll") or
                "onnxruntime_providers_tensorrt" in name)


def collect_ort_binaries(profile: str) -> list[tuple[str, str]]:
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or spec.origin is None:
        raise RuntimeError("ONNX Runtime package is not installed.")
    root = Path(spec.origin).parent
    paths = sorted(path for path in root.rglob("*") if path.is_file() and
                   (path.suffix.lower() in (".dll", ".pyd", ".dylib", ".so") or ".so." in path.name))
    # TensorRT was never a supported Room36 build profile. CUDA is separate.
    paths = [path for path in paths if keep_native_for_profile(path.name, profile)]
    validate_native_components(paths, profile)
    return [(str(path), str(Path("onnxruntime") / path.relative_to(root).parent)) for path in paths]
