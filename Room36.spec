# -*- mode: python ; coding: utf-8 -*-
import json
import os
from pathlib import Path
import sys

project_root = Path(SPECPATH).resolve()
sys.path.insert(0, str(project_root))
from build_support.onnx_packaging import collect_ort_binaries, resolve_build_profile

profile = resolve_build_profile()
ort_binaries = collect_ort_binaries(profile)
cuda_binaries = []
if profile == "cuda":
    cuda_root = Path(os.environ.get("GPU_RUNTIME_DLL_DIR", str(project_root / "gpu_runtime_dll")))
    if not cuda_root.is_absolute():
        cuda_root = project_root / cuda_root
    cuda_binaries = [(str(path), "gpu_runtime_dll") for path in sorted(cuda_root.glob("*.dll"))]

manifest_dir = project_root / "build" / ("onnx-profile-" + profile)
manifest_dir.mkdir(parents=True, exist_ok=True)
manifest = manifest_dir / "onnx_build_profile.json"
manifest.write_text(json.dumps({
    "schema": 1, "profile": profile,
    "native_components": sorted(Path(path).name for path, destination in ort_binaries),
}), encoding="utf-8")
model_datas = [(str(path), "onnx_models") for path in sorted((project_root / "onnx_models").iterdir())
               if path.is_file() and (path.suffix.lower() in (".onnx", ".json", ".npz") or
                                      (path.name.endswith(".onnx.data") and
                                       (path.parent / path.name[:-5]).is_file()))]
if not (project_root / "onnx_models" / "open_closed_eye.onnx").is_file():
    raise RuntimeError("The bundled eye model is required for packaged inference acceptance.")

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=ort_binaries + cuda_binaries,
    datas=model_datas + [(str(project_root / "qml"), "qml"),
                         (str(project_root / "assets"), "assets"),
                         (str(manifest), ".")],
    hiddenimports=["PySide6.QtQuickControls2"],
    hookspath=[str(project_root / "build_support" / "hooks")],
    excludes=["torch", "tensorflow", "onnxruntime.backend", "onnxruntime.datasets",
              "onnxruntime.quantization", "onnxruntime.tools", "onnxruntime.transformers"],
    noarchive=False,
)
# Guard against dependencies/hooks accidentally reintroducing another GPU stack.
if profile != "cuda":
    a.binaries = [entry for entry in a.binaries
                  if not Path(entry[0]).name.lower().startswith(
                      ("cudart", "cudnn", "cublas", "cufft", "nvrtc", "onnxruntime_providers_cuda",
                       "onnxruntime_providers_tensorrt"))]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Room36",
          console=True, debug=False, strip=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Room36")
