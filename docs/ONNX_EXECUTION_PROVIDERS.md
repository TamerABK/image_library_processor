# Room 36 ONNX execution providers

`face_detector.onnx_runtime` selects the provider for all production ONNX
sessions. On Windows, automatic selection tries separate sessions in this order:

```text
CUDA + CPU → MIGraphX + CPU → DirectML + CPU → OpenVINO + CPU → CPU
```

Only providers reported by the installed ONNX Runtime distribution are tried.
Linux and macOS use their platform-specific orders. An explicit `providers=`
argument is filtered against availability, and `ROOM36_ONNX_PROVIDER` can prefer
`cuda`, `migraphx`, `directml`, or `cpu`. CPU is the final fallback. A runtime that
silently activates CPU during an accelerator attempt is retried with the next
accelerator. Session diagnostics report the requested and actual providers.

Each attempt has its own session options and model source. Sequential execution,
disabled memory patterns, serialized `Run`, dynamic batch overrides, and the
temporary nonzero-Reshape correction apply only to a DirectML attempt. CUDA
sessions use the original model without DirectML restrictions.

Install exactly one ONNX Runtime distribution in a clean environment:

| Target | Requirements file | Runtime package |
| --- | --- | --- |
| CPU | `requirements-cpu.txt` | `onnxruntime` |
| NVIDIA | `requirements-cuda.txt` | `onnxruntime-gpu` |
| Windows AMD / DX12 | `requirements-windows-directml.txt` | `onnxruntime-directml` |

`requirements-directml.txt` is the existing DirectML profile. Do not install
`onnxruntime-gpu` and `onnxruntime-directml` together. The selected interpreter
must be the one that starts Room 36.

## Building Room 36 releases

| Build | Spec | Requirements | Expected EP |
| --- | --- | --- | --- |
| CPU | `Room36-cpu.spec` | `requirements-cpu.txt` | `CPUExecutionProvider` |
| NVIDIA | `Room36-cuda.spec` | `requirements-cuda.txt` | `CUDAExecutionProvider` |
| AMD/Windows | `Room36-directml.spec` | `requirements-windows-directml.txt` | `DmlExecutionProvider` |

Build each release from a separate clean Python 3.12 environment containing
exactly one ORT distribution. The release spec fixes its profile and rejects a
conflicting `ROOM36_ONNX_BUILD_PROFILE` override. `Room36.spec` remains a
developer/auto build; use the three named specs for releases. Run these commands
from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv-cpu
.\.venv-cpu\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
pip install pyinstaller
pyinstaller --clean Room36-cpu.spec
python -m build_support.validate_onnx_package dist\Room36-CPU --profile cpu
```

```powershell
py -3.12 -m venv .venv-cuda
.\.venv-cuda\Scripts\Activate.ps1
pip install -r requirements-cuda.txt
pip install pyinstaller
pyinstaller --clean Room36-cuda.spec
python -m build_support.validate_onnx_package dist\Room36-CUDA --profile cuda
```

```powershell
py -3.12 -m venv .venv-directml
.\.venv-directml\Scripts\Activate.ps1
pip install -r requirements-windows-directml.txt
pip install pyinstaller
pyinstaller --clean Room36-directml.spec
python -m build_support.validate_onnx_package dist\Room36-DirectML --profile directml
```

Before building, place the production ONNX files under `onnx_models/`, including
`open_closed_eye.onnx`, `scrfd_10g_bnkps.onnx`, `glintr100.onnx`, and
`sixdrepnet.onnx`. The spec discovers the other model assets automatically and
bundles `qml/` and `assets/` once each. CUDA builds can additionally bundle DLLs
from `gpu_runtime_dll/` (or `GPU_RUNTIME_DLL_DIR`); CPU and DirectML builds never
use that folder. Build acceptance checks the manifest and native files, launches
the frozen `Room36.exe`, runs real bundled-model inference, and rejects CPU
fallback for accelerator profiles. CUDA and DirectML acceptance require their
respective GPU hardware, drivers, and runtime DLLs.

Provider-policy tests use mocked runtimes and run on any host. Native accelerator
acceptance requires the target GPU, driver, and packaged runtime.
