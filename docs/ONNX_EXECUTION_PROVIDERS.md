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

`Room36.spec` uses `build_support/onnx_packaging.py` and the custom ONNX hook to
collect native files from the single installed runtime distribution. Set
`ROOM36_ONNX_BUILD_PROFILE=cpu|cuda|directml` to require a matching build. Model
files must be present under `onnx_models` for packaging. After building, run
`python -m build_support.validate_onnx_package <distribution> --profile <profile>`;
the validator launches the frozen executable and rejects source-Python or CPU
fallback masquerading as accelerator success.

Provider-policy tests use mocked runtimes and run on any host. Native accelerator
acceptance requires the target GPU, driver, and packaged runtime.
