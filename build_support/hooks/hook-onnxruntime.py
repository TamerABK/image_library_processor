# Override the generic hook: collect native files from the one active ORT wheel.
from build_support.onnx_packaging import collect_ort_binaries, resolve_build_profile

binaries = collect_ort_binaries(resolve_build_profile())
hiddenimports = [
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_inference_collection",
    "onnxruntime.capi.onnxruntime_pybind11_state",
]
excludedimports = ["onnxruntime.backend", "onnxruntime.datasets", "onnxruntime.quantization",
                   "onnxruntime.tools", "onnxruntime.transformers"]
