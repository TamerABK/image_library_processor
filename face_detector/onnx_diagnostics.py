"""Small diagnostic executed by both source main.py and the frozen Room36.exe."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np

from app_paths import model_path, resource_path
from .onnx_runtime import create_inference_session, startup_diagnostics

MODELS = {
    "eye": ("open_closed_eye.onnx", 32),
    "detector": ("scrfd_10g_bnkps.onnx", 640),
    "embedding": ("glintr100.onnx", 112),
    "pose": ("sixdrepnet.onnx", 224),
    "vibe": ("vibe_semantic_v1.onnx", 224),
}
PROVIDERS = {
    "cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider", "migraphx": "MIGraphXExecutionProvider",
}


def synthetic_input(session, side: int, batch_size: int = 1) -> np.ndarray:
    shape = list(session.get_inputs()[0].shape)
    if len(shape) != 4:
        raise ValueError("Unsupported diagnostic model input rank.")
    shape = [
        (batch_size if i == 0 else 3 if i == 1 else side)
        if not isinstance(n, int) or n <= 0 else n
        for i, n in enumerate(shape)
    ]
    if shape[0] != batch_size:
        raise ValueError("Model has a fixed batch dimension incompatible with the requested batch.")
    # A deterministic normalized tensor; never requires private photographs.
    return np.linspace(-0.45, 0.45, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)


def validate_outputs(outputs) -> None:
    if not outputs or any(np.asarray(value).size == 0 or not np.isfinite(value).all()
                          for value in outputs):
        raise ValueError("Model returned empty or non-finite output.")


def run_diagnostic(*, model: str = "eye", expect_provider: str | None = None,
                   iterations: int = 1, warmup: int = 1, batch_size: int = 1) -> dict:
    info = startup_diagnostics()
    report = {
        "ort_version": info["runtime_version"],
        "available_providers": info["available_providers"],
        "selected_provider": info["selected_providers"][0],
        "frozen": bool(getattr(sys, "frozen", False)),
        "model": model,
        "batch_size": batch_size,
        "warmup": warmup,
        "test_inference": "FAIL",
    }
    manifest = resource_path("onnx_build_profile.json")
    if manifest.is_file():
        report["build_profile"] = json.loads(manifest.read_text(encoding="utf-8"))["profile"]
    else:
        report["build_profile"] = "source"
    expected = PROVIDERS.get(expect_provider, expect_provider)
    if expect_provider is None and report["build_profile"] == "directml":
        expected = PROVIDERS["directml"]
    report["expected_provider"] = expected
    if expected is not None and expected not in info["available_providers"]:
        report["error"] = "Expected execution provider is unavailable."
        return report
    try:
        filename, side = MODELS[model]
        session = create_inference_session(
            model_path(filename), providers=[expected] if expected else None)
        report["test_model_provider"] = session.get_providers()[0]
        report["active_providers"] = session.get_providers()
        report["session_creation_ms"] = session.diagnostics["session_creation_seconds"] * 1000
        report["fallbacks"] = session.diagnostics["fallbacks"]
        if expected and session.diagnostics["active_provider"] != expected:
            report["error"] = "Expected accelerator did not initialize; CPU fallback is not accepted."
            return report
        tensor = synthetic_input(session, side, batch_size)
        feed = {session.get_inputs()[0].name: tensor}
        for _ in range(warmup):
            validate_outputs(session.run(None, feed))
        times = []
        for _ in range(iterations):
            started = time.perf_counter()
            outputs = session.run(None, feed)
            times.append(time.perf_counter() - started)
            validate_outputs(outputs)
        report["test_model_provider"] = session.get_providers()[0]
        report["active_providers"] = session.get_providers()
        report["fallbacks"] = session.diagnostics.get("last_run_fallbacks",
                                                     session.diagnostics["fallbacks"])
        if expected and report["test_model_provider"] != expected:
            report["error"] = "Expected accelerator did not execute; CPU fallback is not accepted."
            return report
        report["output_shapes"] = [list(np.asarray(output).shape) for output in outputs]
        report["mean_latency_ms"] = statistics.mean(times) * 1000
        report["median_latency_ms"] = statistics.median(times) * 1000
        report["throughput_per_second"] = batch_size / statistics.mean(times)
        report["iterations"] = iterations
        report["test_inference"] = "PASS"
    except Exception as exc:
        # Stable support diagnostic, never leak paths or native exception text.
        report["error"] = "Model inference failed: " + type(exc).__name__
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Room36 ONNX runtime diagnostic (no GUI).")
    parser.add_argument("--diagnose-onnx", action="store_true")
    parser.add_argument("--onnx-model", choices=MODELS, default="eye")
    parser.add_argument("--expect-provider", choices=PROVIDERS)
    parser.add_argument("--benchmark-onnx", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 100 or not 0 <= args.warmup <= 20 or not 1 <= args.batch_size <= 64:
        parser.error("Iterations, warmup or batch size outside diagnostic bounds.")
    report = run_diagnostic(model=args.onnx_model, expect_provider=args.expect_provider,
                            iterations=args.iterations if args.benchmark_onnx else 1,
                            warmup=args.warmup if args.benchmark_onnx else 0,
                            batch_size=args.batch_size)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["test_inference"] == "PASS" else 1
