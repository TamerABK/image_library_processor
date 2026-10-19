"""Acceptance must launch the frozen artifact, never import development ORT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from build_support.onnx_packaging import EXPECTED_PROVIDERS, validate_native_components
from build_support.room36_spec import REQUIRED_MODELS


def validate_package(distribution: Path, profile: str, *, models=("eye",)) -> dict:
    if profile not in EXPECTED_PROVIDERS:
        raise ValueError("Unknown ONNX build profile: " + profile)
    distribution = distribution.resolve()
    if not distribution.is_dir():
        raise RuntimeError("Distribution directory does not exist: " + str(distribution))
    executables = list(distribution.rglob("Room36.exe"))
    if len(executables) != 1:
        raise RuntimeError("Expected exactly one packaged Room36.exe.")
    manifests = list(distribution.rglob("onnx_build_profile.json"))
    if len(manifests) != 1:
        raise RuntimeError("Expected exactly one build profile manifest.")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest["profile"] != profile:
        raise RuntimeError("Packaged build profile does not match acceptance target.")
    if manifest.get("expected_provider") != EXPECTED_PROVIDERS[profile]:
        raise RuntimeError("Packaged expected provider does not match acceptance target.")
    for folder in ("qml", "assets", "onnx_models"):
        if not any(path.is_dir() for path in distribution.rglob(folder)):
            raise RuntimeError("Packaged application resource directory is missing: " + folder)
    model_dir = next(path for path in distribution.rglob("onnx_models") if path.is_dir())
    missing_models = [name for name in REQUIRED_MODELS if not (model_dir / name).is_file()]
    if missing_models:
        raise RuntimeError("Packaged production models are missing: " + ", ".join(missing_models))
    native = validate_native_components(
        [path for path in distribution.rglob("*") if path.is_file()],
        profile, manifest["native_components"])
    reports = []
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["ROOM36_ONNX_PROVIDER"] = profile
    for model in models:
        try:
            process = subprocess.run(
                [str(executables[0]), "--diagnose-onnx", "--expect-provider", profile,
                 "--onnx-model", model],
                cwd=distribution, env=environment, capture_output=True, text=True, timeout=180,
            )
        except OSError as exc:
            if getattr(exc, "winerror", None) == 4551:
                raise RuntimeError(
                    "Windows Application Control blocked Room36.exe before startup. "
                    "Validate a signed build or run under an authorized policy."
                ) from exc
            raise
        try:
            report = json.loads(process.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError("Packaged executable did not return a diagnostic report.") from exc
        if not report.get("frozen"):
            raise RuntimeError("Diagnostic used a development Python environment.")
        if process.returncode or report.get("test_inference") != "PASS":
            raise RuntimeError("Packaged " + model + " inference failed: " +
                               report.get("error", "unknown failure"))
        if report.get("build_profile") != profile:
            raise RuntimeError("Frozen runtime profile mismatch.")
        expected = EXPECTED_PROVIDERS[profile]
        if expected not in report["available_providers"] or report["test_model_provider"] != expected:
            raise RuntimeError("Packaged provider acceptance failed; CPU fallback is not accepted.")
        reports.append(report)
    return {"profile": profile, "native_components_checked": len(manifest["native_components"]),
            "directml_dll_present": "directml.dll" in native, "models": reports, "status": "PASS"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("distribution", type=Path)
    parser.add_argument("--profile", required=True, choices=("cpu", "cuda", "directml"))
    parser.add_argument("--models", nargs="+", default=["eye"])
    args = parser.parse_args()
    try:
        print(json.dumps(validate_package(args.distribution, args.profile, models=args.models),
                         sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
