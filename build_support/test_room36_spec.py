from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from build_support.onnx_packaging import (PROFILES, keep_native_for_profile,
                                           resolve_build_profile, validate_native_components)
from build_support.room36_spec import (DIST_NAMES, REQUIRED_MODELS, build_room36_spec,
                                        collect_cuda_runtime_binaries, discover_model_datas,
                                        filter_analysis_binaries)
from build_support.validate_onnx_package import validate_package


ROOT = Path(__file__).resolve().parents[1]


def installed_wheels(*names):
    def version(name):
        if name in names:
            return "test"
        raise importlib.metadata.PackageNotFoundError(name)
    return version


class SpecContractTests(unittest.TestCase):
    def test_developer_spec_keeps_auto_profile(self):
        source = (ROOT / "Room36.spec").read_text(encoding="utf-8")
        with patch("build_support.room36_spec.build_room36_spec", return_value="built") as build:
            namespace = {"SPECPATH": str(ROOT)}
            exec(compile(source, "Room36.spec", "exec"), namespace)
        self.assertEqual(namespace["coll"], "built")
        self.assertIsNone(build.call_args.args[0])

    def test_each_release_spec_selects_its_own_profile(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                source = (ROOT / ("Room36-" + profile + ".spec")).read_text(encoding="utf-8")
                with patch("build_support.room36_spec.build_room36_spec", return_value="built") as build:
                    namespace = {"SPECPATH": str(ROOT)}
                    exec(compile(source, profile + ".spec", "exec"), namespace)
                self.assertEqual(namespace["coll"], "built")
                self.assertEqual(build.call_args.args[0], profile)
                self.assertEqual(DIST_NAMES[profile], {"cpu": "Room36-CPU", "cuda": "Room36-CUDA",
                                                       "directml": "Room36-DirectML"}[profile])

    def test_wrong_or_multiple_wheels_are_rejected(self):
        cases = (("directml", ("onnxruntime-gpu",)),
                 ("cuda", ("onnxruntime-directml",)),
                 ("cpu", ("onnxruntime", "onnxruntime-gpu")))
        for profile, distributions in cases:
            with self.subTest(profile=profile, distributions=distributions):
                with patch.object(importlib.metadata, "version", side_effect=installed_wheels(*distributions)):
                    with patch.dict(os.environ, {"ROOM36_ONNX_BUILD_PROFILE": "auto"}):
                        with self.assertRaises(RuntimeError):
                            resolve_build_profile(profile)

    def test_explicit_spec_rejects_conflicting_override(self):
        with patch.object(importlib.metadata, "version", side_effect=installed_wheels("onnxruntime-directml")):
            with patch.dict(os.environ, {"ROOM36_ONNX_BUILD_PROFILE": "cuda"}):
                with self.assertRaisesRegex(RuntimeError, "conflicts with explicit"):
                    resolve_build_profile("directml")

    def test_native_component_isolation(self):
        self.assertFalse(keep_native_for_profile("DirectML.dll", "cuda"))
        self.assertFalse(keep_native_for_profile("cudnn64_9.dll", "directml"))
        self.assertFalse(keep_native_for_profile("onnxruntime_providers_cuda.dll", "cpu"))
        self.assertTrue(keep_native_for_profile("onnxruntime_providers_cuda.dll", "cuda"))
        entries = [("DirectML.dll", ".", "BINARY"), ("onnxruntime_providers_cuda.dll", ".", "BINARY")]
        self.assertEqual([entry[0] for entry in filter_analysis_binaries(entries, "cuda")],
                         ["onnxruntime_providers_cuda.dll"])
        with self.assertRaisesRegex(RuntimeError, "Unexpected native components"):
            validate_native_components(["onnxruntime_pybind11_state.pyd", "DirectML.dll"], "cpu")
        with self.assertRaisesRegex(RuntimeError, "onnxruntime_providers_cuda.dll"):
            validate_native_components(["onnxruntime_pybind11_state.pyd"], "cuda")

    def test_model_discovery_keeps_external_data_with_parent_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "onnx_models"
            model_dir.mkdir()
            for name in REQUIRED_MODELS:
                (model_dir / name).touch()
            (model_dir / "open_closed_eye.onnx.data").touch()
            (model_dir / "orphan.onnx.data").touch()
            names = {Path(path).name for path, _ in discover_model_datas(Path(temporary))}
            self.assertIn("open_closed_eye.onnx.data", names)
            self.assertNotIn("orphan.onnx.data", names)

    def test_cuda_runtime_folder_is_only_used_by_cuda_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dlls = root / "gpu_runtime_dll"
            dlls.mkdir()
            (dlls / "cudart64_13.dll").touch()
            (dlls / "cudnn64_9.dll").touch()
            with patch.dict(os.environ, {"GPU_RUNTIME_DLL_DIR": str(dlls)}):
                cuda = collect_cuda_runtime_binaries(root, "cuda")
                self.assertEqual({Path(path).name for path, _ in cuda},
                                 {"cudart64_13.dll", "cudnn64_9.dll"})
                self.assertEqual(collect_cuda_runtime_binaries(root, "cpu"), [])
                self.assertEqual(collect_cuda_runtime_binaries(root, "directml"), [])

    def test_shared_builder_writes_profile_manifest_and_distinct_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "onnx_models").mkdir()
            for name in REQUIRED_MODELS:
                (root / "onnx_models" / name).touch()
            (root / "qml").mkdir()
            (root / "assets").mkdir()
            native = root / "onnxruntime_pybind11_state.pyd"
            native.touch()
            class Analysis:
                def __init__(self, *_args, **kwargs):
                    self.binaries = [(str(native), "onnxruntime/capi", "BINARY")]
                    self.datas = kwargs["datas"]
                    self.pure = []
                    self.scripts = []
            def collect(*_args, **kwargs):
                return kwargs["name"]
            namespace = {"SPECPATH": str(root), "Analysis": Analysis,
                         "PYZ": lambda *_: None, "EXE": lambda *_a, **_k: None,
                         "COLLECT": collect}
            with patch("build_support.room36_spec.resolve_build_profile", return_value="cpu"):
                with patch("build_support.room36_spec.collect_ort_binaries",
                           return_value=[(str(native), "onnxruntime/capi")]):
                    result = build_room36_spec("cpu", namespace)
            self.assertEqual(result, "Room36-CPU")
            manifest = json.loads((root / "build" / "onnx-profile-cpu" /
                                   "onnx_build_profile.json").read_text())
            self.assertEqual(manifest["profile"], "cpu")
            self.assertEqual(manifest["expected_provider"], "CPUExecutionProvider")


class PackageValidatorTests(unittest.TestCase):
    def test_accelerator_cpu_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Room36.exe").touch()
            for folder in ("qml", "assets", "onnx_models"):
                (root / folder).mkdir()
            for name in REQUIRED_MODELS:
                (root / "onnx_models" / name).touch()
            for name in ("DirectML.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll",
                         "onnxruntime_pybind11_state.pyd"):
                (root / name).touch()
            (root / "onnx_build_profile.json").write_text(json.dumps({
                "profile": "directml", "expected_provider": "DmlExecutionProvider",
                "native_components": ["DirectML.dll", "onnxruntime.dll",
                                      "onnxruntime_providers_shared.dll",
                                      "onnxruntime_pybind11_state.pyd"]}))
            report = {"frozen": True, "build_profile": "directml", "test_inference": "PASS",
                      "available_providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
                      "test_model_provider": "CPUExecutionProvider"}
            process = SimpleNamespace(returncode=0, stdout=json.dumps(report))
            with patch("build_support.validate_onnx_package.subprocess.run", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                    validate_package(root, "directml")


if __name__ == "__main__":
    unittest.main()
