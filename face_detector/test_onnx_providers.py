from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from build_support.onnx_packaging import validate_native_components
from build_support.validate_onnx_package import validate_package
from face_detector import onnx_runtime as runtime
from face_detector.onnx_directml_compat import correct_nonzero_reshapes
from face_detector.onnx_diagnostics import run_diagnostic

CPU = runtime.CPU
DML = runtime.DML
CUDA = "CUDAExecutionProvider"
MIG = "MIGraphXExecutionProvider"


class ProviderPolicyTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"ROOM36_ONNX_PROVIDER": "auto"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def select(self, available, *, platform="win32", explicit=None):
        with patch.object(runtime, "get_available_providers", return_value=available):
            return runtime.select_providers(explicit, platform=platform)

    def test_nvidia(self):
        self.assertEqual(self.select([CPU, CUDA]), [CUDA, CPU])

    def test_directml(self):
        self.assertEqual(self.select([CPU, DML]), [DML, CPU])

    def test_migraphx_before_directml(self):
        self.assertEqual(self.select([DML, CPU, MIG]), [MIG, DML, CPU])

    def test_windows_order_is_deterministic(self):
        self.assertEqual(self.select([CPU, DML, MIG, CUDA, "OpenVINOExecutionProvider"]),
                         [CUDA, MIG, DML, "OpenVINOExecutionProvider", CPU])

    def test_linux_excludes_directml(self):
        self.assertEqual(self.select([DML, CPU, CUDA, MIG], platform="linux"), [CUDA, MIG, CPU])

    def test_legacy_rocm_linux_only(self):
        available = ["ROCMExecutionProvider", CPU]
        self.assertEqual(self.select(available, platform="linux"), available)
        self.assertEqual(self.select(available), [CPU])

    def test_macos_coreml(self):
        self.assertEqual(self.select([CPU, CUDA, "CoreMLExecutionProvider"], platform="darwin"),
                         ["CoreMLExecutionProvider", CPU])

    def test_cpu_only(self):
        self.assertEqual(self.select([CPU]), [CPU])

    def test_force_cpu(self):
        with patch.dict(os.environ, {"ROOM36_ONNX_PROVIDER": "cpu"}):
            self.assertEqual(self.select([CPU, CUDA, DML]), [CPU])

    def test_override_preferences_and_safe_fallback(self):
        for requested, expected in (("cuda", CUDA), ("directml", DML), ("migraphx", MIG)):
            with self.subTest(requested=requested):
                with patch.dict(os.environ, {"ROOM36_ONNX_PROVIDER": requested}):
                    self.assertEqual(self.select([CPU, CUDA, DML, MIG])[0], expected)
                    self.assertEqual(self.select([CPU]), [CPU])

    def test_invalid_override_logs_and_uses_auto(self):
        with patch.dict(os.environ, {"ROOM36_ONNX_PROVIDER": "banana"}):
            with self.assertLogs(runtime.LOGGER, level="WARNING"):
                self.assertEqual(self.select([CPU, DML]), [DML, CPU])

    def test_explicit_options_precede_environment(self):
        provider = (DML, {"device_id": 1})
        with patch.dict(os.environ, {"ROOM36_ONNX_PROVIDER": "cpu"}):
            self.assertEqual(self.select([DML, CPU], explicit=[provider]), [provider, CPU])
            self.assertEqual(self.select([DML, CPU], explicit=[]), [CPU])

    def test_unavailable_explicit_provider(self):
        self.assertEqual(self.select([CPU, DML], explicit=[CUDA]), [CPU])

    def test_compiled_provider_names_are_not_availability(self):
        fake = SimpleNamespace(get_all_providers=lambda: [DML, CPU])
        with patch.object(runtime, "ort", fake):
            self.assertEqual(runtime.get_available_providers(), [CPU])


class SessionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.fail = set()
        self.silent_cpu = set()
        self.fake = SimpleNamespace(
            SessionOptions=lambda: SimpleNamespace(enable_mem_pattern=True, execution_mode="parallel"),
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            get_available_providers=lambda: [CUDA, MIG, DML, CPU],
            InferenceSession=self.construct,
        )
        self.patch = patch.object(runtime, "ort", self.fake)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        env = patch.dict(os.environ, {"ROOM36_ONNX_PROVIDER": "auto"})
        env.start()
        self.addCleanup(env.stop)

    def construct(self, path, *, sess_options, providers):
        names = [p[0] if isinstance(p, tuple) else p for p in providers]
        self.calls.append((names, sess_options))
        if names[0] in self.fail:
            raise RuntimeError("test initialization failure")
        active = [CPU] if names[0] in self.silent_cpu else names
        return SimpleNamespace(get_providers=lambda: active, run=lambda *args, **kwargs: [1])

    def test_directml_options(self):
        options = self.fake.SessionOptions()
        session = runtime.create_inference_session("eye.onnx", providers=[DML], session_options=options)
        self.assertEqual(options.execution_mode, "sequential")
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(session.diagnostics["active_provider"], DML)

    def test_each_inference_logs_actual_provider_and_device(self):
        logger = MagicMock()
        with patch.object(runtime, "get_scan_logger", return_value=logger):
            gpu = runtime.create_inference_session("eye.onnx", providers=[DML])
            gpu.run(None, {"input": SimpleNamespace(shape=(4, 3, 32, 32))})
            self.fail.add(DML)
            with self.assertLogs(runtime.LOGGER, level="WARNING"):
                cpu = runtime.create_inference_session("eye.onnx", providers=[DML])
            cpu.run(None, {"input": SimpleNamespace(shape=(2, 3, 32, 32))})
        messages = [call.args[0] % call.args[1:] for call in logger.info.call_args_list]
        self.assertTrue(any("onnx_inference model=eye" in message and
                            "provider=DmlExecutionProvider device=GPU" in message and
                            "batch_size=4" in message for message in messages))
        self.assertTrue(any("onnx_inference model=eye" in message and
                            "provider=CPUExecutionProvider device=CPU" in message and
                            "batch_size=2" in message for message in messages))
        self.assertTrue(any("onnx_session_ready model=eye" in message and
                            "active=CPUExecutionProvider device=CPU" in message for message in messages))

    def test_secondary_directml_also_requires_safe_options(self):
        runtime.create_inference_session("model.onnx", providers=[CUDA, DML])
        self.assertFalse(self.calls[0][1].enable_mem_pattern)

    def test_cpu_does_not_receive_directml_restrictions(self):
        runtime.create_inference_session("model.onnx", providers=[CPU])
        self.assertTrue(self.calls[0][1].enable_mem_pattern)

    def test_migraphx_failure_tries_directml(self):
        self.fail.add(MIG)
        with self.assertLogs(runtime.LOGGER, level="WARNING"):
            result = runtime.create_inference_session("model.onnx", providers=[MIG, DML])
        self.assertEqual([call[0][0] for call in self.calls], [MIG, DML])
        self.assertEqual(result.diagnostics["active_provider"], DML)
        self.assertEqual(result.diagnostics["fallbacks"][0]["provider"], MIG)

    def test_directml_failure_tries_cpu(self):
        self.fail.add(DML)
        with self.assertLogs(runtime.LOGGER, level="WARNING"):
            result = runtime.create_inference_session("model.onnx", providers=[DML])
        self.assertEqual([call[0][0] for call in self.calls], [DML, CPU])
        self.assertEqual(result.diagnostics["active_provider"], CPU)

    def test_ort_internal_cpu_fallback_still_tries_next_accelerator(self):
        self.silent_cpu.add(CUDA)
        with self.assertLogs(runtime.LOGGER, level="WARNING"):
            result = runtime.create_inference_session("model.onnx", providers=[CUDA, DML])
        self.assertEqual(result.diagnostics["active_provider"], DML)
        self.assertEqual([call[0][0] for call in self.calls], [CUDA, DML])

    def test_cpu_initialization_failure_is_clear(self):
        self.fail.add(CPU)
        with self.assertLogs(runtime.LOGGER, level="WARNING"):
            with self.assertRaisesRegex(RuntimeError, "Unable to initialize ONNX model"):
                runtime.create_inference_session("model.onnx", providers=[CPU])

    def test_profiler_records_actual_fallback_provider(self):
        profiler = MagicMock()
        self.fail.add(DML)
        with patch.object(runtime, "get_active_profiler", return_value=profiler):
            with self.assertLogs(runtime.LOGGER, level="WARNING"):
                runtime.create_inference_session("eye.onnx", providers=[DML])
        self.assertEqual(profiler.record_model_session_creation.call_args.kwargs["provider"], CPU)

    def test_directml_run_is_serialized_per_session(self):
        active = 0
        peak = 0
        counter_lock = threading.Lock()

        def run(*args):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.015)
            with counter_lock:
                active -= 1
            return [1]

        session = runtime.InferenceSession(
            SimpleNamespace(run=run, get_providers=lambda: [DML, CPU]),
            {"active_providers": [DML, CPU]})
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: session.run(None, {}), range(8)))
        self.assertEqual(peak, 1)
        self.assertEqual(results, [[1]] * 8)

    def test_unrelated_directml_sessions_are_not_globally_serialized(self):
        barrier = threading.Barrier(2)
        raw = SimpleNamespace(run=lambda *args: barrier.wait(timeout=2),
                              get_providers=lambda: [DML])
        sessions = [runtime.InferenceSession(raw, {"active_providers": [DML]}) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda s: s.run(None, {}), sessions))

    def test_cuda_has_no_run_lock(self):
        session = runtime.InferenceSession(SimpleNamespace(), {"active_providers": [CUDA, CPU]})
        self.assertIsNone(session._run_lock)

    def test_directml_async_api_cannot_bypass_lock(self):
        session = runtime.InferenceSession(SimpleNamespace(), {"active_providers": [DML]})
        with self.assertRaisesRegex(RuntimeError, "background worker"):
            session.run_async(None, {}, None, None)

    def test_dynamic_batch_variants_are_bounded_and_reused(self):
        created = []
        self.fake.SessionOptions = lambda: SimpleNamespace(
            enable_mem_pattern=True, execution_mode="parallel",
            add_free_dimension_override_by_name=lambda *args: None)
        def factory(*args, **kwargs):
            created.append(kwargs["session_options"])
            batch = len(created) + 1
            fake = SimpleNamespace(
                diagnostics={"fallbacks": [], "active_providers": [DML]},
                get_providers=lambda: [DML],
                run=lambda *a, **k: [batch],
            )
            return fake
        raw = SimpleNamespace(
            get_providers=lambda: [DML],
            get_inputs=lambda: [SimpleNamespace(shape=["None", 3, 112, 112])],
            run=lambda *a, **k: [0],
        )
        session = runtime.InferenceSession(
            raw, {"active_providers": [DML], "fallbacks": []},
            model_path="arc.onnx", providers=[DML, CPU],
            options=self.fake.SessionOptions())
        with patch.object(runtime, "create_inference_session", side_effect=factory):
            for batch in (8, 4, 8, 2):
                session.run(None, {"input": SimpleNamespace(shape=(batch, 3, 112, 112))})
        self.assertEqual(len(created), 3)
        self.assertEqual(list(session._batch_sessions), [8, 2])


class PackagingContractTests(unittest.TestCase):
    def test_missing_directml_dll_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "directml.dll"):
            validate_native_components(["onnxruntime_pybind11_state.pyd", "onnxruntime.dll"], "directml")

    def test_native_components_accept_arbitrary_layout(self):
        paths = ["a/DirectML.dll", "b/onnxruntime.dll", "c/onnxruntime_providers_shared.dll",
                 "d/onnxruntime_pybind11_state.pyd"]
        self.assertIn("directml.dll", validate_native_components(paths, "directml"))

    def test_package_cannot_pass_via_source_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "Room36.exe").touch()
            (folder / "onnxruntime_pybind11_state.pyd").touch()
            (folder / "onnx_build_profile.json").write_text(
                '{"profile":"cpu","native_components":["onnxruntime_pybind11_state.pyd"]}')
            process = SimpleNamespace(returncode=0, stdout='{"frozen":false,"test_inference":"PASS"}')
            with patch("build_support.validate_onnx_package.subprocess.run", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "development Python"):
                    validate_package(folder, "cpu")

    def test_strict_diagnostic_rejects_unavailable_provider(self):
        with patch("face_detector.onnx_diagnostics.startup_diagnostics", return_value={
            "runtime_version": "test", "available_providers": [CPU], "selected_providers": [CPU]}):
            report = run_diagnostic(expect_provider="directml")
        self.assertEqual(report["test_inference"], "FAIL")
        self.assertIn("unavailable", report["error"])

    def test_strict_diagnostic_rejects_cpu_fallback(self):
        session = SimpleNamespace(
            diagnostics={"active_provider": CPU, "session_creation_seconds": .1,
                         "fallbacks": [{"provider": DML}]}, get_providers=lambda: [CPU])
        with patch("face_detector.onnx_diagnostics.startup_diagnostics", return_value={
            "runtime_version": "test", "available_providers": [DML, CPU], "selected_providers": [DML]}):
            with patch("face_detector.onnx_diagnostics.create_inference_session", return_value=session):
                report = run_diagnostic(expect_provider="directml")
        self.assertEqual(report["test_inference"], "FAIL")
        self.assertIn("not accepted", report["error"])


class DirectMLModelCompatibilityTests(unittest.TestCase):
    def test_only_nonzero_reshape_shapes_are_corrected(self):
        import onnx
        from onnx import TensorProto, helper
        valid = helper.make_node("Constant", [], ["valid_shape"],
                                 value=helper.make_tensor("shape", TensorProto.INT64,
                                                          [2], [1, -1]))
        zero = helper.make_node("Constant", [], ["zero_shape"],
                                value=helper.make_tensor("shape", TensorProto.INT64,
                                                         [2], [0, -1]))
        nodes = [valid, zero,
                 helper.make_node("Reshape", ["input", "valid_shape"], ["a"],
                                  name="safe", allowzero=1),
                 helper.make_node("Reshape", ["input", "zero_shape"], ["b"],
                                  name="unsafe", allowzero=1)]
        graph = helper.make_graph(nodes, "test", [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])], [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [0, 2])])
        model = helper.make_model(graph)
        self.assertEqual(correct_nonzero_reshapes(model), 1)
        self.assertEqual([n.attribute[0].i for n in model.graph.node
                          if n.op_type == "Reshape"], [0, 1])


if __name__ == "__main__":
    unittest.main()
