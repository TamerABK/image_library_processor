"""Optional native tests: deterministic temporary tensors, no external corpus."""
from __future__ import annotations

import gc
import unittest

import numpy as np

from app_paths import model_path
from face_detector.onnx_diagnostics import MODELS, synthetic_input, validate_outputs
from face_detector.onnx_runtime import create_inference_session, create_session_options, get_available_providers


class NativeModelTests(unittest.TestCase):
    def compare(self, model, provider, *, batch_size=1):
        if provider not in get_available_providers():
            self.skipTest(provider + " is not installed.")
        filename, side = MODELS[model]
        if not model_path(filename).is_file():
            self.skipTest("Optional production model is not installed: " + model)
        options = create_session_options([provider])
        options.intra_op_num_threads = 2
        accelerator = create_inference_session(model_path(filename), providers=[provider],
                                               session_options=options)
        if accelerator.get_providers()[0] != provider:
            self.skipTest(provider + " cannot initialize on this hardware/runtime.")
        tensor = synthetic_input(accelerator, side, batch_size)
        actual = accelerator.run(None, {accelerator.get_inputs()[0].name: tensor})
        validate_outputs(actual)
        del accelerator
        gc.collect()
        cpu_options = create_session_options(["CPUExecutionProvider"])
        cpu_options.intra_op_num_threads = 2
        cpu = create_inference_session(model_path(filename), providers=["CPUExecutionProvider"],
                                      session_options=cpu_options)
        expected = cpu.run(None, {cpu.get_inputs()[0].name: tensor})
        validate_outputs(expected)
        self.assertEqual(len(actual), len(expected))
        for output, reference in zip(actual, expected):
            np.testing.assert_allclose(output, reference, rtol=0.01, atol=0.002)
        if model in ("embedding", "vibe"):
            a, b = actual[0].reshape(-1), expected[0].reshape(-1)
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            self.assertGreater(cosine, 0.999)
        if model == "eye":
            self.assertEqual(int(np.argmax(actual[0])), int(np.argmax(expected[0])))
            self.assertTrue(np.all(np.asarray(actual[0]) >= 0))
        if model == "pose":
            from face_analyzer.head_pose import SixDRepNetOnnxEstimator
            a = SixDRepNetOnnxEstimator._rotation_matrix(np.asarray(actual[0])[0].squeeze())
            b = SixDRepNetOnnxEstimator._rotation_matrix(np.asarray(expected[0])[0].squeeze())
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            np.testing.assert_allclose(SixDRepNetOnnxEstimator._euler_xyz(a),
                                       SixDRepNetOnnxEstimator._euler_xyz(b), atol=0.01)
        if model == "detector":
            # SCRFD score tensors retain valid confidence semantics.
            for scores in actual[:3]:
                self.assertTrue(np.all(np.asarray(scores) >= 0))
                self.assertTrue(np.all(np.asarray(scores) <= 1))
        print("ONNX_NATIVE model=" + model + " provider=" + provider +
              " cpu_comparison=PASS")


def _make_test(model, provider):
    def test(self):
        self.compare(model, provider)
    return test


for _model in MODELS:
    for _alias, _provider in (
        ("cpu", "CPUExecutionProvider"), ("cuda", "CUDAExecutionProvider"),
        ("directml", "DmlExecutionProvider"), ("migraphx", "MIGraphXExecutionProvider"),
    ):
        setattr(NativeModelTests, "test_" + _model + "_" + _alias, _make_test(_model, _provider))

for _model in ("embedding", "pose", "vibe"):
    def _batch_test(self, model=_model):
        self.compare(model, "DmlExecutionProvider", batch_size=8)
    setattr(NativeModelTests, "test_" + _model + "_directml_batch8", _batch_test)


def _embedding_batch32(self):
    self.compare("embedding", "DmlExecutionProvider", batch_size=32)


NativeModelTests.test_embedding_directml_batch32 = _embedding_batch32


if __name__ == "__main__":
    unittest.main()
