"""Experimental expression service invariants; no network or GPU required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from face_processing import expression_insights as expression
from app_paths import model_path


class _Input:
    name = "data"
    shape = [1, 3, 112, 112]


class _Session:
    def __init__(self, *args, **kwargs):
        self.providers = kwargs["providers"]

    def get_inputs(self):
        return [_Input()]

    def get_providers(self):
        return self.providers

    def run(self, outputs, feeds):
        assert feeds["data"].shape == (1, 3, 112, 112)
        return [np.array([[0, 0, 0, 1, 0, 0, 0]], np.float32)]


class ExpressionInsightsTests(unittest.TestCase):
    def test_model_uses_application_onnx_directory(self):
        self.assertEqual(expression.model_directory() / expression.NAME,
                         model_path(expression.NAME))

    def test_model_hash_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / expression.NAME
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                expression.ExpressionService(path)

    def test_output_and_missing_landmarks(self):
        with patch.object(expression, "verify", return_value=True):
            service = expression.ExpressionService("model.onnx", providers=["CPUExecutionProvider"],
                session_factory=_Session)
        image = np.zeros((112, 112, 3), np.uint8)
        self.assertEqual(service.predict(image, None)["status"], "unavailable")
        landmarks = expression.TEMPLATE.copy()
        result = service.predict(image, landmarks)
        self.assertEqual(result["label"], "happy")
        self.assertEqual(result["status"], "estimated")
        self.assertEqual(len(result["probabilities"]), 7)

    def test_invalid_output(self):
        with self.assertRaises(ValueError):
            expression.probabilities(np.full(7, np.nan))
        with self.assertRaises(ValueError):
            expression.probabilities([1, 2])

    def test_benchmark_preprocessing_parity(self):
        from app.tools.benchmark_expression_models import align_official, preprocess
        image = np.full((112, 112, 3), 70, np.uint8)
        np.testing.assert_array_equal(expression.align_official(image, expression.TEMPLATE),
                                      align_official(image, expression.TEMPLATE))
        np.testing.assert_array_equal(expression.preprocess(image), preprocess(image))


if __name__ == "__main__":
    unittest.main()
