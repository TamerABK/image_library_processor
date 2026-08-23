from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from automatic_scan.config import ExecutionProvider
from automatic_scan.services import _resolve_onnx_providers
from face_detector.arc_embedder import ArcFaceEmbedder
from face_detector.onnx_runtime import create_inference_session
from face_detector.scrfd_face_detector import SCRFDFaceDetector


class ProviderResolutionTests(unittest.TestCase):
    @patch("automatic_scan.services.select_providers")
    def test_gpu_provider_resolution_filters_to_available_runtime(self, select_providers_mock) -> None:
        select_providers_mock.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        providers = _resolve_onnx_providers(ExecutionProvider.GPU)

        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        select_providers_mock.assert_called_once()

    @patch("face_detector.scrfd_face_detector.select_providers")
    def test_scrfd_detector_filters_explicit_provider_list(self, select_providers_mock) -> None:
        select_providers_mock.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        providers = SCRFDFaceDetector._select_providers(
            ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
        )

        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        select_providers_mock.assert_called_once_with(
            ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
        )

    @patch("face_detector.arc_embedder.select_providers")
    def test_arcface_embedder_filters_explicit_provider_list(self, select_providers_mock) -> None:
        select_providers_mock.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        providers = ArcFaceEmbedder._select_providers(
            ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
        )

        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        select_providers_mock.assert_called_once_with(
            ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
        )

    @patch("face_detector.onnx_runtime.ort")
    @patch("face_detector.onnx_runtime.select_providers")
    def test_create_inference_session_filters_explicit_provider_list(
        self,
        select_providers_mock,
        ort_mock,
    ) -> None:
        session = MagicMock()
        ort_mock.InferenceSession.return_value = session
        select_providers_mock.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        result = create_inference_session(
            "model.onnx",
            session_options=object(),
            providers=["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"],
        )

        self.assertIs(result, session)
        ort_mock.InferenceSession.assert_called_once()
        _, kwargs = ort_mock.InferenceSession.call_args
        self.assertEqual(kwargs["providers"], ["CUDAExecutionProvider", "CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
