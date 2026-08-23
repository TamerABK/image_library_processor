from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from automatic_scan.perf import get_active_profiler
from face_detector.onnx_runtime import ort


Provider = str | tuple[str, dict[str, Any]]


class OnnxModel:
    """Small ONNX Runtime wrapper with deterministic provider fallback."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[Provider] | None = None,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Install onnxruntime or "
                "onnxruntime-gpu."
            )

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.model_path = model_path

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        requested = list(providers) if providers else [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        available = set(ort.get_available_providers())
        selected = [
            provider
            for provider in requested
            if (provider[0] if isinstance(provider, tuple) else provider) in available
        ]
        if not selected:
            selected = ["CPUExecutionProvider"]

        session_started = time.perf_counter()
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=selected,
        )
        self.input = self.session.get_inputs()[0]
        self.outputs = self.session.get_outputs()
        profiler = get_active_profiler()
        if profiler is not None:
            provider_name = self.session.get_providers()[0] if self.session.get_providers() else None
            profiler.record_model_session_creation(
                self.model_path.stem,
                time.perf_counter() - session_started,
                provider=provider_name,
            )

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.run_all_outputs(tensor)

    def run_all_outputs(self, tensor: np.ndarray) -> list[np.ndarray]:
        contiguous = np.ascontiguousarray(tensor)
        started = time.perf_counter()
        outputs = self.session.run(
            None,
            {self.input.name: contiguous},
        )
        profiler = get_active_profiler()
        if profiler is not None:
            batch_size = int(contiguous.shape[0]) if contiguous.ndim >= 1 else 1
            provider_name = self.session.get_providers()[0] if self.session.get_providers() else None
            profiler.record_model_call(
                self.model_path.stem,
                batch_size=batch_size,
                inference_seconds=time.perf_counter() - started,
                provider=provider_name,
            )
        return outputs

    def metadata_report(self) -> dict[str, Any]:
        metadata = self.session.get_modelmeta()
        return {
            "model_path": str(self.model_path),
            "file_size_bytes": int(self.model_path.stat().st_size),
            "sha256": self._sha256(self.model_path),
            "producer_name": metadata.producer_name,
            "graph_name": metadata.graph_name,
            "description": metadata.description,
            "custom_metadata": dict(metadata.custom_metadata_map),
            "inputs": [
                {
                    "name": item.name,
                    "shape": list(item.shape),
                    "type": item.type,
                }
                for item in self.session.get_inputs()
            ],
            "outputs": [
                {
                    "name": item.name,
                    "shape": list(item.shape),
                    "type": item.type,
                }
                for item in self.session.get_outputs()
            ],
            "providers": list(self.session.get_providers()),
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
