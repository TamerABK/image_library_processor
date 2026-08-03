from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # Allows importing the package before ORT is installed.
    ort = None  # type: ignore[assignment]


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

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=selected,
        )
        self.input = self.session.get_inputs()[0]
        self.outputs = self.session.get_outputs()

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.run_all_outputs(tensor)

    def run_all_outputs(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.session.run(
            None,
            {self.input.name: np.ascontiguousarray(tensor)},
        )

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
