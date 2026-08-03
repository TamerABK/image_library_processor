from pathlib import Path
from typing import Any

import numpy as np

from face_processing.interfaces import FaceDetector
from face_processing.models import DetectedFace
from .onnx_runtime import (
    create_session_options,
    get_available_providers,
    select_providers,
)
from .scrfd import SCRFD


class SCRFDFaceDetector(FaceDetector):
    _PREFERRED_PROVIDERS = (
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "OpenVINOExecutionProvider",
        "ROCMExecutionProvider",
        "CPUExecutionProvider",
    )

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
    ):
        self._input_size = input_size
        self._available_providers = get_available_providers()
        session_options = create_session_options()

        self._detector = SCRFD(
            str(model_path),
            session_options=session_options,
            providers=self._select_providers(),
        )

        self._detector.prepare(
            ctx_id=0,
            det_thresh=score_threshold,
            nms_thresh=nms_threshold,
            input_size=input_size,
        )

    @classmethod
    def _select_providers(cls) -> list[str]:
        return select_providers(cls._PREFERRED_PROVIDERS)

    def runtime_info(self) -> dict[str, Any]:
        return {
            "selected_providers": self._detector.session.get_providers(),
            "available_providers": list(self._available_providers),
            "input_size": list(self._input_size),
        }

    def detect(
        self,
        image: np.ndarray,
        path: Path,
    ) -> list[DetectedFace]:

        detections, landmarks = self._detector.detect(image)

        if detections is None or len(detections) == 0:
            return []

        if landmarks is None:
            landmarks = [None] * len(detections)

        results: list[DetectedFace] = []

        for det, kps in zip(detections, landmarks):

            x1, y1, x2, y2, score = det

            bbox = (
                int(round(x1)),
                int(round(y1)),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
            )

            results.append(
                DetectedFace(
                    path=path,
                    bbox=bbox,
                    confidence=float(score),
                    landmarks=np.asarray(kps, dtype=np.float32),
                    index=None,
                    analysis=None,
                )
            )

        return results
