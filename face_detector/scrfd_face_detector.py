from pathlib import Path
import time
from typing import Any
from typing import Callable

import numpy as np

from automatic_scan.perf import get_active_profiler
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
        providers: list[str] | tuple[str, ...] | None = None,
    ):
        self._input_size = input_size
        self._timing_callback: Callable[[str, float], None] | None = None
        self._available_providers = get_available_providers()
        session_options = create_session_options()

        self._detector = SCRFD(
            str(model_path),
            session_options=session_options,
            providers=self._select_providers(providers),
        )
        self._detector.set_timing_callback(self._record_timing)

        self._detector.prepare(
            ctx_id=0,
            det_thresh=score_threshold,
            nms_thresh=nms_threshold,
            input_size=input_size,
        )

    def set_timing_callback(
        self,
        callback: Callable[[str, float], None] | None,
    ) -> None:
        self._timing_callback = callback

    @classmethod
    def _select_providers(
        cls,
        providers: list[str] | tuple[str, ...] | None = None,
    ) -> list[str]:
        if providers:
            return select_providers(providers)
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
        phase_totals: dict[str, float] = {}
        previous_callback = self._timing_callback

        def capture_phase(phase: str, seconds: float) -> None:
            phase_totals[phase] = phase_totals.get(phase, 0.0) + seconds
            if previous_callback is not None:
                previous_callback(phase, seconds)

        self._timing_callback = capture_phase
        detect_started = time.perf_counter()
        try:
            detections, landmarks = self._detector.detect(image)
            self._record_timing("detect_model_total", time.perf_counter() - detect_started)

            if detections is None or len(detections) == 0:
                self._record_timing("detect_materialize_faces", 0.0)
                self._record_model_metrics(phase_totals)
                return []

            if landmarks is None:
                landmarks = [None] * len(detections)

            results: list[DetectedFace] = []

            materialize_started = time.perf_counter()
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
            self._record_timing("detect_materialize_faces", time.perf_counter() - materialize_started)
            self._record_model_metrics(phase_totals)
            return results
        finally:
            self._timing_callback = previous_callback

    def _record_model_metrics(self, phase_totals: dict[str, float]) -> None:
        profiler = get_active_profiler()
        if profiler is None:
            return
        preprocess_seconds = (
            phase_totals.get("detect_resize_and_pad", 0.0)
            + phase_totals.get("detect_blob_from_image", 0.0)
        )
        inference_seconds = phase_totals.get("detect_session_run", 0.0)
        postprocess_seconds = sum(
            seconds
            for phase, seconds in phase_totals.items()
            if phase not in {"detect_resize_and_pad", "detect_blob_from_image", "detect_session_run"}
        )
        provider_name = (
            self._detector.session.get_providers()[0]
            if self._detector.session.get_providers()
            else None
        )
        profiler.record_model_call(
            "face_detector",
            batch_size=1,
            preprocess_seconds=preprocess_seconds,
            inference_seconds=inference_seconds,
            postprocess_seconds=postprocess_seconds,
            provider=provider_name,
        )

    def _record_timing(
        self,
        phase: str,
        seconds: float,
    ) -> None:
        profiler = get_active_profiler()
        if profiler is not None:
            profiler.record_model_phase(f"face_detector.{phase}", seconds)
        if self._timing_callback is not None:
            self._timing_callback(phase, seconds)
