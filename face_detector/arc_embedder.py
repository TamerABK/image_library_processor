from pathlib import Path
import time
from typing import Any, Callable

import cv2
import numpy as np

from .face_aligner import FaceAligner
from .interfaces import FaceEmbedder
from .models import DetectedFace, EmbeddedFace
from .onnx_runtime import (
    create_inference_session,
    create_session_options,
    get_available_providers,
    select_providers,
)


class ArcFaceEmbedder(FaceEmbedder):
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
        aligner: FaceAligner,
    ):
        self._aligner = aligner
        self._timing_callback: Callable[[str, float], None] | None = None
        self._available_providers = get_available_providers()
        session_options = create_session_options()

        self._session = create_inference_session(
            model_path,
            session_options=session_options,
            providers=self._select_providers(),
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def set_timing_callback(
        self,
        callback: Callable[[str, float], None] | None,
    ) -> None:
        self._timing_callback = callback

    def runtime_info(self) -> dict[str, Any]:
        return {
            "selected_providers": self._session.get_providers(),
            "available_providers": list(self._available_providers),
        }

    def embed(
        self,
        image: np.ndarray,
        faces: list[DetectedFace],
    ) -> list[EmbeddedFace]:
        return self.embed_requests([(image, face) for face in faces])

    def embed_requests(
        self,
        face_requests: list[tuple[np.ndarray, DetectedFace]],
    ) -> list[EmbeddedFace]:
        if not face_requests:
            return []

        align_started = time.perf_counter()
        aligned_faces = [
            self._aligner.align(image, face.landmarks)
            for image, face in face_requests
        ]
        self._record_timing("align", time.perf_counter() - align_started)

        try:
            embeddings = self._extract_embeddings(aligned_faces)
        except Exception:
            embeddings = self._extract_embeddings_one_by_one(aligned_faces)

        embedded = []
        for (_image, face), embedding in zip(face_requests, embeddings):
            embedded.append(
                EmbeddedFace(
                    path=face.path,
                    bbox=face.bbox,
                    confidence=face.confidence,
                    landmarks=face.landmarks,
                    embedding=embedding,
                )
            )
        return embedded

    @classmethod
    def _select_providers(cls) -> list[str]:
        return select_providers(cls._PREFERRED_PROVIDERS)

    def _extract_embeddings(
        self,
        aligned_faces: list[np.ndarray],
    ) -> list[np.ndarray]:
        preprocess_started = time.perf_counter()
        prepared = [self._prepare_input(aligned) for aligned in aligned_faces]
        batch = np.stack(prepared, axis=0)
        self._record_timing("embed_preprocess", time.perf_counter() - preprocess_started)

        infer_started = time.perf_counter()
        raw_embeddings = self._run_session_raw(batch)
        self._record_timing("embed_infer", time.perf_counter() - infer_started)

        postprocess_started = time.perf_counter()
        embeddings = self._normalize_embeddings(raw_embeddings)
        self._record_timing("embed_postprocess", time.perf_counter() - postprocess_started)

        if len(embeddings) != len(aligned_faces):
            raise RuntimeError(
                f"Expected {len(aligned_faces)} embeddings but got {len(embeddings)}."
            )
        return [embedding for embedding in embeddings]

    def _extract_embeddings_one_by_one(
        self,
        aligned_faces: list[np.ndarray],
    ) -> list[np.ndarray]:
        embeddings = []
        preprocess_seconds = 0.0
        infer_seconds = 0.0
        postprocess_seconds = 0.0
        for aligned in aligned_faces:
            preprocess_started = time.perf_counter()
            single_batch = self._prepare_input(aligned)[None, ...]
            preprocess_seconds += time.perf_counter() - preprocess_started

            infer_started = time.perf_counter()
            raw_embeddings = self._run_session_raw(single_batch)
            infer_seconds += time.perf_counter() - infer_started

            postprocess_started = time.perf_counter()
            embeddings.append(self._normalize_embeddings(raw_embeddings)[0])
            postprocess_seconds += time.perf_counter() - postprocess_started

        self._record_timing("embed_preprocess", preprocess_seconds)
        self._record_timing("embed_infer", infer_seconds)
        self._record_timing("embed_postprocess", postprocess_seconds)
        return embeddings

    def _prepare_input(
        self,
        aligned: np.ndarray,
    ) -> np.ndarray:
        rgb = cv2.cvtColor(
            aligned,
            cv2.COLOR_BGR2RGB,
        )

        blob = rgb.astype(np.float32)
        blob = (blob - 127.5) / 127.5
        return np.transpose(blob, (2, 0, 1))

    def _run_session_raw(
        self,
        batch: np.ndarray,
    ) -> np.ndarray:
        embedding_batch = self._session.run(
            [self._output_name],
            {self._input_name: batch},
        )[0]
        return np.asarray(embedding_batch, dtype=np.float32)

    @staticmethod
    def _normalize_embeddings(
        embedding_batch: np.ndarray,
    ) -> np.ndarray:
        embedding_batch = np.asarray(embedding_batch, dtype=np.float32)
        norms = np.linalg.norm(embedding_batch, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embedding_batch / norms

    def _record_timing(
        self,
        phase: str,
        seconds: float,
    ) -> None:
        if self._timing_callback is not None:
            self._timing_callback(phase, seconds)
