from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from app_paths import model_path

from .cache import compute_model_fingerprint
from .config import VibeGroupingConfig
from .errors import VibeModelLoadError, VibeModelNotFoundError


LOGGER = logging.getLogger(__name__)

_CLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class VibeEmbedder(ABC):
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def model_fingerprint(self) -> str:
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        ...

    @property
    def uses_fallback(self) -> bool:
        return False

    @abstractmethod
    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        ...


class OnnxVibeEmbedder(VibeEmbedder):
    def __init__(
        self,
        model_file: Path,
    ) -> None:
        if not model_file.is_file():
            raise VibeModelNotFoundError(
                f"Missing vibe model: {model_file}. "
                "Install the ONNX model into the onnx_models folder or enable the fallback embedder."
            )

        try:
            from face_detector.onnx_runtime import create_inference_session, create_session_options

            session_options = create_session_options()
            self._session = create_inference_session(
                model_file,
                session_options=session_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as exc:  # pragma: no cover - exercised by integration on real ORT
            raise VibeModelLoadError(f"Failed to load vibe model {model_file}: {exc}") from exc

        self._model_file = model_file
        self._fingerprint = compute_model_fingerprint(model_file)
        self._provider = self._session.get_providers()[0]
        self._input = self._session.get_inputs()[0]
        self._outputs = self._session.get_outputs()
        input_shape = list(self._input.shape)
        if len(input_shape) != 4:
            raise VibeModelLoadError(
                f"Expected a 4D image encoder input tensor, got {input_shape!r}."
            )
        _, channels, height, width = input_shape
        if channels != 3:
            raise VibeModelLoadError(
                f"Expected a 3-channel image encoder input tensor, got {input_shape!r}."
            )
        self._input_height = int(height if isinstance(height, int) else 224)
        self._input_width = int(width if isinstance(width, int) else 224)
        output_shape = list(self._outputs[0].shape)
        self._embedding_dimension = int(output_shape[-1] if output_shape and isinstance(output_shape[-1], int) else 512)
        LOGGER.info(
            "Initialized ONNX vibe embedder with provider=%s model=%s",
            self._provider,
            self._model_file,
        )

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def model_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def provider(self) -> str:
        return self._provider

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self._embedding_dimension), dtype=np.float32)

        batch = np.stack([self._prepare(image) for image in images], axis=0)
        outputs = self._session.run(
            [self._outputs[0].name],
            {self._input.name: np.ascontiguousarray(batch)},
        )[0]
        embeddings = np.asarray(outputs, dtype=np.float32)
        if embeddings.ndim != 2:
            embeddings = embeddings.reshape(len(images), -1)
        return _l2_normalize(embeddings)

    def _prepare(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        resized = ImageOps.fit(
            pil_image,
            (self._input_width, self._input_height),
            method=Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
        tensor = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = (tensor - _CLIP_MEAN) / _CLIP_STD
        tensor = np.transpose(tensor, (2, 0, 1))
        return np.ascontiguousarray(tensor, dtype=np.float32)


class FallbackVisualEmbedder(VibeEmbedder):
    def __init__(self) -> None:
        self._embedding_dimension = 14 * 14 * 4 + 64
        self._fingerprint = "fallback_visual_embedder_v1"
        self._provider = "CPUExecutionProvider"

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def model_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def uses_fallback(self) -> bool:
        return True

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self._embedding_dimension), dtype=np.float32)

        descriptors = [self._describe(image) for image in images]
        return _l2_normalize(np.stack(descriptors, axis=0))

    def _describe(self, image_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        pooled_rgb = cv2.resize(rgb, (14, 14), interpolation=cv2.INTER_AREA).reshape(-1)
        pooled_h = cv2.resize(hsv[..., :1], (14, 14), interpolation=cv2.INTER_AREA).reshape(-1)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(sobel_x, sobel_y)
        pooled_grad = cv2.resize(gradient, (14, 14), interpolation=cv2.INTER_AREA).reshape(-1)

        dct = cv2.dct(cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA))
        descriptor = np.concatenate(
            [
                pooled_rgb,
                pooled_h,
                pooled_grad,
                dct.reshape(-1),
            ]
        ).astype(np.float32)
        return descriptor


def load_embedder(config: VibeGroupingConfig) -> VibeEmbedder:
    candidate = model_path(config.semantic_model_filename)
    if candidate.is_file():
        return OnnxVibeEmbedder(candidate)
    if not config.allow_visual_fallback:
        raise VibeModelNotFoundError(
            f"Missing vibe model {candidate}. "
            "Place an ONNX image encoder in onnx_models or enable the fallback embedder."
        )
    LOGGER.warning(
        "Falling back to the built-in visual descriptor embedder because %s is missing.",
        candidate,
    )
    return FallbackVisualEmbedder()


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms
