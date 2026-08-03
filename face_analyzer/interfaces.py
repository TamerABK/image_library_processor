from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from .models import DetectedFace, EyeState, FaceAnalysisResult, HeadPose


class FaceAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        image: np.ndarray,
        face: DetectedFace,
    ) -> FaceAnalysisResult:
        raise NotImplementedError


class EyeStateEstimator(Protocol):
    def estimate(
        self,
        image: np.ndarray,
        face: DetectedFace,
        *,
        head_pose: HeadPose,
        face_index: int | None = None,
    ) -> EyeState:
        ...


class HeadPoseEstimator(Protocol):
    def estimate(
        self,
        image: np.ndarray,
        face: DetectedFace,
    ) -> HeadPose:
        ...
