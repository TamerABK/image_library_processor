from importlib import import_module
from typing import Any

from .cache import FaceScanCache, ImageFaceAnalysisCache
from .interfaces import (
    EmbeddingSimilarity,
    FaceAnalyzer,
    FaceClusterer,
    FaceDatabase,
    FaceDetector,
    FaceEmbedder,
    FacePreviewRenderer as FacePreviewRendererProtocol,
    FaceQualityAssessor,
    FaceRecognizer,
)
from .models import (
    DetectedFace,
    EmbeddedFace,
    EyeState,
    FaceAnalysisResult,
    FaceProcessorResult,
    HeadPose,
    KnownPerson,
    KnownPersonResult,
    Match,
    Person,
    RecognizedFace,
    StoredEmbedding,
    UnknownCluster,
)
from .processor import FaceProcessor
from .recognition import DefaultFaceRecognizer

__all__ = [
    "ArcFaceEmbedder",
    "ConnectedComponentFaceClusterer",
    "CosineEmbeddingSimilarity",
    "DefaultFaceAnalyzer",
    "DefaultFaceRecognizer",
    "DetectedFace",
    "EmbeddedFace",
    "EmbeddingSimilarity",
    "EyeState",
    "FaceAligner",
    "FaceAnalysisResult",
    "FaceAnalyzer",
    "FaceAnalyzerConfig",
    "FaceClusterer",
    "FaceDatabase",
    "FaceDetector",
    "FaceEmbedder",
    "FacePreviewRenderer",
    "FacePreviewRendererProtocol",
    "FaceProcessor",
    "FaceProcessorResult",
    "FaceQualityAssessor",
    "FaceRecognizer",
    "FaceScanCache",
    "HeadPose",
    "ImageFaceAnalysisCache",
    "KnownPerson",
    "KnownPersonResult",
    "Match",
    "Person",
    "RecognizedFace",
    "SCRFDFaceDetector",
    "SQLiteFaceDatabase",
    "StoredEmbedding",
    "UnknownCluster",
]

_LAZY_EXPORTS = {
    "DefaultFaceAnalyzer": ("face_analyzer", "DefaultFaceAnalyzer"),
    "FaceAnalyzerConfig": ("face_analyzer", "FaceAnalyzerConfig"),
    "ArcFaceEmbedder": ("face_detector.arc_embedder", "ArcFaceEmbedder"),
    "ConnectedComponentFaceClusterer": (
        "face_detector.connected_face_clusterer",
        "ConnectedComponentFaceClusterer",
    ),
    "CosineEmbeddingSimilarity": (
        "face_detector.cosine_similarity",
        "CosineEmbeddingSimilarity",
    ),
    "FaceAligner": ("face_detector.face_aligner", "FaceAligner"),
    "FacePreviewRenderer": ("face_detector.preview_renderer", "FacePreviewRenderer"),
    "SCRFDFaceDetector": ("face_detector.scrfd_face_detector", "SCRFDFaceDetector"),
    "SQLiteFaceDatabase": (
        "face_detector.face_database_sqlite",
        "SQLiteFaceDatabase",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
