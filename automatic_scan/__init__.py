from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "AutomaticDependencyGraph",
    "AutomaticDuplicateGroup",
    "AutomaticScanConfig",
    "AutomaticScanDiagnostics",
    "AutomaticScanOrchestrator",
    "AutomaticScanResult",
    "AutomaticStage",
    "AutomaticStateStore",
    "AutomaticUnknownCluster",
    "BlurPolicy",
    "ExecutionProvider",
    "FaceQualityPreset",
    "ImageFaceQualityAssessment",
    "ManualOverrideKind",
    "PeoplePictureAssessment",
    "PipelineError",
    "PipelineImageState",
    "PipelineImageStatus",
    "PipelineSummary",
    "PipelineWarning",
    "StageDecision",
    "VibeDetail",
]


_LAZY_IMPORTS = {
    "AutomaticDependencyGraph": ("automatic_scan.dependency_graph", "AutomaticDependencyGraph"),
    "AutomaticDuplicateGroup": ("automatic_scan.models", "AutomaticDuplicateGroup"),
    "AutomaticScanConfig": ("automatic_scan.config", "AutomaticScanConfig"),
    "AutomaticScanDiagnostics": ("automatic_scan.models", "AutomaticScanDiagnostics"),
    "AutomaticScanOrchestrator": ("automatic_scan.orchestrator", "AutomaticScanOrchestrator"),
    "AutomaticScanResult": ("automatic_scan.models", "AutomaticScanResult"),
    "AutomaticStage": ("automatic_scan.models", "AutomaticStage"),
    "AutomaticStateStore": ("automatic_scan.state_store", "AutomaticStateStore"),
    "AutomaticUnknownCluster": ("automatic_scan.models", "AutomaticUnknownCluster"),
    "BlurPolicy": ("automatic_scan.config", "BlurPolicy"),
    "ExecutionProvider": ("automatic_scan.config", "ExecutionProvider"),
    "FaceQualityPreset": ("automatic_scan.config", "FaceQualityPreset"),
    "ImageFaceQualityAssessment": ("automatic_scan.models", "ImageFaceQualityAssessment"),
    "ManualOverrideKind": ("automatic_scan.models", "ManualOverrideKind"),
    "PeoplePictureAssessment": ("automatic_scan.models", "PeoplePictureAssessment"),
    "PipelineError": ("automatic_scan.models", "PipelineError"),
    "PipelineImageState": ("automatic_scan.models", "PipelineImageState"),
    "PipelineImageStatus": ("automatic_scan.models", "PipelineImageStatus"),
    "PipelineSummary": ("automatic_scan.models", "PipelineSummary"),
    "PipelineWarning": ("automatic_scan.models", "PipelineWarning"),
    "StageDecision": ("automatic_scan.models", "StageDecision"),
    "VibeDetail": ("automatic_scan.config", "VibeDetail"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
