from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from scan_controls import CancellationToken


ProgressCallback = Callable[[str, int, int | None], None]


@dataclass(frozen=True, slots=True)
class ScanError:
    path: str
    message: str
    fatal: bool = False


@dataclass(frozen=True, slots=True)
class VibeImageFeatures:
    image_path: str
    semantic_embedding: np.ndarray
    capture_timestamp: float | None
    timestamp_source: str
    recognized_person_ids: tuple[str, ...]
    color_features: np.ndarray | None
    composition_features: np.ndarray | None
    face_layout: np.ndarray | None
    face_scale_summary: np.ndarray | None
    subject_scene_embedding: np.ndarray | None
    background_embedding: np.ndarray | None
    action_scores: np.ndarray | None
    scene_scores: np.ndarray | None
    shot_type_scores: np.ndarray | None
    width: int | None
    height: int | None
    file_mtime_ns: int
    file_size: int
    quality_score: float | None = None
    brightness: float | None = None
    face_count: int = 0
    face_area_ratio: float = 0.0
    dominant_people_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VibeDuplicateSubgroup:
    subgroup_id: str
    image_paths: tuple[str, ...]


@dataclass(slots=True)
class VibeGroup:
    group_id: str
    image_paths: list[str]
    representative_path: str
    start_timestamp: float | None
    end_timestamp: float | None
    recognized_person_ids: tuple[str, ...]
    recognized_person_names: tuple[str, ...]
    label: str | None
    cohesion_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    duplicate_subgroups: list[VibeDuplicateSubgroup] = field(default_factory=list)


@dataclass(slots=True)
class VibeGroupingResult:
    groups: list[VibeGroup]
    ungrouped_paths: list[str]
    errors: list[ScanError]
    config_snapshot: dict[str, Any]
    model_fingerprint: str
    provider: str
    cache_hits: int = 0
    cache_misses: int = 0
    stage_timings: dict[str, float] = field(default_factory=dict)
    used_fallback_embedder: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ImageGroupingProcessor(Protocol):
    def group(
        self,
        image_paths: Sequence[str | Path],
        *,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> VibeGroupingResult:
        ...
