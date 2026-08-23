from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from duplicate_detector.models import PhotoInfo

from .config import AutomaticScanConfig
from .models import (
    AutomaticDuplicateGroup,
    DuplicateRankingBreakdown,
    PipelineImageState,
)


def select_duplicate_keepers(
    grouped_photos: list[list[PhotoInfo]],
    *,
    image_states: dict[str, PipelineImageState],
    config: AutomaticScanConfig,
) -> list[AutomaticDuplicateGroup]:
    results: list[AutomaticDuplicateGroup] = []
    keepers_per_group = config.duplicate_selection.keepers_per_group

    for photos in grouped_photos:
        ordered_paths = [str(photo.path) for photo in photos]
        group_id = _group_id(ordered_paths)
        ranking = {
            str(photo.path): _ranking_breakdown(
                photo,
                image_states.get(str(photo.path)),
                photos,
            )
            for photo in photos
        }
        sorted_paths = sorted(
            ordered_paths,
            key=lambda path: _ranking_sort_key(ranking[path]),
            reverse=True,
        )
        keeper_paths = sorted_paths[: min(keepers_per_group, len(sorted_paths))]
        excluded_paths = sorted_paths[min(keepers_per_group, len(sorted_paths)) :]
        results.append(
            AutomaticDuplicateGroup(
                group_id=group_id,
                image_paths=sorted_paths,
                keeper_paths=keeper_paths,
                excluded_paths=excluded_paths,
                ranking=ranking,
            )
        )
    return results


def _ranking_sort_key(breakdown: DuplicateRankingBreakdown) -> tuple[object, ...]:
    timestamp = float("-inf") if breakdown.capture_timestamp is None else breakdown.capture_timestamp
    return (
        breakdown.acceptable_primary_face_count,
        breakdown.worst_primary_score,
        breakdown.weighted_primary_average,
        breakdown.eye_open_score,
        breakdown.pose_visibility_score,
        breakdown.image_sharpness_score,
        breakdown.exposure_contrast_score,
        breakdown.overall_quality_score,
        breakdown.group_visual_centrality,
        -timestamp,
        breakdown.path_key,
    )


def _ranking_breakdown(
    photo: PhotoInfo,
    image_state: PipelineImageState | None,
    group_members: list[PhotoInfo],
) -> DuplicateRankingBreakdown:
    face_quality = None if image_state is None else image_state.face_quality_assessment
    acceptable_primary_face_count = (
        0
        if face_quality is None or not face_quality.is_people_picture
        else face_quality.acceptable_primary_face_count
    )
    worst_primary_score = 0.0 if face_quality is None or face_quality.worst_primary_score is None else face_quality.worst_primary_score
    weighted_primary_average = (
        0.0
        if face_quality is None or face_quality.weighted_primary_average is None
        else face_quality.weighted_primary_average
    )
    eye_open_score = max(0.0, 1.0 - (0.5 if face_quality is not None and face_quality.closed_eye_primary_count else 0.0))

    image_sharpness_score = 0.0
    exposure_contrast_score = 0.0
    overall_quality_score = 0.0
    pose_visibility_score = 0.0
    explanation: list[str] = []

    if image_state is not None and image_state.face_quality_assessment is not None:
        explanation.append("Sharper important faces" if weighted_primary_average >= 0.5 else "Important faces weaker")

    centrality = _group_visual_centrality(photo, group_members)
    explanation.append("Closer to the duplicate group center" if centrality >= 0.5 else "Further from the duplicate group center")

    return DuplicateRankingBreakdown(
        acceptable_primary_face_count=acceptable_primary_face_count,
        worst_primary_score=worst_primary_score,
        weighted_primary_average=weighted_primary_average,
        eye_open_score=eye_open_score,
        pose_visibility_score=pose_visibility_score,
        image_sharpness_score=image_sharpness_score,
        exposure_contrast_score=exposure_contrast_score,
        overall_quality_score=overall_quality_score,
        group_visual_centrality=centrality,
        capture_timestamp=None if image_state is None else image_state.capture_timestamp,
        path_key=str(Path(photo.path).resolve()),
        explanation=tuple(explanation),
    )


def _group_visual_centrality(photo: PhotoInfo, group_members: list[PhotoInfo]) -> float:
    if len(group_members) <= 1:
        return 1.0

    distances: list[float] = []
    for other in group_members:
        if other.path == photo.path:
            continue
        phash_distance = bin(photo.phash ^ other.phash).count("1")
        dhash_distance = bin(photo.dhash ^ other.dhash).count("1")
        distances.append((phash_distance + dhash_distance) / 64.0)
    if not distances:
        return 1.0
    average_distance = sum(distances) / len(distances)
    return max(0.0, min(1.0, 1.0 - average_distance))


def _group_id(paths: list[str]) -> str:
    digest = hashlib.sha1()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
    return digest.hexdigest()[:16]
