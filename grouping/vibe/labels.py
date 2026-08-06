from __future__ import annotations

from collections import Counter

import numpy as np

from grouping.models import VibeImageFeatures

from .prototypes import ScenePrototypeTable


def build_group_label(
    features: list[VibeImageFeatures],
    people_names: tuple[str, ...],
    *,
    prototype_table: ScenePrototypeTable | None = None,
) -> str:
    if prototype_table is not None:
        prototype_label = _prototype_label(features, people_names, prototype_table=prototype_table)
        if prototype_label is not None:
            return prototype_label

    people_part = _people_label(people_names)
    descriptor = _fallback_descriptor_label(features)
    if people_part and descriptor:
        return f"{people_part} · {descriptor}"
    if people_part:
        return people_part
    if descriptor:
        return descriptor
    return "Untitled scene"


def _prototype_label(
    features: list[VibeImageFeatures],
    people_names: tuple[str, ...],
    *,
    prototype_table: ScenePrototypeTable,
) -> str | None:
    if not features:
        return None

    action_scores = _mean_scores(features, attr_name="action_scores")
    scene_scores = _mean_scores(features, attr_name="scene_scores")
    shot_scores = _mean_scores(features, attr_name="shot_type_scores")
    action_matches = prototype_table.top_matches(action_scores, category="action", limit=2)
    scene_matches = prototype_table.top_matches(scene_scores, category="scene", limit=1)
    shot_matches = prototype_table.top_matches(shot_scores, category="shot", limit=1)

    if action_matches:
        top_action = action_matches[0]
        tags = set(top_action.tags)
        if "audience" in tags or "reaction" in tags:
            if len(action_matches) > 1 and "speech" in action_matches[1].tags:
                return f"{action_matches[1].label} · Audience Reaction"
            return "Audience Reaction" if top_action.label == "Audience Listening" else top_action.label
        if "portrait" in tags and people_names:
            people_part = _people_label(people_names)
            if people_part and top_action.label not in {"Portrait", "Solo Portrait"}:
                return f"{people_part} · {top_action.label}"
        return top_action.label

    if scene_matches and shot_matches:
        return f"{scene_matches[0].label} · {shot_matches[0].label}"
    if scene_matches:
        return scene_matches[0].label
    if shot_matches:
        return shot_matches[0].label

    people_part = _people_label(people_names)
    if people_part:
        return f"{people_part} · Scene"
    return None


def _people_label(people_names: tuple[str, ...]) -> str | None:
    if not people_names:
        return None
    if len(people_names) == 1:
        return people_names[0]
    if len(people_names) == 2:
        return f"{people_names[0]} & {people_names[1]}"
    return f"{people_names[0]} & {people_names[1]} +{len(people_names) - 2}"


def _fallback_descriptor_label(features: list[VibeImageFeatures]) -> str:
    if not features:
        return "Untitled scene"

    participant_modes = Counter(str(feature.metadata.get("participant_mode", "none")) for feature in features)
    dominant_mode = participant_modes.most_common(1)[0][0]
    if dominant_mode == "couple":
        return "Couple Portrait"
    if dominant_mode in {"family_group", "small_group", "crowd"}:
        return "Group Portrait"
    if dominant_mode == "solo":
        return "Portrait"
    if any(feature.face_count > 0 for feature in features):
        return "Portrait Sequence"
    return "Untitled scene"


def _mean_scores(
    features: list[VibeImageFeatures],
    *,
    attr_name: str,
) -> np.ndarray | None:
    vectors = [
        np.asarray(getattr(feature, attr_name), dtype=np.float32)
        for feature in features
        if getattr(feature, attr_name) is not None
    ]
    if not vectors:
        return None
    return np.mean(np.stack(vectors, axis=0), axis=0)
