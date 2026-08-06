from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from grouping.models import VibeImageFeatures

from .config import VibeGroupingConfig
from .prototypes import ScenePrototypeTable


_ACTION_FAMILY_ORDER = (
    "rings",
    "kiss",
    "speech",
    "toast",
    "applause",
    "audience",
    "dance",
    "cake",
    "makeup",
    "dressing",
    "portrait",
    "walking",
    "procession",
    "hug",
    "talking",
    "laughing",
    "eating",
    "flowers",
    "bouquet",
    "play",
    "entering",
    "leaving",
)

_ACTION_CONFLICT_FAMILIES = {
    frozenset(("speech", "audience")),
    frozenset(("speech", "applause")),
    frozenset(("rings", "kiss")),
    frozenset(("walking", "rings")),
    frozenset(("walking", "kiss")),
    frozenset(("portrait", "walking")),
    frozenset(("cake", "eating")),
    frozenset(("makeup", "dressing")),
    frozenset(("dressing", "portrait")),
    frozenset(("makeup", "portrait")),
    frozenset(("walking", "portrait")),
}

_ACTION_COMPATIBLE_FAMILIES = {
    frozenset(("speech", "toast")),
    frozenset(("portrait", "flowers")),
    frozenset(("portrait", "hug")),
    frozenset(("walking", "procession")),
}

_SHOT_MODE_COMPATIBILITY = {
    ("main_action", "portrait"): 0.75,
    ("portrait", "main_action"): 0.75,
    ("detail", "main_action"): 0.65,
    ("main_action", "detail"): 0.65,
    ("audience", "reaction"): 0.60,
    ("reaction", "audience"): 0.60,
    ("portrait", "detail"): 0.45,
    ("detail", "portrait"): 0.45,
}


def temporal_similarity(delta_seconds: float, scale_seconds: float) -> float:
    return math.exp(-max(delta_seconds, 0.0) / max(scale_seconds, 1.0))


def person_set_similarity(
    people_a: tuple[str, ...],
    people_b: tuple[str, ...],
) -> float:
    set_a = set(people_a)
    set_b = set(people_b)
    if not set_a and not set_b:
        return 0.5
    if not set_a or not set_b:
        return 0.35
    return len(set_a & set_b) / len(set_a | set_b)


def optional_cosine_similarity(
    vector_a: np.ndarray | None,
    vector_b: np.ndarray | None,
) -> float:
    if vector_a is None or vector_b is None:
        return 0.5
    if vector_a.shape != vector_b.shape:
        return 0.5
    return float(np.clip(np.dot(vector_a, vector_b), 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class ActionProfile:
    top_key: str | None
    top_family: str | None
    top_score: float
    second_score: float
    margin: float
    confident: bool
    strongly_confident: bool
    top_keys: tuple[str, ...]


class ShotConfidence(StrEnum):
    UNCERTAIN = "uncertain"
    CONFIDENT = "confident"
    STRONG = "strong"


class ParticipantConflictStrength(StrEnum):
    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"


@dataclass(frozen=True, slots=True)
class ShotProfile:
    top_key: str | None
    mode: str
    top_score: float
    second_score: float
    margin: float
    confidence: ShotConfidence
    confident: bool
    strongly_confident: bool
    top_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenePairComponents:
    semantic: float
    action: float
    people: float
    layout: float
    subject_scene: float
    background: float
    time: float
    composition: float
    color: float
    action_conflict_penalty: float
    participant_mode_conflict_penalty: float
    shot_mode_conflict_penalty: float
    main_vs_reaction_penalty: float
    temporal_bridge_penalty: float
    transition_penalty: float
    action_hard_conflict: bool
    action_soft_conflict: bool
    participant_mode_hard_conflict: bool
    participant_mode_soft_conflict: bool
    participant_conflict_strength: str
    shot_mode_conflict: bool
    main_vs_reaction_conflict: bool
    main_vs_reaction_confident: bool
    action_confidence_mean: float
    action_margin_mean: float
    action_reliable: bool
    left_shot_margin: float
    right_shot_margin: float
    left_shot_confidence: str
    right_shot_confidence: str
    combined: float


@dataclass(slots=True)
class CombinedSimilarityComputer:
    config: VibeGroupingConfig
    prototype_table: ScenePrototypeTable | None = None
    _action_profiles: dict[str, ActionProfile] = field(default_factory=dict, init=False, repr=False)
    _shot_profiles: dict[str, ShotProfile] = field(default_factory=dict, init=False, repr=False)

    def semantic_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
    ) -> float:
        return float(np.clip(np.dot(left.semantic_embedding, right.semantic_embedding), 0.0, 1.0))

    def action_profile(
        self,
        feature: VibeImageFeatures,
    ) -> ActionProfile:
        cached = self._action_profiles.get(feature.image_path)
        if cached is not None:
            return cached

        if self.prototype_table is None or feature.action_scores is None:
            profile = ActionProfile(
                top_key=None,
                top_family=None,
                top_score=0.0,
                second_score=0.0,
                margin=0.0,
                confident=False,
                strongly_confident=False,
                top_keys=(),
            )
            self._action_profiles[feature.image_path] = profile
            return profile

        matches = self.prototype_table.top_matches(feature.action_scores, category="action", limit=3)
        top_score = 0.0 if not matches else float(matches[0].score)
        second_score = 0.0 if len(matches) < 2 else float(matches[1].score)
        margin = max(0.0, top_score - second_score)
        profile = ActionProfile(
            top_key=None if not matches else matches[0].key,
            top_family=None if not matches else _action_family(matches[0].tags),
            top_score=top_score,
            second_score=second_score,
            margin=margin,
            confident=bool(matches) and margin >= self.config.minimum_action_margin,
            strongly_confident=bool(matches) and margin >= self.config.strong_action_margin,
            top_keys=tuple(match.key for match in matches),
        )
        self._action_profiles[feature.image_path] = profile
        return profile

    def action_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
    ) -> float:
        left_profile = self.action_profile(left)
        right_profile = self.action_profile(right)
        if not left_profile.confident or not right_profile.confident:
            return 0.5
        if left.action_scores is None or right.action_scores is None:
            return 0.5

        centered_similarity = _centered_vector_similarity(left.action_scores, right.action_scores)
        overlap = _top_k_overlap(left_profile.top_keys, right_profile.top_keys)
        family_similarity = _action_family_similarity(left_profile.top_family, right_profile.top_family)
        score = (
            (0.45 * centered_similarity)
            + (0.30 * overlap)
            + (0.25 * family_similarity)
        )
        return float(np.clip(score, 0.0, 1.0))

    def shot_profile(
        self,
        feature: VibeImageFeatures,
    ) -> ShotProfile:
        cached = self._shot_profiles.get(feature.image_path)
        if cached is not None:
            return cached

        if self.prototype_table is None or feature.shot_type_scores is None:
            profile = ShotProfile(
                top_key=None,
                mode="main_action",
                top_score=0.0,
                second_score=0.0,
                margin=0.0,
                confidence=ShotConfidence.UNCERTAIN,
                confident=False,
                strongly_confident=False,
                top_keys=(),
            )
            self._shot_profiles[feature.image_path] = profile
            return profile

        matches = self.prototype_table.top_matches(feature.shot_type_scores, category="shot", limit=3)
        top_score = 0.0 if not matches else float(matches[0].score)
        second_score = 0.0 if len(matches) < 2 else float(matches[1].score)
        margin = max(0.0, top_score - second_score)
        if margin < self.config.minimum_shot_margin:
            confidence = ShotConfidence.UNCERTAIN
        elif margin < self.config.strong_shot_margin:
            confidence = ShotConfidence.CONFIDENT
        else:
            confidence = ShotConfidence.STRONG
        mode = "main_action"
        if matches:
            top_tags = set(matches[0].tags)
            for name in ("main_action", "reaction", "audience", "detail", "portrait", "candid"):
                if name in top_tags:
                    mode = name
                    break
        if confidence is ShotConfidence.UNCERTAIN and _metadata_str(feature, "shot_scale_category", "wide") == "close":
            mode = "portrait"
        profile = ShotProfile(
            top_key=None if not matches else matches[0].key,
            mode=mode,
            top_score=top_score,
            second_score=second_score,
            margin=margin,
            confidence=confidence,
            confident=confidence is not ShotConfidence.UNCERTAIN,
            strongly_confident=confidence is ShotConfidence.STRONG,
            top_keys=tuple(match.key for match in matches),
        )
        self._shot_profiles[feature.image_path] = profile
        return profile

    def people_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
    ) -> float:
        return person_set_similarity(left.recognized_person_ids, right.recognized_person_ids)

    def layout_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
    ) -> float:
        return optional_cosine_similarity(left.face_layout, right.face_layout)

    def subject_scene_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
    ) -> float:
        return optional_cosine_similarity(left.subject_scene_embedding, right.subject_scene_embedding)

    def background_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
    ) -> float:
        return optional_cosine_similarity(left.background_embedding, right.background_embedding)

    def composition_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
    ) -> float:
        vector_similarity = optional_cosine_similarity(left.composition_features, right.composition_features)
        orientation_similarity = _orientation_similarity(
            _metadata_str(left, "orientation", "square"),
            _metadata_str(right, "orientation", "square"),
        )
        shot_scale_similarity = _shot_scale_similarity(
            _shot_scale_category(left),
            _shot_scale_category(right),
        )
        face_bucket_similarity = _face_count_bucket_similarity(
            _metadata_str(left, "face_count_bucket", "zero"),
            _metadata_str(right, "face_count_bucket", "zero"),
        )
        participant_similarity = _participant_mode_similarity(
            self._participant_mode(left),
            self._participant_mode(right),
        )
        left_shot_profile = self.shot_profile(left)
        right_shot_profile = self.shot_profile(right)
        shot_mode_similarity = _shot_mode_similarity(left_shot_profile, right_shot_profile)
        centroid_similarity = _geometry_similarity(
            _metadata_float(left, "subject_centroid_x", 0.5),
            _metadata_float(left, "subject_centroid_y", 0.5),
            _metadata_float(right, "subject_centroid_x", 0.5),
            _metadata_float(right, "subject_centroid_y", 0.5),
            scale=3.0,
        )
        spread_similarity = _spread_similarity(left, right)
        subject_scale_similarity = _scalar_similarity(
            _metadata_float(left, "primary_subject_scale", _metadata_float(left, "largest_face_area_ratio", 0.0)),
            _metadata_float(right, "primary_subject_scale", _metadata_float(right, "largest_face_area_ratio", 0.0)),
            scale=8.0,
        )

        score = (
            (0.14 * vector_similarity)
            + (0.08 * orientation_similarity)
            + (0.14 * shot_scale_similarity)
            + (0.14 * face_bucket_similarity)
            + (0.18 * participant_similarity)
            + (0.10 * centroid_similarity)
            + (0.08 * spread_similarity)
            + (0.08 * subject_scale_similarity)
            + (0.06 * shot_mode_similarity)
        )
        return float(np.clip(score, 0.0, 1.0))

    def color_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
    ) -> float:
        return optional_cosine_similarity(left.color_features, right.color_features)

    def pair_components(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
        *,
        hard_boundary: bool = False,
        soft_boundary: bool = False,
    ) -> ScenePairComponents:
        semantic = self.semantic_similarity(left, right)
        action = self.action_similarity(left, right)
        people = self.people_similarity(left, right)
        layout = self.layout_similarity(left, right)
        subject_scene = self.subject_scene_similarity(left, right)
        background = self.background_similarity(left, right)
        time_score, delta_seconds = self._time_similarity(left, right)
        composition = self.composition_similarity(left, right)
        color = self.color_similarity(left, right)
        left_action = self.action_profile(left)
        right_action = self.action_profile(right)
        left_shot_profile = self.shot_profile(left)
        right_shot_profile = self.shot_profile(right)
        action_reliable = (
            left.action_scores is not None
            and right.action_scores is not None
            and left_action.confident
            and right_action.confident
        )

        available_names = {
            "semantic",
            "time",
            "people",
            "composition",
        }
        if action_reliable:
            available_names.add("action")
        if left.face_layout is not None and right.face_layout is not None:
            available_names.add("layout")
        if left.subject_scene_embedding is not None and right.subject_scene_embedding is not None:
            available_names.add("subject_scene")
        if left.background_embedding is not None and right.background_embedding is not None:
            available_names.add("background")
        if left.color_features is not None and right.color_features is not None:
            available_names.add("color")

        weights = self.config.normalized_weights(available_names)
        component_values = {
            "semantic": semantic,
            "action": action,
            "people": people,
            "layout": layout,
            "subject_scene": subject_scene,
            "background": background,
            "time": time_score,
            "composition": composition,
            "color": color,
        }
        score = 0.0
        for name, weight in weights.items():
            score += weight * component_values[name]

        action_soft_conflict = _action_conflict(left_action, right_action)
        action_hard_conflict = (
            action_soft_conflict
            and left_action.strongly_confident
            and right_action.strongly_confident
        )
        if action_hard_conflict:
            action_penalty = self.config.strong_action_conflict_penalty
        elif action_soft_conflict:
            action_penalty = self.config.action_conflict_penalty
        else:
            action_penalty = 0.0

        left_mode = self._participant_mode(left)
        right_mode = self._participant_mode(right)
        participant_conflict_strength = _participant_conflict_strength(
            left,
            right,
            left_mode,
            right_mode,
            layout_similarity=layout,
            subject_scene_similarity=subject_scene,
        )
        participant_mode_hard_conflict, participant_mode_soft_conflict = _participant_mode_conflicts(
            left_mode,
            right_mode,
            participant_conflict_strength=participant_conflict_strength,
        )
        if participant_conflict_strength is ParticipantConflictStrength.STRONG:
            participant_penalty = self.config.participant_mode_strong_penalty
        elif participant_conflict_strength is ParticipantConflictStrength.WEAK:
            participant_penalty = self.config.participant_mode_weak_penalty
        else:
            participant_penalty = 0.0

        left_shot_mode = left_shot_profile.mode
        right_shot_mode = right_shot_profile.mode
        shot_mode_conflict = _shot_mode_conflict(left_shot_mode, right_shot_mode)
        shot_conflict_confident = left_shot_profile.confident and right_shot_profile.confident
        shot_mode_penalty = (
            self.config.shot_mode_conflict_penalty
            if shot_mode_conflict and shot_conflict_confident
            else 0.0
        )
        main_vs_reaction_conflict = _main_vs_reaction_conflict(
            left_shot_profile,
            right_shot_profile,
            left_mode,
            right_mode,
        )
        if main_vs_reaction_conflict and left_shot_profile.strongly_confident and right_shot_profile.strongly_confident:
            main_vs_reaction_penalty = self.config.main_vs_reaction_strong_penalty
        elif main_vs_reaction_conflict and shot_conflict_confident:
            main_vs_reaction_penalty = self.config.main_vs_reaction_supported_penalty
        elif main_vs_reaction_conflict:
            main_vs_reaction_penalty = self.config.main_vs_reaction_weak_penalty
        else:
            main_vs_reaction_penalty = 0.0

        temporal_bridge_penalty = 0.0
        if (
            delta_seconds is not None
            and delta_seconds > self.config.maximum_soft_time_gap_seconds
            and semantic < self.config.strong_pair_similarity
            and action < 0.78
            and subject_scene < self.config.cross_soft_boundary_similarity
        ):
            temporal_bridge_penalty = self.config.temporal_bridge_penalty

        transition_penalty = 0.0
        if hard_boundary:
            transition_penalty = self.config.hard_transition_penalty
        elif soft_boundary:
            transition_penalty = self.config.soft_transition_penalty

        if (
            delta_seconds is not None
            and delta_seconds > self.config.maximum_hard_time_gap_seconds
            and semantic < self.config.strong_pair_similarity
            and subject_scene < self.config.long_range_subject_scene_threshold
            and people < 0.75
        ):
            return ScenePairComponents(
                semantic=semantic,
                action=action,
                people=people,
                layout=layout,
                subject_scene=subject_scene,
                background=background,
                time=time_score,
                composition=composition,
                color=color,
                action_conflict_penalty=action_penalty,
                participant_mode_conflict_penalty=participant_penalty,
                shot_mode_conflict_penalty=shot_mode_penalty,
                main_vs_reaction_penalty=main_vs_reaction_penalty,
                temporal_bridge_penalty=temporal_bridge_penalty,
                transition_penalty=transition_penalty,
                action_hard_conflict=action_hard_conflict,
                action_soft_conflict=action_soft_conflict,
                participant_mode_hard_conflict=participant_mode_hard_conflict,
                participant_mode_soft_conflict=participant_mode_soft_conflict,
                participant_conflict_strength=participant_conflict_strength.value,
                shot_mode_conflict=shot_mode_conflict,
                main_vs_reaction_conflict=main_vs_reaction_conflict,
                main_vs_reaction_confident=shot_conflict_confident,
                action_confidence_mean=_mean([1.0 if left_action.confident else 0.0, 1.0 if right_action.confident else 0.0]),
                action_margin_mean=_mean([left_action.margin, right_action.margin]),
                action_reliable=action_reliable,
                left_shot_margin=left_shot_profile.margin,
                right_shot_margin=right_shot_profile.margin,
                left_shot_confidence=left_shot_profile.confidence.value,
                right_shot_confidence=right_shot_profile.confidence.value,
                combined=0.0,
            )

        combined = (
            score
            - action_penalty
            - participant_penalty
            - shot_mode_penalty
            - main_vs_reaction_penalty
            - temporal_bridge_penalty
            - transition_penalty
        )
        return ScenePairComponents(
            semantic=semantic,
            action=action,
            people=people,
            layout=layout,
            subject_scene=subject_scene,
            background=background,
            time=time_score,
            composition=composition,
            color=color,
            action_conflict_penalty=action_penalty,
            participant_mode_conflict_penalty=participant_penalty,
            shot_mode_conflict_penalty=shot_mode_penalty,
            main_vs_reaction_penalty=main_vs_reaction_penalty,
            temporal_bridge_penalty=temporal_bridge_penalty,
            transition_penalty=transition_penalty,
            action_hard_conflict=action_hard_conflict,
            action_soft_conflict=action_soft_conflict,
            participant_mode_hard_conflict=participant_mode_hard_conflict,
            participant_mode_soft_conflict=participant_mode_soft_conflict,
            participant_conflict_strength=participant_conflict_strength.value,
            shot_mode_conflict=shot_mode_conflict,
            main_vs_reaction_conflict=main_vs_reaction_conflict,
            main_vs_reaction_confident=shot_conflict_confident,
            action_confidence_mean=_mean([1.0 if left_action.confident else 0.0, 1.0 if right_action.confident else 0.0]),
            action_margin_mean=_mean([left_action.margin, right_action.margin]),
            action_reliable=action_reliable,
            left_shot_margin=left_shot_profile.margin,
            right_shot_margin=right_shot_profile.margin,
            left_shot_confidence=left_shot_profile.confidence.value,
            right_shot_confidence=right_shot_profile.confidence.value,
            combined=float(np.clip(combined, 0.0, 1.0)),
        )

    def pair_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
        *,
        hard_boundary: bool = False,
        soft_boundary: bool = False,
    ) -> float:
        return self.pair_components(
            left,
            right,
            hard_boundary=hard_boundary,
            soft_boundary=soft_boundary,
        ).combined

    def _time_similarity(
        self,
        left: VibeImageFeatures,
        right: VibeImageFeatures,
    ) -> tuple[float, float | None]:
        if left.capture_timestamp is None or right.capture_timestamp is None:
            return 0.5, None
        delta_seconds = abs(left.capture_timestamp - right.capture_timestamp)
        return (
            temporal_similarity(delta_seconds, self.config.maximum_soft_time_gap_seconds),
            delta_seconds,
        )

    def _participant_mode(
        self,
        feature: VibeImageFeatures,
    ) -> str:
        action_profile = self.action_profile(feature)
        shot_profile = self.shot_profile(feature)
        if shot_profile.mode == "audience" and shot_profile.confident:
            return "audience"
        if action_profile.top_family in {"speech", "toast"} and feature.face_count <= 2:
            return "speaker"
        metadata_mode = _metadata_str(feature, "participant_mode", "")
        if metadata_mode == "crowd":
            metadata_mode = "large_group"
        if metadata_mode == "family_group":
            return "small_group"
        if metadata_mode:
            return metadata_mode
        if feature.face_count <= 0:
            return "none"
        if feature.face_count == 1:
            return "solo"
        if feature.face_count == 2:
            return "couple"
        if feature.face_count <= 4:
            return "small_group"
        return "large_group"

    def _shot_mode(
        self,
        feature: VibeImageFeatures,
    ) -> str:
        return self.shot_profile(feature).mode


def _action_family(tags: tuple[str, ...]) -> str:
    tag_set = set(tags)
    for name in _ACTION_FAMILY_ORDER:
        if name in tag_set:
            return name
    return next(iter(tag_set), "generic")


def _action_family_similarity(
    left_family: str | None,
    right_family: str | None,
) -> float:
    if left_family is None or right_family is None:
        return 0.5
    if left_family == right_family:
        return 1.0
    if frozenset((left_family, right_family)) in _ACTION_COMPATIBLE_FAMILIES:
        return 0.65
    if frozenset((left_family, right_family)) in _ACTION_CONFLICT_FAMILIES:
        return 0.15
    return 0.35


def _action_conflict(
    left: ActionProfile,
    right: ActionProfile,
) -> bool:
    if not left.confident or not right.confident:
        return False
    if left.top_key is None or right.top_key is None:
        return False
    if left.top_key == right.top_key:
        return False
    if left.top_family is None or right.top_family is None:
        return False
    if left.top_family == right.top_family:
        return False
    return frozenset((left.top_family, right.top_family)) in _ACTION_CONFLICT_FAMILIES


def _participant_mode_conflicts(
    left_mode: str,
    right_mode: str,
    *,
    participant_conflict_strength: ParticipantConflictStrength,
) -> tuple[bool, bool]:
    if left_mode == right_mode or participant_conflict_strength is ParticipantConflictStrength.NONE:
        return False, False
    if participant_conflict_strength is ParticipantConflictStrength.STRONG:
        return True, False
    return False, True


def _shot_mode_conflict(
    left_mode: str,
    right_mode: str,
) -> bool:
    if left_mode == right_mode:
        return False
    if {left_mode, right_mode} == {"main_action", "reaction"}:
        return True
    if {left_mode, right_mode} == {"main_action", "audience"}:
        return True
    if {left_mode, right_mode} == {"portrait", "detail"}:
        return True
    return False


def _main_vs_reaction_conflict(
    left_shot_profile: ShotProfile,
    right_shot_profile: ShotProfile,
    left_participant_mode: str,
    right_participant_mode: str,
) -> bool:
    if not left_shot_profile.confident or not right_shot_profile.confident:
        return False
    left_shot_mode = left_shot_profile.mode
    right_shot_mode = right_shot_profile.mode
    if {left_shot_mode, right_shot_mode} == {"main_action", "reaction"}:
        return True
    if {left_shot_mode, right_shot_mode} == {"main_action", "audience"}:
        return True
    return {left_participant_mode, right_participant_mode} == {"speaker", "audience"}


def _participant_conflict_strength(
    left: VibeImageFeatures,
    right: VibeImageFeatures,
    left_mode: str,
    right_mode: str,
    *,
    layout_similarity: float,
    subject_scene_similarity: float,
) -> ParticipantConflictStrength:
    if left_mode == right_mode:
        return ParticipantConflictStrength.NONE
    normalized = frozenset((left_mode, right_mode))
    strong_pairs = {
        frozenset(("couple", "audience")),
        frozenset(("solo", "large_group")),
        frozenset(("solo", "audience")),
        frozenset(("speaker", "audience")),
        frozenset(("speaker", "large_group")),
        frozenset(("none", "large_group")),
        frozenset(("none", "audience")),
    }
    weak_pairs = {
        frozenset(("solo", "couple")),
        frozenset(("couple", "small_group")),
        frozenset(("small_group", "large_group")),
        frozenset(("solo", "small_group")),
        frozenset(("couple", "large_group")),
        frozenset(("none", "solo")),
        frozenset(("none", "couple")),
    }
    strength = (
        ParticipantConflictStrength.STRONG
        if normalized in strong_pairs
        else ParticipantConflictStrength.WEAK
        if normalized in weak_pairs or left_mode != right_mode
        else ParticipantConflictStrength.NONE
    )
    face_delta = abs(int(left.face_count) - int(right.face_count))
    faces_tiny = max(float(left.face_area_ratio or 0.0), float(right.face_area_ratio or 0.0)) < 0.05
    if strength is ParticipantConflictStrength.STRONG and (
        face_delta <= 1
        or (layout_similarity >= 0.88 and subject_scene_similarity >= 0.88)
        or faces_tiny
    ):
        return ParticipantConflictStrength.WEAK
    return strength


def _centered_vector_similarity(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
) -> float:
    left = np.asarray(left_scores, dtype=np.float32)
    right = np.asarray(right_scores, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 1:
        return 0.5
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.5
    cosine = float(np.dot(left_centered / left_norm, right_centered / right_norm))
    return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))


def _top_k_overlap(
    left_keys: tuple[str, ...],
    right_keys: tuple[str, ...],
    *,
    limit: int = 3,
) -> float:
    if not left_keys or not right_keys:
        return 0.5
    left = set(left_keys[:limit])
    right = set(right_keys[:limit])
    if not left and not right:
        return 0.5
    return len(left & right) / max(len(left | right), 1)


def _orientation_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if "square" in {left, right}:
        return 0.55
    return 0.35


def _shot_scale_similarity(left: str, right: str) -> float:
    order = {"detail": 0, "close": 1, "medium": 2, "full_body": 3, "wide": 4}
    if left == right:
        return 1.0
    distance = abs(order.get(left, 4) - order.get(right, 4))
    if distance == 1:
        return 0.78
    if distance == 2:
        return 0.52
    return 0.22


def _face_count_bucket_similarity(left: str, right: str) -> float:
    order = {"zero": 0, "one": 1, "couple": 2, "small_group": 3, "crowd": 4}
    if left == right:
        return 1.0
    distance = abs(order.get(left, 4) - order.get(right, 4))
    if distance == 1:
        return 0.72
    if distance == 2:
        return 0.38
    return 0.12


def _participant_mode_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if frozenset((left, right)) in {
        frozenset(("solo", "couple")),
        frozenset(("small_group", "large_group")),
        frozenset(("speaker", "solo")),
    }:
        return 0.48
    if frozenset((left, right)) in {
        frozenset(("couple", "small_group")),
        frozenset(("speaker", "audience")),
        frozenset(("solo", "large_group")),
        frozenset(("couple", "large_group")),
    }:
        return 0.12
    return 0.30


def _shot_mode_similarity(left: ShotProfile, right: ShotProfile) -> float:
    if not left.confident or not right.confident:
        return 0.5
    if left.mode == right.mode:
        return 1.0
    return _SHOT_MODE_COMPATIBILITY.get((left.mode, right.mode), 0.20)


def _geometry_similarity(
    left_x: float,
    left_y: float,
    right_x: float,
    right_y: float,
    *,
    scale: float,
) -> float:
    distance = abs(left_x - right_x) + abs(left_y - right_y)
    return float(np.clip(math.exp(-scale * distance), 0.0, 1.0))


def _spread_similarity(left: VibeImageFeatures, right: VibeImageFeatures) -> float:
    horizontal = _scalar_similarity(
        _metadata_float(left, "subject_horizontal_spread", _metadata_float(left, "face_horizontal_spread", 0.0)),
        _metadata_float(right, "subject_horizontal_spread", _metadata_float(right, "face_horizontal_spread", 0.0)),
        scale=6.0,
    )
    vertical = _scalar_similarity(
        _metadata_float(left, "subject_vertical_spread", _metadata_float(left, "face_vertical_spread", 0.0)),
        _metadata_float(right, "subject_vertical_spread", _metadata_float(right, "face_vertical_spread", 0.0)),
        scale=6.0,
    )
    return (horizontal + vertical) / 2.0


def _scalar_similarity(left: float, right: float, *, scale: float) -> float:
    return float(np.clip(math.exp(-scale * abs(left - right)), 0.0, 1.0))


def _metadata_str(feature: VibeImageFeatures, key: str, default: str) -> str:
    value = feature.metadata.get(key, default)
    return str(value)


def _metadata_float(feature: VibeImageFeatures, key: str, default: float) -> float:
    value = feature.metadata.get(key, default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _shot_scale_category(feature: VibeImageFeatures) -> str:
    scale = _metadata_str(feature, "shot_scale_category", "wide") if feature.metadata else "wide"
    if scale == "wide" and feature.face_count <= 0 and _metadata_str(feature, "orientation", "landscape") == "portrait":
        return "detail"
    return scale


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
