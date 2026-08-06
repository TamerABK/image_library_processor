from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from grouping.models import VibeImageFeatures

from .config import VibeGroupingConfig
from .similarity import CombinedSimilarityComputer, ParticipantConflictStrength, person_set_similarity


class BoundaryReliability(StrEnum):
    NONE = "none"
    WEAK = "weak"
    SUPPORTED = "supported"
    HARD = "hard"


@dataclass(frozen=True, slots=True)
class BoundaryStrength:
    crossed_hard_boundary: bool
    crossed_soft_boundary: bool
    strongest_transition: float
    strongest_soft_transition: float
    strongest_hard_transition: float


@dataclass(frozen=True, slots=True)
class SceneTransitionScore:
    left_image: str
    right_image: str
    semantic_change: float
    action_change: float
    people_change: float
    layout_change: float
    subject_scene_change: float
    background_change: float
    composition_change: float
    temporal_gap: float
    combined_transition: float
    boundary_reliability: str
    is_hard_boundary: bool
    is_soft_boundary: bool
    continuity_override_applied: bool
    continuity_score: float
    visual_support_score: float
    continuity_reasons: tuple[str, ...]
    participant_conflict_strength: str
    shot_conflict_confident: bool
    left_shot_margin: float
    right_shot_margin: float
    left_shot_confidence: str
    right_shot_confidence: str
    recovery_crossable: bool
    hard_reasons: tuple[str, ...]
    soft_reasons: tuple[str, ...]
    accepted_hard_boundary: bool = False
    accepted_soft_boundary: bool = False


@dataclass(frozen=True, slots=True)
class SceneSegment:
    member_indices: tuple[int, ...]
    boundary_before: SceneTransitionScore | None
    boundary_after: SceneTransitionScore | None


@dataclass(frozen=True, slots=True)
class TemporalSession:
    ordered_features: tuple[VibeImageFeatures, ...]
    transitions: tuple[SceneTransitionScore, ...]
    scene_segments: tuple[SceneSegment, ...]
    recovery_events: tuple[dict[str, object], ...] = ()
    tiny_segment_merge_events: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalSegmentation:
    sessions: list[TemporalSession]
    untimed: list[VibeImageFeatures]


def segment_by_time(
    features: list[VibeImageFeatures],
    *,
    config: VibeGroupingConfig,
    similarity: CombinedSimilarityComputer,
) -> TemporalSegmentation:
    timed = sorted(
        [feature for feature in features if feature.capture_timestamp is not None],
        key=lambda item: (item.capture_timestamp, item.image_path),
    )
    untimed = sorted(
        [feature for feature in features if feature.capture_timestamp is None],
        key=lambda item: item.image_path,
    )
    if not timed:
        return TemporalSegmentation(sessions=[], untimed=untimed)

    broad_sessions = _build_broad_sessions(timed, config=config, similarity=similarity)
    return TemporalSegmentation(
        sessions=[
            _segment_session(session_features, config=config, similarity=similarity)
            for session_features in broad_sessions
        ],
        untimed=untimed,
    )


def crosses_hard_boundary(
    left_timeline_index: int,
    right_timeline_index: int,
    transitions: Sequence[SceneTransitionScore],
) -> bool:
    return strongest_crossed_boundary(left_timeline_index, right_timeline_index, transitions).crossed_hard_boundary


def strongest_crossed_boundary(
    left_index: int,
    right_index: int,
    transitions: Sequence[SceneTransitionScore],
) -> BoundaryStrength:
    if left_index == right_index or not transitions:
        return BoundaryStrength(
            crossed_hard_boundary=False,
            crossed_soft_boundary=False,
            strongest_transition=0.0,
            strongest_soft_transition=0.0,
            strongest_hard_transition=0.0,
        )
    start = max(0, min(left_index, right_index))
    end = min(max(left_index, right_index), len(transitions))
    crossed = list(transitions[start:end])
    if not crossed:
        return BoundaryStrength(
            crossed_hard_boundary=False,
            crossed_soft_boundary=False,
            strongest_transition=0.0,
            strongest_soft_transition=0.0,
            strongest_hard_transition=0.0,
        )
    strongest_transition = max(item.combined_transition for item in crossed)
    soft_values = [item.combined_transition for item in crossed if item.accepted_soft_boundary]
    hard_values = [item.combined_transition for item in crossed if item.accepted_hard_boundary]
    return BoundaryStrength(
        crossed_hard_boundary=bool(hard_values),
        crossed_soft_boundary=bool(soft_values or hard_values),
        strongest_transition=float(strongest_transition),
        strongest_soft_transition=0.0 if not soft_values else float(max(soft_values)),
        strongest_hard_transition=0.0 if not hard_values else float(max(hard_values)),
    )


def _build_broad_sessions(
    timed: list[VibeImageFeatures],
    *,
    config: VibeGroupingConfig,
    similarity: CombinedSimilarityComputer,
) -> list[list[VibeImageFeatures]]:
    sessions: list[list[VibeImageFeatures]] = [[timed[0]]]
    for feature in timed[1:]:
        current_session = sessions[-1]
        previous = current_session[-1]
        delta_seconds = abs((feature.capture_timestamp or 0.0) - (previous.capture_timestamp or 0.0))
        semantic = similarity.semantic_similarity(previous, feature)
        action = similarity.action_similarity(previous, feature)
        people = person_set_similarity(previous.recognized_person_ids, feature.recognized_person_ids)
        day_break = _day_boundary(previous.capture_timestamp, feature.capture_timestamp)
        start_new = delta_seconds > config.maximum_hard_time_gap_seconds
        if not start_new and day_break and delta_seconds >= config.session_gap_seconds:
            start_new = True
        if (
            not start_new
            and delta_seconds > config.session_gap_seconds
            and semantic < config.strong_pair_similarity
            and action < 0.72
            and people < 0.60
        ):
            start_new = True
        if start_new:
            sessions.append([feature])
        else:
            current_session.append(feature)
    return sessions


def _segment_session(
    session_features: list[VibeImageFeatures],
    *,
    config: VibeGroupingConfig,
    similarity: CombinedSimilarityComputer,
) -> TemporalSession:
    ordered = tuple(session_features)
    if len(ordered) <= 1:
        return TemporalSession(
            ordered_features=ordered,
            transitions=(),
            scene_segments=(SceneSegment(member_indices=(0,), boundary_before=None, boundary_after=None),)
            if ordered
            else (),
            recovery_events=(),
            tiny_segment_merge_events=(),
        )

    transitions = tuple(
        _compute_transition(ordered[index], ordered[index + 1], config=config, similarity=similarity)
        for index in range(len(ordered) - 1)
    )
    boundary_flags = [
        _accept_boundary(
            ordered,
            transitions,
            index,
            config=config,
            similarity=similarity,
        )
        for index in range(len(transitions))
    ]

    transitions = tuple(
        replace(
            transition,
            accepted_hard_boundary=transition.is_hard_boundary and boundary_flags[index],
            accepted_soft_boundary=transition.is_soft_boundary and boundary_flags[index],
        )
        for index, transition in enumerate(transitions)
    )

    raw_segments = _segments_from_boundaries(len(ordered), boundary_flags)
    smoothed_segments, recovery_events = _recover_singletons(
        raw_segments,
        transitions,
        ordered,
        config=config,
        similarity=similarity,
    )
    final_segments, tiny_segment_merge_events = _recover_tiny_segments(
        smoothed_segments,
        transitions,
        ordered,
        config=config,
        similarity=similarity,
    )
    scene_segments: list[SceneSegment] = []
    for segment in final_segments:
        first = segment[0]
        last = segment[-1]
        boundary_before = transitions[first - 1] if first > 0 and boundary_flags[first - 1] else None
        boundary_after = transitions[last] if last < len(transitions) and boundary_flags[last] else None
        scene_segments.append(
            SceneSegment(
                member_indices=tuple(segment),
                boundary_before=boundary_before,
                boundary_after=boundary_after,
            )
        )
    return TemporalSession(
        ordered_features=ordered,
        transitions=transitions,
        scene_segments=tuple(scene_segments),
        recovery_events=tuple(recovery_events),
        tiny_segment_merge_events=tuple(tiny_segment_merge_events),
    )


def _compute_transition(
    left: VibeImageFeatures,
    right: VibeImageFeatures,
    *,
    config: VibeGroupingConfig,
    similarity: CombinedSimilarityComputer,
) -> SceneTransitionScore:
    pair = similarity.pair_components(left, right)
    detail_transition = _is_detail_transition(left, right, similarity=similarity)
    semantic_similarity = pair.semantic
    subject_scene_similarity = pair.subject_scene
    layout_similarity = pair.layout
    background_similarity = pair.background
    composition_similarity = pair.composition
    semantic_change = 1.0 - semantic_similarity
    action_change = 0.0 if not pair.action_reliable else 1.0 - pair.action
    people_change = 1.0 - pair.people
    layout_change = 1.0 - layout_similarity
    subject_scene_change = 1.0 - subject_scene_similarity
    background_change = 1.0 - background_similarity
    composition_change = 1.0 - composition_similarity
    temporal_gap = 1.0 - pair.time
    transition_weights = dict(config.normalized_transition_weights())
    if not pair.action_reliable:
        transition_weights.pop("action", None)
    total_transition_weight = sum(transition_weights.values())
    if total_transition_weight > 0.0:
        transition_weights = {
            key: value / total_transition_weight
            for key, value in transition_weights.items()
        }
    combined = (
        (transition_weights.get("semantic", 0.0) * semantic_change)
        + (transition_weights.get("action", 0.0) * action_change)
        + (transition_weights.get("people", 0.0) * people_change)
        + (transition_weights.get("layout", 0.0) * layout_change)
        + (transition_weights.get("background", 0.0) * background_change)
        + (transition_weights.get("composition", 0.0) * composition_change)
        + (transition_weights.get("temporal", 0.0) * temporal_gap)
    )

    continuity_reasons: list[str] = []
    semantic_subject_continuity = (
        semantic_similarity >= config.continuity_semantic_similarity
        and subject_scene_similarity >= config.continuity_subject_scene_similarity
    )
    if semantic_subject_continuity:
        continuity_reasons.append("semantic_subject_scene")
    background_comp_layout_continuity = (
        background_similarity >= config.continuity_background_similarity
        and composition_similarity >= config.continuity_composition_similarity
        and layout_similarity >= config.continuity_layout_similarity
    )
    if background_comp_layout_continuity:
        continuity_reasons.append("background_composition_layout")
    very_strong_visual_continuity = (
        semantic_similarity >= max(config.continuity_semantic_similarity, 0.92)
        and subject_scene_similarity >= max(config.continuity_subject_scene_similarity, 0.92)
        and composition_similarity >= max(config.continuity_composition_similarity, 0.90)
        and layout_similarity >= max(config.continuity_layout_similarity, 0.88)
    )
    if very_strong_visual_continuity:
        continuity_reasons.append("all_signal_continuity")
    strong_visual_continuity = bool(
        semantic_subject_continuity
        or background_comp_layout_continuity
        or very_strong_visual_continuity
    )
    continuity_score = max(
        (semantic_similarity + subject_scene_similarity) / 2.0,
        (background_similarity + composition_similarity + layout_similarity) / 3.0,
        (semantic_similarity + subject_scene_similarity + composition_similarity + layout_similarity) / 4.0,
    )

    visual_change_supports_boundary = (
        semantic_change >= config.conflict_visual_support_threshold
        or subject_scene_change >= config.conflict_subject_scene_support_threshold
        or composition_change >= config.conflict_composition_support_threshold
        or layout_change >= config.conflict_layout_support_threshold
    )
    visual_change_strongly_supports_boundary = (
        semantic_change >= config.conflict_strong_visual_support_threshold
        or subject_scene_change >= config.conflict_subject_scene_strong_support_threshold
        or (
            composition_change >= config.conflict_composition_strong_support_threshold
            and layout_change >= config.conflict_layout_strong_support_threshold
        )
    )
    visual_support_score = max(
        semantic_change,
        subject_scene_change,
        composition_change,
        layout_change,
    )

    weak_conflict_reasons: list[str] = []
    if pair.participant_conflict_strength != ParticipantConflictStrength.NONE.value:
        weak_conflict_reasons.append("participant_mode_conflict")
    if pair.shot_mode_conflict and not pair.main_vs_reaction_conflict:
        weak_conflict_reasons.append("shot_mode_conflict")
    if pair.main_vs_reaction_conflict:
        weak_conflict_reasons.append("main_vs_reaction")
    if pair.action_soft_conflict and not pair.action_reliable:
        weak_conflict_reasons.append("uncertain_action_conflict")

    objective_hard_reasons: list[str] = []
    if (
        semantic_change >= config.semantic_hard_boundary_threshold
        and composition_change >= config.composition_soft_boundary_threshold
        and not detail_transition
    ):
        objective_hard_reasons.append("semantic_plus_composition")
    if composition_change >= config.composition_hard_boundary_threshold and not detail_transition:
        objective_hard_reasons.append("composition_threshold")
    if (
        semantic_change >= config.semantic_soft_boundary_threshold + 0.05
        and background_change >= config.background_hard_support_threshold
        and layout_change >= config.layout_hard_support_threshold
        and not detail_transition
    ):
        objective_hard_reasons.append("semantic_background_layout")
    if (
        semantic_change >= 0.38
        and subject_scene_change >= config.conflict_subject_scene_strong_support_threshold
        and layout_change >= config.layout_hard_support_threshold
        and not detail_transition
    ):
        objective_hard_reasons.append("semantic_subject_scene_layout")
    if (
        left.capture_timestamp is not None
        and right.capture_timestamp is not None
        and abs(left.capture_timestamp - right.capture_timestamp) > config.maximum_adjacent_gap_within_scene_seconds
        and visual_change_supports_boundary
    ):
        objective_hard_reasons.append("time_gap_with_visual_change")

    continuity_override_applied = False
    reliability = BoundaryReliability.NONE
    selected_reasons: list[str] = []

    if objective_hard_reasons:
        reliability = BoundaryReliability.HARD
        selected_reasons = objective_hard_reasons
    elif (
        strong_visual_continuity
        and weak_conflict_reasons
        and (
            not visual_change_supports_boundary
            or (
                pair.participant_conflict_strength == ParticipantConflictStrength.WEAK.value
                and visual_support_score < 0.35
            )
        )
        and pair.participant_conflict_strength != ParticipantConflictStrength.STRONG.value
        and not (pair.main_vs_reaction_conflict and pair.main_vs_reaction_confident)
    ):
        continuity_override_applied = True
        reliability = BoundaryReliability.NONE
        selected_reasons = []
    elif pair.action_hard_conflict and visual_change_strongly_supports_boundary and not detail_transition:
        reliability = BoundaryReliability.HARD
        selected_reasons = ["action_conflict"]
    elif pair.action_soft_conflict and pair.action_reliable and visual_change_supports_boundary and not detail_transition:
        reliability = BoundaryReliability.SUPPORTED
        selected_reasons = ["action_conflict"]
    elif pair.participant_conflict_strength == ParticipantConflictStrength.STRONG.value:
        if (
            visual_change_strongly_supports_boundary
            and pair.main_vs_reaction_confident
            and not detail_transition
        ):
            reliability = BoundaryReliability.HARD
            selected_reasons = ["participant_mode_conflict"]
        elif visual_change_strongly_supports_boundary and not detail_transition:
            reliability = BoundaryReliability.SUPPORTED
            selected_reasons = ["participant_mode_conflict"]
        elif visual_change_supports_boundary and not detail_transition:
            reliability = BoundaryReliability.SUPPORTED
            selected_reasons = ["participant_mode_conflict"]
        else:
            reliability = BoundaryReliability.WEAK
            selected_reasons = ["participant_mode_conflict"]
    elif pair.main_vs_reaction_conflict:
        if (
            pair.main_vs_reaction_confident
            and visual_change_strongly_supports_boundary
            and pair.participant_conflict_strength == ParticipantConflictStrength.STRONG.value
            and not detail_transition
        ):
            reliability = BoundaryReliability.HARD
            selected_reasons = ["main_vs_reaction"]
        elif pair.main_vs_reaction_confident and visual_change_strongly_supports_boundary and not detail_transition:
            reliability = BoundaryReliability.SUPPORTED
            selected_reasons = ["main_vs_reaction"]
        elif pair.main_vs_reaction_confident and visual_change_supports_boundary and not detail_transition:
            reliability = BoundaryReliability.SUPPORTED
            selected_reasons = ["main_vs_reaction"]
        else:
            reliability = BoundaryReliability.WEAK
            selected_reasons = ["main_vs_reaction"]
    elif combined >= config.hard_boundary_threshold and (
        not detail_transition or combined >= config.hard_boundary_threshold + 0.10
    ):
        reliability = BoundaryReliability.HARD
        selected_reasons = ["combined_threshold"]
    elif combined >= config.soft_boundary_threshold:
        reliability = BoundaryReliability.SUPPORTED
        selected_reasons = ["combined_threshold"]
    elif weak_conflict_reasons:
        reliability = BoundaryReliability.WEAK
        selected_reasons = sorted(set(weak_conflict_reasons))

    hard = reliability is BoundaryReliability.HARD
    soft = reliability is BoundaryReliability.SUPPORTED
    hard_reasons = tuple(sorted(set(selected_reasons))) if hard else ()
    soft_reasons = tuple(sorted(set(selected_reasons))) if reliability in {BoundaryReliability.SUPPORTED, BoundaryReliability.WEAK} else ()
    return SceneTransitionScore(
        left_image=left.image_path,
        right_image=right.image_path,
        semantic_change=float(max(0.0, min(1.0, semantic_change))),
        action_change=float(max(0.0, min(1.0, action_change))),
        people_change=float(max(0.0, min(1.0, people_change))),
        layout_change=float(max(0.0, min(1.0, layout_change))),
        subject_scene_change=float(max(0.0, min(1.0, subject_scene_change))),
        background_change=float(max(0.0, min(1.0, background_change))),
        composition_change=float(max(0.0, min(1.0, composition_change))),
        temporal_gap=float(max(0.0, min(1.0, temporal_gap))),
        combined_transition=float(max(0.0, min(1.0, combined))),
        boundary_reliability=reliability.value,
        is_hard_boundary=hard,
        is_soft_boundary=soft,
        continuity_override_applied=continuity_override_applied,
        continuity_score=float(max(0.0, min(1.0, continuity_score))),
        visual_support_score=float(max(0.0, min(1.0, visual_support_score))),
        continuity_reasons=tuple(sorted(set(continuity_reasons))),
        participant_conflict_strength=pair.participant_conflict_strength,
        shot_conflict_confident=pair.main_vs_reaction_confident,
        left_shot_margin=float(max(0.0, pair.left_shot_margin)),
        right_shot_margin=float(max(0.0, pair.right_shot_margin)),
        left_shot_confidence=pair.left_shot_confidence,
        right_shot_confidence=pair.right_shot_confidence,
        recovery_crossable=reliability in {BoundaryReliability.NONE, BoundaryReliability.WEAK},
        hard_reasons=hard_reasons,
        soft_reasons=soft_reasons,
    )


def _accept_boundary(
    ordered: tuple[VibeImageFeatures, ...],
    transitions: tuple[SceneTransitionScore, ...],
    boundary_index: int,
    *,
    config: VibeGroupingConfig,
    similarity: CombinedSimilarityComputer,
) -> bool:
    transition = transitions[boundary_index]
    reliability = BoundaryReliability(transition.boundary_reliability)
    if reliability is BoundaryReliability.HARD:
        return True
    if reliability is not BoundaryReliability.SUPPORTED:
        return False
    if _detail_context_supports_continuity(
        ordered,
        boundary_index,
        config=config,
        similarity=similarity,
    ):
        return False

    support = _boundary_support(
        ordered,
        boundary_index,
        config=config,
        similarity=similarity,
    )
    if support >= config.boundary_support_margin:
        return True
    if transition.combined_transition >= config.soft_boundary_threshold + 0.08:
        return True
    if (
        transition.semantic_change >= config.semantic_hard_boundary_threshold - 0.02
        and transition.composition_change >= config.composition_soft_boundary_threshold
    ):
        return True
    if transition.composition_change >= config.composition_hard_boundary_threshold - 0.04:
        return True
    return False


def _detail_context_supports_continuity(
    ordered: tuple[VibeImageFeatures, ...],
    boundary_index: int,
    *,
    config: VibeGroupingConfig,
    similarity: CombinedSimilarityComputer,
) -> bool:
    left = ordered[boundary_index]
    right = ordered[boundary_index + 1]
    left_is_detail = similarity._shot_mode(left) == "detail"
    right_is_detail = similarity._shot_mode(right) == "detail"
    if not left_is_detail and not right_is_detail:
        return False
    if right_is_detail and boundary_index + 2 < len(ordered):
        if similarity.pair_similarity(left, ordered[boundary_index + 2]) >= config.strong_pair_similarity:
            return True
    if left_is_detail and boundary_index - 1 >= 0:
        if similarity.pair_similarity(ordered[boundary_index - 1], right) >= config.strong_pair_similarity:
            return True
    return False


def _boundary_support(
    ordered: tuple[VibeImageFeatures, ...],
    boundary_index: int,
    *,
    config: VibeGroupingConfig,
    similarity: CombinedSimilarityComputer,
) -> float:
    left_pairs: list[float] = []
    right_pairs: list[float] = []
    start = max(0, boundary_index - config.boundary_context_window + 1)
    end = min(len(ordered) - 2, boundary_index + config.boundary_context_window)
    for index in range(start, boundary_index):
        left_pairs.append(similarity.pair_similarity(ordered[index], ordered[index + 1]))
    for index in range(boundary_index + 1, end + 1):
        right_pairs.append(similarity.pair_similarity(ordered[index], ordered[index + 1]))
    cross = similarity.pair_similarity(ordered[boundary_index], ordered[boundary_index + 1])
    if not left_pairs or not right_pairs:
        return max(0.0, 0.5 - cross)
    return max(
        0.0,
        ((sum(left_pairs) / len(left_pairs)) + (sum(right_pairs) / len(right_pairs))) / 2.0 - cross,
    )


def _segments_from_boundaries(
    feature_count: int,
    boundary_flags: list[bool],
) -> list[list[int]]:
    segments: list[list[int]] = []
    current = [0]
    for index, is_boundary in enumerate(boundary_flags):
        next_member = index + 1
        if is_boundary:
            segments.append(current)
            current = [next_member]
        else:
            current.append(next_member)
    segments.append(current)
    return segments


def _recover_singletons(
    segments: list[list[int]],
    transitions: tuple[SceneTransitionScore, ...],
    ordered: tuple[VibeImageFeatures, ...],
    *,
    config: VibeGroupingConfig,
    similarity: CombinedSimilarityComputer,
) -> tuple[list[list[int]], list[dict[str, object]]]:
    if len(segments) < 2:
        return segments, []

    recovered: list[list[int]] = []
    recovery_events: list[dict[str, object]] = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if len(segment) != 1:
            recovered.append(segment)
            index += 1
            continue

        singleton = segment[0]
        left_segment = recovered[-1] if recovered else None
        right_segment = segments[index + 1] if index + 1 < len(segments) else None
        before_transition = transitions[singleton - 1] if singleton > 0 else None
        after_transition = transitions[singleton] if singleton < len(transitions) else None
        left_similarity = 0.0
        right_similarity = 0.0
        left_crossable = False
        right_crossable = False

        if left_segment and right_segment:
            left_similarity = _segment_anchor_similarity(
                ordered,
                singleton,
                left_segment,
                similarity=similarity,
            )
            right_similarity = _segment_anchor_similarity(
                ordered,
                singleton,
                right_segment,
                similarity=similarity,
            )
            left_crossable = _can_cross_recovery_boundary(
                before_transition,
                merge_score=left_similarity,
                anchor=ordered[left_segment[-1]],
                candidate=ordered[singleton],
                similarity=similarity,
                config=config,
            )
            right_crossable = _can_cross_recovery_boundary(
                after_transition,
                merge_score=right_similarity,
                anchor=ordered[right_segment[0]],
                candidate=ordered[singleton],
                similarity=similarity,
                config=config,
            )
            if left_crossable and right_crossable:
                margin = abs(left_similarity - right_similarity)
                if (
                    max(left_similarity, right_similarity) >= config.singleton_recovery_similarity
                    and (
                        margin >= config.singleton_recovery_margin
                        or _strong_recovery_continuity(
                            ordered[left_segment[-1]],
                            ordered[singleton],
                            similarity=similarity,
                            config=config,
                        )
                        != _strong_recovery_continuity(
                            ordered[right_segment[0]],
                            ordered[singleton],
                            similarity=similarity,
                            config=config,
                        )
                    )
                ):
                    choose_left = left_similarity >= right_similarity
                    target = left_segment if choose_left else right_segment
                    target.append(singleton)
                    target.sort()
                    recovery_events.append(
                        {
                            "singleton_path": ordered[singleton].image_path,
                            "left_merge_score": round(left_similarity, 4),
                            "right_merge_score": round(right_similarity, 4),
                            "selected_side": "left" if choose_left else "right",
                            "boundary_reliability_crossed": (
                                before_transition.boundary_reliability
                                if choose_left and before_transition is not None
                                else after_transition.boundary_reliability
                                if after_transition is not None
                                else BoundaryReliability.NONE.value
                            ),
                            "recovery_result": "merged",
                        }
                    )
                    index += 1
                    continue

        if left_segment:
            left_similarity = _segment_anchor_similarity(
                ordered,
                singleton,
                left_segment,
                similarity=similarity,
            )
            if (
                left_similarity >= config.singleton_recovery_similarity
                and _can_cross_recovery_boundary(
                    before_transition,
                    merge_score=left_similarity,
                    anchor=ordered[left_segment[-1]],
                    candidate=ordered[singleton],
                    similarity=similarity,
                    config=config,
                )
                and _can_absorb_singleton(
                    ordered[left_segment[-1]],
                    ordered[singleton],
                    similarity=similarity,
                    config=config,
                )
            ):
                left_segment.append(singleton)
                left_segment.sort()
                recovery_events.append(
                    {
                        "singleton_path": ordered[singleton].image_path,
                        "left_merge_score": round(left_similarity, 4),
                        "right_merge_score": 0.0,
                        "selected_side": "left",
                        "boundary_reliability_crossed": (
                            before_transition.boundary_reliability
                            if before_transition is not None
                            else BoundaryReliability.NONE.value
                        ),
                        "recovery_result": "merged",
                    }
                )
                index += 1
                continue

        if right_segment:
            right_similarity = _segment_anchor_similarity(
                ordered,
                singleton,
                right_segment,
                similarity=similarity,
            )
            if (
                right_similarity >= config.singleton_recovery_similarity
                and _can_cross_recovery_boundary(
                    after_transition,
                    merge_score=right_similarity,
                    anchor=ordered[right_segment[0]],
                    candidate=ordered[singleton],
                    similarity=similarity,
                    config=config,
                )
                and _can_absorb_singleton(
                    ordered[right_segment[0]],
                    ordered[singleton],
                    similarity=similarity,
                    config=config,
                )
            ):
                right_segment.append(singleton)
                right_segment.sort()
                recovery_events.append(
                    {
                        "singleton_path": ordered[singleton].image_path,
                        "left_merge_score": 0.0,
                        "right_merge_score": round(right_similarity, 4),
                        "selected_side": "right",
                        "boundary_reliability_crossed": (
                            after_transition.boundary_reliability
                            if after_transition is not None
                            else BoundaryReliability.NONE.value
                        ),
                        "recovery_result": "merged",
                    }
                )
                index += 1
                continue

        recovered.append(segment)
        index += 1
    return recovered, recovery_events


def _recover_tiny_segments(
    segments: list[list[int]],
    transitions: tuple[SceneTransitionScore, ...],
    ordered: tuple[VibeImageFeatures, ...],
    *,
    config: VibeGroupingConfig,
    similarity: CombinedSimilarityComputer,
) -> tuple[list[list[int]], list[dict[str, object]]]:
    if len(segments) < 2:
        return segments, []

    merged = [list(segment) for segment in segments]
    events: list[dict[str, object]] = []
    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(merged):
            segment = merged[index]
            if len(segment) == 0 or len(segment) > config.tiny_segment_max_size:
                index += 1
                continue
            left_segment = merged[index - 1] if index > 0 else None
            right_segment = merged[index + 1] if index + 1 < len(merged) else None
            if left_segment is None and right_segment is None:
                index += 1
                continue

            left_score = 0.0
            right_score = 0.0
            left_allowed = False
            right_allowed = False
            if left_segment is not None:
                left_boundary = transitions[segment[0] - 1] if segment[0] > 0 else None
                left_score = _segment_to_segment_similarity(
                    ordered,
                    left_segment,
                    segment,
                    similarity=similarity,
                )
                left_allowed = _can_cross_segment_boundary(
                    left_boundary,
                    merge_score=left_score,
                    left_segment=left_segment,
                    right_segment=segment,
                    ordered=ordered,
                    similarity=similarity,
                    config=config,
                )
            if right_segment is not None:
                right_boundary = transitions[segment[-1]] if segment[-1] < len(transitions) else None
                right_score = _segment_to_segment_similarity(
                    ordered,
                    segment,
                    right_segment,
                    similarity=similarity,
                )
                right_allowed = _can_cross_segment_boundary(
                    right_boundary,
                    merge_score=right_score,
                    left_segment=segment,
                    right_segment=right_segment,
                    ordered=ordered,
                    similarity=similarity,
                    config=config,
                )

            if not left_allowed and not right_allowed:
                events.append(
                    {
                        "segment_id": f"{segment[0]}-{segment[-1]}",
                        "member_count": len(segment),
                        "merge_score": round(max(left_score, right_score), 4),
                        "merge_target": None,
                        "merge_rejected_reason": "boundary_not_crossable",
                    }
                )
                index += 1
                continue

            choose_left = left_allowed and (not right_allowed or left_score >= right_score)
            chosen_score = left_score if choose_left else right_score
            minimum_score = config.tiny_segment_merge_similarity
            if chosen_score < minimum_score:
                events.append(
                    {
                        "segment_id": f"{segment[0]}-{segment[-1]}",
                        "member_count": len(segment),
                        "merge_score": round(chosen_score, 4),
                        "merge_target": "left" if choose_left else "right",
                        "merge_rejected_reason": "low_similarity",
                    }
                )
                index += 1
                continue

            if choose_left and left_segment is not None:
                left_segment.extend(segment)
                left_segment.sort()
                del merged[index]
                events.append(
                    {
                        "segment_id": f"{segment[0]}-{segment[-1]}",
                        "member_count": len(segment),
                        "merge_score": round(chosen_score, 4),
                        "merge_target": "left",
                        "merge_rejected_reason": None,
                    }
                )
            elif right_segment is not None:
                right_segment.extend(segment)
                right_segment.sort()
                del merged[index]
                events.append(
                    {
                        "segment_id": f"{segment[0]}-{segment[-1]}",
                        "member_count": len(segment),
                        "merge_score": round(chosen_score, 4),
                        "merge_target": "right",
                        "merge_rejected_reason": None,
                    }
                )
            changed = True
    return merged, events


def _segment_anchor_similarity(
    ordered: tuple[VibeImageFeatures, ...],
    singleton_index: int,
    segment: list[int],
    *,
    similarity: CombinedSimilarityComputer,
) -> float:
    anchors = segment[:2] + segment[-2:]
    values = [
        similarity.pair_similarity(ordered[singleton_index], ordered[index])
        for index in anchors
    ]
    return sum(values) / max(len(values), 1)


def _can_absorb_singleton(
    anchor: VibeImageFeatures,
    singleton: VibeImageFeatures,
    *,
    similarity: CombinedSimilarityComputer,
    config: VibeGroupingConfig,
) -> bool:
    components = similarity.pair_components(anchor, singleton)
    if components.action_hard_conflict:
        return False
    if (
        components.participant_conflict_strength == ParticipantConflictStrength.STRONG.value
        and not _strong_recovery_continuity(anchor, singleton, similarity=similarity, config=config)
    ):
        return False
    if components.main_vs_reaction_conflict and not _strong_recovery_continuity(
        anchor,
        singleton,
        similarity=similarity,
        config=config,
    ):
        return False
    if (
        anchor.capture_timestamp is not None
        and singleton.capture_timestamp is not None
        and abs(anchor.capture_timestamp - singleton.capture_timestamp) > config.maximum_adjacent_gap_within_scene_seconds
    ):
        return False
    return True


def _can_cross_recovery_boundary(
    transition: SceneTransitionScore | None,
    *,
    merge_score: float,
    anchor: VibeImageFeatures,
    candidate: VibeImageFeatures,
    similarity: CombinedSimilarityComputer,
    config: VibeGroupingConfig,
) -> bool:
    if transition is None:
        return True
    reliability = BoundaryReliability(transition.boundary_reliability)
    if reliability is BoundaryReliability.HARD:
        return False
    continuity = _strong_recovery_continuity(anchor, candidate, similarity=similarity, config=config)
    if reliability is BoundaryReliability.SUPPORTED:
        return merge_score >= config.supported_boundary_merge_similarity and continuity
    return merge_score >= config.weak_boundary_merge_similarity


def _can_cross_segment_boundary(
    transition: SceneTransitionScore | None,
    *,
    merge_score: float,
    left_segment: list[int],
    right_segment: list[int],
    ordered: tuple[VibeImageFeatures, ...],
    similarity: CombinedSimilarityComputer,
    config: VibeGroupingConfig,
) -> bool:
    if transition is None:
        return True
    reliability = BoundaryReliability(transition.boundary_reliability)
    if reliability is BoundaryReliability.HARD:
        return False
    continuity = _strong_recovery_continuity(
        ordered[left_segment[-1]],
        ordered[right_segment[0]],
        similarity=similarity,
        config=config,
    )
    if reliability is BoundaryReliability.SUPPORTED:
        return merge_score >= config.supported_boundary_merge_similarity and continuity
    return merge_score >= config.weak_boundary_merge_similarity


def _strong_recovery_continuity(
    left: VibeImageFeatures,
    right: VibeImageFeatures,
    *,
    similarity: CombinedSimilarityComputer,
    config: VibeGroupingConfig,
) -> bool:
    components = similarity.pair_components(left, right)
    return bool(
        (
            components.semantic >= config.continuity_semantic_similarity
            and components.subject_scene >= config.continuity_subject_scene_similarity
        )
        or (
            components.background >= config.continuity_background_similarity
            and components.composition >= config.continuity_composition_similarity
            and components.layout >= config.continuity_layout_similarity
        )
    )


def _segment_to_segment_similarity(
    ordered: tuple[VibeImageFeatures, ...],
    left_segment: list[int],
    right_segment: list[int],
    *,
    similarity: CombinedSimilarityComputer,
) -> float:
    left_anchor = ordered[left_segment[-1]]
    right_anchor = ordered[right_segment[0]]
    pair = similarity.pair_components(left_anchor, right_anchor)
    return (
        (0.30 * pair.semantic)
        + (0.25 * pair.subject_scene)
        + (0.20 * pair.composition)
        + (0.15 * pair.layout)
        + (0.10 * pair.background)
    )


def _is_detail_transition(
    left: VibeImageFeatures,
    right: VibeImageFeatures,
    *,
    similarity: CombinedSimilarityComputer,
) -> bool:
    prototype_table = similarity.prototype_table
    if prototype_table is None:
        return False
    left_shot = prototype_table.top_matches(left.shot_type_scores, category="shot", limit=1)
    right_shot = prototype_table.top_matches(right.shot_type_scores, category="shot", limit=1)
    left_action = prototype_table.top_matches(left.action_scores, category="action", limit=1)
    right_action = prototype_table.top_matches(right.action_scores, category="action", limit=1)
    tags = set()
    for matches in (left_shot, right_shot, left_action, right_action):
        if matches:
            tags.update(matches[0].tags)
            tags.add(matches[0].key)
    return "detail" in tags


def _day_boundary(left_timestamp: float | None, right_timestamp: float | None) -> bool:
    if left_timestamp is None or right_timestamp is None:
        return False
    return int(left_timestamp // 86400) != int(right_timestamp // 86400)
