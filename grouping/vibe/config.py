from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable


class VibeGroupingPreset(StrEnum):
    SESSION = "session"
    BALANCED_SCENES = "balanced_scene"
    TIGHT_SCENES = "tight_scene"

    @property
    def display_name(self) -> str:
        if self is VibeGroupingPreset.SESSION:
            return "Session"
        if self is VibeGroupingPreset.TIGHT_SCENES:
            return "Tight Scenes"
        return "Balanced Scenes"


@dataclass(frozen=True, slots=True)
class VibeGroupingConfig:
    semantic_weight: float = 0.27
    action_weight: float = 0.14
    people_weight: float = 0.08
    layout_weight: float = 0.13
    subject_scene_weight: float = 0.19
    background_weight: float = 0.04
    time_weight: float = 0.04
    composition_weight: float = 0.11
    color_weight: float = 0.0

    minimum_pair_similarity: float = 0.74
    strong_pair_similarity: float = 0.86
    minimum_group_cohesion: float = 0.76

    session_gap_seconds: int = 30 * 60
    maximum_soft_time_gap_seconds: int = 20 * 60
    maximum_hard_time_gap_seconds: int = 2 * 60 * 60
    maximum_scene_span_seconds: int = 10 * 60
    maximum_adjacent_gap_within_scene_seconds: int = 180

    hard_boundary_threshold: float = 0.32
    soft_boundary_threshold: float = 0.22
    semantic_hard_boundary_threshold: float = 0.42
    semantic_soft_boundary_threshold: float = 0.33
    composition_hard_boundary_threshold: float = 0.72
    composition_soft_boundary_threshold: float = 0.48
    layout_support_threshold: float = 0.18
    layout_hard_support_threshold: float = 0.25
    background_hard_support_threshold: float = 0.38
    boundary_support_margin: float = 0.04
    boundary_context_window: int = 2

    scene_neighbor_count: int = 6
    adjacent_timeline_radius: int = 3
    cross_boundary_candidate_count: int = 1
    maximum_timeline_edge_distance: int = 6
    allowed_internal_outliers: int = 2
    maximum_internal_timeline_gaps: int = 2
    maximum_internal_gap_size: int = 2

    minimum_group_size: int = 2
    maximum_group_size: int = 40
    target_group_size: int = 18

    minimum_action_margin: float = 0.012
    strong_action_margin: float = 0.025
    minimum_shot_margin: float = 0.025
    strong_shot_margin: float = 0.045
    action_conflict_penalty: float = 0.20
    strong_action_conflict_penalty: float = 0.30
    participant_mode_conflict_penalty: float = 0.08
    participant_mode_weak_penalty: float = 0.03
    participant_mode_strong_penalty: float = 0.10
    shot_mode_conflict_penalty: float = 0.03
    main_vs_reaction_penalty: float = 0.08
    main_vs_reaction_weak_penalty: float = 0.03
    main_vs_reaction_supported_penalty: float = 0.08
    main_vs_reaction_strong_penalty: float = 0.16
    hard_transition_penalty: float = 0.20
    soft_transition_penalty: float = 0.07
    temporal_bridge_penalty: float = 0.08
    soft_cross_boundary_penalty: float = 0.14
    cross_soft_boundary_similarity: float = 0.86
    internal_soft_boundary_penalty: float = 0.10
    conflict_visual_support_threshold: float = 0.22
    conflict_strong_visual_support_threshold: float = 0.32
    conflict_subject_scene_support_threshold: float = 0.20
    conflict_subject_scene_strong_support_threshold: float = 0.30
    conflict_composition_support_threshold: float = 0.30
    conflict_composition_strong_support_threshold: float = 0.42
    conflict_layout_support_threshold: float = 0.22
    conflict_layout_strong_support_threshold: float = 0.20
    continuity_semantic_similarity: float = 0.88
    continuity_subject_scene_similarity: float = 0.88
    continuity_composition_similarity: float = 0.86
    continuity_layout_similarity: float = 0.82
    continuity_background_similarity: float = 0.92

    transition_semantic_weight: float = 0.28
    transition_action_weight: float = 0.16
    transition_people_weight: float = 0.08
    transition_layout_weight: float = 0.15
    transition_background_weight: float = 0.08
    transition_composition_weight: float = 0.21
    transition_temporal_weight: float = 0.04

    prototype_confidence_threshold: float = 0.50
    adjacent_merge_similarity: float = 0.84
    singleton_recovery_similarity: float = 0.78
    singleton_recovery_margin: float = 0.04
    tiny_segment_max_size: int = 2
    tiny_segment_merge_similarity: float = 0.80
    weak_boundary_merge_similarity: float = 0.82
    supported_boundary_merge_similarity: float = 0.88
    long_range_semantic_threshold: float = 0.90
    long_range_subject_scene_threshold: float = 0.88

    include_people: bool = True
    include_color: bool = True
    include_composition: bool = True
    include_subject_scene_embedding: bool = True
    include_background_embedding: bool = True
    merge_non_adjacent_scenes: bool = False

    batch_size: int = 16
    random_seed: int = 1337
    semantic_model_filename: str = "vibe_semantic_v1.onnx"
    allow_visual_fallback: bool = True
    preprocessing_version: int = 2
    background_preprocessing_version: int = 1
    subject_scene_preprocessing_version: int = 1
    composition_feature_version: int = 2
    feature_version: int = 4
    transition_version: int = 3
    boundary_reliability_version: int = 1
    shot_confidence_version: int = 1
    singleton_recovery_version: int = 1
    prototype_version: int = 1
    result_cache_version: int = 3
    algorithm_version: int = 4

    def __post_init__(self) -> None:
        weights = {
            "semantic_weight": self.semantic_weight,
            "action_weight": self.action_weight,
            "people_weight": self.people_weight,
            "layout_weight": self.layout_weight,
            "subject_scene_weight": self.subject_scene_weight,
            "background_weight": self.background_weight,
            "time_weight": self.time_weight,
            "composition_weight": self.composition_weight,
            "color_weight": self.color_weight,
        }
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("Vibe grouping weights must be non-negative.")

        positive_int_fields = {
            "batch_size": self.batch_size,
            "scene_neighbor_count": self.scene_neighbor_count,
            "adjacent_timeline_radius": self.adjacent_timeline_radius,
            "maximum_timeline_edge_distance": self.maximum_timeline_edge_distance,
            "minimum_group_size": self.minimum_group_size,
            "maximum_group_size": self.maximum_group_size,
            "target_group_size": self.target_group_size,
            "session_gap_seconds": self.session_gap_seconds,
            "maximum_soft_time_gap_seconds": self.maximum_soft_time_gap_seconds,
            "maximum_hard_time_gap_seconds": self.maximum_hard_time_gap_seconds,
            "maximum_scene_span_seconds": self.maximum_scene_span_seconds,
            "maximum_adjacent_gap_within_scene_seconds": self.maximum_adjacent_gap_within_scene_seconds,
            "boundary_context_window": self.boundary_context_window,
            "preprocessing_version": self.preprocessing_version,
            "background_preprocessing_version": self.background_preprocessing_version,
            "subject_scene_preprocessing_version": self.subject_scene_preprocessing_version,
            "composition_feature_version": self.composition_feature_version,
            "feature_version": self.feature_version,
            "transition_version": self.transition_version,
            "boundary_reliability_version": self.boundary_reliability_version,
            "shot_confidence_version": self.shot_confidence_version,
            "singleton_recovery_version": self.singleton_recovery_version,
            "prototype_version": self.prototype_version,
            "result_cache_version": self.result_cache_version,
            "algorithm_version": self.algorithm_version,
        }
        if any(value <= 0 for value in positive_int_fields.values()):
            raise ValueError("Scene-grouping integer settings must be positive.")

        nonnegative_int_fields = {
            "cross_boundary_candidate_count": self.cross_boundary_candidate_count,
            "allowed_internal_outliers": self.allowed_internal_outliers,
            "maximum_internal_timeline_gaps": self.maximum_internal_timeline_gaps,
            "maximum_internal_gap_size": self.maximum_internal_gap_size,
            "tiny_segment_max_size": self.tiny_segment_max_size,
        }
        if any(value < 0 for value in nonnegative_int_fields.values()):
            raise ValueError("Scene-grouping counters must be non-negative.")

        if self.maximum_group_size < self.minimum_group_size:
            raise ValueError("maximum_group_size must be at least minimum_group_size.")
        if self.target_group_size < self.minimum_group_size:
            raise ValueError("target_group_size must be at least minimum_group_size.")
        if self.maximum_soft_time_gap_seconds > self.maximum_hard_time_gap_seconds:
            raise ValueError("maximum_soft_time_gap_seconds cannot exceed maximum_hard_time_gap_seconds.")
        if self.maximum_scene_span_seconds < self.maximum_adjacent_gap_within_scene_seconds:
            raise ValueError("maximum_scene_span_seconds must allow at least one adjacent gap.")

        bounded = {
            "minimum_pair_similarity": self.minimum_pair_similarity,
            "strong_pair_similarity": self.strong_pair_similarity,
            "minimum_group_cohesion": self.minimum_group_cohesion,
            "hard_boundary_threshold": self.hard_boundary_threshold,
            "soft_boundary_threshold": self.soft_boundary_threshold,
            "semantic_hard_boundary_threshold": self.semantic_hard_boundary_threshold,
            "semantic_soft_boundary_threshold": self.semantic_soft_boundary_threshold,
            "composition_hard_boundary_threshold": self.composition_hard_boundary_threshold,
            "composition_soft_boundary_threshold": self.composition_soft_boundary_threshold,
            "layout_support_threshold": self.layout_support_threshold,
            "layout_hard_support_threshold": self.layout_hard_support_threshold,
            "background_hard_support_threshold": self.background_hard_support_threshold,
            "boundary_support_margin": self.boundary_support_margin,
            "minimum_action_margin": self.minimum_action_margin,
            "strong_action_margin": self.strong_action_margin,
            "minimum_shot_margin": self.minimum_shot_margin,
            "strong_shot_margin": self.strong_shot_margin,
            "prototype_confidence_threshold": self.prototype_confidence_threshold,
            "adjacent_merge_similarity": self.adjacent_merge_similarity,
            "singleton_recovery_similarity": self.singleton_recovery_similarity,
            "singleton_recovery_margin": self.singleton_recovery_margin,
            "tiny_segment_merge_similarity": self.tiny_segment_merge_similarity,
            "weak_boundary_merge_similarity": self.weak_boundary_merge_similarity,
            "supported_boundary_merge_similarity": self.supported_boundary_merge_similarity,
            "long_range_semantic_threshold": self.long_range_semantic_threshold,
            "long_range_subject_scene_threshold": self.long_range_subject_scene_threshold,
            "action_conflict_penalty": self.action_conflict_penalty,
            "strong_action_conflict_penalty": self.strong_action_conflict_penalty,
            "participant_mode_conflict_penalty": self.participant_mode_conflict_penalty,
            "participant_mode_weak_penalty": self.participant_mode_weak_penalty,
            "participant_mode_strong_penalty": self.participant_mode_strong_penalty,
            "shot_mode_conflict_penalty": self.shot_mode_conflict_penalty,
            "main_vs_reaction_penalty": self.main_vs_reaction_penalty,
            "main_vs_reaction_weak_penalty": self.main_vs_reaction_weak_penalty,
            "main_vs_reaction_supported_penalty": self.main_vs_reaction_supported_penalty,
            "main_vs_reaction_strong_penalty": self.main_vs_reaction_strong_penalty,
            "hard_transition_penalty": self.hard_transition_penalty,
            "soft_transition_penalty": self.soft_transition_penalty,
            "temporal_bridge_penalty": self.temporal_bridge_penalty,
            "soft_cross_boundary_penalty": self.soft_cross_boundary_penalty,
            "cross_soft_boundary_similarity": self.cross_soft_boundary_similarity,
            "internal_soft_boundary_penalty": self.internal_soft_boundary_penalty,
            "conflict_visual_support_threshold": self.conflict_visual_support_threshold,
            "conflict_strong_visual_support_threshold": self.conflict_strong_visual_support_threshold,
            "conflict_subject_scene_support_threshold": self.conflict_subject_scene_support_threshold,
            "conflict_subject_scene_strong_support_threshold": self.conflict_subject_scene_strong_support_threshold,
            "conflict_composition_support_threshold": self.conflict_composition_support_threshold,
            "conflict_composition_strong_support_threshold": self.conflict_composition_strong_support_threshold,
            "conflict_layout_support_threshold": self.conflict_layout_support_threshold,
            "conflict_layout_strong_support_threshold": self.conflict_layout_strong_support_threshold,
            "continuity_semantic_similarity": self.continuity_semantic_similarity,
            "continuity_subject_scene_similarity": self.continuity_subject_scene_similarity,
            "continuity_composition_similarity": self.continuity_composition_similarity,
            "continuity_layout_similarity": self.continuity_layout_similarity,
            "continuity_background_similarity": self.continuity_background_similarity,
            "transition_semantic_weight": self.transition_semantic_weight,
            "transition_action_weight": self.transition_action_weight,
            "transition_people_weight": self.transition_people_weight,
            "transition_layout_weight": self.transition_layout_weight,
            "transition_background_weight": self.transition_background_weight,
            "transition_composition_weight": self.transition_composition_weight,
            "transition_temporal_weight": self.transition_temporal_weight,
        }
        if any(value < 0.0 or value > 1.0 for value in bounded.values()):
            raise ValueError("Scene-grouping thresholds and penalties must be between 0.0 and 1.0.")

        if self.soft_boundary_threshold > self.hard_boundary_threshold:
            raise ValueError("soft_boundary_threshold cannot exceed hard_boundary_threshold.")
        if self.semantic_soft_boundary_threshold > self.semantic_hard_boundary_threshold:
            raise ValueError("semantic_soft_boundary_threshold cannot exceed semantic_hard_boundary_threshold.")
        if self.composition_soft_boundary_threshold > self.composition_hard_boundary_threshold:
            raise ValueError("composition_soft_boundary_threshold cannot exceed composition_hard_boundary_threshold.")
        if self.minimum_action_margin > self.strong_action_margin:
            raise ValueError("minimum_action_margin cannot exceed strong_action_margin.")
        if self.minimum_shot_margin > self.strong_shot_margin:
            raise ValueError("minimum_shot_margin cannot exceed strong_shot_margin.")
        if self.action_conflict_penalty > self.strong_action_conflict_penalty:
            raise ValueError("action_conflict_penalty cannot exceed strong_action_conflict_penalty.")
        if self.participant_mode_weak_penalty > self.participant_mode_strong_penalty:
            raise ValueError("participant_mode_weak_penalty cannot exceed participant_mode_strong_penalty.")
        if self.main_vs_reaction_weak_penalty > self.main_vs_reaction_supported_penalty:
            raise ValueError("main_vs_reaction_weak_penalty cannot exceed main_vs_reaction_supported_penalty.")
        if self.main_vs_reaction_supported_penalty > self.main_vs_reaction_strong_penalty:
            raise ValueError("main_vs_reaction_supported_penalty cannot exceed main_vs_reaction_strong_penalty.")

        if not self.active_weight_names():
            raise ValueError("At least one scene-grouping signal must be enabled.")

    def active_weight_names(self) -> tuple[str, ...]:
        active = ["semantic"]
        if self.action_weight > 0.0:
            active.append("action")
        if self.include_people and self.people_weight > 0.0:
            active.append("people")
        if self.layout_weight > 0.0:
            active.append("layout")
        if self.include_subject_scene_embedding and self.subject_scene_weight > 0.0:
            active.append("subject_scene")
        if self.include_background_embedding and self.background_weight > 0.0:
            active.append("background")
        if self.time_weight > 0.0:
            active.append("time")
        if self.include_composition and self.composition_weight > 0.0:
            active.append("composition")
        if self.include_color and self.color_weight > 0.0:
            active.append("color")
        return tuple(active)

    def normalized_weights(
        self,
        available_names: Iterable[str] | None = None,
    ) -> dict[str, float]:
        weights = {
            "semantic": self.semantic_weight,
            "action": self.action_weight,
            "people": self.people_weight if self.include_people else 0.0,
            "layout": self.layout_weight,
            "subject_scene": self.subject_scene_weight if self.include_subject_scene_embedding else 0.0,
            "background": self.background_weight if self.include_background_embedding else 0.0,
            "time": self.time_weight,
            "composition": self.composition_weight if self.include_composition else 0.0,
            "color": self.color_weight if self.include_color else 0.0,
        }
        if available_names is not None:
            allowed = set(available_names)
            weights = {key: value for key, value in weights.items() if key in allowed}
        total = sum(value for value in weights.values() if value > 0.0)
        if total <= 0.0:
            raise ValueError("At least one scene-grouping weight must be active.")
        return {
            key: value / total
            for key, value in weights.items()
            if value > 0.0
        }

    def normalized_transition_weights(self) -> dict[str, float]:
        weights = {
            "semantic": self.transition_semantic_weight,
            "action": self.transition_action_weight,
            "people": self.transition_people_weight,
            "layout": self.transition_layout_weight,
            "background": self.transition_background_weight,
            "composition": self.transition_composition_weight,
            "temporal": self.transition_temporal_weight,
        }
        total = sum(value for value in weights.values() if value > 0.0)
        if total <= 0.0:
            raise ValueError("At least one transition weight must be active.")
        return {
            key: value / total
            for key, value in weights.items()
            if value > 0.0
        }

    def cache_signature(self) -> dict[str, object]:
        return {
            "minimum_pair_similarity": self.minimum_pair_similarity,
            "strong_pair_similarity": self.strong_pair_similarity,
            "minimum_group_cohesion": self.minimum_group_cohesion,
            "session_gap_seconds": self.session_gap_seconds,
            "maximum_soft_time_gap_seconds": self.maximum_soft_time_gap_seconds,
            "maximum_hard_time_gap_seconds": self.maximum_hard_time_gap_seconds,
            "maximum_scene_span_seconds": self.maximum_scene_span_seconds,
            "maximum_adjacent_gap_within_scene_seconds": self.maximum_adjacent_gap_within_scene_seconds,
            "hard_boundary_threshold": self.hard_boundary_threshold,
            "soft_boundary_threshold": self.soft_boundary_threshold,
            "semantic_hard_boundary_threshold": self.semantic_hard_boundary_threshold,
            "semantic_soft_boundary_threshold": self.semantic_soft_boundary_threshold,
            "composition_hard_boundary_threshold": self.composition_hard_boundary_threshold,
            "composition_soft_boundary_threshold": self.composition_soft_boundary_threshold,
            "layout_support_threshold": self.layout_support_threshold,
            "layout_hard_support_threshold": self.layout_hard_support_threshold,
            "background_hard_support_threshold": self.background_hard_support_threshold,
            "boundary_support_margin": self.boundary_support_margin,
            "boundary_context_window": self.boundary_context_window,
            "scene_neighbor_count": self.scene_neighbor_count,
            "adjacent_timeline_radius": self.adjacent_timeline_radius,
            "cross_boundary_candidate_count": self.cross_boundary_candidate_count,
            "maximum_timeline_edge_distance": self.maximum_timeline_edge_distance,
            "allowed_internal_outliers": self.allowed_internal_outliers,
            "maximum_internal_timeline_gaps": self.maximum_internal_timeline_gaps,
            "maximum_internal_gap_size": self.maximum_internal_gap_size,
            "tiny_segment_max_size": self.tiny_segment_max_size,
            "minimum_group_size": self.minimum_group_size,
            "maximum_group_size": self.maximum_group_size,
            "target_group_size": self.target_group_size,
            "weights": self.normalized_weights(),
            "penalties": {
                "minimum_action_margin": self.minimum_action_margin,
                "strong_action_margin": self.strong_action_margin,
                "minimum_shot_margin": self.minimum_shot_margin,
                "strong_shot_margin": self.strong_shot_margin,
                "action_conflict_penalty": self.action_conflict_penalty,
                "strong_action_conflict_penalty": self.strong_action_conflict_penalty,
                "participant_mode_conflict_penalty": self.participant_mode_conflict_penalty,
                "participant_mode_weak_penalty": self.participant_mode_weak_penalty,
                "participant_mode_strong_penalty": self.participant_mode_strong_penalty,
                "shot_mode_conflict_penalty": self.shot_mode_conflict_penalty,
                "main_vs_reaction_penalty": self.main_vs_reaction_penalty,
                "main_vs_reaction_weak_penalty": self.main_vs_reaction_weak_penalty,
                "main_vs_reaction_supported_penalty": self.main_vs_reaction_supported_penalty,
                "main_vs_reaction_strong_penalty": self.main_vs_reaction_strong_penalty,
                "hard_transition_penalty": self.hard_transition_penalty,
                "soft_transition_penalty": self.soft_transition_penalty,
                "temporal_bridge_penalty": self.temporal_bridge_penalty,
                "soft_cross_boundary_penalty": self.soft_cross_boundary_penalty,
                "cross_soft_boundary_similarity": self.cross_soft_boundary_similarity,
                "internal_soft_boundary_penalty": self.internal_soft_boundary_penalty,
                "singleton_recovery_margin": self.singleton_recovery_margin,
                "tiny_segment_merge_similarity": self.tiny_segment_merge_similarity,
                "weak_boundary_merge_similarity": self.weak_boundary_merge_similarity,
                "supported_boundary_merge_similarity": self.supported_boundary_merge_similarity,
                "conflict_visual_support_threshold": self.conflict_visual_support_threshold,
                "conflict_strong_visual_support_threshold": self.conflict_strong_visual_support_threshold,
                "conflict_subject_scene_support_threshold": self.conflict_subject_scene_support_threshold,
                "conflict_subject_scene_strong_support_threshold": self.conflict_subject_scene_strong_support_threshold,
                "conflict_composition_support_threshold": self.conflict_composition_support_threshold,
                "conflict_composition_strong_support_threshold": self.conflict_composition_strong_support_threshold,
                "conflict_layout_support_threshold": self.conflict_layout_support_threshold,
                "conflict_layout_strong_support_threshold": self.conflict_layout_strong_support_threshold,
                "continuity_semantic_similarity": self.continuity_semantic_similarity,
                "continuity_subject_scene_similarity": self.continuity_subject_scene_similarity,
                "continuity_composition_similarity": self.continuity_composition_similarity,
                "continuity_layout_similarity": self.continuity_layout_similarity,
                "continuity_background_similarity": self.continuity_background_similarity,
            },
            "transition_weights": self.normalized_transition_weights(),
            "include_people": self.include_people,
            "include_color": self.include_color,
            "include_composition": self.include_composition,
            "include_subject_scene_embedding": self.include_subject_scene_embedding,
            "include_background_embedding": self.include_background_embedding,
            "merge_non_adjacent_scenes": self.merge_non_adjacent_scenes,
            "composition_feature_version": self.composition_feature_version,
            "subject_scene_preprocessing_version": self.subject_scene_preprocessing_version,
            "feature_version": self.feature_version,
            "transition_version": self.transition_version,
            "boundary_reliability_version": self.boundary_reliability_version,
            "shot_confidence_version": self.shot_confidence_version,
            "singleton_recovery_version": self.singleton_recovery_version,
            "prototype_version": self.prototype_version,
            "result_cache_version": self.result_cache_version,
            "algorithm_version": self.algorithm_version,
        }

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["normalized_weights"] = self.normalized_weights()
        payload["normalized_transition_weights"] = self.normalized_transition_weights()
        payload["preset"] = infer_preset(self).value
        return payload


_PRESET_VALUES = {
    VibeGroupingPreset.SESSION: dict(
        semantic_weight=0.42,
        action_weight=0.10,
        people_weight=0.18,
        layout_weight=0.06,
        subject_scene_weight=0.06,
        background_weight=0.03,
        time_weight=0.15,
        composition_weight=0.04,
        color_weight=0.02,
        minimum_pair_similarity=0.60,
        strong_pair_similarity=0.76,
        minimum_group_cohesion=0.63,
        session_gap_seconds=35 * 60,
        maximum_soft_time_gap_seconds=25 * 60,
        maximum_scene_span_seconds=60 * 60,
        maximum_adjacent_gap_within_scene_seconds=10 * 60,
        hard_boundary_threshold=0.76,
        soft_boundary_threshold=0.58,
        scene_neighbor_count=16,
        adjacent_timeline_radius=5,
        cross_boundary_candidate_count=2,
        maximum_timeline_edge_distance=12,
        maximum_group_size=80,
        target_group_size=24,
        include_subject_scene_embedding=False,
        include_background_embedding=False,
    ),
    VibeGroupingPreset.BALANCED_SCENES: dict(
        semantic_weight=0.27,
        action_weight=0.14,
        people_weight=0.08,
        layout_weight=0.13,
        subject_scene_weight=0.19,
        background_weight=0.04,
        time_weight=0.04,
        composition_weight=0.11,
        color_weight=0.0,
        minimum_pair_similarity=0.74,
        strong_pair_similarity=0.86,
        minimum_group_cohesion=0.76,
        hard_boundary_threshold=0.32,
        soft_boundary_threshold=0.22,
        scene_neighbor_count=6,
        adjacent_timeline_radius=3,
        cross_boundary_candidate_count=1,
        maximum_timeline_edge_distance=6,
        maximum_group_size=40,
        target_group_size=18,
        maximum_scene_span_seconds=600,
        maximum_adjacent_gap_within_scene_seconds=180,
        allowed_internal_outliers=2,
        minimum_action_margin=0.012,
        strong_action_margin=0.025,
        minimum_shot_margin=0.025,
        strong_shot_margin=0.045,
        action_conflict_penalty=0.20,
        strong_action_conflict_penalty=0.30,
        participant_mode_conflict_penalty=0.08,
        participant_mode_weak_penalty=0.03,
        participant_mode_strong_penalty=0.10,
        shot_mode_conflict_penalty=0.03,
        main_vs_reaction_penalty=0.08,
        main_vs_reaction_weak_penalty=0.03,
        main_vs_reaction_supported_penalty=0.08,
        main_vs_reaction_strong_penalty=0.16,
        conflict_visual_support_threshold=0.22,
        conflict_strong_visual_support_threshold=0.32,
        conflict_subject_scene_support_threshold=0.20,
        conflict_subject_scene_strong_support_threshold=0.30,
        conflict_composition_support_threshold=0.30,
        conflict_composition_strong_support_threshold=0.42,
        conflict_layout_support_threshold=0.22,
        conflict_layout_strong_support_threshold=0.20,
        continuity_semantic_similarity=0.88,
        continuity_subject_scene_similarity=0.88,
        continuity_composition_similarity=0.86,
        continuity_layout_similarity=0.82,
        continuity_background_similarity=0.92,
        soft_cross_boundary_penalty=0.14,
        cross_soft_boundary_similarity=0.86,
        internal_soft_boundary_penalty=0.10,
        boundary_context_window=2,
        boundary_support_margin=0.04,
        include_subject_scene_embedding=True,
        include_background_embedding=True,
        adjacent_merge_similarity=0.84,
        singleton_recovery_similarity=0.78,
        singleton_recovery_margin=0.04,
        tiny_segment_max_size=2,
        tiny_segment_merge_similarity=0.80,
        weak_boundary_merge_similarity=0.82,
        supported_boundary_merge_similarity=0.88,
    ),
    VibeGroupingPreset.TIGHT_SCENES: dict(
        semantic_weight=0.24,
        action_weight=0.16,
        people_weight=0.07,
        layout_weight=0.15,
        subject_scene_weight=0.22,
        background_weight=0.03,
        time_weight=0.02,
        composition_weight=0.11,
        color_weight=0.0,
        minimum_pair_similarity=0.78,
        strong_pair_similarity=0.89,
        minimum_group_cohesion=0.80,
        hard_boundary_threshold=0.28,
        soft_boundary_threshold=0.18,
        scene_neighbor_count=4,
        adjacent_timeline_radius=2,
        cross_boundary_candidate_count=0,
        maximum_timeline_edge_distance=4,
        maximum_group_size=25,
        target_group_size=12,
        maximum_scene_span_seconds=300,
        maximum_adjacent_gap_within_scene_seconds=90,
        allowed_internal_outliers=1,
        minimum_action_margin=0.010,
        strong_action_margin=0.022,
        minimum_shot_margin=0.022,
        strong_shot_margin=0.040,
        action_conflict_penalty=0.22,
        strong_action_conflict_penalty=0.32,
        participant_mode_conflict_penalty=0.08,
        participant_mode_weak_penalty=0.03,
        participant_mode_strong_penalty=0.10,
        shot_mode_conflict_penalty=0.04,
        main_vs_reaction_penalty=0.08,
        main_vs_reaction_weak_penalty=0.03,
        main_vs_reaction_supported_penalty=0.08,
        main_vs_reaction_strong_penalty=0.16,
        continuity_semantic_similarity=0.90,
        continuity_subject_scene_similarity=0.90,
        continuity_composition_similarity=0.88,
        continuity_layout_similarity=0.84,
        continuity_background_similarity=0.93,
        soft_cross_boundary_penalty=0.18,
        cross_soft_boundary_similarity=0.90,
        internal_soft_boundary_penalty=0.12,
        boundary_context_window=2,
        boundary_support_margin=0.03,
        include_subject_scene_embedding=True,
        include_background_embedding=True,
        adjacent_merge_similarity=0.88,
        singleton_recovery_similarity=0.81,
        singleton_recovery_margin=0.05,
        tiny_segment_max_size=2,
        tiny_segment_merge_similarity=0.83,
        weak_boundary_merge_similarity=0.85,
        supported_boundary_merge_similarity=0.91,
        transition_semantic_weight=0.26,
        transition_action_weight=0.18,
        transition_people_weight=0.07,
        transition_layout_weight=0.16,
        transition_background_weight=0.06,
        transition_composition_weight=0.24,
        transition_temporal_weight=0.03,
    ),
}


def preset_config(
    preset: VibeGroupingPreset,
    **overrides: object,
) -> VibeGroupingConfig:
    values = dict(_PRESET_VALUES[preset])
    values.update(overrides)
    return VibeGroupingConfig(**values)


def infer_preset(config: VibeGroupingConfig) -> VibeGroupingPreset:
    for preset, values in _PRESET_VALUES.items():
        if all(getattr(config, key) == value for key, value in values.items()):
            return preset
    return VibeGroupingPreset.BALANCED_SCENES
