from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_detector.face_database_sqlite import SQLiteFaceDatabase
from grouping.models import VibeGroup, VibeGroupingResult, VibeImageFeatures
from grouping.vibe.cache import VibeFeatureCache, VibeGroupingResultCache
from grouping.vibe.clusterer import VibeClusterer
from grouping.vibe.config import VibeGroupingConfig, VibeGroupingPreset, preset_config
from grouping.vibe.embedder import FallbackVisualEmbedder, VibeEmbedder, load_embedder
from grouping.vibe.labels import build_group_label
from grouping.vibe.processor import VibeGroupingProcessor
from grouping.vibe.prototypes import PrototypeConcept, ScenePrototypeTable
from grouping.vibe.similarity import CombinedSimilarityComputer, ScenePairComponents
from grouping.vibe.temporal_segments import (
    SceneSegment,
    SceneTransitionScore,
    TemporalSegmentation,
    TemporalSession,
    _compute_transition,
    _recover_singletons,
    _recover_tiny_segments,
    segment_by_time,
)
from scan_controls import CancellationToken, ScanCancelledError
from ui.models import ResultGroup, ResultItem, ScanResultMessage
from ui.view_model import PhotoCleanerViewModel


ACTION_KEYS = (
    "aisle",
    "rings",
    "kiss",
    "applause",
    "speech",
    "audience",
    "couple_portrait",
    "family_portrait",
    "walking",
    "makeup",
    "dress",
    "cake",
    "detail",
    "eating",
    "dancing",
)
SCENE_KEYS = ("ceremony", "window", "outdoor", "mirror", "reception", "table")
SHOT_KEYS = ("portrait", "main_action", "audience", "detail", "reaction")

ACTION_INDEX = {key: index for index, key in enumerate(ACTION_KEYS)}
SCENE_INDEX = {key: index for index, key in enumerate(SCENE_KEYS)}
SHOT_INDEX = {key: index for index, key in enumerate(SHOT_KEYS)}


def _normalized(values: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _score_vector(index_map: dict[str, int], values: dict[str, float] | None) -> np.ndarray | None:
    if not values:
        return None
    vector = np.zeros(len(index_map), dtype=np.float32)
    for key, score in values.items():
        vector[index_map[key]] = score
    return _normalized(vector)


def _prototype_table() -> ScenePrototypeTable:
    dim = 4
    action_concepts = tuple(
        PrototypeConcept(
            key=key,
            phrase=key.replace("_", " "),
            label={
                "aisle": "Bride Walking Down The Aisle",
                "rings": "Ring Exchange",
                "kiss": "First Kiss",
                "applause": "Guests Applauding",
                "speech": "Speech",
                "audience": "Audience Listening",
                "couple_portrait": "Couple Portrait",
                "family_portrait": "Family Portrait",
                "walking": "Walking Together",
                "makeup": "Bride Makeup",
                "dress": "Getting Dressed",
                "cake": "Cake Cutting",
                "detail": "Detail",
                "eating": "Eating",
                "dancing": "Dancing",
            }[key],
            category="action",
            tags={
                "aisle": ("aisle", "walking", "ceremony", "main_action"),
                "rings": ("rings", "ceremony", "interaction", "main_action"),
                "kiss": ("kiss", "ceremony", "interaction", "main_action"),
                "applause": ("applause", "reaction", "audience"),
                "speech": ("speech", "speaker", "main_action"),
                "audience": ("audience", "speech", "reaction"),
                "couple_portrait": ("portrait", "couple", "posed"),
                "family_portrait": ("portrait", "family", "group", "posed"),
                "walking": ("walking", "movement", "couple"),
                "makeup": ("makeup", "preparation", "interaction"),
                "dress": ("dressing", "preparation"),
                "cake": ("cake", "interaction", "main_action"),
                "detail": ("detail",),
                "eating": ("eating", "group"),
                "dancing": ("dance", "group"),
            }[key],
        )
        for key in ACTION_KEYS
    )
    scene_concepts = tuple(
        PrototypeConcept(
            key=key,
            phrase=key.replace("_", " "),
            label=key.replace("_", " ").title(),
            category="scene",
            tags=(key,),
        )
        for key in SCENE_KEYS
    )
    shot_concepts = tuple(
        PrototypeConcept(
            key=key,
            phrase=key.replace("_", " "),
            label=key.replace("_", " ").title(),
            category="shot",
            tags=(key,),
        )
        for key in SHOT_KEYS
    )
    return ScenePrototypeTable(
        metadata={
            "sha256": "test-prototypes-v1",
            "semantic_model_fingerprint": "test-model",
            "embedding_dimension": dim,
        },
        action_embeddings=np.zeros((len(action_concepts), dim), dtype=np.float32),
        scene_embeddings=np.zeros((len(scene_concepts), dim), dtype=np.float32),
        shot_embeddings=np.zeros((len(shot_concepts), dim), dtype=np.float32),
        action_concepts=action_concepts,
        scene_concepts=scene_concepts,
        shot_concepts=shot_concepts,
    )


TEST_PROTOTYPE_TABLE = _prototype_table()


def _feature(
    path: str,
    *,
    semantic: list[float],
    action: dict[str, float] | None = None,
    scene: dict[str, float] | None = None,
    shot: dict[str, float] | None = None,
    timestamp: float | None = None,
    people: tuple[str, ...] = (),
    people_names: tuple[str, ...] = (),
    participant_mode: str | None = None,
    face_count: int = 0,
    brightness: float = 0.5,
    quality: float | None = None,
    background: list[float] | None = None,
    subject_scene: list[float] | None = None,
    layout: list[float] | None = None,
) -> VibeImageFeatures:
    color = _normalized([brightness, 1.0 - brightness, 0.25, 0.75])
    composition = _normalized([face_count / 8.0, 0.5, 0.25, brightness, 0.2, 0.1])
    layout_vector = None if layout is None else _normalized(layout)
    if layout_vector is None and face_count > 0:
        layout_vector = _normalized(
            [
                face_count / 8.0,
                1.0 if (participant_mode or "") == "solo" else 0.0,
                1.0 if (participant_mode or "") == "couple" else 0.0,
                1.0 if (participant_mode or "") in {"family_group", "small_group"} else 0.0,
                0.4 if face_count > 0 else 0.0,
                0.2 if face_count > 2 else 0.0,
            ]
        )
    metadata = {
        "participant_mode": participant_mode or (
            "solo"
            if face_count == 1
            else "couple"
            if face_count == 2
            else "family_group"
            if face_count >= 3
            else "none"
        ),
        "orientation": "landscape",
        "shot_scale_category": (
            "close"
            if face_count == 1
            else "medium"
            if face_count == 2
            else "wide"
        ),
        "face_count_bucket": (
            "zero"
            if face_count <= 0
            else "one"
            if face_count == 1
            else "couple"
            if face_count == 2
            else "small_group"
            if face_count <= 4
            else "crowd"
        ),
        "subject_centroid_x": 0.5,
        "subject_centroid_y": 0.5,
        "subject_horizontal_spread": 0.05 if face_count <= 2 else 0.18,
        "subject_vertical_spread": 0.05 if face_count <= 2 else 0.14,
        "primary_subject_scale": 0.12 if face_count <= 2 else 0.05,
        "warmth": 0.55,
        "black_and_white_likelihood": 0.0,
    }
    return VibeImageFeatures(
        image_path=path,
        semantic_embedding=_normalized(semantic),
        capture_timestamp=timestamp,
        timestamp_source="exif" if timestamp is not None else "missing",
        recognized_person_ids=people,
        color_features=color,
        composition_features=composition,
        face_layout=layout_vector,
        face_scale_summary=None if layout_vector is None else layout_vector[-3:],
        subject_scene_embedding=_normalized(subject_scene if subject_scene is not None else semantic),
        background_embedding=None if background is None else _normalized(background),
        action_scores=_score_vector(ACTION_INDEX, action),
        scene_scores=_score_vector(SCENE_INDEX, scene),
        shot_type_scores=_score_vector(SHOT_INDEX, shot),
        width=1200,
        height=800,
        file_mtime_ns=1,
        file_size=1,
        quality_score=quality,
        brightness=brightness,
        face_count=face_count,
        face_area_ratio=0.05 * face_count,
        dominant_people_names=people_names,
        metadata=metadata,
    )


def _transition(
    left_image: str,
    right_image: str,
    *,
    combined: float = 0.0,
    semantic_change: float = 0.0,
    action_change: float = 0.0,
    people_change: float = 0.0,
    layout_change: float = 0.0,
    subject_scene_change: float = 0.0,
    background_change: float = 0.0,
    composition_change: float = 0.0,
    temporal_gap: float = 0.0,
    boundary_reliability: str | None = None,
    hard_boundary: bool = False,
    soft_boundary: bool = False,
    accepted_hard_boundary: bool | None = None,
    accepted_soft_boundary: bool | None = None,
) -> SceneTransitionScore:
    reliability = boundary_reliability or ("hard" if hard_boundary else "supported" if soft_boundary else "none")
    return SceneTransitionScore(
        left_image=left_image,
        right_image=right_image,
        semantic_change=semantic_change,
        action_change=action_change,
        people_change=people_change,
        layout_change=layout_change,
        subject_scene_change=subject_scene_change,
        background_change=background_change,
        composition_change=composition_change,
        temporal_gap=temporal_gap,
        combined_transition=combined,
        boundary_reliability=reliability,
        is_hard_boundary=hard_boundary,
        is_soft_boundary=soft_boundary,
        continuity_override_applied=False,
        continuity_score=0.0,
        visual_support_score=0.0,
        continuity_reasons=(),
        participant_conflict_strength="none",
        shot_conflict_confident=False,
        left_shot_margin=0.0,
        right_shot_margin=0.0,
        left_shot_confidence="uncertain",
        right_shot_confidence="uncertain",
        recovery_crossable=reliability in {"none", "weak"},
        hard_reasons=(),
        soft_reasons=(),
        accepted_hard_boundary=hard_boundary if accepted_hard_boundary is None else accepted_hard_boundary,
        accepted_soft_boundary=soft_boundary if accepted_soft_boundary is None else accepted_soft_boundary,
    )


def _manual_segmentation(
    features: list[VibeImageFeatures],
    transitions: list[SceneTransitionScore],
    *,
    segments: list[tuple[int, ...]] | None = None,
) -> TemporalSegmentation:
    scene_segments = []
    for member_indices in segments or [tuple(range(len(features)))]:
        first = member_indices[0]
        last = member_indices[-1]
        boundary_before = (
            transitions[first - 1]
            if first > 0 and (transitions[first - 1].accepted_hard_boundary or transitions[first - 1].accepted_soft_boundary)
            else None
        )
        boundary_after = (
            transitions[last]
            if last < len(transitions) and (transitions[last].accepted_hard_boundary or transitions[last].accepted_soft_boundary)
            else None
        )
        scene_segments.append(
            SceneSegment(
                member_indices=member_indices,
                boundary_before=boundary_before,
                boundary_after=boundary_after,
            )
        )
    return TemporalSegmentation(
        sessions=[
            TemporalSession(
                ordered_features=tuple(features),
                transitions=tuple(transitions),
                scene_segments=tuple(scene_segments),
            )
        ],
        untimed=[],
    )


class CountingEmbedder(VibeEmbedder):
    def __init__(self, *, fingerprint: str = "test-model") -> None:
        self.calls = 0
        self._fingerprint = fingerprint

    @property
    def embedding_dimension(self) -> int:
        return 4

    @property
    def model_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def provider(self) -> str:
        return "CPUExecutionProvider"

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        self.calls += 1
        vectors = []
        for image in images:
            means = image.mean(axis=(0, 1)).astype(np.float32) / 255.0
            vectors.append(_normalized([float(means[2]), float(means[1]), float(means[0]), 0.2]))
        return np.stack(vectors, axis=0)


def _scoring_prototype_table(*, fingerprint: str) -> ScenePrototypeTable:
    action_concepts = (
        PrototypeConcept("warm", "warm", "Warm Portrait", "action", ("portrait",)),
        PrototypeConcept("cool", "cool", "Cool Portrait", "action", ("portrait",)),
    )
    scene_concepts = (
        PrototypeConcept("indoor", "indoor", "Indoor", "scene", ("indoor",)),
        PrototypeConcept("outdoor", "outdoor", "Outdoor", "scene", ("outdoor",)),
    )
    shot_concepts = (
        PrototypeConcept("portrait", "portrait", "Portrait", "shot", ("portrait",)),
        PrototypeConcept("detail", "detail", "Detail", "shot", ("detail",)),
    )
    return ScenePrototypeTable(
        metadata={
            "sha256": fingerprint,
            "semantic_model_fingerprint": "test-model",
            "embedding_dimension": 4,
        },
        action_embeddings=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        scene_embeddings=np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        shot_embeddings=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        action_concepts=action_concepts,
        scene_concepts=scene_concepts,
        shot_concepts=shot_concepts,
    )


class VibeConfigTests(unittest.TestCase):
    def test_normalizes_scene_weights(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        weights = config.normalized_weights()
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertIn("action", weights)
        self.assertIn("layout", weights)

    def test_preset_configures_tight_scenes(self) -> None:
        config = preset_config(VibeGroupingPreset.TIGHT_SCENES)
        self.assertAlmostEqual(config.minimum_pair_similarity, 0.78)
        self.assertTrue(config.include_background_embedding)
        self.assertEqual(config.scene_neighbor_count, 4)
        self.assertAlmostEqual(config.hard_boundary_threshold, 0.28)

    def test_load_embedder_uses_fallback_when_model_missing(self) -> None:
        config = VibeGroupingConfig(semantic_model_filename="does_not_exist.onnx", allow_visual_fallback=True)
        embedder = load_embedder(config)
        self.assertTrue(embedder.uses_fallback)
        self.assertEqual(embedder.provider, "CPUExecutionProvider")


class SceneGroupingTests(unittest.TestCase):
    def _cluster(
        self,
        features: list[VibeImageFeatures],
        *,
        config: VibeGroupingConfig | None = None,
    ) -> tuple[list[tuple[str, ...]], list[str]]:
        config = config or preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        segmentation = segment_by_time(features, config=config, similarity=similarity)
        clusterer = VibeClusterer(config)
        groups, ungrouped = clusterer.cluster(segmentation, similarity=similarity)
        return (
            [tuple(member.image_path for member in group.members) for group in groups],
            [item.image_path for item in ungrouped],
        )

    def test_ceremony_splits_immediate_actions(self) -> None:
        groups, _ungrouped = self._cluster(
            [
                _feature("aisle.jpg", semantic=[1.0, 0.1, 0.0, 0.2], action={"aisle": 1.0}, scene={"ceremony": 1.0}, shot={"main_action": 1.0}, timestamp=10.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
                _feature("rings.jpg", semantic=[0.92, 0.12, 0.0, 0.25], action={"rings": 1.0}, scene={"ceremony": 1.0}, shot={"main_action": 1.0}, timestamp=20.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
                _feature("kiss.jpg", semantic=[0.90, 0.2, 0.0, 0.18], action={"kiss": 1.0}, scene={"ceremony": 1.0}, shot={"main_action": 1.0}, timestamp=28.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
                _feature("applause.jpg", semantic=[0.82, 0.25, 0.15, 0.1], action={"applause": 1.0}, scene={"ceremony": 1.0}, shot={"reaction": 1.0}, timestamp=36.0, people=("bride", "groom"), face_count=5, participant_mode="crowd"),
            ]
        )
        self.assertEqual(groups, [("aisle.jpg",), ("rings.jpg",), ("kiss.jpg",), ("applause.jpg",)])

    def test_speech_splits_reaction_sequences(self) -> None:
        groups, _ungrouped = self._cluster(
            [
                _feature("speaker_1.jpg", semantic=[1.0, 0.0, 0.1, 0.0], action={"speech": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=10.0, people=("speaker",), face_count=1, participant_mode="solo"),
                _feature("speaker_2.jpg", semantic=[0.99, 0.02, 0.08, 0.0], action={"speech": 1.0}, scene={"reception": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("speaker",), face_count=1, participant_mode="solo"),
                _feature("speaker_3.jpg", semantic=[0.97, 0.03, 0.06, 0.0], action={"speech": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=26.0, people=("speaker",), face_count=1, participant_mode="solo"),
                _feature("audience.jpg", semantic=[0.82, 0.22, 0.2, 0.0], action={"audience": 1.0}, scene={"reception": 1.0}, shot={"audience": 1.0}, timestamp=34.0, face_count=6, participant_mode="crowd"),
                _feature("applause.jpg", semantic=[0.80, 0.28, 0.22, 0.0], action={"applause": 1.0}, scene={"reception": 1.0}, shot={"reaction": 1.0}, timestamp=42.0, face_count=6, participant_mode="crowd"),
                _feature("speaker_resume.jpg", semantic=[0.98, 0.01, 0.09, 0.0], action={"speech": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=56.0, people=("speaker",), face_count=1, participant_mode="solo"),
            ]
        )
        self.assertEqual(
            groups,
            [
                ("speaker_1.jpg", "speaker_2.jpg", "speaker_3.jpg"),
                ("audience.jpg",),
                ("applause.jpg",),
                ("speaker_resume.jpg",),
            ],
        )

    def test_portrait_setup_keeps_window_sequence_and_splits_family(self) -> None:
        groups, _ungrouped = self._cluster(
            [
                _feature("window_1.jpg", semantic=[1.0, 0.1, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
                _feature("window_2.jpg", semantic=[0.98, 0.12, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
                _feature("window_close.jpg", semantic=[0.97, 0.15, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=26.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
                _feature("outdoor_walk.jpg", semantic=[0.80, 0.15, 0.18, 0.2], action={"walking": 1.0}, scene={"outdoor": 1.0}, shot={"main_action": 1.0}, timestamp=60.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
                _feature("family.jpg", semantic=[0.76, 0.2, 0.18, 0.0], action={"family_portrait": 1.0}, scene={"outdoor": 1.0}, shot={"portrait": 1.0}, timestamp=74.0, people=("bride", "groom", "parent1", "parent2"), face_count=5, participant_mode="family_group"),
            ]
        )
        self.assertEqual(
            groups,
            [
                ("window_1.jpg", "window_2.jpg", "window_close.jpg"),
                ("outdoor_walk.jpg",),
                ("family.jpg",),
            ],
        )

    def test_preparation_splits_makeup_dress_and_mirror(self) -> None:
        groups, _ungrouped = self._cluster(
            [
                _feature("makeup_1.jpg", semantic=[1.0, 0.0, 0.1, 0.0], action={"makeup": 1.0}, scene={"mirror": 1.0}, shot={"main_action": 1.0}, timestamp=10.0, people=("bride", "artist"), face_count=2, participant_mode="couple"),
                _feature("makeup_2.jpg", semantic=[0.98, 0.0, 0.12, 0.0], action={"makeup": 1.0}, scene={"mirror": 1.0}, shot={"detail": 1.0}, timestamp=18.0, people=("bride", "artist"), face_count=2, participant_mode="couple"),
                _feature("dress.jpg", semantic=[0.82, 0.15, 0.1, 0.0], action={"dress": 1.0}, scene={"mirror": 1.0}, shot={"main_action": 1.0}, timestamp=34.0, people=("bride",), face_count=1, participant_mode="solo"),
                _feature("mirror_portrait.jpg", semantic=[0.74, 0.38, 0.02, 0.0], action={"couple_portrait": 1.0}, scene={"mirror": 1.0}, shot={"portrait": 1.0}, timestamp=52.0, people=("bride",), face_count=1, participant_mode="solo"),
            ]
        )
        self.assertEqual(groups, [("makeup_1.jpg", "makeup_2.jpg"), ("dress.jpg",), ("mirror_portrait.jpg",)])

    def test_cake_cutting_multi_angle_stays_together(self) -> None:
        groups, _ungrouped = self._cluster(
            [
                _feature("cake_wide.jpg", semantic=[1.0, 0.0, 0.0, 0.12], action={"cake": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=10.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
                _feature("cake_close.jpg", semantic=[0.98, 0.0, 0.0, 0.16], action={"cake": 1.0}, scene={"reception": 1.0}, shot={"detail": 0.4, "main_action": 0.7}, timestamp=18.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
                _feature("cake_side.jpg", semantic=[0.96, 0.0, 0.0, 0.2], action={"cake": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=26.0, people=("bride", "groom"), face_count=2, participant_mode="couple"),
            ]
        )
        self.assertEqual(groups, [("cake_wide.jpg", "cake_close.jpg", "cake_side.jpg")])

    def test_same_venue_different_activities_do_not_merge(self) -> None:
        groups, _ungrouped = self._cluster(
            [
                _feature("eating.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"eating": 1.0}, scene={"reception": 1.0}, shot={"audience": 1.0}, timestamp=10.0, face_count=6, participant_mode="crowd"),
                _feature("speech.jpg", semantic=[0.92, 0.15, 0.0, 0.0], action={"speech": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=20.0, face_count=1, participant_mode="solo"),
                _feature("dancing.jpg", semantic=[0.84, 0.18, 0.2, 0.0], action={"dancing": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=30.0, face_count=6, participant_mode="crowd"),
                _feature("cake.jpg", semantic=[0.86, 0.0, 0.0, 0.22], action={"cake": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=40.0, face_count=2, participant_mode="couple"),
            ]
        )
        self.assertEqual(groups, [("eating.jpg",), ("speech.jpg",), ("dancing.jpg",), ("cake.jpg",)])

    def test_one_anomalous_detail_frame_does_not_create_three_portrait_groups(self) -> None:
        groups, ungrouped = self._cluster(
            [
                _feature("portrait_1.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("a", "b"), face_count=2, participant_mode="couple"),
                _feature("portrait_2.jpg", semantic=[0.99, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("a", "b"), face_count=2, participant_mode="couple"),
                _feature("detail.jpg", semantic=[0.7, 0.05, 0.0, 0.35], action={"detail": 1.0}, scene={"window": 1.0}, shot={"detail": 1.0}, timestamp=24.0, people=(), face_count=0, participant_mode="none"),
                _feature("portrait_3.jpg", semantic=[0.99, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=30.0, people=("a", "b"), face_count=2, participant_mode="couple"),
                _feature("portrait_4.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=38.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            ]
        )
        self.assertLessEqual(len(groups) + len(ungrouped), 2)
        self.assertTrue(any(group[0] == "portrait_1.jpg" for group in groups))

    def test_missing_timestamps_group_only_when_action_and_people_align(self) -> None:
        groups, ungrouped = self._cluster(
            [
                _feature("untimed_a.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"makeup": 1.0}, scene={"mirror": 1.0}, shot={"main_action": 1.0}, timestamp=None, people=("bride", "artist"), face_count=2, participant_mode="couple"),
                _feature("untimed_b.jpg", semantic=[0.98, 0.0, 0.1, 0.0], action={"makeup": 1.0}, scene={"mirror": 1.0}, shot={"detail": 0.4, "main_action": 0.6}, timestamp=None, people=("bride", "artist"), face_count=2, participant_mode="couple"),
                _feature("untimed_c.jpg", semantic=[0.55, 0.0, 0.7, 0.1], action={"eating": 1.0}, scene={"table": 1.0}, shot={"audience": 1.0}, timestamp=None, people=("guest1", "guest2"), face_count=5, participant_mode="crowd"),
            ]
        )
        self.assertEqual(groups[0], ("untimed_a.jpg", "untimed_b.jpg"))
        self.assertEqual(ungrouped, ["untimed_c.jpg"])

    def test_result_is_deterministic_for_input_order(self) -> None:
        features = [
            _feature("b.jpg", semantic=[0.98, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=20.0, people=("p1", "p2"), face_count=2, participant_mode="couple"),
            _feature("a.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("p1", "p2"), face_count=2, participant_mode="couple"),
            _feature("c.jpg", semantic=[0.0, 1.0, 0.0, 0.0], action={"speech": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=2000.0, people=("speaker",), face_count=1, participant_mode="solo"),
        ]
        groups_a, ungrouped_a = self._cluster(features)
        groups_b, ungrouped_b = self._cluster(list(reversed(features)))
        self.assertEqual(groups_a, groups_b)
        self.assertEqual(ungrouped_a, ungrouped_b)


class SceneBoundaryRuleTests(unittest.TestCase):
    class _StubTransitionSimilarity:
        def __init__(self, components: ScenePairComponents, *, shot_mode: str = "portrait") -> None:
            self._components = components
            self._shot_mode_value = shot_mode
            self.prototype_table = None

        def pair_components(self, _left: VibeImageFeatures, _right: VibeImageFeatures) -> ScenePairComponents:
            return self._components

        def _shot_mode(self, _feature: VibeImageFeatures) -> str:
            return self._shot_mode_value

    @staticmethod
    def _components(
        *,
        semantic: float,
        action: float = 1.0,
        people: float = 1.0,
        layout: float = 1.0,
        subject_scene: float = 1.0,
        background: float = 1.0,
        time: float = 1.0,
        composition: float = 1.0,
    ) -> ScenePairComponents:
        return ScenePairComponents(
            semantic=semantic,
            action=action,
            people=people,
            layout=layout,
            subject_scene=subject_scene,
            background=background,
            time=time,
            composition=composition,
            color=1.0,
            action_conflict_penalty=0.0,
            participant_mode_conflict_penalty=0.0,
            shot_mode_conflict_penalty=0.0,
            main_vs_reaction_penalty=0.0,
            temporal_bridge_penalty=0.0,
            transition_penalty=0.0,
            action_hard_conflict=False,
            action_soft_conflict=False,
            participant_mode_hard_conflict=False,
            participant_mode_soft_conflict=False,
            participant_conflict_strength="none",
            shot_mode_conflict=False,
            main_vs_reaction_conflict=False,
            main_vs_reaction_confident=False,
            action_confidence_mean=1.0,
            action_margin_mean=0.05,
            action_reliable=True,
            left_shot_margin=0.0,
            right_shot_margin=0.0,
            left_shot_confidence="strong",
            right_shot_confidence="strong",
            combined=0.8,
        )

    def test_midrange_transition_can_become_hard_boundary(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        left = _feature("left.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("a", "b"), face_count=2, participant_mode="couple")
        right = _feature("right.jpg", semantic=[0.52, 0.85, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("a", "b"), face_count=2, participant_mode="couple")
        similarity = self._StubTransitionSimilarity(
            self._components(
                semantic=0.52,
                action=0.90,
                people=0.95,
                layout=0.75,
                background=0.85,
                time=0.90,
                composition=0.27,
            )
        )
        transition = _compute_transition(left, right, config=config, similarity=similarity)
        self.assertGreaterEqual(transition.combined_transition, 0.36)
        self.assertLess(transition.combined_transition, 0.40)
        self.assertTrue(transition.is_hard_boundary)
        self.assertIn("composition_threshold", transition.hard_reasons)

    def test_composition_hard_trigger_overrides_combined_threshold(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        left = _feature("left.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("a", "b"), face_count=2, participant_mode="couple")
        right = _feature("right.jpg", semantic=[0.52, 0.85, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("a", "b"), face_count=2, participant_mode="couple")
        similarity = self._StubTransitionSimilarity(
            self._components(
                semantic=0.52,
                composition=0.27,
            )
        )
        transition = _compute_transition(left, right, config=config, similarity=similarity)
        self.assertLess(transition.combined_transition, config.hard_boundary_threshold)
        self.assertTrue(transition.is_hard_boundary)
        self.assertIn("composition_threshold", transition.hard_reasons)

    def test_action_uncertainty_is_neutral(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        left = _feature(
            "flat_a.jpg",
            semantic=[1.0, 0.0, 0.0, 0.0],
            action={"speech": 0.201, "audience": 0.200, "applause": 0.199, "walking": 0.198},
            scene={"reception": 1.0},
            shot={"main_action": 1.0},
            timestamp=10.0,
            face_count=1,
            participant_mode="solo",
        )
        right = _feature(
            "flat_b.jpg",
            semantic=[0.99, 0.0, 0.0, 0.0],
            action={"speech": 0.200, "audience": 0.199, "applause": 0.198, "walking": 0.197},
            scene={"reception": 1.0},
            shot={"main_action": 1.0},
            timestamp=18.0,
            face_count=1,
            participant_mode="solo",
        )
        self.assertFalse(similarity.action_profile(left).confident)
        self.assertFalse(similarity.action_profile(right).confident)
        self.assertEqual(similarity.action_similarity(left, right), 0.5)

    def test_confident_action_conflict_is_explicit(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        rings = _feature("rings.jpg", semantic=[1.0, 0.0, 0.1, 0.0], action={"rings": 1.0}, scene={"ceremony": 1.0}, shot={"main_action": 1.0}, timestamp=10.0, people=("bride", "groom"), face_count=2, participant_mode="couple")
        kiss = _feature("kiss.jpg", semantic=[0.92, 0.0, 0.1, 0.0], action={"kiss": 1.0}, scene={"ceremony": 1.0}, shot={"main_action": 1.0}, timestamp=18.0, people=("bride", "groom"), face_count=2, participant_mode="couple")
        pair = similarity.pair_components(rings, kiss)
        self.assertTrue(pair.action_hard_conflict)
        self.assertGreaterEqual(pair.action_conflict_penalty, config.strong_action_conflict_penalty)

    def test_subject_scene_similarity_prefers_same_local_setup(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        anchor = _feature("anchor.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("a", "b"), face_count=2, participant_mode="couple", subject_scene=[1.0, 0.0, 0.0, 0.0])
        same_setup = _feature("same.jpg", semantic=[0.99, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("a", "b"), face_count=2, participant_mode="couple", subject_scene=[0.98, 0.02, 0.0, 0.0])
        other_setup = _feature("other.jpg", semantic=[0.99, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=26.0, people=("a", "b"), face_count=2, participant_mode="couple", subject_scene=[0.0, 1.0, 0.0, 0.0])
        self.assertGreater(
            similarity.subject_scene_similarity(anchor, same_setup),
            similarity.subject_scene_similarity(anchor, other_setup),
        )

    def test_strong_continuity_suppresses_weak_participant_conflict(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        left = _feature("left.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("a", "b"), face_count=2, participant_mode="couple", subject_scene=[1.0, 0.0, 0.0, 0.0], background=[1.0, 0.0, 0.0, 0.0], layout=[0.25, 0.75, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1])
        right = _feature("right.jpg", semantic=[0.98, 0.02, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("a", "b"), face_count=3, participant_mode="small_group", subject_scene=[0.98, 0.02, 0.0, 0.0], background=[0.98, 0.02, 0.0, 0.0], layout=[0.25, 0.74, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1])
        transition = _compute_transition(left, right, config=config, similarity=similarity)
        self.assertEqual(transition.participant_conflict_strength, "weak")
        self.assertEqual(transition.boundary_reliability, "none")
        self.assertFalse(transition.accepted_soft_boundary)
        self.assertFalse(transition.accepted_hard_boundary)

    def test_strong_continuity_suppresses_uncertain_main_reaction_flip(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        left = _feature("left.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"speech": 1.0}, scene={"reception": 1.0}, shot={"main_action": 0.51, "reaction": 0.50}, timestamp=10.0, people=("speaker",), face_count=1, participant_mode="solo", subject_scene=[1.0, 0.0, 0.0, 0.0], background=[1.0, 0.0, 0.0, 0.0], layout=[0.6, 0.0, 0.0, 0.0])
        right = _feature("right.jpg", semantic=[0.99, 0.0, 0.0, 0.0], action={"speech": 1.0}, scene={"reception": 1.0}, shot={"reaction": 0.51, "main_action": 0.50}, timestamp=18.0, people=("speaker",), face_count=1, participant_mode="solo", subject_scene=[0.99, 0.01, 0.0, 0.0], background=[0.99, 0.01, 0.0, 0.0], layout=[0.6, 0.0, 0.0, 0.0])
        transition = _compute_transition(left, right, config=config, similarity=similarity)
        self.assertEqual(transition.left_shot_confidence, "uncertain")
        self.assertEqual(transition.right_shot_confidence, "uncertain")
        self.assertFalse(transition.shot_conflict_confident)
        self.assertEqual(transition.boundary_reliability, "none")

    def test_confident_main_reaction_with_visual_support_still_splits(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        speaker = _feature("speaker.jpg", semantic=[1.0, 0.0, 0.1, 0.0], action={"speech": 1.0}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=10.0, people=("speaker",), face_count=1, participant_mode="solo")
        audience = _feature("audience.jpg", semantic=[0.82, 0.22, 0.2, 0.0], action={"audience": 1.0}, scene={"reception": 1.0}, shot={"audience": 1.0}, timestamp=18.0, face_count=6, participant_mode="crowd")
        transition = _compute_transition(speaker, audience, config=config, similarity=similarity)
        self.assertIn(transition.boundary_reliability, {"supported", "hard"})
        self.assertFalse(transition.continuity_override_applied)

    def test_uncertain_action_transition_contributes_zero_action_change(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        left = _feature("flat_a.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"speech": 0.201, "audience": 0.200, "applause": 0.199, "walking": 0.198}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=10.0, face_count=1, participant_mode="solo")
        right = _feature("flat_b.jpg", semantic=[0.95, 0.05, 0.0, 0.0], action={"speech": 0.200, "audience": 0.199, "applause": 0.198, "walking": 0.197}, scene={"reception": 1.0}, shot={"main_action": 1.0}, timestamp=18.0, face_count=1, participant_mode="solo")
        transition = _compute_transition(left, right, config=config, similarity=similarity)
        self.assertEqual(transition.action_change, 0.0)


class SceneStructureTests(unittest.TestCase):
    def test_boundary_bypass_edges_are_rejected(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        features = [
            _feature("a.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("b.jpg", semantic=[0.99, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("c.jpg", semantic=[0.99, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=26.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("d.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=34.0, people=("a", "b"), face_count=2, participant_mode="couple"),
        ]
        segmentation = _manual_segmentation(
            features,
            [
                _transition("a.jpg", "b.jpg"),
                _transition("b.jpg", "c.jpg", combined=0.38, hard_boundary=True),
                _transition("c.jpg", "d.jpg"),
            ],
        )
        clusterer = VibeClusterer(config)
        groups, _ungrouped = clusterer.cluster(segmentation, similarity=similarity)
        self.assertEqual(
            [tuple(member.image_path for member in group.members) for group in groups],
            [("a.jpg", "b.jpg"), ("c.jpg", "d.jpg")],
        )
        self.assertIn(
            "crossed_hard_boundary",
            {entry["reason"] for entry in clusterer.last_diagnostics["rejected_edges"]},
        )

    def test_non_contiguous_group_is_split_by_timeline_islands(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        selected = {0, 1, 2, 7, 8, 14}
        features = []
        for index in range(15):
            if index in selected:
                features.append(
                    _feature(
                        f"scene_{index}.jpg",
                        semantic=[1.0, 0.0, 0.0, 0.0],
                        action={"couple_portrait": 1.0},
                        scene={"window": 1.0},
                        shot={"portrait": 1.0},
                        timestamp=float(index * 10),
                        people=("a", "b"),
                        face_count=2,
                        participant_mode="couple",
                    )
                )
            else:
                features.append(
                    _feature(
                        f"filler_{index}.jpg",
                        semantic=[0.0, 1.0, 0.0, 0.0],
                        action={"speech": 1.0},
                        scene={"reception": 1.0},
                        shot={"main_action": 1.0},
                        timestamp=float(index * 10),
                        people=("speaker",),
                        face_count=1,
                        participant_mode="solo",
                    )
                )
        segmentation = _manual_segmentation(
            features,
            [
                _transition(features[index].image_path, features[index + 1].image_path)
                for index in range(len(features) - 1)
            ],
            segments=[(0, 1, 2, 7, 8, 14)],
        )
        clusterer = VibeClusterer(config)
        groups, ungrouped = clusterer.cluster(segmentation, similarity=similarity)
        grouped_paths = [tuple(member.image_path for member in group.members) for group in groups]
        name_to_index = {feature.image_path: index for index, feature in enumerate(features)}
        for group in grouped_paths:
            indices = sorted(name_to_index[path] for path in group)
            span = indices[-1] - indices[0] + 1
            missing = span - len(indices)
            self.assertLessEqual(missing, config.allowed_internal_outliers)
        self.assertFalse(any({"scene_0.jpg", "scene_14.jpg"}.issubset(set(group)) for group in grouped_paths))

    def test_maximum_scene_span_splits_long_groups(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        features = [
            _feature(f"long_{index}.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=float(index * 150), people=("a", "b"), face_count=2, participant_mode="couple")
            for index in range(6)
        ]
        segmentation = _manual_segmentation(
            features,
            [_transition(features[index].image_path, features[index + 1].image_path) for index in range(len(features) - 1)],
        )
        clusterer = VibeClusterer(config)
        groups, ungrouped = clusterer.cluster(segmentation, similarity=similarity)
        self.assertGreaterEqual(len(groups) + len(ungrouped), 2)
        for group in groups:
            timestamps = [member.capture_timestamp for member in group.members if member.capture_timestamp is not None]
            self.assertLessEqual(max(timestamps) - min(timestamps), config.maximum_scene_span_seconds)

    def test_singleton_recovery_does_not_cross_hard_boundary(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        features = [
            _feature("portrait_a.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("applause.jpg", semantic=[0.7, 0.25, 0.2, 0.0], action={"applause": 1.0}, scene={"reception": 1.0}, shot={"reaction": 1.0}, timestamp=18.0, face_count=6, participant_mode="crowd"),
            _feature("portrait_b.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=26.0, people=("a", "b"), face_count=2, participant_mode="couple"),
        ]
        segmentation = segment_by_time(features, config=config, similarity=similarity)
        session = segmentation.sessions[0]
        self.assertEqual(len(session.scene_segments), 3)
        self.assertEqual(sum(1 for item in session.transitions if item.accepted_hard_boundary), 2)

    def test_session_preset_remains_broader_than_balanced(self) -> None:
        window_1 = _feature("window_1.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("bride", "groom"), face_count=2, participant_mode="couple", subject_scene=[1.0, 0.0, 0.0, 0.0], background=[1.0, 0.0, 0.0, 0.0])
        window_2 = _feature("window_2.jpg", semantic=[0.99, 0.02, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=20.0, people=("bride", "groom"), face_count=2, participant_mode="couple", subject_scene=[0.98, 0.02, 0.0, 0.0], background=[0.98, 0.02, 0.0, 0.0])
        mirror_metadata = dict(window_1.metadata)
        mirror_metadata.update(
            {
                "orientation": "portrait",
                "shot_scale_category": "wide",
                "subject_centroid_x": 0.78,
                "subject_centroid_y": 0.35,
                "subject_horizontal_spread": 0.28,
                "subject_vertical_spread": 0.20,
                "primary_subject_scale": 0.04,
            }
        )
        mirror_1 = replace(
            window_1,
            image_path="mirror_1.jpg",
            capture_timestamp=30.0,
            semantic_embedding=_normalized([0.72, 0.69, 0.0, 0.0]),
            subject_scene_embedding=_normalized([0.0, 1.0, 0.0, 0.0]),
            background_embedding=_normalized([0.0, 1.0, 0.0, 0.0]),
            composition_features=_normalized([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
            metadata=mirror_metadata,
        )
        mirror_2 = replace(
            window_2,
            image_path="mirror_2.jpg",
            capture_timestamp=40.0,
            semantic_embedding=_normalized([0.70, 0.71, 0.0, 0.0]),
            subject_scene_embedding=_normalized([0.0, 0.98, 0.02, 0.0]),
            background_embedding=_normalized([0.0, 0.98, 0.02, 0.0]),
            composition_features=_normalized([0.0, 0.98, 0.02, 0.0, 1.0, 0.0]),
            metadata=mirror_metadata,
        )
        features = [window_1, window_2, mirror_1, mirror_2]
        balanced_groups, balanced_ungrouped = SceneGroupingTests()._cluster(
            features,
            config=preset_config(VibeGroupingPreset.BALANCED_SCENES),
        )
        session_groups, session_ungrouped = SceneGroupingTests()._cluster(
            features,
            config=preset_config(VibeGroupingPreset.SESSION),
        )
        self.assertLess(
            len(session_groups) + len(session_ungrouped),
            len(balanced_groups) + len(balanced_ungrouped),
        )

    def test_singleton_recovery_crosses_weak_boundary(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        ordered = (
            _feature("a1.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("single.jpg", semantic=[0.99, 0.01, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("a", "b"), face_count=3, participant_mode="small_group"),
            _feature("a2.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=26.0, people=("a", "b"), face_count=2, participant_mode="couple"),
        )
        transitions = (
            _transition("a1.jpg", "single.jpg", boundary_reliability="weak"),
            _transition("single.jpg", "a2.jpg", boundary_reliability="weak"),
        )
        segments, events = _recover_singletons(
            [[0], [1], [2]],
            transitions,
            ordered,
            config=config,
            similarity=similarity,
        )
        self.assertLess(len(segments), 3)
        self.assertTrue(any(1 in segment for segment in segments))
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(all(event["recovery_result"] == "merged" for event in events))

    def test_tiny_segment_recovery_crosses_supported_boundary_when_continuous(self) -> None:
        config = preset_config(VibeGroupingPreset.BALANCED_SCENES)
        similarity = CombinedSimilarityComputer(config, prototype_table=TEST_PROTOTYPE_TABLE)
        ordered = (
            _feature("left_1.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=10.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("left_2.jpg", semantic=[0.99, 0.01, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=18.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("tiny_1.jpg", semantic=[0.99, 0.01, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=26.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("tiny_2.jpg", semantic=[0.98, 0.02, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=34.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("right_1.jpg", semantic=[0.99, 0.01, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=42.0, people=("a", "b"), face_count=2, participant_mode="couple"),
            _feature("right_2.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"couple_portrait": 1.0}, scene={"window": 1.0}, shot={"portrait": 1.0}, timestamp=50.0, people=("a", "b"), face_count=2, participant_mode="couple"),
        )
        transitions = (
            _transition("left_1.jpg", "left_2.jpg"),
            _transition("left_2.jpg", "tiny_1.jpg", boundary_reliability="supported", soft_boundary=True, accepted_soft_boundary=True),
            _transition("tiny_1.jpg", "tiny_2.jpg"),
            _transition("tiny_2.jpg", "right_1.jpg", boundary_reliability="supported", soft_boundary=True, accepted_soft_boundary=True),
            _transition("right_1.jpg", "right_2.jpg"),
        )
        segments, events = _recover_tiny_segments(
            [[0, 1], [2, 3], [4, 5]],
            transitions,
            ordered,
            config=config,
            similarity=similarity,
        )
        self.assertLess(len(segments), 3)
        self.assertTrue(any(2 in segment and 3 in segment for segment in segments))
        self.assertTrue(any(event["merge_rejected_reason"] is None for event in events))


class LabelTests(unittest.TestCase):
    def test_labels_prefer_specific_action_names(self) -> None:
        label = build_group_label(
            [
                _feature("a.jpg", semantic=[1.0, 0.0, 0.0, 0.0], action={"rings": 1.0}, scene={"ceremony": 1.0}, shot={"main_action": 1.0}, timestamp=10.0, people=("bride", "groom"), people_names=("Alice", "Bob"), face_count=2, participant_mode="couple"),
                _feature("b.jpg", semantic=[0.98, 0.0, 0.0, 0.0], action={"rings": 1.0}, scene={"ceremony": 1.0}, shot={"main_action": 1.0}, timestamp=20.0, people=("bride", "groom"), people_names=("Alice", "Bob"), face_count=2, participant_mode="couple"),
            ],
            ("Alice", "Bob"),
            prototype_table=TEST_PROTOTYPE_TABLE,
        )
        self.assertEqual(label, "Ring Exchange")


class VibeProcessorCacheTests(unittest.TestCase):
    def _make_processor(
        self,
        temp_dir: Path,
        *,
        config: VibeGroupingConfig,
        embedder: VibeEmbedder,
        prototype_table: ScenePrototypeTable,
    ) -> VibeGroupingProcessor:
        cache_db = temp_dir / "cache.sqlite3"
        return VibeGroupingProcessor(
            config=config,
            feature_cache=VibeFeatureCache(cache_db),
            result_cache=VibeGroupingResultCache(cache_db),
            embedder=embedder,
            face_database=SQLiteFaceDatabase(temp_dir / "faces.sqlite3", CosineEmbeddingSimilarity()),
            prototype_table=prototype_table,
        )

    @staticmethod
    def _write_image(path: Path, color: tuple[int, int, int]) -> None:
        image = Image.new("RGB", (64, 64), color=color)
        image.save(path)

    def test_prototype_change_reuses_cached_semantic_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            temp_dir = Path(tmp_text)
            self._write_image(temp_dir / "a.jpg", (255, 32, 32))
            self._write_image(temp_dir / "b.jpg", (32, 32, 255))
            embedder = CountingEmbedder()
            processor_a = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=False, include_subject_scene_embedding=False),
                embedder=embedder,
                prototype_table=_scoring_prototype_table(fingerprint="proto-a"),
            )
            first = processor_a.scan_folder(temp_dir)
            self.assertEqual(embedder.calls, 1)

            processor_b = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=False, include_subject_scene_embedding=False),
                embedder=embedder,
                prototype_table=_scoring_prototype_table(fingerprint="proto-b"),
            )
            second = processor_b.scan_folder(temp_dir)
            self.assertEqual(embedder.calls, 1)
            self.assertEqual(second.cache_hits, 2)
            self.assertNotIn("cache_load_seconds", second.stage_timings)
            self.assertEqual(len(first.groups), len(second.groups))

    def test_background_setting_invalidates_feature_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            temp_dir = Path(tmp_text)
            self._write_image(temp_dir / "a.jpg", (255, 32, 32))
            embedder = CountingEmbedder()

            processor_a = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=False, include_subject_scene_embedding=False),
                embedder=embedder,
                prototype_table=_scoring_prototype_table(fingerprint="proto-a"),
            )
            processor_a.scan_folder(temp_dir)
            self.assertEqual(embedder.calls, 1)

            processor_b = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=True, include_subject_scene_embedding=False),
                embedder=embedder,
                prototype_table=_scoring_prototype_table(fingerprint="proto-a"),
            )
            processor_b.scan_folder(temp_dir)
            self.assertEqual(embedder.calls, 3)

    def test_algorithm_version_invalidates_result_cache_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            temp_dir = Path(tmp_text)
            self._write_image(temp_dir / "a.jpg", (120, 120, 255))
            embedder = CountingEmbedder()
            prototype_table = _scoring_prototype_table(fingerprint="proto-a")

            processor_a = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=False, include_subject_scene_embedding=False, algorithm_version=2),
                embedder=embedder,
                prototype_table=prototype_table,
            )
            processor_a.scan_folder(temp_dir)
            self.assertEqual(embedder.calls, 1)

            processor_b = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=False, include_subject_scene_embedding=False, algorithm_version=3),
                embedder=embedder,
                prototype_table=prototype_table,
            )
            second = processor_b.scan_folder(temp_dir)
            self.assertEqual(embedder.calls, 1)
            self.assertEqual(second.cache_hits, 1)
            self.assertNotIn("cache_load_seconds", second.stage_timings)

    def test_reuses_folder_result_cache_when_inputs_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            temp_dir = Path(tmp_text)
            self._write_image(temp_dir / "a.jpg", (255, 120, 90))
            self._write_image(temp_dir / "b.jpg", (250, 122, 88))
            embedder = CountingEmbedder()
            processor = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=False, include_subject_scene_embedding=False),
                embedder=embedder,
                prototype_table=_scoring_prototype_table(fingerprint="proto-a"),
            )

            processor.scan_folder(temp_dir)
            second = processor.scan_folder(temp_dir)
            self.assertIn("cache_load_seconds", second.stage_timings)
            self.assertEqual(embedder.calls, 1)

    def test_cancel_before_scan_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            temp_dir = Path(tmp_text)
            self._write_image(temp_dir / "a.jpg", (255, 120, 90))
            processor = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=False, include_subject_scene_embedding=False),
                embedder=CountingEmbedder(),
                prototype_table=_scoring_prototype_table(fingerprint="proto-a"),
            )
            token = CancellationToken()
            token.cancel()
            with self.assertRaises(ScanCancelledError):
                processor.scan_folder(temp_dir, cancellation_token=token)

    def test_corrupted_file_is_reported_without_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            temp_dir = Path(tmp_text)
            self._write_image(temp_dir / "a.jpg", (255, 120, 90))
            (temp_dir / "broken.jpg").write_text("not an image", encoding="utf-8")
            processor = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=False, include_subject_scene_embedding=False),
                embedder=CountingEmbedder(),
                prototype_table=_scoring_prototype_table(fingerprint="proto-a"),
            )
            result = processor.scan_folder(temp_dir)
            self.assertEqual(len(result.errors), 1)

    def test_diagnostics_include_image_and_session_debug_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            temp_dir = Path(tmp_text)
            self._write_image(temp_dir / "a.jpg", (255, 32, 32))
            self._write_image(temp_dir / "b.jpg", (250, 48, 48))
            processor = self._make_processor(
                temp_dir,
                config=VibeGroupingConfig(include_background_embedding=False, include_subject_scene_embedding=False),
                embedder=CountingEmbedder(),
                prototype_table=_scoring_prototype_table(fingerprint="proto-a"),
            )

            result = processor.scan_folder(temp_dir)

            self.assertIn("images", result.diagnostics)
            self.assertIn("sessions", result.diagnostics)
            self.assertEqual(len(result.diagnostics["images"]), 2)
            self.assertEqual(len(result.diagnostics["sessions"]), 1)
            first_image = result.diagnostics["images"][0]
            self.assertIn("top_action_scores", first_image)
            self.assertIn("scene_segment_index", first_image)
            self.assertIn("group_label", first_image)
            first_session = result.diagnostics["sessions"][0]
            self.assertIn("scene_segments", first_session)
            self.assertTrue(first_session["scene_segments"])


class ViewModelDispatchTests(unittest.TestCase):
    def test_vibe_mode_dispatches_to_vibe_processor(self) -> None:
        class StubVibeProcessor:
            def scan_folder(self, *_args, **_kwargs) -> VibeGroupingResult:
                return VibeGroupingResult(
                    groups=[
                        VibeGroup(
                            group_id="group1",
                            image_paths=["/tmp/a.jpg", "/tmp/b.jpg"],
                            representative_path="/tmp/a.jpg",
                            start_timestamp=10.0,
                            end_timestamp=20.0,
                            recognized_person_ids=(),
                            recognized_person_names=(),
                            label="Ring Exchange",
                            cohesion_score=0.7,
                            metadata={},
                        )
                    ],
                    ungrouped_paths=[],
                    errors=[],
                    config_snapshot={},
                    model_fingerprint="fp",
                    provider="CPUExecutionProvider",
                    diagnostics={
                        "images": [
                            {
                                "image_path": "/tmp/a.jpg",
                                "group_label": "Ring Exchange",
                            }
                        ],
                        "sessions": [],
                        "untimed_images": [],
                        "transitions": [],
                        "groups": [
                            {
                                "label": "Ring Exchange",
                                "members": ["/tmp/a.jpg", "/tmp/b.jpg"],
                            }
                        ],
                    },
                )

        with tempfile.TemporaryDirectory() as tmp_text:
            temp_dir = Path(tmp_text)
            view_model = PhotoCleanerViewModel(vibe_processor_factory=lambda _config: StubVibeProcessor())
            view_model._scan_worker(temp_dir, "vibe", None, None, False, None)
            message = view_model.poll_background_message()
            self.assertIsInstance(message, ScanResultMessage)
            assert isinstance(message, ScanResultMessage)
            self.assertEqual(message.mode, "vibe")
            self.assertEqual(message.results[0].title, "Ring Exchange")
            self.assertIsNotNone(message.debug_payload)
            assert message.debug_payload is not None
            self.assertEqual(message.debug_payload["input_folder"], str(temp_dir.resolve()))
            self.assertIn("diagnostics", message.debug_payload)
            self.assertIn("groups", message.debug_payload)

    def test_vibe_debug_payload_can_be_exported_after_scan_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            temp_dir = Path(tmp_text)
            output_path = temp_dir / "vibe_debug.json"
            view_model = PhotoCleanerViewModel()
            view_model.set_mode("vibe")

            message = ScanResultMessage(
                mode="vibe",
                results=[
                    ResultGroup(
                        title="Ring Exchange",
                        items=[
                            ResultItem(
                                path=temp_dir / "a.jpg",
                                title="a.jpg",
                                detail="detail",
                            )
                        ],
                        group_type="vibe",
                    )
                ],
                known_people_only=False,
                summary="Found 1 vibe group.",
                warning=None,
                debug_payload={
                    "source": "gui_app",
                    "input_folder": str(temp_dir.resolve()),
                    "groups": [{"label": "Ring Exchange"}],
                    "diagnostics": {
                        "images": [{"image_path": str((temp_dir / "a.jpg").resolve())}],
                        "sessions": [],
                        "untimed_images": [],
                        "transitions": [],
                        "groups": [{"label": "Ring Exchange"}],
                    },
                },
            )

            view_model.handle_scan_result_message(message)

            self.assertTrue(view_model.can_export_vibe_debug())
            self.assertIn(temp_dir.name, view_model.suggest_vibe_debug_filename())
            written_path = view_model.export_vibe_debug(output_path)
            payload = json.loads(written_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["input_folder"], str(temp_dir.resolve()))
            self.assertIn("exported_at_utc", payload)
            self.assertEqual(payload["groups"][0]["label"], "Ring Exchange")


if __name__ == "__main__":
    unittest.main()
