from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app_paths import model_path


LOGGER = logging.getLogger(__name__)

PROTOTYPE_METADATA_FILENAME = "vibe_scene_prototypes_v1.json"
PROTOTYPE_DATA_FILENAME = "vibe_scene_prototypes_v1.npz"
PROTOTYPE_VERSION = 1
MODEL_NAME = "ViT-B-32"
PRETRAINED_NAME = "laion2b_s34b_b79k"
PROMPT_TEMPLATES = (
    "a photo of {}",
    "a professional photograph of {}",
    "an event photograph showing {}",
    "a candid photograph of {}",
)


@dataclass(frozen=True, slots=True)
class PrototypeConcept:
    key: str
    phrase: str
    label: str
    category: str
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrototypeMatch:
    key: str
    label: str
    score: float
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenePrototypeTable:
    metadata: dict[str, Any]
    action_embeddings: np.ndarray
    scene_embeddings: np.ndarray
    shot_embeddings: np.ndarray
    action_concepts: tuple[PrototypeConcept, ...]
    scene_concepts: tuple[PrototypeConcept, ...]
    shot_concepts: tuple[PrototypeConcept, ...]

    @property
    def fingerprint(self) -> str:
        return str(self.metadata["sha256"])

    @property
    def semantic_model_fingerprint(self) -> str:
        return str(self.metadata["semantic_model_fingerprint"])

    @property
    def embedding_dimension(self) -> int:
        return int(self.metadata["embedding_dimension"])

    def score_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Expected a 2D embedding matrix.")
        return (
            _normalize_scores(matrix @ self.action_embeddings.T),
            _normalize_scores(matrix @ self.scene_embeddings.T),
            _normalize_scores(matrix @ self.shot_embeddings.T),
        )

    def top_matches(
        self,
        scores: np.ndarray | None,
        *,
        category: str,
        limit: int = 3,
    ) -> list[PrototypeMatch]:
        if scores is None:
            return []
        vector = np.asarray(scores, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            return []
        concepts = self._concepts_for(category)
        limit = max(0, min(limit, len(concepts)))
        if limit <= 0:
            return []
        order = np.argsort(-vector, kind="stable")[:limit]
        matches: list[PrototypeMatch] = []
        for index in order:
            score = float(vector[int(index)])
            if score <= 0.0:
                continue
            concept = concepts[int(index)]
            matches.append(
                PrototypeMatch(
                    key=concept.key,
                    label=concept.label,
                    score=score,
                    tags=concept.tags,
                )
            )
        return matches

    def top_label(
        self,
        scores: np.ndarray | None,
        *,
        category: str,
    ) -> str | None:
        matches = self.top_matches(scores, category=category, limit=1)
        return None if not matches else matches[0].label

    def action_conflict(
        self,
        left_scores: np.ndarray | None,
        right_scores: np.ndarray | None,
        *,
        confidence_threshold: float,
    ) -> bool:
        left = self.top_matches(left_scores, category="action", limit=1)
        right = self.top_matches(right_scores, category="action", limit=1)
        if not left or not right:
            return False
        left_top = left[0]
        right_top = right[0]
        if left_top.key == right_top.key:
            return False
        if min(left_top.score, right_top.score) < confidence_threshold:
            return False
        left_family = _primary_action_family(left_top.tags)
        right_family = _primary_action_family(right_top.tags)
        if left_family == right_family:
            return False
        if {"audience", "speech"} == {left_family, right_family}:
            return True
        if {"applause", "speech"} == {left_family, right_family}:
            return True
        if {"portrait", "walking"} == {left_family, right_family}:
            return True
        return True

    def describe_scores(
        self,
        scores: np.ndarray | None,
        *,
        category: str,
        limit: int = 3,
    ) -> list[dict[str, object]]:
        return [
            {
                "key": match.key,
                "label": match.label,
                "score": round(match.score, 4),
                "tags": list(match.tags),
            }
            for match in self.top_matches(scores, category=category, limit=limit)
        ]

    def _concepts_for(self, category: str) -> tuple[PrototypeConcept, ...]:
        if category == "action":
            return self.action_concepts
        if category == "scene":
            return self.scene_concepts
        if category == "shot":
            return self.shot_concepts
        raise ValueError(f"Unknown prototype category: {category}")

    @classmethod
    def from_files(cls, metadata_path: Path, data_path: Path) -> "ScenePrototypeTable":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload = np.load(data_path, allow_pickle=False)
        return cls(
            metadata=metadata,
            action_embeddings=np.asarray(payload["action_embeddings"], dtype=np.float32),
            scene_embeddings=np.asarray(payload["scene_embeddings"], dtype=np.float32),
            shot_embeddings=np.asarray(payload["shot_embeddings"], dtype=np.float32),
            action_concepts=tuple(
                PrototypeConcept(
                    key=str(item["key"]),
                    phrase=str(item["phrase"]),
                    label=str(item["label"]),
                    category="action",
                    tags=tuple(item.get("tags", [])),
                )
                for item in metadata["action_concepts"]
            ),
            scene_concepts=tuple(
                PrototypeConcept(
                    key=str(item["key"]),
                    phrase=str(item["phrase"]),
                    label=str(item["label"]),
                    category="scene",
                    tags=tuple(item.get("tags", [])),
                )
                for item in metadata["scene_concepts"]
            ),
            shot_concepts=tuple(
                PrototypeConcept(
                    key=str(item["key"]),
                    phrase=str(item["phrase"]),
                    label=str(item["label"]),
                    category="shot",
                    tags=tuple(item.get("tags", [])),
                )
                for item in metadata["shot_concepts"]
            ),
        )


ACTION_CONCEPTS: tuple[PrototypeConcept, ...] = (
    PrototypeConcept("portrait_pose", "people posing for a portrait", "Portrait", "action", ("portrait", "posed")),
    PrototypeConcept("solo_walk", "a person walking", "Walking", "action", ("walking", "solo", "movement")),
    PrototypeConcept("group_walk", "people walking together", "Walking Together", "action", ("walking", "group", "movement")),
    PrototypeConcept("speech", "someone giving a speech", "Speech", "action", ("speech", "speaker", "main_action")),
    PrototypeConcept("speech_audience", "people listening to a speech", "Audience Listening", "action", ("audience", "speech", "reaction")),
    PrototypeConcept("applause", "people applauding", "Guests Applauding", "action", ("applause", "reaction", "audience")),
    PrototypeConcept("dance", "people dancing", "Dancing", "action", ("dance", "movement", "group")),
    PrototypeConcept("couple_dance", "a couple dancing", "Couple Dancing", "action", ("dance", "couple", "main_action")),
    PrototypeConcept("hug", "people hugging", "Hugging", "action", ("hug", "interaction", "main_action")),
    PrototypeConcept("kiss", "people kissing", "Kissing", "action", ("kiss", "interaction", "main_action")),
    PrototypeConcept("rings", "people exchanging rings", "Ring Exchange", "action", ("rings", "ceremony", "interaction", "main_action")),
    PrototypeConcept("cake_cutting", "people cutting a cake", "Cake Cutting", "action", ("cake", "interaction", "main_action")),
    PrototypeConcept("eating", "people eating", "Eating", "action", ("eating", "group")),
    PrototypeConcept("makeup", "someone applying makeup", "Makeup", "action", ("makeup", "preparation", "interaction")),
    PrototypeConcept("dressing", "someone getting dressed", "Getting Dressed", "action", ("dressing", "preparation")),
    PrototypeConcept("flowers", "someone holding flowers", "Holding Flowers", "action", ("flowers", "portrait")),
    PrototypeConcept("sitting", "people sitting together", "Seated Group", "action", ("sitting", "group")),
    PrototypeConcept("standing", "people standing together", "Standing Group", "action", ("standing", "group", "posed")),
    PrototypeConcept("laughing", "people laughing", "Laughing", "action", ("laughing", "reaction")),
    PrototypeConcept("talking", "people talking", "Talking", "action", ("talking", "interaction")),
    PrototypeConcept("procession", "a procession", "Procession", "action", ("procession", "movement", "ceremony")),
    PrototypeConcept("group_portrait", "a group portrait", "Group Portrait", "action", ("portrait", "group", "posed")),
    PrototypeConcept("couple_portrait", "a couple portrait", "Couple Portrait", "action", ("portrait", "couple", "posed")),
    PrototypeConcept("solo_portrait", "a solo portrait", "Solo Portrait", "action", ("portrait", "solo", "posed")),
    PrototypeConcept("candid_interaction", "a candid interaction", "Candid Interaction", "action", ("candid", "interaction")),
    PrototypeConcept("children_playing", "children playing", "Children Playing", "action", ("play", "movement", "group")),
    PrototypeConcept("entering_room", "people entering a room", "Entering", "action", ("entering", "movement")),
    PrototypeConcept("leaving_room", "people leaving", "Leaving", "action", ("leaving", "movement")),
    PrototypeConcept("aisle_walk", "a bride walking down the aisle", "Bride Walking Down The Aisle", "action", ("aisle", "walking", "ceremony", "main_action")),
    PrototypeConcept("altar_waiting", "a groom waiting at the altar", "Groom Waiting At The Altar", "action", ("altar", "ceremony", "portrait")),
    PrototypeConcept("wedding_ring_exchange", "a wedding ring exchange", "Ring Exchange", "action", ("rings", "ceremony", "interaction", "main_action", "wedding")),
    PrototypeConcept("wedding_first_kiss", "a wedding first kiss", "First Kiss", "action", ("kiss", "ceremony", "interaction", "main_action", "wedding")),
    PrototypeConcept("ceremony_reading", "a wedding ceremony reading", "Ceremony Reading", "action", ("ceremony", "speech", "main_action", "wedding")),
    PrototypeConcept("wedding_speech", "a wedding speech", "Wedding Speech", "action", ("speech", "speaker", "main_action", "wedding")),
    PrototypeConcept("wedding_toast", "a wedding toast", "Wedding Toast", "action", ("speech", "toast", "main_action", "wedding")),
    PrototypeConcept("wedding_cake_cutting", "a wedding cake cutting", "Cake Cutting", "action", ("cake", "interaction", "main_action", "wedding")),
    PrototypeConcept("bouquet_toss", "a wedding bouquet toss", "Bouquet Toss", "action", ("bouquet", "movement", "main_action", "wedding")),
    PrototypeConcept("first_dance", "a wedding first dance", "First Dance", "action", ("dance", "couple", "main_action", "wedding")),
    PrototypeConcept("bride_makeup", "a bride getting makeup applied", "Bride Makeup", "action", ("makeup", "preparation", "interaction", "wedding")),
    PrototypeConcept("bride_dress", "a bride putting on a wedding dress", "Bride Dressing", "action", ("dressing", "preparation", "wedding")),
    PrototypeConcept("bridesmaids_pose", "a bride posing with bridesmaids", "Bride With Bridesmaids", "action", ("portrait", "group", "posed", "wedding")),
    PrototypeConcept("groomsmen_pose", "a groom posing with groomsmen", "Groom With Groomsmen", "action", ("portrait", "group", "posed", "wedding")),
    PrototypeConcept("family_wedding_portrait", "a family wedding portrait", "Family Portrait", "action", ("portrait", "family", "group", "posed", "wedding")),
    PrototypeConcept("wedding_couple_portrait", "a wedding couple portrait", "Couple Portrait", "action", ("portrait", "couple", "posed", "wedding")),
    PrototypeConcept("wedding_guests_applauding", "wedding guests applauding", "Guests Applauding", "action", ("applause", "reaction", "audience", "wedding")),
    PrototypeConcept("wedding_guests_dancing", "wedding guests dancing", "Dance Floor", "action", ("dance", "group", "wedding")),
)

SCENE_CONCEPTS: tuple[PrototypeConcept, ...] = (
    PrototypeConcept("indoor_ceremony", "an indoor ceremony", "Indoor Ceremony", "scene", ("ceremony", "indoor")),
    PrototypeConcept("outdoor_ceremony", "an outdoor ceremony", "Outdoor Ceremony", "scene", ("ceremony", "outdoor")),
    PrototypeConcept("dance_floor", "a dance floor", "Dance Floor", "scene", ("dance", "venue")),
    PrototypeConcept("reception_hall", "a reception hall", "Reception Hall", "scene", ("reception", "venue")),
    PrototypeConcept("church_interior", "a church interior", "Church Interior", "scene", ("church", "venue")),
    PrototypeConcept("beach", "a beach", "Beach", "scene", ("beach", "outdoor")),
    PrototypeConcept("forest", "a forest", "Forest", "scene", ("forest", "outdoor")),
    PrototypeConcept("garden", "a garden", "Garden", "scene", ("garden", "outdoor")),
    PrototypeConcept("city_street", "a city street", "City Street", "scene", ("street", "outdoor")),
    PrototypeConcept("hotel_room", "a hotel room", "Hotel Room", "scene", ("room", "indoor")),
    PrototypeConcept("dressing_room", "a dressing room", "Dressing Room", "scene", ("dressing_room", "indoor")),
    PrototypeConcept("dining_table", "a dining table", "Dining Table", "scene", ("table", "indoor")),
    PrototypeConcept("stage", "a stage", "Stage", "scene", ("stage", "venue")),
    PrototypeConcept("sunset_field", "a sunset field", "Sunset Field", "scene", ("field", "outdoor", "sunset")),
    PrototypeConcept("dark_indoor", "a dark indoor venue", "Dark Indoor Venue", "scene", ("indoor", "dark")),
    PrototypeConcept("bright_outdoor", "a bright outdoor location", "Bright Outdoor Location", "scene", ("outdoor", "bright")),
)

SHOT_TYPE_CONCEPTS: tuple[PrototypeConcept, ...] = (
    PrototypeConcept("close_up_portrait", "a close-up portrait", "Close-up Portrait", "shot", ("close_up", "portrait")),
    PrototypeConcept("medium_portrait", "a medium portrait", "Medium Portrait", "shot", ("medium", "portrait")),
    PrototypeConcept("full_body_portrait", "a full-body portrait", "Full-body Portrait", "shot", ("full_body", "portrait")),
    PrototypeConcept("wide_environmental_portrait", "a wide environmental portrait", "Wide Environmental Portrait", "shot", ("wide", "portrait")),
    PrototypeConcept("group_portrait_shot", "a group portrait", "Group Portrait", "shot", ("group", "portrait")),
    PrototypeConcept("candid_shot", "a candid photograph", "Candid", "shot", ("candid",)),
    PrototypeConcept("detail_shot", "a detail photograph", "Detail", "shot", ("detail",)),
    PrototypeConcept("reaction_shot", "a reaction photograph", "Reaction", "shot", ("reaction",)),
    PrototypeConcept("audience_shot", "an audience photograph", "Audience", "shot", ("audience",)),
    PrototypeConcept("main_action_shot", "a photograph of the main action", "Main Action", "shot", ("main_action",)),
)


def prototype_spec() -> dict[str, object]:
    return {
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED_NAME,
        "prototype_version": PROTOTYPE_VERSION,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "action_concepts": [_concept_dict(concept) for concept in ACTION_CONCEPTS],
        "scene_concepts": [_concept_dict(concept) for concept in SCENE_CONCEPTS],
        "shot_concepts": [_concept_dict(concept) for concept in SHOT_TYPE_CONCEPTS],
    }


def load_prototype_table(
    *,
    semantic_model_fingerprint: str,
    embedding_dimension: int,
    metadata_filename: str = PROTOTYPE_METADATA_FILENAME,
    data_filename: str = PROTOTYPE_DATA_FILENAME,
) -> ScenePrototypeTable | None:
    metadata_path = model_path(metadata_filename)
    data_path = model_path(data_filename)
    if not metadata_path.is_file() or not data_path.is_file():
        LOGGER.warning(
            "Scene prototype resources are missing at %s and %s.",
            metadata_path,
            data_path,
        )
        return None

    table = ScenePrototypeTable.from_files(metadata_path, data_path)
    if table.semantic_model_fingerprint != semantic_model_fingerprint:
        LOGGER.warning(
            "Scene prototype fingerprint mismatch. Expected model fingerprint %s, found %s.",
            semantic_model_fingerprint,
            table.semantic_model_fingerprint,
        )
        return None
    if table.embedding_dimension != embedding_dimension:
        LOGGER.warning(
            "Scene prototype dimension mismatch. Expected %s, found %s.",
            embedding_dimension,
            table.embedding_dimension,
        )
        return None
    return table


def prototype_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_variants(concept: PrototypeConcept) -> list[str]:
    return [template.format(concept.phrase) for template in PROMPT_TEMPLATES]


def _concept_dict(concept: PrototypeConcept) -> dict[str, object]:
    return {
        "key": concept.key,
        "phrase": concept.phrase,
        "label": concept.label,
        "tags": list(concept.tags),
    }


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float32)
    clipped = np.clip(matrix, 0.0, None)
    norms = np.linalg.norm(clipped, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return clipped / norms


def _primary_action_family(tags: Iterable[str]) -> str:
    ordered = (
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
    tag_set = set(tags)
    for name in ordered:
        if name in tag_set:
            return name
    return next(iter(tag_set), "generic")
