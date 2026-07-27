from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: np.ndarray
    path: Path
    
@dataclass(slots=True)
class EmbeddedFace:

    bbox: tuple[int, int, int, int]

    confidence: float

    landmarks: np.ndarray

    embedding: np.ndarray
    path: Path

@dataclass(slots=True)
class RecognizedFace:
    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: np.ndarray
    embedding: np.ndarray
    person_id: int
    path: Path


@dataclass(slots=True)
class UnknownCluster:
    id: int
    faces: list[EmbeddedFace]
    representative: EmbeddedFace
    preview: np.ndarray | None

@dataclass(slots=True)
class KnownPerson:
    person_id: int
    name: str
    faces: list[RecognizedFace]


@dataclass(slots=True)
class KnownPersonResult:
    person_id: int
    name: str
    photos: list[Path]

@dataclass(slots=True)
class FaceProcessorResult:
    known_people: list[KnownPersonResult]
    unknown_clusters: list[UnknownCluster]



@dataclass(slots=True)
class Person:
    id: int
    name: str


@dataclass(slots=True)
class StoredEmbedding:
    id: int
    person_id: int
    embedding: np.ndarray


@dataclass(slots=True)
class Match:
    person_id: int

    score: float
