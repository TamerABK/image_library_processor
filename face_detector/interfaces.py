from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .models import DetectedFace, EmbeddedFace, Match, Person, StoredEmbedding, RecognizedFace, UnknownCluster


class FaceDetector(ABC):

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        path: Path,
    ) -> list[DetectedFace]:
        pass

class FaceEmbedder(ABC):

    @abstractmethod
    def embed(
            self,
            image: np.ndarray,
            faces: list[DetectedFace],
    ) -> list[EmbeddedFace]:
        pass

    @abstractmethod
    def embed_requests(
        self,
        face_requests: list[tuple[np.ndarray, DetectedFace]],
    ) -> list[EmbeddedFace]:
        ...


class FaceDatabase(ABC):

    @abstractmethod
    def contains_people(self) -> bool:
        ...

    @abstractmethod
    def cache_signature(self) -> str:
        ...

    @abstractmethod
    def find_nearest_embedding(
        self,
        embedding: np.ndarray,
    ) -> Match | None:
        ...

    @abstractmethod
    def add_person(
        self,
        name: str,
    ) -> Person:
        ...

    @abstractmethod
    def add_embedding(
        self,
        person_id: int,
        embedding: np.ndarray,
    ) -> StoredEmbedding:
        ...

    @abstractmethod
    def get_person(
        self,
        person_id: int,
    ) -> Person | None:
        ...

    @abstractmethod
    def list_people_names(self) -> list[str]:
        ...



class EmbeddingSimilarity(ABC):

    @abstractmethod
    def score(
            self,
            embedding1: np.ndarray,
            embedding2: np.ndarray,
    ) -> float:
        """
        Higher is better.
        """
        ...

    @abstractmethod
    def distance(
            self,
            embedding1: np.ndarray,
            embedding2: np.ndarray,
    ) -> float:
        """
        Lower is better.
        """
        ...

    @property
    @abstractmethod
    def default_threshold(self) -> float:
        pass

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        pass


class FaceRecognizer(ABC):

    @abstractmethod
    def recognize(
        self,
        faces: list[EmbeddedFace],
    ) -> tuple[list[RecognizedFace], list[EmbeddedFace]]:
        ...

class FaceClusterer(ABC):

    @abstractmethod
    def cluster(
        self,
        faces: list[EmbeddedFace],
    ) -> list[UnknownCluster]:
        ...

class FacePreviewRenderer:

    def render(
        self,
        face: EmbeddedFace,
    ) -> np.ndarray:
        ...

class FaceQualityAssessor(ABC):

    @abstractmethod
    def score(
        self,
        aligned_face: np.ndarray,
    ) -> float:
        ...
