from .interfaces import FaceDatabase
from .models import (
    EmbeddedFace,
    RecognizedFace,
)


class DefaultFaceRecognizer:

    def __init__(
        self,
        database: FaceDatabase,
        threshold: float,
    ):
        self._database = database
        self._threshold = threshold

    def recognize(
        self,
        faces: list[EmbeddedFace],
    ) -> tuple[
        list[RecognizedFace],
        list[EmbeddedFace],
    ]:

        recognized = []
        unknown = []

        if not self._database.contains_people():
            return [], faces

        for face in faces:

            match = self._database.find_nearest_embedding(
                face.embedding,
            )

            if (
                match is None
                or match.score < self._threshold
            ):
                unknown.append(face)
                continue

            recognized.append(
                RecognizedFace(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    landmarks=face.landmarks,
                    embedding=face.embedding,
                    person_id=match.person_id,
                    path=face.path,
                )
            )

        return recognized, unknown