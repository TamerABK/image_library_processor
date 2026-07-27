from dataclasses import dataclass

import numpy as np

from .interfaces import EmbeddingSimilarity


@dataclass(slots=True)
class CosineEmbeddingSimilarity(EmbeddingSimilarity):

    threshold: float = 0.60

    def score(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:

        return float(
            np.dot(
                embedding1,
                embedding2,
            )
        )

    def distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:

        return 1.0 - self.score(
            embedding1,
            embedding2,
        )

    @property
    def default_threshold(self) -> float:
        return self.threshold

    @property
    def higher_is_better(self) -> bool:
        return True