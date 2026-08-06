"""
High-level duplicate detection pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from scan_controls import CancellationToken

from .candidate_generator import CandidateGenerator
from .config import DetectorConfig
from .indexer import ImageIndexer
from .models import DuplicateGroup, ProgressPhase
from .orb import OrbVerifier
from .union_find import UnionFind

ProgressCallback = Callable[
    [ProgressPhase, int, int | None],
    None,
]


class DuplicateDetector:
    """
    High-level duplicate detection engine.
    """

    def __init__(self, config: DetectorConfig | None = None):

        self._config = config or DetectorConfig()

        self._indexer = ImageIndexer(self._config)
        self._generator = CandidateGenerator(self._config)
        self._verifier = OrbVerifier(self._config)

    # ------------------------------------------------------------------

    def find_duplicates(
        self,
        folder: str | Path,
        progress_callback: ProgressCallback | None = None,
        file_extensions: tuple[str, ...] | None = None,
        orientation_filter: str | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[DuplicateGroup]:

        def index_progress(done: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback(
                    ProgressPhase.INDEXING,
                    done,
                    total,
                )

        photos = self._indexer.index(
            folder,
            index_progress,
            file_extensions=file_extensions,
            orientation_filter=orientation_filter,
            cancellation_token=cancellation_token,
        )

        if len(photos) < 2:
            return []

        photo_to_index = {
            photo: i
            for i, photo in enumerate(photos)
        }

        uf = UnionFind(len(photos))

        for pair in self._generator.generate(
                photos,
                progress_callback,
                cancellation_token=cancellation_token,
        ):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()

            if self._verifier.verify(pair):
                uf.union(
                    photo_to_index[pair.left],
                    photo_to_index[pair.right],
                )

        groups = []

        for component in uf.non_trivial_components():

            group = DuplicateGroup()

            for index in component:
                group.add(photos[index])

            group.sort()

            groups.append(group)

        groups.sort(
            key=lambda g: len(g),
            reverse=True,
        )

        self._verifier.clear_cache()

        return groups
