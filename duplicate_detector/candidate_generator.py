"""
Candidate generation using perceptual hashes.

This module is responsible for generating candidate duplicate pairs.
It performs NO image loading and NO ORB verification.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Iterator

from scan_controls import CancellationToken

from .config import DetectorConfig
from .models import CandidatePair, PhotoInfo, ProgressPhase
from .utils import hamming_distance

ProgressCallback = Callable[
    [ProgressPhase, int, int],
    None,
]


class CandidateGenerator:
    """
    Generates candidate pairs using pHash and dHash filtering.

    The generator yields candidates lazily to avoid allocating large
    intermediate lists.
    """

    def __init__(self, config: DetectorConfig):
        self._config = config

    # ------------------------------------------------------------------

    def generate(
        self,
        photos: list[PhotoInfo],
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Iterator[CandidatePair]:
        """
        Lazily yield candidate duplicate pairs.
        """

        if len(photos) < 2:
            return

        # Sorting improves locality and makes results deterministic.
        ordered = sorted(photos, key=lambda p: p.phash)

        count = len(ordered)

        for i in range(count - 1):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()

            left = ordered[i]

            for j in range(i + 1, count):

                right = ordered[j]

                phash_distance = hamming_distance(
                    left.phash,
                    right.phash,
                )

                if phash_distance > self._config.phash_threshold:
                    continue

                dhash_distance = hamming_distance(
                    left.dhash,
                    right.dhash,
                )

                if dhash_distance > self._config.dhash_threshold:
                    continue

                yield CandidatePair(
                    left=left,
                    right=right,
                    phash_distance=phash_distance,
                    dhash_distance=dhash_distance,
                )

            if (
                progress_callback is not None
                and (i % 25 == 0 or i == count - 2)
            ):
                progress_callback(
                    ProgressPhase.MATCHING,
                    i + 1,
                    count - 1,
                )
