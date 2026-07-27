"""
Configuration for the duplicate photo detector.

Every tunable parameter used by the detector should live here.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from image_loader import default_image_loader


@dataclass(slots=True)
class DetectorConfig:
    """
    Configuration for the duplicate detector.

    The default values are conservative and designed to work well for
    personal photo libraries (≈6,000 images).
    """

    # ------------------------------------------------------------------
    # Hash thresholds
    # ------------------------------------------------------------------

    #: Maximum Hamming distance between perceptual hashes.
    phash_threshold: int = 18

    #: Maximum Hamming distance between difference hashes.
    dhash_threshold: int = 18

    # ------------------------------------------------------------------
    # ORB verification
    # ------------------------------------------------------------------

    #: Number of ORB features to extract.
    orb_features: int = 1000

    #: Lowe ratio test threshold.
    orb_ratio: float = 0.75

    #: Minimum normalized ORB score required to consider
    #: two images duplicates.
    orb_min_score: float = 0.18

    enable_orb_verification: bool = True

    # ------------------------------------------------------------------
    # Descriptor cache
    # ------------------------------------------------------------------

    #: Maximum number of cached ORB descriptor sets.
    cache_size: int = 256

    # ------------------------------------------------------------------
    # Parallelism
    # ------------------------------------------------------------------

    #: Number of worker threads used during hashing.
    #: None = automatically use all logical CPUs.
    max_workers: int | None = None

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    #: Supported file extensions.
    supported_extensions: tuple[str, ...] = default_image_loader.supported_extensions()

    #: Maximum decode size used for duplicate hashes.
    hash_decode_dimension: int = 2048

    #: Maximum decode size used for ORB verification.
    orb_decode_dimension: int = 1600

    #: Ignore unreadable/corrupted images instead of raising.
    ignore_load_errors: bool = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:

        if not (0 <= self.phash_threshold <= 64):
            raise ValueError("phash_threshold must be between 0 and 64.")

        if not (0 <= self.dhash_threshold <= 64):
            raise ValueError("dhash_threshold must be between 0 and 64.")

        if self.orb_features <= 0:
            raise ValueError("orb_features must be positive.")

        if not (0.0 < self.orb_ratio < 1.0):
            raise ValueError("orb_ratio must be between 0 and 1.")

        if not (0.0 <= self.orb_min_score <= 1.0):
            raise ValueError("orb_min_score must be between 0 and 1.")

        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive.")

        if self.max_workers is not None and self.max_workers <= 0:
            raise ValueError("max_workers must be positive.")

        if not self.supported_extensions:
            raise ValueError("supported_extensions cannot be empty.")

        if self.hash_decode_dimension <= 0:
            raise ValueError("hash_decode_dimension must be positive.")

        if self.orb_decode_dimension <= 0:
            raise ValueError("orb_decode_dimension must be positive.")

    @property
    def workers(self) -> int:
        """
        Returns the number of worker threads that should be used.
        """
        if self.max_workers is not None:
            return self.max_workers

        return max(1, os.cpu_count() or 1)
