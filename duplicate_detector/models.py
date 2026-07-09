"""
Core data models used by the duplicate detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class MatchType(Enum):
    EXACT = "exact"
    NEAR_DUPLICATE = "near_duplicate"
    SIMILAR = "similar"



class ProgressPhase(Enum):
    INDEXING = "indexing"
    MATCHING = "matching"

# ---------------------------------------------------------------------------
# Photo metadata
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class PhotoInfo:
    """
    Metadata extracted from an image.

    This object is immutable and intentionally does NOT contain
    ORB descriptors or OpenCV objects.
    """

    path: Path

    width: int
    height: int
    file_size: int
    phash: int
    dhash: int

    @property
    def pixels(self) -> int:
        """Total number of pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Image aspect ratio."""
        return self.width / self.height if self.height else 0.0

    @property
    def filename(self) -> str:
        """Filename only."""
        return self.path.name

    @property
    def extension(self) -> str:
        """Lowercase file extension."""
        return self.path.suffix.lower()


# ---------------------------------------------------------------------------
# Candidate pair
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class CandidatePair:
    """
    Candidate duplicate pair after hash filtering.
    """

    left: PhotoInfo
    right: PhotoInfo

    phash_distance: int
    dhash_distance: int


# ---------------------------------------------------------------------------
# Verified pair
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class VerifiedPair:
    """
    Candidate pair after ORB verification.
    """

    left: PhotoInfo
    right: PhotoInfo

    phash_distance: int
    dhash_distance: int

    match_type: MatchType
    orb_score: float


# ---------------------------------------------------------------------------
# Duplicate group
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DuplicateGroup:
    """
    A connected component of duplicate images.
    """

    photos: list[PhotoInfo] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.photos)

    def __iter__(self):
        return iter(self.photos)

    def add(self, photo: PhotoInfo) -> None:
        self.photos.append(photo)

    @property
    def size(self) -> int:
        return len(self.photos)

    @property
    def best(self) -> PhotoInfo:
        """
        Return the highest resolution image.

        This can later be replaced with a smarter quality score
        (blur detection, compression, etc.).
        """
        return max(self.photos, key=lambda p: p.pixels)

    def sort(self) -> None:
        """
        Sort by descending resolution.
        """
        self.photos.sort(
            key=lambda p: (p.pixels, p.filename),
            reverse=True,
        )