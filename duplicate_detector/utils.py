"""
Utility functions used throughout the duplicate detector.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from image_loader import default_image_loader

from .config import DetectorConfig


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def imagehash_to_uint64(hash_obj: imagehash.ImageHash) -> int:
    """
    Convert an ImageHash into a native Python integer.

    Using integers allows extremely fast Hamming distance calculations via
    XOR + int.bit_count().
    """
    return int(str(hash_obj), 16)


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Compute the Hamming distance between two 64-bit hashes.
    """
    return (hash1 ^ hash2).bit_count()


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def is_supported_image(path: Path, config: DetectorConfig) -> bool:
    """
    Return True if the file extension is supported.
    """
    return path.suffix.lower() in config.supported_extensions


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_pil(path: Path) -> Image.Image:
    """
    Load an image using Pillow.
    """
    image = default_image_loader.load_pil_for_hashing(path)
    if image is None:
        raise ValueError(f"Could not open image: {path}")
    return image


def load_cv(path: Path):
    """
    Load an image using OpenCV.
    """
    return default_image_loader.load_for_scan(path)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def image_dimensions(image: Image.Image) -> tuple[int, int]:
    """
    Return (width, height).
    """
    return image.size


def aspect_ratio(width: int, height: int) -> float:
    """
    Compute aspect ratio.
    """
    return width / height if height else 0.0


def megapixels(width: int, height: int) -> float:
    """
    Compute megapixels.
    """
    return (width * height) / 1_000_000.0


# ---------------------------------------------------------------------------
# Hash generation
# ---------------------------------------------------------------------------

def compute_phash(image: Image.Image) -> int:
    """
    Compute perceptual hash.
    """
    return imagehash_to_uint64(imagehash.phash(image,hash_size=16))


def compute_dhash(image: Image.Image) -> int:
    """
    Compute difference hash.
    """
    return imagehash_to_uint64(imagehash.dhash(image))
