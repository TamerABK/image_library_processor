"""
ORB-based candidate verification.

This module verifies whether two candidate images should belong to the
same duplicate group.

The detector never receives ORB scores or OpenCV objects.
It simply receives True/False.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from image_loader import default_image_loader

from .config import DetectorConfig
from .models import CandidatePair


class OrbVerifier:
    """
    Verifies candidate pairs using ORB feature matching.
    """

    def __init__(self, config: DetectorConfig):

        self._config = config

        self._orb = cv2.ORB_create(
            nfeatures=config.orb_features
        )

        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # path -> (keypoints, descriptors)
        self._cache: OrderedDict[
            Path,
            tuple[list[cv2.KeyPoint], np.ndarray | None]
        ] = OrderedDict()

    # ------------------------------------------------------------------

    def verify(self, pair: CandidatePair) -> bool:
        """
        Returns True if the images are considered near duplicates.
        """

        kp1, des1 = self._get_descriptors(pair.left.path)
        kp2, des2 = self._get_descriptors(pair.right.path)

        if des1 is None or des2 is None:
            return False

        if len(des1) < 8 or len(des2) < 8:
            return False

        matches = self._matcher.knnMatch(
            des1,
            des2,
            k=2,
        )

        good_matches = []

        ratio = self._config.orb_ratio

        for match_pair in matches:

            if len(match_pair) != 2:
                continue

            m, n = match_pair

            if m.distance < ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            return False

        src = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        dst = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            5.0,
        )

        if mask is None:
            return False

        inliers = int(mask.sum())

        score = inliers / len(good_matches)

        return score >= self._config.orb_min_score

    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """
        Frees all cached descriptors.
        """

        self._cache.clear()

    # ------------------------------------------------------------------

    def _get_descriptors(
        self,
        path: Path,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:

        cached = self._cache.get(path)

        if cached is not None:

            self._cache.move_to_end(path)

            return cached

        image = default_image_loader.load_grayscale(
            path,
            max_dimension=self._config.orb_decode_dimension,
        )

        if image is None:

            result = ([], None)

        else:

            result = self._orb.detectAndCompute(
                image,
                None,
            )

        self._cache[path] = result

        while len(self._cache) > self._config.cache_size:
            self._cache.popitem(last=False)

        return result
