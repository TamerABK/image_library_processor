from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from automatic_scan.project_pipeline import MAX_UNKNOWN_FACES, unknown_face_limit
from face_detector.connected_face_clusterer import (
    ClusterCapacityError, ConnectedComponentFaceClusterer)
from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_processing.models import EmbeddedFace
from scan_controls import CancellationToken, ScanCancelledError


class ClusterCapacityTests(unittest.TestCase):
    def test_original_limit_is_raised_eightfold_but_configurable_downward(self):
        self.assertEqual(MAX_UNKNOWN_FACES, 16_000)
        with patch.dict(os.environ, {"ROOM36_MAX_UNKNOWN_FACES": "9000"}):
            self.assertEqual(unknown_face_limit(), 9000)
        with patch.dict(os.environ, {"ROOM36_MAX_UNKNOWN_FACES": "999999"}):
            self.assertEqual(unknown_face_limit(), 16_000)

    def test_dense_graph_fails_safely_before_unbounded_edge_growth(self):
        matrix = np.ones((12, 12), dtype=np.float32)
        clusterer = ConnectedComponentFaceClusterer(CosineEmbeddingSimilarity())
        with patch("face_detector.connected_face_clusterer.MAX_GRAPH_EDGES", 20):
            with self.assertRaises(ClusterCapacityError):
                clusterer._build_graphs(matrix)

    def test_clustering_honors_cancellation(self):
        vectors = np.eye(32, dtype=np.float32)
        faces = [EmbeddedFace((0, 0, 10, 10), 1.0,
            np.zeros((5, 2), dtype=np.float32), vector, Path(f"{index}.jpg"))
            for index, vector in enumerate(vectors)]
        token = CancellationToken()
        token.cancel()
        clusterer = ConnectedComponentFaceClusterer(CosineEmbeddingSimilarity())
        with self.assertRaises(ScanCancelledError):
            clusterer.cluster(faces, cancellation_token=token)

    def test_clustering_can_cancel_during_graph_construction(self):
        class CancelAfterChecks:
            checks = 0

            def raise_if_canceled(self):
                self.checks += 1
                if self.checks >= 4:
                    raise ScanCancelledError("cancelled during clustering")

        vectors = np.eye(128, dtype=np.float32)
        faces = [EmbeddedFace((0, 0, 10, 10), 1.0,
            np.zeros((5, 2), dtype=np.float32), vector, Path(f"{index}.jpg"))
            for index, vector in enumerate(vectors)]
        with self.assertRaises(ScanCancelledError):
            ConnectedComponentFaceClusterer(CosineEmbeddingSimilarity()).cluster(
                faces, cancellation_token=CancelAfterChecks())


if __name__ == "__main__":
    unittest.main()
