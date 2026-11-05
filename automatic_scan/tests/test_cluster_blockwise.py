"""Dense pre-change reference against the production blockwise path."""
from collections import defaultdict
from pathlib import Path
import unittest

import numpy as np

from face_detector.connected_face_clusterer import ConnectedComponentFaceClusterer, effective_block_rows
from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_processing.models import EmbeddedFace
from scan_controls import ScanCancelledError


def _faces(vectors):
    return [EmbeddedFace((0, 0, 10, 10), 1., np.zeros((5, 2), np.float32),
                         vector, Path(f"{i}.jpg")) for i, vector in enumerate(vectors)]


def _dense_reference(clusterer, faces):
    embeddings = np.asarray([face.embedding for face in faces], dtype=np.float32)
    matrix = embeddings @ embeddings.T
    strong, weak = defaultdict(list), []
    for i in range(len(faces) - 1):
        for j in range(i + 1, len(faces)):
            score = matrix[i, j]
            if score >= clusterer._strong:
                strong[i].append(j)
                strong[j].append(i)
            elif score >= clusterer._weak:
                weak.append((i, j))
    labels = clusterer._strong_components(strong, len(faces))
    labels = clusterer._attach_weak_faces(labels, strong, weak)
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    answer = []
    for indices in groups.values():
        scores = matrix[np.ix_(indices, indices)].sum(axis=1)
        answer.append((tuple(indices), indices[int(np.argmax(scores))]))
    answer.sort(key=lambda item: len(item[0]), reverse=True)
    return answer, strong, weak, labels


class BlockwiseClusteringTests(unittest.TestCase):
    def test_budget_policy(self):
        self.assertEqual(effective_block_rows(8219), 256)
        self.assertEqual(effective_block_rows(16000, budget_bytes=1024), 1)
        self.assertEqual(effective_block_rows(4000, max_rows=128), 128)
        with self.assertRaises(ValueError):
            effective_block_rows(100, max_rows=0)

    def test_dense_reference_matches_all_block_sizes(self):
        rng = np.random.default_rng(371)
        templates = rng.normal(size=(8, 32)).astype(np.float32)
        templates /= np.linalg.norm(templates, axis=1, keepdims=True)
        vectors = np.vstack([templates[i % 8] + rng.normal(0, .012, 32)
                             for i in range(35)]).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        faces = _faces(vectors)
        dense_matrix = vectors @ vectors.T
        for rows in (1, 2, 16, 64, 128, 256):
            with self.subTest(rows=rows):
                maximum_difference = 0.0
                for start in range(0, len(vectors), rows):
                    block = vectors[start:start + rows] @ vectors[start:].T
                    reference = dense_matrix[start:start + rows, start:]
                    maximum_difference = max(maximum_difference,
                                             float(np.max(np.abs(block - reference))))
                self.assertLessEqual(maximum_difference, 1e-6)
                clusterer = ConnectedComponentFaceClusterer(CosineEmbeddingSimilarity(),
                    strong_threshold=.85, weak_threshold=.75, block_rows=rows)
                expected, strong, weak, labels = _dense_reference(clusterer, faces)
                actual_strong, actual_weak, stats = clusterer._build_graphs_blockwise(vectors, rows)
                self.assertEqual(dict(strong), dict(actual_strong))
                self.assertEqual(weak, actual_weak)
                self.assertEqual(sum(len(x) for x in strong.values()) // 2, stats["strong_edges"])
                self.assertEqual(labels, clusterer._attach_weak_faces(
                    clusterer._strong_components(actual_strong, len(faces)), actual_strong, actual_weak))
                clusters = clusterer.cluster(faces)
                positions = {id(face): index for index, face in enumerate(faces)}
                self.assertEqual(expected, [(tuple(positions[id(face)] for face in c.faces),
                                            positions[id(c.representative)]) for c in clusters])
                self.assertLessEqual(clusterer.last_diagnostics["max_similarity_block_bytes"],
                                     rows * len(faces) * 4)

    def test_empty_single_and_uneven_blocks(self):
        clusterer = ConnectedComponentFaceClusterer(CosineEmbeddingSimilarity(), block_rows=2)
        self.assertEqual(clusterer.cluster([]), [])
        vectors = np.eye(5, dtype=np.float32)
        faces = _faces(vectors)
        self.assertIs(clusterer.cluster(faces[:1])[0].representative, faces[0])
        self.assertEqual(len(clusterer.cluster(faces)), 5)
        self.assertEqual(clusterer.last_diagnostics["block_count"], 3)

    def test_exact_thresholds_and_weak_edges(self):
        vectors = np.array([[1, 0], [.63, .55], [.55, .63], [0, 1]], dtype=np.float32)
        faces = _faces(vectors)
        for rows in (1, 2, 16):
            clusterer = ConnectedComponentFaceClusterer(CosineEmbeddingSimilarity(),
                strong_threshold=.63, weak_threshold=.55, block_rows=rows)
            expected, strong, weak, _ = _dense_reference(clusterer, faces)
            actual_strong, actual_weak, _ = clusterer._build_graphs_blockwise(vectors, rows)
            self.assertEqual(dict(strong), dict(actual_strong))
            self.assertEqual(weak, actual_weak)
            positions = {id(face): index for index, face in enumerate(faces)}
            actual = [(tuple(positions[id(face)] for face in group.faces),
                       positions[id(group.representative)]) for group in clusterer.cluster(faces)]
            self.assertEqual(expected, actual)

    def test_large_cluster_representative_matches_dense(self):
        rng = np.random.default_rng(728)
        vectors = rng.normal(size=(300, 32)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        faces = _faces(vectors)
        clusterer = ConnectedComponentFaceClusterer(CosineEmbeddingSimilarity(),
            strong_threshold=-2, weak_threshold=-3, block_rows=16)
        expected, _, _, _ = _dense_reference(clusterer, faces)
        actual = clusterer.cluster(faces)
        self.assertEqual(len(actual), 1)
        self.assertIs(actual[0].representative, faces[expected[0][1]])

    def test_representative_selection_can_be_cancelled_between_blocks(self):
        class CancelOnSecondCheck:
            checks = 0

            def raise_if_canceled(self):
                self.checks += 1
                if self.checks == 2:
                    raise ScanCancelledError("cancelled during representative selection")

        vectors = np.ones((5, 2), dtype=np.float32)
        faces = _faces(vectors)
        with self.assertRaises(ScanCancelledError):
            ConnectedComponentFaceClusterer._representative(
                list(range(5)), vectors, faces, 2, CancelOnSecondCheck())


if __name__ == "__main__":
    unittest.main()
