from collections import defaultdict, deque

import numpy as np

from face_processing.interfaces import EmbeddingSimilarity, FaceClusterer
from face_processing.models import EmbeddedFace, UnknownCluster

MAX_GRAPH_EDGES = 4_000_000


class ClusterCapacityError(RuntimeError):
    """A dense similarity graph would consume unsafe amounts of memory."""


class ConnectedComponentFaceClusterer(FaceClusterer):

    def __init__(
        self,
        similarity: EmbeddingSimilarity,
        strong_threshold: float = 0.85,
        weak_threshold: float = 0.75,
    ):
        if weak_threshold >= strong_threshold:
            raise ValueError(
                "weak_threshold must be smaller than strong_threshold."
            )

        self._similarity = similarity
        self._strong = strong_threshold
        self._weak = weak_threshold

    def cluster(
        self,
        faces: list[EmbeddedFace],
        cancellation_token=None,
    ) -> list[UnknownCluster]:

        if not faces:
            return []

        if len(faces) == 1:
            return [
                UnknownCluster(
                    id=0,
                    faces=faces,
                    representative=faces[0],
                    preview=None,
                )
            ]

        if cancellation_token is not None:
            cancellation_token.raise_if_canceled()
        similarity_matrix = self._build_similarity_matrix(faces)
        if cancellation_token is not None:
            cancellation_token.raise_if_canceled()
        strong_graph, weak_edges = self._build_graphs(similarity_matrix, cancellation_token)

        labels = self._strong_components(strong_graph, len(faces), cancellation_token)

        labels = self._attach_weak_faces(
            labels,
            strong_graph,
            weak_edges,
            cancellation_token,
        )

        grouped_indices = defaultdict(list)

        for index, label in enumerate(labels):
            grouped_indices[label].append(index)

        clusters = []

        for cluster_id, cluster_face_indices in grouped_indices.items():
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            cluster_faces = [faces[index] for index in cluster_face_indices]
            representative = self._representative(
                cluster_face_indices,
                similarity_matrix,
                faces,
            )

            clusters.append(
                UnknownCluster(
                    id=cluster_id,
                    faces=cluster_faces,
                    representative=representative,
                    preview=None,
                )
            )

        clusters.sort(
            key=lambda c: len(c.faces),
            reverse=True,
        )

        return clusters

    @staticmethod
    def _build_similarity_matrix(
        faces: list[EmbeddedFace],
    ) -> np.ndarray:
        embeddings = np.asarray(
            [face.embedding for face in faces],
            dtype=np.float32,
        )
        return embeddings @ embeddings.T

    def _build_graphs(
        self,
        similarity_matrix: np.ndarray,
        cancellation_token=None,
    ) -> tuple[dict[int, list[int]], list[tuple[int, int]]]:

        strong = defaultdict(list)
        weak = []
        edge_count = 0
        # At 10,000 faces, triu_indices_from alone allocates about 800 MiB.
        # Iterate rows to preserve edge order and thresholds with bounded
        # temporary arrays; the similarity matrix remains the main allocation.
        for i in range(len(similarity_matrix) - 1):
            if cancellation_token is not None and i % 32 == 0:
                cancellation_token.raise_if_canceled()
            scores = similarity_matrix[i, i + 1:]
            strong_offsets = np.flatnonzero(scores >= self._strong)
            weak_offsets = np.flatnonzero((scores >= self._weak) & (scores < self._strong))
            edge_count += len(strong_offsets) + len(weak_offsets)
            if edge_count > MAX_GRAPH_EDGES:
                raise ClusterCapacityError("Unknown-face similarity graph exceeds its memory safety limit")
            for offset in strong_offsets:
                j = i + 1 + int(offset)
                strong[i].append(j)
                strong[j].append(i)
            for offset in weak_offsets:
                weak.append((i, i + 1 + int(offset)))

        return strong, weak

    def _strong_components(
            self,
            graph,
            num_faces,
            cancellation_token=None,
    ):

        labels = [-1] * num_faces
        visited = [False] * num_faces

        cluster = 0

        for start in range(num_faces):
            if cancellation_token is not None and start % 256 == 0:
                cancellation_token.raise_if_canceled()

            if visited[start]:
                continue

            queue = deque([start])
            visited[start] = True

            while queue:

                node = queue.popleft()

                labels[node] = cluster

                for neighbor in graph[node]:

                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            cluster += 1

        return labels

    def _attach_weak_faces(
        self,
        labels,
        strong_graph,
        weak_edges,
        cancellation_token=None,
    ):

        changed = True

        while changed:

            changed = False

            for index, (a, b) in enumerate(weak_edges):
                if cancellation_token is not None and index % 1024 == 0:
                    cancellation_token.raise_if_canceled()

                la = labels[a]
                lb = labels[b]

                if la == lb:
                    continue

                if self._has_strong_neighbor_in_cluster(
                    a,
                    lb,
                    labels,
                    strong_graph,
                ):
                    labels[a] = lb
                    changed = True

                elif self._has_strong_neighbor_in_cluster(
                    b,
                    la,
                    labels,
                    strong_graph,
                ):
                    labels[b] = la
                    changed = True

        return labels

    @staticmethod
    def _has_strong_neighbor_in_cluster(
        node,
        cluster,
        labels,
        graph,
    ):

        count = 0

        for neighbor in graph[node]:

            if labels[neighbor] == cluster:
                count += 1

        return count >= 1

    @staticmethod
    def _representative(
        face_indices: list[int],
        similarity_matrix: np.ndarray,
        faces: list[EmbeddedFace],
    ) -> EmbeddedFace:

        if len(face_indices) == 1:
            return faces[face_indices[0]]

        cluster_scores = similarity_matrix[np.ix_(face_indices, face_indices)]
        best_local_index = int(np.argmax(cluster_scores.sum(axis=1)))
        return faces[face_indices[best_local_index]]
