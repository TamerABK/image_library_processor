from collections import defaultdict, deque

from .interfaces import EmbeddingSimilarity, FaceClusterer
from .models import EmbeddedFace, UnknownCluster
import numpy as np


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

        similarity_matrix = self._build_similarity_matrix(faces)
        strong_graph, weak_edges = self._build_graphs(similarity_matrix)

        labels = self._strong_components(strong_graph, len(faces))

        labels = self._attach_weak_faces(
            labels,
            strong_graph,
            weak_edges,
        )

        grouped_indices = defaultdict(list)

        for index, label in enumerate(labels):
            grouped_indices[label].append(index)

        clusters = []

        for cluster_id, cluster_face_indices in grouped_indices.items():
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
    ) -> tuple[dict[int, list[int]], list[tuple[int, int]]]:

        strong = defaultdict(list)
        weak = []
        upper_indices = np.triu_indices_from(similarity_matrix, k=1)
        pair_scores = similarity_matrix[upper_indices]

        strong_mask = pair_scores >= self._strong
        for i, j in zip(upper_indices[0][strong_mask], upper_indices[1][strong_mask]):
            strong[int(i)].append(int(j))
            strong[int(j)].append(int(i))

        weak_mask = (pair_scores >= self._weak) & ~strong_mask
        weak.extend(
            (int(i), int(j))
            for i, j in zip(upper_indices[0][weak_mask], upper_indices[1][weak_mask])
        )

        return strong, weak

    def _strong_components(
            self,
            graph,
            num_faces,
    ):

        labels = [-1] * num_faces
        visited = [False] * num_faces

        cluster = 0

        for start in range(num_faces):

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
    ):

        changed = True

        while changed:

            changed = False

            for a, b in weak_edges:

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
