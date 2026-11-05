from collections import defaultdict, deque
import time

import numpy as np

from face_processing.interfaces import EmbeddingSimilarity, FaceClusterer
from face_processing.models import EmbeddedFace, UnknownCluster

MAX_GRAPH_EDGES = 4_000_000
DEFAULT_BLOCK_ROWS = 256
DEFAULT_BLOCK_BUDGET_BYTES = 16 * 1024 * 1024


def _process_memory():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return info.rss, getattr(info, "private", None)
    except (ImportError, OSError):
        return None, None


def effective_block_rows(face_count: int, *, max_rows: int = DEFAULT_BLOCK_ROWS,
                         budget_bytes: int = DEFAULT_BLOCK_BUDGET_BYTES) -> int:
    if max_rows < 1 or budget_bytes < 4:
        raise ValueError("Clustering block rows and budget must be positive")
    return max(1, min(max_rows, budget_bytes // max(4, face_count * 4)))


class ClusterCapacityError(RuntimeError):
    """A dense similarity graph would consume unsafe amounts of memory."""


class ConnectedComponentFaceClusterer(FaceClusterer):

    def __init__(
        self,
        similarity: EmbeddingSimilarity,
        strong_threshold: float = 0.85,
        weak_threshold: float = 0.75,
        block_rows: int = DEFAULT_BLOCK_ROWS,
        block_budget_bytes: int = DEFAULT_BLOCK_BUDGET_BYTES,
    ):
        if weak_threshold >= strong_threshold:
            raise ValueError(
                "weak_threshold must be smaller than strong_threshold."
            )

        self._similarity = similarity
        self._strong = strong_threshold
        self._weak = weak_threshold
        self._block_rows = block_rows
        self._block_budget_bytes = block_budget_bytes
        self.last_diagnostics: dict[str, int | float] = {}

    def cluster(
        self,
        faces: list[EmbeddedFace],
        cancellation_token=None,
    ) -> list[UnknownCluster]:
        self.last_diagnostics = {}
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
        rss_before, private_before = _process_memory()
        started = time.perf_counter()
        embeddings = np.asarray([face.embedding for face in faces], dtype=np.float32)
        rows = effective_block_rows(len(faces), max_rows=self._block_rows,
                                    budget_bytes=self._block_budget_bytes)
        if cancellation_token is not None:
            cancellation_token.raise_if_canceled()
        strong_graph, weak_edges, graph_stats = self._build_graphs_blockwise(
            embeddings, rows, cancellation_token)

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
        representative_started = time.perf_counter()

        for cluster_id, cluster_face_indices in grouped_indices.items():
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            cluster_faces = [faces[index] for index in cluster_face_indices]
            representative = self._representative(
                cluster_face_indices,
                embeddings,
                faces,
                rows,
                cancellation_token,
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

        self.last_diagnostics = dict(face_count=len(faces), embedding_dimension=embeddings.shape[1],
            embedding_bytes=embeddings.nbytes, block_rows=rows,
            block_budget_bytes=self._block_budget_bytes,
            max_similarity_block_bytes=min(rows, len(faces)) * len(faces) * 4,
            block_count=(len(faces) + rows - 1) // rows,
            largest_component=max(map(len, grouped_indices.values())),
            representative_seconds=time.perf_counter() - representative_started,
            total_seconds=time.perf_counter() - started, **graph_stats)
        rss_after, private_after = _process_memory()
        if rss_before is not None and rss_after is not None:
            self.last_diagnostics.update(process_rss_before_bytes=rss_before,
                                         process_rss_after_bytes=rss_after)
        if private_before is not None and private_after is not None:
            self.last_diagnostics.update(process_private_before_bytes=private_before,
                                         process_private_after_bytes=private_after)
        return clusters

    def _build_graphs_blockwise(self, embeddings, rows, cancellation_token=None):
        strong = defaultdict(list)
        weak = []
        strong_count = weak_count = max_adjacency = 0
        similarity_seconds = graph_seconds = 0.0
        n = len(embeddings)
        for start in range(0, n, rows):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            tick = time.perf_counter()
            # Earlier columns cannot contain j > i for any row in this block.
            block = embeddings[start:start + rows] @ embeddings[start:].T
            similarity_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            for local, i in enumerate(range(start, min(start + rows, n))):
                if cancellation_token is not None and i % 32 == 0:
                    cancellation_token.raise_if_canceled()
                scores = block[local, local + 1:]
                strong_offsets = np.flatnonzero(scores >= self._strong)
                weak_offsets = np.flatnonzero((scores >= self._weak) & (scores < self._strong))
                if strong_count + weak_count + len(strong_offsets) + len(weak_offsets) > MAX_GRAPH_EDGES:
                    raise ClusterCapacityError("Unknown-face similarity graph exceeds its memory safety limit")
                strong_count += len(strong_offsets)
                weak_count += len(weak_offsets)
                for offset in strong_offsets:
                    j = i + 1 + int(offset)
                    strong[i].append(j)
                    strong[j].append(i)
                    max_adjacency = max(max_adjacency, len(strong[i]), len(strong[j]))
                for offset in weak_offsets:
                    weak.append((i, i + 1 + int(offset)))
            graph_seconds += time.perf_counter() - tick
            del block
        return strong, weak, dict(strong_edges=strong_count, weak_edges=weak_count,
            total_edges=strong_count + weak_count, max_adjacency_length=max_adjacency,
            similarity_seconds=similarity_seconds, graph_seconds=graph_seconds)

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
            traversed = 0
            while queue:
                if cancellation_token is not None and traversed % 1024 == 0:
                    cancellation_token.raise_if_canceled()
                node = queue.popleft()
                traversed += 1

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
        embeddings: np.ndarray,
        faces: list[EmbeddedFace],
        rows: int = DEFAULT_BLOCK_ROWS,
        cancellation_token=None,
    ) -> EmbeddedFace:

        if len(face_indices) == 1:
            return faces[face_indices[0]]

        selected = embeddings[face_indices]
        scores = np.empty(len(face_indices), dtype=np.float32)
        for start in range(0, len(face_indices), rows):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()
            block = selected[start:start + rows] @ selected.T
            scores[start:start + len(block)] = block.sum(axis=1)
            del block
        best_local_index = int(np.argmax(scores))
        return faces[face_indices[best_local_index]]
