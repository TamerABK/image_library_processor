from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from grouping.models import VibeImageFeatures

from .config import VibeGroupingConfig
from .similarity import CombinedSimilarityComputer, ScenePairComponents
from .temporal_segments import (
    SceneTransitionScore,
    TemporalSegmentation,
    TemporalSession,
    strongest_crossed_boundary,
)


@dataclass(frozen=True, slots=True)
class SceneCohesion:
    semantic: float
    action: float
    people: float
    layout: float
    subject_scene: float
    background: float
    composition: float
    lower_percentile: float
    transition_penalty: float
    timeline_contiguity_score: float
    scene_duration_score: float
    action_confidence_score: float
    internal_soft_boundary_count: int
    internal_hard_boundary_count: int
    combined: float


@dataclass(frozen=True, slots=True)
class ClusteredGroup:
    members: tuple[VibeImageFeatures, ...]
    cohesion: SceneCohesion
    medoid_path: str
    boundary_before: SceneTransitionScore | None
    boundary_after: SceneTransitionScore | None

    @property
    def cohesion_score(self) -> float:
        return self.cohesion.combined


class VibeClusterer:
    def __init__(self, config: VibeGroupingConfig) -> None:
        self._config = config
        self.last_diagnostics: dict[str, object] = {"rejected_edges": []}
        self._rejected_edges: dict[tuple[str, str], dict[str, object]] = {}

    def cluster(
        self,
        segmentation: TemporalSegmentation,
        *,
        similarity: CombinedSimilarityComputer,
    ) -> tuple[list[ClusteredGroup], list[VibeImageFeatures]]:
        self._rejected_edges = {}
        groups: list[ClusteredGroup] = []
        ungrouped: list[VibeImageFeatures] = []

        for session in segmentation.sessions:
            session_groups, session_ungrouped = self._cluster_session(
                session,
                similarity=similarity,
            )
            groups.extend(session_groups)
            ungrouped.extend(session_ungrouped)

        if segmentation.untimed:
            untimed_groups, untimed_ungrouped = self._cluster_untimed(
                segmentation.untimed,
                similarity=similarity,
            )
            groups.extend(untimed_groups)
            ungrouped.extend(untimed_ungrouped)

        groups.sort(
            key=lambda group: (
                group.members[0].capture_timestamp is None,
                float("inf") if group.members[0].capture_timestamp is None else group.members[0].capture_timestamp,
                group.members[0].image_path,
            )
        )
        ungrouped.sort(key=lambda item: item.image_path)
        self.last_diagnostics = {
            "rejected_edges": list(self._rejected_edges.values()),
        }
        return groups, ungrouped

    def _cluster_session(
        self,
        session: TemporalSession,
        *,
        similarity: CombinedSimilarityComputer,
    ) -> tuple[list[ClusteredGroup], list[VibeImageFeatures]]:
        features = list(session.ordered_features)
        if not features:
            return [], []

        memo = _SimilarityMemo(features, similarity)
        provisional_groups: list[list[int]] = []
        ungrouped: list[VibeImageFeatures] = []

        for scene_segment in session.scene_segments:
            local_groups, local_ungrouped = self._cluster_segment(
                features,
                list(scene_segment.member_indices),
                session=session,
                memo=memo,
            )
            provisional_groups.extend(local_groups)
            ungrouped.extend(local_ungrouped)

        provisional_groups = self._merge_adjacent_groups(
            provisional_groups,
            session=session,
            memo=memo,
        )

        groups: list[ClusteredGroup] = []
        promoted_ungrouped_paths = {item.image_path for item in ungrouped}
        pending_groups = [sorted(set(item)) for item in provisional_groups]
        while pending_groups:
            indices = pending_groups.pop(0)
            indices = sorted(set(indices))
            if not indices:
                continue
            structural_split = self._split_on_structure(indices, session=session, memo=memo)
            if structural_split is not None:
                for child in structural_split:
                    if len(child) >= self._config.minimum_group_size:
                        pending_groups.append(child)
                    else:
                        promoted_ungrouped_paths.update(features[index].image_path for index in child)
                continue
            cohesion, medoid_index = self._cohesion(indices, session=session, memo=memo)
            if (
                len(indices) > 1
                and (
                    cohesion.combined < self._config.minimum_group_cohesion
                    or cohesion.internal_hard_boundary_count > 0
                )
            ):
                promoted_ungrouped_paths.update(features[index].image_path for index in indices)
                continue
            members = tuple(
                sorted(
                    (features[index] for index in indices),
                    key=lambda feature: (
                        feature.capture_timestamp is None,
                        feature.capture_timestamp or 0.0,
                        feature.image_path,
                    ),
                )
            )
            first_index = min(indices)
            last_index = max(indices)
            groups.append(
                ClusteredGroup(
                    members=members,
                    cohesion=cohesion,
                    medoid_path=features[medoid_index].image_path,
                    boundary_before=session.transitions[first_index - 1] if first_index > 0 else None,
                    boundary_after=session.transitions[last_index] if last_index < len(session.transitions) else None,
                )
            )

        final_ungrouped = [
            feature
            for feature in features
            if feature.image_path in promoted_ungrouped_paths
        ]
        return groups, final_ungrouped

    def _cluster_untimed(
        self,
        features: list[VibeImageFeatures],
        *,
        similarity: CombinedSimilarityComputer,
    ) -> tuple[list[ClusteredGroup], list[VibeImageFeatures]]:
        if not features:
            return [], []
        if len(features) == 1:
            only = features[0]
            return [
                ClusteredGroup(
                    members=(only,),
                    cohesion=SceneCohesion(
                        semantic=1.0,
                        action=1.0,
                        people=1.0,
                        layout=1.0,
                        subject_scene=1.0,
                        background=1.0,
                        composition=1.0,
                        lower_percentile=1.0,
                        transition_penalty=0.0,
                        timeline_contiguity_score=1.0,
                        scene_duration_score=1.0,
                        action_confidence_score=1.0,
                        internal_soft_boundary_count=0,
                        internal_hard_boundary_count=0,
                        combined=1.0,
                    ),
                    medoid_path=only.image_path,
                    boundary_before=None,
                    boundary_after=None,
                )
            ], []
        if len(features) < self._config.minimum_group_size:
            return [], list(features)

        ordered = sorted(features, key=lambda item: item.image_path)
        memo = _SimilarityMemo(ordered, similarity)
        indices = list(range(len(ordered)))
        groups: list[ClusteredGroup] = []
        ungrouped: list[VibeImageFeatures] = []
        edges = self._build_candidate_graph(ordered, indices, session=None, memo=memo)
        components = self._connected_components(indices, edges)

        for component in components:
            refined = self._refine_component(component, session=None, memo=memo)
            for item in refined:
                if not item:
                    continue
                if len(item) < self._config.minimum_group_size:
                    ungrouped.extend(ordered[index] for index in item)
                    continue
                cohesion, medoid_index = self._cohesion(item, session=None, memo=memo)
                if len(item) > 1 and cohesion.combined < self._config.minimum_group_cohesion:
                    ungrouped.extend(ordered[index] for index in item)
                    continue
                members = tuple(ordered[index] for index in sorted(item))
                groups.append(
                    ClusteredGroup(
                        members=members,
                        cohesion=cohesion,
                        medoid_path=ordered[medoid_index].image_path,
                        boundary_before=None,
                        boundary_after=None,
                    )
                )
        return groups, ungrouped

    def _cluster_segment(
        self,
        features: list[VibeImageFeatures],
        indices: list[int],
        *,
        session: TemporalSession,
        memo: "_SimilarityMemo",
    ) -> tuple[list[list[int]], list[VibeImageFeatures]]:
        if len(indices) == 1:
            return [indices], []
        if len(indices) < self._config.minimum_group_size:
            return [], [features[index] for index in indices]

        neighbor_edges = self._build_candidate_graph(features, indices, session=session, memo=memo)
        connected_components = self._connected_components(indices, neighbor_edges)
        groups: list[list[int]] = []
        ungrouped: list[VibeImageFeatures] = []
        for component in connected_components:
            if len(component) == 1:
                groups.append(component)
                continue
            if len(component) < self._config.minimum_group_size:
                ungrouped.extend(features[index] for index in component)
                continue
            refined = self._refine_component(component, session=session, memo=memo)
            for item in refined:
                if len(item) == 1:
                    groups.append(sorted(item))
                elif len(item) < self._config.minimum_group_size:
                    ungrouped.extend(features[index] for index in item)
                else:
                    groups.append(sorted(item))
        return groups, ungrouped

    def _build_candidate_graph(
        self,
        features: list[VibeImageFeatures],
        indices: list[int],
        *,
        session: TemporalSession | None,
        memo: "_SimilarityMemo",
    ) -> dict[tuple[int, int], float]:
        local_features = [features[index] for index in indices]
        embeddings = np.stack([feature.semantic_embedding for feature in local_features], axis=0)
        count = len(indices)
        neighbor_count = min(self._config.scene_neighbor_count, max(count - 1, 0))
        edges: dict[tuple[int, int], float] = {}

        if neighbor_count > 0:
            chunk_size = 256
            for start in range(0, count, chunk_size):
                end = min(start + chunk_size, count)
                chunk = embeddings[start:end]
                scores = chunk @ embeddings.T
                row_indices = np.arange(start, end)
                scores[np.arange(end - start), row_indices] = -1.0
                kth = min(neighbor_count - 1, max(scores.shape[1] - 1, 0))
                topk_indices = np.argpartition(-scores, kth=kth, axis=1)[:, :neighbor_count]
                for local_row, neighbors in enumerate(topk_indices):
                    left_index = indices[start + local_row]
                    for neighbor in neighbors:
                        right_index = indices[int(neighbor)]
                        self._maybe_add_edge(
                            features,
                            left_index,
                            right_index,
                            session=session,
                            memo=memo,
                            threshold=self._config.minimum_pair_similarity,
                            edges=edges,
                        )

        for offset, left_index in enumerate(indices):
            for step in range(1, self._config.adjacent_timeline_radius + 1):
                if offset + step >= len(indices):
                    break
                right_index = indices[offset + step]
                self._maybe_add_edge(
                    features,
                    left_index,
                    right_index,
                    session=session,
                    memo=memo,
                    threshold=self._config.minimum_pair_similarity - 0.03,
                    edges=edges,
                )
        return edges

    def _maybe_add_edge(
        self,
        features: list[VibeImageFeatures],
        left_index: int,
        right_index: int,
        *,
        session: TemporalSession | None,
        memo: "_SimilarityMemo",
        threshold: float,
        edges: dict[tuple[int, int], float],
    ) -> None:
        if left_index == right_index:
            return
        key = _edge_key(left_index, right_index)
        components = memo.components(*key)
        raw_score = components.combined
        adjusted_score = raw_score

        if session is not None:
            boundary = strongest_crossed_boundary(left_index, right_index, session.transitions)
            if boundary.crossed_hard_boundary:
                self._record_rejected_edge(features, key, "crossed_hard_boundary", raw_score, adjusted_score)
                return
            if abs(left_index - right_index) > self._config.maximum_timeline_edge_distance:
                if not (
                    components.semantic >= self._config.long_range_semantic_threshold
                    and components.subject_scene >= self._config.long_range_subject_scene_threshold
                    and not boundary.crossed_soft_boundary
                    and not components.action_hard_conflict
                ):
                    self._record_rejected_edge(features, key, "timeline_distance", raw_score, adjusted_score)
                    return
            if components.action_hard_conflict:
                self._record_rejected_edge(features, key, "action_conflict", raw_score, adjusted_score)
                return
            if boundary.crossed_soft_boundary:
                adjusted_score = max(0.0, adjusted_score - self._config.soft_cross_boundary_penalty)
                if adjusted_score < self._config.cross_soft_boundary_similarity:
                    self._record_rejected_edge(
                        features,
                        key,
                        "crossed_soft_boundary_below_threshold",
                        raw_score,
                        adjusted_score,
                    )
                    return
        else:
            if components.action_hard_conflict:
                self._record_rejected_edge(features, key, "action_conflict", raw_score, adjusted_score)
                return

        if adjusted_score < threshold:
            self._record_rejected_edge(features, key, "low_similarity", raw_score, adjusted_score)
            return
        edges[key] = max(edges.get(key, 0.0), adjusted_score)

    def _record_rejected_edge(
        self,
        features: list[VibeImageFeatures],
        edge: tuple[int, int],
        reason: str,
        raw_score: float,
        adjusted_score: float,
    ) -> None:
        left = features[edge[0]].image_path
        right = features[edge[1]].image_path
        key = (left, right)
        if key in self._rejected_edges:
            return
        self._rejected_edges[key] = {
            "left_image": left,
            "right_image": right,
            "reason": reason,
            "raw_similarity": round(raw_score, 4),
            "adjusted_similarity": round(adjusted_score, 4),
            "timeline_distance": abs(edge[0] - edge[1]),
        }

    def _connected_components(
        self,
        indices: list[int],
        edges: dict[tuple[int, int], float],
    ) -> list[list[int]]:
        adjacency: dict[int, list[int]] = defaultdict(list)
        for (left, right), _score in edges.items():
            adjacency[left].append(right)
            adjacency[right].append(left)

        seen: set[int] = set()
        components: list[list[int]] = []
        for start in indices:
            if start in seen:
                continue
            stack = [start]
            component: list[int] = []
            while stack:
                index = stack.pop()
                if index in seen:
                    continue
                seen.add(index)
                component.append(index)
                stack.extend(adjacency.get(index, ()))
            components.append(sorted(component))
        return components

    def _refine_component(
        self,
        indices: list[int],
        *,
        session: TemporalSession | None,
        memo: "_SimilarityMemo",
    ) -> list[list[int]]:
        indices = sorted(indices)
        if len(indices) < 2:
            return [indices]

        structural_split = self._split_on_structure(indices, session=session, memo=memo)
        if structural_split is not None:
            return [
                child
                for item in structural_split
                for child in self._refine_component(item, session=session, memo=memo)
            ]

        cohesion, _medoid = self._cohesion(indices, session=session, memo=memo)
        if (
            len(indices) <= self._config.maximum_group_size
            and cohesion.combined >= self._config.minimum_group_cohesion
            and cohesion.internal_hard_boundary_count == 0
        ):
            return [indices]

        if len(indices) <= 2:
            return [[index] for index in indices]

        labels = self._agglomerative_split(indices, memo)
        left = [index for index, label in zip(indices, labels) if label == 0]
        right = [index for index, label in zip(indices, labels) if label == 1]
        if not left or not right or left == indices or right == indices:
            midpoint = len(indices) // 2
            left = indices[:midpoint]
            right = indices[midpoint:]
        return [
            child
            for item in (left, right)
            for child in self._refine_component(item, session=session, memo=memo)
        ]

    def _split_on_structure(
        self,
        indices: list[int],
        *,
        session: TemporalSession | None,
        memo: "_SimilarityMemo",
    ) -> list[list[int]] | None:
        ordered = sorted(indices)
        split_positions: set[int] = set()
        missing_inside_span = 0
        internal_gap_count = 0

        for position in range(len(ordered) - 1):
            left_index = ordered[position]
            right_index = ordered[position + 1]
            gap_size = max(0, right_index - left_index - 1)
            if gap_size > 0:
                missing_inside_span += gap_size
                internal_gap_count += 1
            if gap_size > self._config.maximum_internal_gap_size:
                split_positions.add(position)
            if session is None:
                continue
            if (
                session.ordered_features[left_index].capture_timestamp is not None
                and session.ordered_features[right_index].capture_timestamp is not None
            ):
                adjacent_gap = abs(
                    session.ordered_features[right_index].capture_timestamp
                    - session.ordered_features[left_index].capture_timestamp
                )
                if adjacent_gap > self._config.maximum_adjacent_gap_within_scene_seconds:
                    split_positions.add(position)
            if right_index == left_index + 1:
                transition = session.transitions[left_index]
                if transition.accepted_hard_boundary:
                    split_positions.add(position)

        if missing_inside_span > self._config.allowed_internal_outliers:
            for position in range(len(ordered) - 1):
                if ordered[position + 1] != ordered[position] + 1:
                    split_positions.add(position)
        if internal_gap_count > self._config.maximum_internal_timeline_gaps:
            for position in range(len(ordered) - 1):
                if ordered[position + 1] != ordered[position] + 1:
                    split_positions.add(position)

        if session is not None and not split_positions:
            duration_seconds = self._scene_duration_seconds(ordered, session=session)
            if duration_seconds > self._config.maximum_scene_span_seconds:
                split_positions.add(self._best_split_position(ordered, session=session))

        if not split_positions and len(ordered) > self._config.maximum_group_size:
            split_positions.add(len(ordered) // 2 - 1)

        if not split_positions:
            return None
        return _split_indices(ordered, sorted(split_positions))

    def _best_split_position(
        self,
        ordered: list[int],
        *,
        session: TemporalSession,
    ) -> int:
        best_position = max(0, len(ordered) // 2 - 1)
        best_score = -1.0
        for position in range(len(ordered) - 1):
            left_index = ordered[position]
            right_index = ordered[position + 1]
            score = 0.0
            if right_index == left_index + 1:
                score = session.transitions[left_index].combined_transition
            else:
                score = 0.5 + min(0.5, (right_index - left_index - 1) / 10.0)
            if score > best_score:
                best_score = score
                best_position = position
        return best_position

    def _agglomerative_split(
        self,
        indices: list[int],
        memo: "_SimilarityMemo",
    ) -> np.ndarray:
        distance_matrix = np.zeros((len(indices), len(indices)), dtype=np.float32)
        for row in range(len(indices)):
            for column in range(row + 1, len(indices)):
                score = memo.score(indices[row], indices[column])
                distance = 1.0 - score
                distance_matrix[row, column] = distance
                distance_matrix[column, row] = distance
        clustering = AgglomerativeClustering(
            n_clusters=2,
            metric="precomputed",
            linkage="average",
        )
        return clustering.fit_predict(distance_matrix)

    def _cohesion(
        self,
        indices: list[int],
        *,
        session: TemporalSession | None,
        memo: "_SimilarityMemo",
    ) -> tuple[SceneCohesion, int]:
        if len(indices) == 1:
            action_profile = memo.action_profile(indices[0])
            cohesion = SceneCohesion(
                semantic=1.0,
                action=1.0,
                people=1.0,
                layout=1.0,
                subject_scene=1.0,
                background=1.0,
                composition=1.0,
                lower_percentile=1.0,
                transition_penalty=0.0,
                timeline_contiguity_score=1.0,
                scene_duration_score=1.0,
                action_confidence_score=1.0 if action_profile.confident else 0.5,
                internal_soft_boundary_count=0,
                internal_hard_boundary_count=0,
                combined=1.0,
            )
            return cohesion, indices[0]

        pair_scores: list[float] = []
        semantic_scores: list[float] = []
        action_scores: list[float] = []
        people_scores: list[float] = []
        layout_scores: list[float] = []
        subject_scene_scores: list[float] = []
        background_scores: list[float] = []
        composition_scores: list[float] = []
        mean_scores = np.ones((len(indices), len(indices)), dtype=np.float32)

        for row in range(len(indices)):
            for column in range(row + 1, len(indices)):
                components = memo.components(indices[row], indices[column])
                score = components.combined
                pair_scores.append(score)
                semantic_scores.append(components.semantic)
                action_scores.append(components.action)
                people_scores.append(components.people)
                layout_scores.append(components.layout)
                subject_scene_scores.append(components.subject_scene)
                background_scores.append(components.background)
                composition_scores.append(components.composition)
                mean_scores[row, column] = score
                mean_scores[column, row] = score

        means = mean_scores.mean(axis=1)
        medoid_offset = int(np.argmax(means))
        medoid_index = indices[medoid_offset]
        lower_percentile = 0.0 if not pair_scores else float(np.percentile(pair_scores, 20))
        transition_penalty, internal_soft_count, internal_hard_count = self._internal_transition_penalty(
            indices,
            session=session,
        )
        timeline_contiguity_score = self._timeline_contiguity_score(indices)
        scene_duration_score = self._scene_duration_score(indices, session=session)
        action_confidence_score = _mean(
            [
                1.0 if memo.action_profile(index).confident else 0.0
                for index in indices
            ],
            default=0.5,
        )

        combined = (
            (0.16 * _mean(semantic_scores, default=0.5))
            + (0.12 * _mean(action_scores, default=0.5))
            + (0.08 * _mean(people_scores, default=0.5))
            + (0.10 * _mean(layout_scores, default=0.5))
            + (0.12 * _mean(subject_scene_scores, default=0.5))
            + (0.06 * _mean(background_scores, default=0.5))
            + (0.10 * _mean(composition_scores, default=0.5))
            + (0.12 * lower_percentile)
            + (0.08 * timeline_contiguity_score)
            + (0.04 * scene_duration_score)
            + (0.04 * action_confidence_score)
            - transition_penalty
        )
        if internal_hard_count > 0:
            combined = min(combined, 0.0)

        cohesion = SceneCohesion(
            semantic=_mean(semantic_scores, default=0.5),
            action=_mean(action_scores, default=0.5),
            people=_mean(people_scores, default=0.5),
            layout=_mean(layout_scores, default=0.5),
            subject_scene=_mean(subject_scene_scores, default=0.5),
            background=_mean(background_scores, default=0.5),
            composition=_mean(composition_scores, default=0.5),
            lower_percentile=lower_percentile,
            transition_penalty=transition_penalty,
            timeline_contiguity_score=timeline_contiguity_score,
            scene_duration_score=scene_duration_score,
            action_confidence_score=action_confidence_score,
            internal_soft_boundary_count=internal_soft_count,
            internal_hard_boundary_count=internal_hard_count,
            combined=float(np.clip(combined, 0.0, 1.0)),
        )
        return cohesion, medoid_index

    def _internal_transition_penalty(
        self,
        indices: list[int],
        *,
        session: TemporalSession | None,
    ) -> tuple[float, int, int]:
        if session is None:
            return 0.0, 0, 0
        ordered = sorted(indices)
        internal_soft = 0
        internal_hard = 0
        max_soft_value = 0.0
        for position in range(len(ordered) - 1):
            left_index = ordered[position]
            right_index = ordered[position + 1]
            if right_index != left_index + 1:
                continue
            transition = session.transitions[left_index]
            if transition.accepted_hard_boundary:
                internal_hard += 1
            elif transition.accepted_soft_boundary:
                internal_soft += 1
                max_soft_value = max(max_soft_value, transition.combined_transition)
        penalty = (
            (1.0 if internal_hard > 0 else 0.0)
            + (internal_soft * self._config.internal_soft_boundary_penalty)
            + (0.05 * max_soft_value if internal_soft > 0 else 0.0)
        )
        return penalty, internal_soft, internal_hard

    def _timeline_contiguity_score(self, indices: list[int]) -> float:
        if len(indices) <= 1:
            return 1.0
        ordered = sorted(indices)
        span = ordered[-1] - ordered[0] + 1
        missing = span - len(ordered)
        gap_count = sum(1 for left, right in zip(ordered, ordered[1:]) if right != left + 1)
        score = 1.0 - min(0.7, (missing / max(span, 1))) - min(0.3, gap_count * 0.1)
        return float(np.clip(score, 0.0, 1.0))

    def _scene_duration_score(
        self,
        indices: list[int],
        *,
        session: TemporalSession | None,
    ) -> float:
        if session is None or len(indices) <= 1:
            return 1.0
        duration = self._scene_duration_seconds(indices, session=session)
        if duration <= self._config.maximum_scene_span_seconds:
            return 1.0
        excess = duration - self._config.maximum_scene_span_seconds
        return float(np.clip(np.exp(-excess / max(self._config.maximum_scene_span_seconds, 1)), 0.0, 1.0))

    def _scene_duration_seconds(
        self,
        indices: list[int],
        *,
        session: TemporalSession,
    ) -> float:
        ordered = sorted(indices)
        first = session.ordered_features[ordered[0]].capture_timestamp
        last = session.ordered_features[ordered[-1]].capture_timestamp
        if first is None or last is None:
            return 0.0
        return float(max(0.0, last - first))

    def _merge_adjacent_groups(
        self,
        provisional_groups: list[list[int]],
        *,
        session: TemporalSession,
        memo: "_SimilarityMemo",
    ) -> list[list[int]]:
        if not provisional_groups:
            return []

        groups = [sorted(group) for group in provisional_groups]
        groups.sort(key=lambda item: item[0])
        merged: list[list[int]] = [groups[0]]

        for current in groups[1:]:
            previous = merged[-1]
            if not self._are_adjacent(previous, current):
                merged.append(current)
                continue
            boundary = session.transitions[previous[-1]]
            if boundary.accepted_hard_boundary:
                merged.append(current)
                continue
            bridge_score = self._adjacent_group_similarity(previous, current, memo=memo)
            if boundary.accepted_soft_boundary and bridge_score < self._config.cross_soft_boundary_similarity:
                merged.append(current)
                continue
            if bridge_score < self._config.adjacent_merge_similarity:
                merged.append(current)
                continue
            candidate = sorted(set(previous + current))
            if self._split_on_structure(candidate, session=session, memo=memo) is not None:
                merged.append(current)
                continue
            cohesion, _medoid = self._cohesion(candidate, session=session, memo=memo)
            if cohesion.combined < self._config.minimum_group_cohesion:
                merged.append(current)
                continue
            merged[-1] = candidate
        return merged

    @staticmethod
    def _are_adjacent(left: list[int], right: list[int]) -> bool:
        return max(left) + 1 == min(right)

    def _adjacent_group_similarity(
        self,
        left: list[int],
        right: list[int],
        *,
        memo: "_SimilarityMemo",
    ) -> float:
        anchors = [left[-1], left[max(0, len(left) - 2)], right[0], right[min(len(right) - 1, 1)]]
        comparisons = [
            memo.score(anchors[0], anchors[2]),
            memo.score(anchors[0], anchors[3]),
            memo.score(anchors[1], anchors[2]),
        ]
        return sum(comparisons) / len(comparisons)


class _SimilarityMemo:
    def __init__(
        self,
        features: list[VibeImageFeatures],
        similarity: CombinedSimilarityComputer,
    ) -> None:
        self._features = features
        self._similarity = similarity

    @lru_cache(maxsize=250000)
    def score(self, left_index: int, right_index: int) -> float:
        if left_index == right_index:
            return 1.0
        if left_index > right_index:
            left_index, right_index = right_index, left_index
        return self._similarity.pair_similarity(
            self._features[left_index],
            self._features[right_index],
        )

    @lru_cache(maxsize=250000)
    def components(self, left_index: int, right_index: int) -> ScenePairComponents:
        if left_index == right_index:
            return ScenePairComponents(
                semantic=1.0,
                action=1.0,
                people=1.0,
                layout=1.0,
                subject_scene=1.0,
                background=1.0,
                time=1.0,
                composition=1.0,
                color=1.0,
                action_conflict_penalty=0.0,
                participant_mode_conflict_penalty=0.0,
                shot_mode_conflict_penalty=0.0,
                main_vs_reaction_penalty=0.0,
                temporal_bridge_penalty=0.0,
                transition_penalty=0.0,
                action_hard_conflict=False,
                action_soft_conflict=False,
                participant_mode_hard_conflict=False,
                participant_mode_soft_conflict=False,
                participant_conflict_strength="none",
                shot_mode_conflict=False,
                main_vs_reaction_conflict=False,
                main_vs_reaction_confident=False,
                action_confidence_mean=1.0,
                action_margin_mean=0.0,
                action_reliable=True,
                left_shot_margin=0.0,
                right_shot_margin=0.0,
                left_shot_confidence="strong",
                right_shot_confidence="strong",
                combined=1.0,
            )
        if left_index > right_index:
            left_index, right_index = right_index, left_index
        return self._similarity.pair_components(
            self._features[left_index],
            self._features[right_index],
        )

    def action_profile(self, index: int):
        return self._similarity.action_profile(self._features[index])


def _edge_key(left_index: int, right_index: int) -> tuple[int, int]:
    return (left_index, right_index) if left_index < right_index else (right_index, left_index)


def _mean(values: list[float], *, default: float) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def _split_indices(indices: list[int], split_positions: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    last = 0
    for boundary in sorted(set(split_positions)):
        result.append(indices[last:boundary + 1])
        last = boundary + 1
    result.append(indices[last:])
    return [item for item in result if item]
