from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from app_paths import app_data_path
from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_detector.face_database_sqlite import SQLiteFaceDatabase
from image_file_utils import find_supported_files
from image_loader import default_image_loader
from scan_controls import CancellationToken, ScanCancelledError

from grouping.models import ProgressCallback, ScanError, VibeGroup, VibeGroupingResult, VibeImageFeatures

from .cache import VibeFeatureCache, VibeGroupingResultCache
from .clusterer import ClusteredGroup, VibeClusterer
from .config import VibeGroupingConfig
from .embedder import VibeEmbedder, load_embedder
from .features import ExtractionSummary, VibeFeatureExtractor
from .labels import build_group_label
from .prototypes import ScenePrototypeTable, load_prototype_table
from .similarity import CombinedSimilarityComputer
from .temporal_segments import segment_by_time


LOGGER = logging.getLogger(__name__)


class VibeGroupingProcessor:
    supported_extensions = default_image_loader.supported_extensions()

    def __init__(
        self,
        config: VibeGroupingConfig | None = None,
        *,
        feature_cache: VibeFeatureCache | None = None,
        result_cache: VibeGroupingResultCache | None = None,
        embedder: VibeEmbedder | None = None,
        face_database: SQLiteFaceDatabase | None = None,
        similarity: CosineEmbeddingSimilarity | None = None,
        prototype_table: ScenePrototypeTable | None = None,
    ) -> None:
        self._config = config or VibeGroupingConfig()
        self._feature_cache = feature_cache or VibeFeatureCache()
        self._result_cache = result_cache or VibeGroupingResultCache()
        self._similarity = similarity or CosineEmbeddingSimilarity()
        self._face_database = face_database or SQLiteFaceDatabase(
            app_data_path("face_embeddings.sqlite3"),
            self._similarity,
        )
        self._embedder = embedder or load_embedder(self._config)
        self._prototype_table = prototype_table
        if self._prototype_table is None:
            self._prototype_table = load_prototype_table(
                semantic_model_fingerprint=self._embedder.model_fingerprint,
                embedding_dimension=self._embedder.embedding_dimension,
            )
        self._extractor = VibeFeatureExtractor(
            config=self._config,
            embedder=self._embedder,
            feature_cache=self._feature_cache,
            face_database=self._face_database,
            similarity=self._similarity,
        )
        self._clusterer = VibeClusterer(self._config)

    @property
    def config(self) -> VibeGroupingConfig:
        return self._config

    @property
    def provider(self) -> str:
        return self._embedder.provider

    @property
    def model_fingerprint(self) -> str:
        return self._embedder.model_fingerprint

    @property
    def prototype_fingerprint(self) -> str | None:
        return None if self._prototype_table is None else self._prototype_table.fingerprint

    def scan_folder(
        self,
        folder: str | Path,
        *,
        file_extensions: tuple[str, ...] | None = None,
        orientation_filter: str | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> VibeGroupingResult:
        folder_path = Path(folder).expanduser().resolve()
        image_paths = find_supported_files(
            folder_path,
            self.supported_extensions,
            file_extensions,
            orientation_filter=orientation_filter,
        )
        return self.group(
            image_paths,
            folder_path=folder_path,
            progress_callback=progress_callback,
            cancellation_token=cancellation_token,
        )

    def group(
        self,
        image_paths: list[str | Path],
        *,
        folder_path: Path | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> VibeGroupingResult:
        started_at = time.perf_counter()
        paths = sorted(Path(path).resolve() for path in image_paths)
        stage_timings: dict[str, float] = {}

        if cancellation_token is not None:
            cancellation_token.raise_if_canceled()

        folder_key = folder_path or _common_parent(paths)
        cache_key = self._build_cache_key(paths)
        cached = self._result_cache.get(folder_key, cache_key)
        if cached is not None:
            cached.stage_timings = {"cache_load_seconds": round(time.perf_counter() - started_at, 4)}
            return cached

        def report(phase: str, done: int, total: int | None) -> None:
            if progress_callback is not None:
                safe_total = None if total is None else max(total, 1)
                progress_callback(phase, done, safe_total)

        report("loading_visual_features", 0, len(paths))
        extraction_started = time.perf_counter()
        features, errors, extraction_summary = self._extractor.extract(
            paths,
            progress_callback=report,
            cancellation_token=cancellation_token,
        )
        stage_timings["feature_extraction_seconds"] = round(time.perf_counter() - extraction_started, 4)
        if cancellation_token is not None:
            cancellation_token.raise_if_canceled()

        if not features:
            result = VibeGroupingResult(
                groups=[],
                ungrouped_paths=[],
                errors=errors,
                config_snapshot=self._config.to_dict(),
                model_fingerprint=self.model_fingerprint,
                provider=self.provider,
                cache_hits=extraction_summary.cache_hits,
                cache_misses=extraction_summary.cache_misses,
                stage_timings=stage_timings,
                used_fallback_embedder=self._embedder.uses_fallback,
                diagnostics={
                    "images": [],
                    "sessions": [],
                    "untimed_images": [],
                    "transitions": [],
                    "groups": [],
                    "transition_distribution": {},
                    "rejected_edges": [],
                },
            )
            self._result_cache.put(folder_key, cache_key, result)
            return result

        report("analyzing_actions_and_scenes", 0, len(features))
        prototype_started = time.perf_counter()
        features = self._attach_prototype_scores(features)
        stage_timings["prototype_scoring_seconds"] = round(time.perf_counter() - prototype_started, 4)
        report("analyzing_actions_and_scenes", len(features), len(features))
        if cancellation_token is not None:
            cancellation_token.raise_if_canceled()

        report("detecting_scene_changes", 0, len(features))
        segmentation_started = time.perf_counter()
        similarity = CombinedSimilarityComputer(self._config, prototype_table=self._prototype_table)
        segmentation = segment_by_time(features, config=self._config, similarity=similarity)
        stage_timings["temporal_segmentation_seconds"] = round(
            time.perf_counter() - segmentation_started,
            4,
        )
        report("detecting_scene_changes", len(features), len(features))
        if cancellation_token is not None:
            cancellation_token.raise_if_canceled()

        report("building_scene_groups", 0, len(features))
        clustering_started = time.perf_counter()
        clustered_groups, ungrouped = self._clusterer.cluster(
            segmentation,
            similarity=similarity,
        )
        stage_timings["clustering_seconds"] = round(time.perf_counter() - clustering_started, 4)
        report("building_scene_groups", len(features), len(features))
        if cancellation_token is not None:
            cancellation_token.raise_if_canceled()

        report("checking_group_coherence", 0, max(len(clustered_groups), 1))
        diagnostics = self._build_diagnostics(segmentation, clustered_groups, similarity=similarity)
        report("checking_group_coherence", len(clustered_groups), max(len(clustered_groups), 1))

        report("choosing_previews", 0, len(clustered_groups))
        groups_started = time.perf_counter()
        groups = [
            self._build_group(group, similarity=similarity)
            for group in clustered_groups
        ]
        stage_timings["representative_selection_seconds"] = round(
            time.perf_counter() - groups_started,
            4,
        )

        report("finalizing_results", len(groups), len(groups))
        result = VibeGroupingResult(
            groups=groups,
            ungrouped_paths=[feature.image_path for feature in ungrouped],
            errors=errors,
            config_snapshot=self._config.to_dict(),
            model_fingerprint=self.model_fingerprint,
            provider=self.provider,
                cache_hits=extraction_summary.cache_hits,
                cache_misses=extraction_summary.cache_misses,
                stage_timings={
                    **stage_timings,
                    "total_seconds": round(time.perf_counter() - started_at, 4),
                },
                used_fallback_embedder=self._embedder.uses_fallback,
                diagnostics=diagnostics,
            )
        self._result_cache.put(folder_key, cache_key, result)
        return result

    def _build_group(
        self,
        cluster: ClusteredGroup,
        *,
        similarity: CombinedSimilarityComputer,
    ) -> VibeGroup:
        features = list(cluster.members)
        representative = self._select_representative(features, similarity=similarity)
        people_ids, people_names = self._group_people(features)
        start_timestamp = min(
            (feature.capture_timestamp for feature in features if feature.capture_timestamp is not None),
            default=None,
        )
        end_timestamp = max(
            (feature.capture_timestamp for feature in features if feature.capture_timestamp is not None),
            default=None,
        )
        group_id = self._group_id(features)
        metadata = {
            "photo_count": len(features),
            "time_range_seconds": (
                None
                if start_timestamp is None or end_timestamp is None
                else float(end_timestamp - start_timestamp)
            ),
            "duplicate_subgroup_count": 0,
            "contains_untimed_images": any(feature.capture_timestamp is None for feature in features),
            "scene_cohesion": {
                "semantic": round(cluster.cohesion.semantic, 4),
                "action": round(cluster.cohesion.action, 4),
                "people": round(cluster.cohesion.people, 4),
                "layout": round(cluster.cohesion.layout, 4),
                "background": round(cluster.cohesion.background, 4),
                "lower_percentile": round(cluster.cohesion.lower_percentile, 4),
                "transition_penalty": round(cluster.cohesion.transition_penalty, 4),
                "combined": round(cluster.cohesion.combined, 4),
            },
            "top_action_scores": self._aggregate_prototype_scores(features, category="action"),
            "top_scene_scores": self._aggregate_prototype_scores(features, category="scene"),
            "top_shot_scores": self._aggregate_prototype_scores(features, category="shot"),
            "boundary_before": self._serialize_transition(cluster.boundary_before),
            "boundary_after": self._serialize_transition(cluster.boundary_after),
        }
        label = build_group_label(features, people_names, prototype_table=self._prototype_table)
        return VibeGroup(
            group_id=group_id,
            image_paths=[feature.image_path for feature in features],
            representative_path=representative.image_path,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            recognized_person_ids=people_ids,
            recognized_person_names=people_names,
            label=label,
            cohesion_score=cluster.cohesion_score,
            metadata=metadata,
            duplicate_subgroups=[],
        )

    def _select_representative(
        self,
        features: list[VibeImageFeatures],
        *,
        similarity: CombinedSimilarityComputer,
    ) -> VibeImageFeatures:
        if len(features) == 1:
            return features[0]

        matrix = np.ones((len(features), len(features)), dtype=np.float32)
        for row in range(len(features)):
            for column in range(row + 1, len(features)):
                score = similarity.pair_similarity(features[row], features[column])
                matrix[row, column] = score
                matrix[column, row] = score
        medoid_offset = int(np.argmax(matrix.mean(axis=1)))
        medoid_scores = matrix[medoid_offset]

        timed_values = [
            feature.capture_timestamp
            for feature in features
            if feature.capture_timestamp is not None
        ]
        median_timestamp = None
        if timed_values:
            ordered = sorted(timed_values)
            median_timestamp = ordered[len(ordered) // 2]

        best_key = None
        best_feature = features[0]
        for index, feature in enumerate(features):
            timestamp_distance = float("inf")
            if median_timestamp is not None and feature.capture_timestamp is not None:
                timestamp_distance = abs(feature.capture_timestamp - median_timestamp)
            key = (
                -(feature.quality_score if feature.quality_score is not None else -1.0),
                -float(medoid_scores[index]),
                -feature.face_count,
                timestamp_distance,
                feature.image_path,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_feature = feature
        return best_feature

    def _group_people(
        self,
        features: list[VibeImageFeatures],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        counts: Counter[tuple[str, str]] = Counter()
        for feature in features:
            for person_id, person_name in zip(feature.recognized_person_ids, feature.dominant_people_names):
                counts[(person_id, person_name)] += 1
        if not counts:
            return (), ()
        ordered = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][1].casefold(), item[0][0]),
        )
        ids = tuple(item[0][0] for item in ordered)
        names = tuple(item[0][1] for item in ordered)
        return ids, names

    def _group_id(self, features: list[VibeImageFeatures]) -> str:
        payload = json.dumps(
            {
                "paths": [feature.image_path for feature in sorted(features, key=lambda item: item.image_path)],
                "algorithm_version": self._config.algorithm_version,
                "model_fingerprint": self.model_fingerprint,
                "prototype_fingerprint": self.prototype_fingerprint,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _build_cache_key(self, paths: list[Path]) -> str:
        membership = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                membership.append({"path": str(path), "missing": True})
                continue
            membership.append(
                {
                    "path": str(path),
                    "file_size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        payload = json.dumps(
            {
                "membership": membership,
                "config": self._config.cache_signature(),
                "model_fingerprint": self.model_fingerprint,
                "prototype_fingerprint": self.prototype_fingerprint,
                "provider": self.provider,
                "face_database_signature": self._face_database.cache_signature() if self._config.include_people else None,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _attach_prototype_scores(
        self,
        features: list[VibeImageFeatures],
    ) -> list[VibeImageFeatures]:
        if self._prototype_table is None or not features:
            return features
        embedding_matrix = np.stack([feature.semantic_embedding for feature in features], axis=0)
        action_scores, scene_scores, shot_scores = self._prototype_table.score_embeddings(embedding_matrix)
        return [
            replace(
                feature,
                action_scores=action_scores[index],
                scene_scores=scene_scores[index],
                shot_type_scores=shot_scores[index],
            )
            for index, feature in enumerate(features)
        ]

    def _aggregate_prototype_scores(
        self,
        features: list[VibeImageFeatures],
        *,
        category: str,
        limit: int = 3,
    ) -> list[dict[str, object]]:
        if self._prototype_table is None or not features:
            return []
        attr = {
            "action": "action_scores",
            "scene": "scene_scores",
            "shot": "shot_type_scores",
        }[category]
        vectors = [
            np.asarray(getattr(feature, attr), dtype=np.float32)
            for feature in features
            if getattr(feature, attr) is not None
        ]
        if not vectors:
            return []
        mean_vector = np.mean(np.stack(vectors, axis=0), axis=0)
        return self._prototype_table.describe_scores(mean_vector, category=category, limit=limit)

    def _build_diagnostics(
        self,
        segmentation,
        clustered_groups: list[ClusteredGroup],
        *,
        similarity: CombinedSimilarityComputer,
    ) -> dict[str, object]:
        memberships: dict[str, dict[str, object]] = {}
        for group_index, group in enumerate(clustered_groups, start=1):
            label = build_group_label(list(group.members), (), prototype_table=self._prototype_table)
            group_id = self._group_id(list(group.members))
            for member in group.members:
                memberships[member.image_path] = {
                    "group_id": group_id,
                    "group_index": group_index,
                    "group_label": label,
                }

        sessions_payload: list[dict[str, object]] = []
        images_payload: list[dict[str, object]] = []
        for session_index, session in enumerate(segmentation.sessions, start=1):
            sessions_payload.append(
                self._serialize_session(
                    session,
                    session_index=session_index,
                    memberships=memberships,
                )
            )
            segment_index_by_member: dict[int, int] = {}
            for segment_index, segment in enumerate(session.scene_segments, start=1):
                for member_index in segment.member_indices:
                    segment_index_by_member[member_index] = segment_index
            for timeline_index, feature in enumerate(session.ordered_features, start=1):
                images_payload.append(
                    self._serialize_feature(
                        feature,
                        similarity=similarity,
                        memberships=memberships.get(feature.image_path),
                        session_index=session_index,
                        timeline_index=timeline_index,
                        scene_segment_index=segment_index_by_member.get(timeline_index - 1),
                        is_untimed=False,
                    )
                )

        untimed_payload: list[dict[str, object]] = []
        for timeline_index, feature in enumerate(segmentation.untimed, start=1):
            serialized = self._serialize_feature(
                feature,
                similarity=similarity,
                memberships=memberships.get(feature.image_path),
                session_index=None,
                timeline_index=timeline_index,
                scene_segment_index=None,
                is_untimed=True,
            )
            images_payload.append(serialized)
            untimed_payload.append(serialized)

        by_path = {item["image_path"]: item for item in images_payload}
        groups_payload: list[dict[str, object]] = []
        for group_index, group in enumerate(clustered_groups, start=1):
            label = build_group_label(list(group.members), (), prototype_table=self._prototype_table)
            group_id = self._group_id(list(group.members))
            members = [member.image_path for member in group.members]
            timed_rows = [by_path[path] for path in members if path in by_path and not by_path[path]["is_untimed"]]
            timeline_indices = [int(row["timeline_index"]) for row in timed_rows if row.get("timeline_index") is not None]
            timeline_start = None if not timeline_indices else min(timeline_indices)
            timeline_end = None if not timeline_indices else max(timeline_indices)
            timeline_span = None if timeline_start is None or timeline_end is None else (timeline_end - timeline_start + 1)
            internal_missing_count = None if timeline_span is None else (timeline_span - len(timeline_indices))
            duration_seconds = None
            timestamps = [
                float(member.capture_timestamp)
                for member in group.members
                if member.capture_timestamp is not None
            ]
            if timestamps:
                duration_seconds = max(timestamps) - min(timestamps)
            action_profiles = [similarity.action_profile(member) for member in group.members]
            action_margin_mean = 0.0 if not action_profiles else sum(item.margin for item in action_profiles) / len(action_profiles)
            action_confidence_mean = 0.0 if not action_profiles else sum(1.0 if item.confident else 0.0 for item in action_profiles) / len(action_profiles)
            groups_payload.append(
                {
                    "group_id": group_id,
                    "group_index": group_index,
                    "label": label,
                    "medoid_path": group.medoid_path,
                    "members": members,
                    "member_count": len(members),
                    "start_timestamp": group.members[0].capture_timestamp if group.members else None,
                    "end_timestamp": group.members[-1].capture_timestamp if group.members else None,
                    "timeline_start": timeline_start,
                    "timeline_end": timeline_end,
                    "timeline_span": timeline_span,
                    "internal_missing_count": internal_missing_count,
                    "contiguity_score": round(group.cohesion.timeline_contiguity_score, 4),
                    "duration_seconds": None if duration_seconds is None else round(duration_seconds, 4),
                    "internal_soft_boundaries": group.cohesion.internal_soft_boundary_count,
                    "internal_hard_boundaries": group.cohesion.internal_hard_boundary_count,
                    "action_confidence_mean": round(action_confidence_mean, 4),
                    "action_margin_mean": round(action_margin_mean, 4),
                    "semantic_cohesion": round(group.cohesion.semantic, 4),
                    "action_cohesion": round(group.cohesion.action, 4),
                    "people_cohesion": round(group.cohesion.people, 4),
                    "layout_cohesion": round(group.cohesion.layout, 4),
                    "subject_scene_cohesion": round(group.cohesion.subject_scene, 4),
                    "background_cohesion": round(group.cohesion.background, 4),
                    "composition_consistency": round(group.cohesion.composition, 4),
                    "lower_percentile_cohesion": round(group.cohesion.lower_percentile, 4),
                    "transition_penalty": round(group.cohesion.transition_penalty, 4),
                    "scene_duration_score": round(group.cohesion.scene_duration_score, 4),
                    "action_confidence_score": round(group.cohesion.action_confidence_score, 4),
                    "combined_cohesion": round(group.cohesion.combined, 4),
                    "top_action_scores": self._aggregate_prototype_scores(list(group.members), category="action"),
                    "top_scene_scores": self._aggregate_prototype_scores(list(group.members), category="scene"),
                    "top_shot_scores": self._aggregate_prototype_scores(list(group.members), category="shot"),
                    "boundary_before": self._serialize_transition(group.boundary_before),
                    "boundary_after": self._serialize_transition(group.boundary_after),
                }
            )

        transitions_payload = [
            self._serialize_transition(transition)
            for session in segmentation.sessions
            for transition in session.transitions
        ]
        return {
            "images": images_payload,
            "sessions": sessions_payload,
            "untimed_images": untimed_payload,
            "transitions": transitions_payload,
            "transition_distribution": self._summarize_transition_distribution(transitions_payload),
            "groups": groups_payload,
            "rejected_edges": list(self._clusterer.last_diagnostics.get("rejected_edges", [])),
        }

    @staticmethod
    def _serialize_transition(transition) -> dict[str, object] | None:
        if transition is None:
            return None
        return {
            "left_image": transition.left_image,
            "right_image": transition.right_image,
            "semantic_change": round(transition.semantic_change, 4),
            "action_change": round(transition.action_change, 4),
            "people_change": round(transition.people_change, 4),
            "layout_change": round(transition.layout_change, 4),
            "subject_scene_change": round(getattr(transition, "subject_scene_change", 0.0), 4),
            "background_change": round(transition.background_change, 4),
            "composition_change": round(transition.composition_change, 4),
            "temporal_gap": round(transition.temporal_gap, 4),
            "combined_transition": round(transition.combined_transition, 4),
            "boundary_reliability": getattr(transition, "boundary_reliability", "none"),
            "hard_boundary": transition.is_hard_boundary,
            "soft_boundary": transition.is_soft_boundary,
            "accepted_hard_boundary": getattr(transition, "accepted_hard_boundary", transition.is_hard_boundary),
            "accepted_soft_boundary": getattr(transition, "accepted_soft_boundary", transition.is_soft_boundary),
            "continuity_override_applied": bool(getattr(transition, "continuity_override_applied", False)),
            "continuity_score": round(float(getattr(transition, "continuity_score", 0.0)), 4),
            "visual_support_score": round(float(getattr(transition, "visual_support_score", 0.0)), 4),
            "continuity_reasons": list(getattr(transition, "continuity_reasons", ()) or ()),
            "participant_conflict_strength": getattr(transition, "participant_conflict_strength", "none"),
            "shot_conflict_confident": bool(getattr(transition, "shot_conflict_confident", False)),
            "left_shot_margin": round(float(getattr(transition, "left_shot_margin", 0.0)), 4),
            "right_shot_margin": round(float(getattr(transition, "right_shot_margin", 0.0)), 4),
            "left_shot_confidence": getattr(transition, "left_shot_confidence", "uncertain"),
            "right_shot_confidence": getattr(transition, "right_shot_confidence", "uncertain"),
            "recovery_crossable": bool(getattr(transition, "recovery_crossable", False)),
            "hard_reasons": list(getattr(transition, "hard_reasons", ()) or ()),
            "soft_reasons": list(getattr(transition, "soft_reasons", ()) or ()),
        }

    def _serialize_session(
        self,
        session,
        *,
        session_index: int,
        memberships: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        scene_segments: list[dict[str, object]] = []
        for segment_index, segment in enumerate(session.scene_segments, start=1):
            members = [session.ordered_features[item].image_path for item in segment.member_indices]
            matching_groups = [
                memberships[path]
                for path in members
                if path in memberships
            ]
            scene_segments.append(
                {
                    "segment_index": segment_index,
                    "members": members,
                    "member_count": len(members),
                    "start_timestamp": None
                    if not segment.member_indices
                    else session.ordered_features[segment.member_indices[0]].capture_timestamp,
                    "end_timestamp": None
                    if not segment.member_indices
                    else session.ordered_features[segment.member_indices[-1]].capture_timestamp,
                    "group_ids": sorted(
                        {str(group["group_id"]) for group in matching_groups},
                    ),
                    "group_labels": sorted(
                        {str(group["group_label"]) for group in matching_groups},
                    ),
                    "boundary_before": self._serialize_transition(segment.boundary_before),
                    "boundary_after": self._serialize_transition(segment.boundary_after),
                }
            )

        return {
            "session_index": session_index,
            "image_count": len(session.ordered_features),
            "segment_count": len(session.scene_segments),
            "start_timestamp": session.ordered_features[0].capture_timestamp if session.ordered_features else None,
            "end_timestamp": session.ordered_features[-1].capture_timestamp if session.ordered_features else None,
            "image_paths": [feature.image_path for feature in session.ordered_features],
            "transition_count": len(session.transitions),
            "hard_boundary_count": sum(1 for item in session.transitions if item.is_hard_boundary),
            "soft_boundary_count": sum(1 for item in session.transitions if item.is_soft_boundary),
            "accepted_hard_boundary_count": sum(
                1 for item in session.transitions if getattr(item, "accepted_hard_boundary", item.is_hard_boundary)
            ),
            "accepted_soft_boundary_count": sum(
                1 for item in session.transitions if getattr(item, "accepted_soft_boundary", item.is_soft_boundary)
            ),
            "recovery_events": list(getattr(session, "recovery_events", ()) or ()),
            "tiny_segment_merge_events": list(getattr(session, "tiny_segment_merge_events", ()) or ()),
            "scene_segments": scene_segments,
        }

    def _serialize_feature(
        self,
        feature: VibeImageFeatures,
        *,
        similarity: CombinedSimilarityComputer,
        memberships: dict[str, object] | None,
        session_index: int | None,
        timeline_index: int,
        scene_segment_index: int | None,
        is_untimed: bool,
    ) -> dict[str, object]:
        action_profile = similarity.action_profile(feature)
        shot_profile = similarity.shot_profile(feature)
        return {
            "image_path": feature.image_path,
            "capture_timestamp": feature.capture_timestamp,
            "timestamp_source": feature.timestamp_source,
            "session_index": session_index,
            "timeline_index": timeline_index,
            "scene_segment_index": scene_segment_index,
            "is_untimed": is_untimed,
            "group_id": None if memberships is None else memberships.get("group_id"),
            "group_index": None if memberships is None else memberships.get("group_index"),
            "group_label": None if memberships is None else memberships.get("group_label"),
            "recognized_person_ids": list(feature.recognized_person_ids),
            "recognized_person_names": list(feature.dominant_people_names),
            "participant_mode": similarity._participant_mode(feature),
            "shot_mode": similarity._shot_mode(feature),
            "shot_margin": self._round_float(shot_profile.margin),
            "shot_confidence": shot_profile.confidence.value,
            "shot_is_confident": shot_profile.confident,
            "shot_is_strongly_confident": shot_profile.strongly_confident,
            "face_count": feature.face_count,
            "face_area_ratio": self._round_float(feature.face_area_ratio),
            "width": feature.width,
            "height": feature.height,
            "file_size": feature.file_size,
            "file_mtime_ns": feature.file_mtime_ns,
            "quality_score": self._round_float(feature.quality_score),
            "brightness": self._round_float(feature.brightness),
            "color_features": self._round_vector(feature.color_features),
            "composition_features": self._round_vector(feature.composition_features),
            "face_layout": self._round_vector(feature.face_layout),
            "face_scale_summary": self._round_vector(feature.face_scale_summary),
            "has_subject_scene_embedding": feature.subject_scene_embedding is not None,
            "has_background_embedding": feature.background_embedding is not None,
            "top_action_key": action_profile.top_key,
            "top_action_family": action_profile.top_family,
            "action_margin": self._round_float(action_profile.margin),
            "action_is_confident": action_profile.confident,
            "action_is_strongly_confident": action_profile.strongly_confident,
            "top_action_scores": self._describe_scores(feature.action_scores, category="action"),
            "top_scene_scores": self._describe_scores(feature.scene_scores, category="scene"),
            "top_shot_scores": self._describe_scores(feature.shot_type_scores, category="shot"),
            "metadata": self._serialize_metadata(feature.metadata),
        }

    def _describe_scores(
        self,
        scores: np.ndarray | None,
        *,
        category: str,
        limit: int = 3,
    ) -> list[dict[str, object]]:
        if self._prototype_table is None:
            return []
        return self._prototype_table.describe_scores(scores, category=category, limit=limit)

    @staticmethod
    def _round_float(value: float | None, *, digits: int = 4) -> float | None:
        if value is None:
            return None
        return round(float(value), digits)

    @staticmethod
    def _round_vector(
        vector: np.ndarray | None,
        *,
        digits: int = 4,
    ) -> list[float] | None:
        if vector is None:
            return None
        return [round(float(item), digits) for item in np.asarray(vector, dtype=np.float32).tolist()]

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, object]:
        serialized: dict[str, object] = {}
        for key, value in sorted(metadata.items()):
            if value is None or isinstance(value, (bool, int, str)):
                serialized[str(key)] = value
            elif isinstance(value, float):
                serialized[str(key)] = round(value, 4)
            elif isinstance(value, (list, tuple)):
                items: list[object] = []
                supported = True
                for item in value:
                    if item is None or isinstance(item, (bool, int, str)):
                        items.append(item)
                    elif isinstance(item, float):
                        items.append(round(item, 4))
                    else:
                        supported = False
                        break
                if supported:
                    serialized[str(key)] = items
        return serialized

    @staticmethod
    def _summarize_transition_distribution(
        transitions: list[dict[str, object] | None],
    ) -> dict[str, object]:
        values = [
            float(item["combined_transition"])
            for item in transitions
            if item is not None and item.get("combined_transition") is not None
        ]
        if not values:
            return {
                "minimum": 0.0,
                "median": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "maximum": 0.0,
                "soft_boundary_count": 0,
                "hard_boundary_count": 0,
                "raw_soft_boundary_count": 0,
                "raw_hard_boundary_count": 0,
            }
        ordered = sorted(values)
        def percentile(fraction: float) -> float:
            index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
            return ordered[index]
        return {
            "minimum": round(ordered[0], 4),
            "median": round(float(np.median(np.asarray(ordered, dtype=np.float32))), 4),
            "p75": round(percentile(0.75), 4),
            "p90": round(percentile(0.90), 4),
            "p95": round(percentile(0.95), 4),
            "maximum": round(ordered[-1], 4),
            "soft_boundary_count": sum(1 for item in transitions if item and item.get("accepted_soft_boundary")),
            "hard_boundary_count": sum(1 for item in transitions if item and item.get("accepted_hard_boundary")),
            "raw_soft_boundary_count": sum(1 for item in transitions if item and item.get("soft_boundary")),
            "raw_hard_boundary_count": sum(1 for item in transitions if item and item.get("hard_boundary")),
        }


def _common_parent(paths: list[Path]) -> Path:
    if not paths:
        return Path(".").resolve()
    current = paths[0].parent
    for path in paths[1:]:
        while current not in path.parents and current != path.parent:
            if current.parent == current:
                return current
            current = current.parent
    return current
