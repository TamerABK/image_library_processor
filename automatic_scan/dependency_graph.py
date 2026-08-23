from __future__ import annotations

from collections import deque

from .models import AutomaticStage, PipelineImageState


class AutomaticDependencyGraph:
    def __init__(self) -> None:
        self._edges: dict[AutomaticStage, tuple[AutomaticStage, ...]] = {
            AutomaticStage.BLUR_ANALYSIS: (
                AutomaticStage.DUPLICATE_CANDIDATES,
                AutomaticStage.FACE_DETECTION_ANALYSIS,
            ),
            AutomaticStage.DUPLICATE_CANDIDATES: (
                AutomaticStage.DUPLICATE_FINALIZATION,
            ),
            AutomaticStage.FACE_DETECTION_ANALYSIS: (
                AutomaticStage.PEOPLE_IMAGE_CLASSIFICATION,
            ),
            AutomaticStage.PEOPLE_IMAGE_CLASSIFICATION: (
                AutomaticStage.FACE_QUALITY_GATE,
            ),
            AutomaticStage.FACE_QUALITY_GATE: (
                AutomaticStage.DUPLICATE_FINALIZATION,
            ),
            AutomaticStage.DUPLICATE_FINALIZATION: (
                AutomaticStage.FACE_EMBEDDING,
                AutomaticStage.VIBE_GROUPING,
            ),
            AutomaticStage.FACE_EMBEDDING: (
                AutomaticStage.FACE_RECOGNITION,
            ),
            AutomaticStage.FACE_RECOGNITION: (
                AutomaticStage.UNKNOWN_FACE_CLUSTERING,
                AutomaticStage.VIBE_GROUPING,
            ),
            AutomaticStage.UNKNOWN_FACE_CLUSTERING: (),
            AutomaticStage.VIBE_GROUPING: (
                AutomaticStage.RESULT_FINALIZATION,
            ),
            AutomaticStage.RESULT_FINALIZATION: (),
        }

    def required_downstream_stages(self, stage: AutomaticStage) -> tuple[AutomaticStage, ...]:
        ordered: list[AutomaticStage] = []
        queue: deque[AutomaticStage] = deque(self._edges.get(stage, ()))
        seen: set[AutomaticStage] = set()
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            ordered.append(current)
            queue.extend(self._edges.get(current, ()))
        return tuple(ordered)

    def invalidate_from(
        self,
        stage: AutomaticStage,
        image_states: dict[str, PipelineImageState],
        image_paths: list[str],
    ) -> None:
        affected_stages = {stage, *self.required_downstream_stages(stage)}
        for image_path in image_paths:
            image = image_states.get(image_path)
            if image is None:
                continue
            for stage_name in affected_stages:
                image.completed_stages.discard(stage_name.value)
                image.skipped_stages.discard(stage_name.value)

    def can_reuse_cache(self, stage: AutomaticStage, image: PipelineImageState) -> bool:
        return stage.value in image.cache_hits and stage.value not in image.cache_misses
