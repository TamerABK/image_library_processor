from __future__ import annotations

from pathlib import Path

from grouping.models import VibeGroup, VibeGroupingResult

def group_vibes_with_fallback(
    services,
    image_paths: list[str],
    *,
    progress_callback: callable | None = None,
    cancellation_token=None,
) -> tuple[VibeGroupingResult, bool, str | None]:
    try:
        return (
            services.group_vibes(
                image_paths,
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
            ),
            False,
            None,
        )
    except Exception as exc:
        fallback_result = _simple_vibe_fallback(image_paths)
        return fallback_result, True, str(exc)


def _simple_vibe_fallback(image_paths: list[str]) -> VibeGroupingResult:
    if not image_paths:
        return VibeGroupingResult(
            groups=[],
            ungrouped_paths=[],
            errors=[],
            config_snapshot={},
            model_fingerprint="automatic_fallback_v1",
            provider="CPUExecutionProvider",
            used_fallback_embedder=True,
            diagnostics={"fallback": "simple_chronological"},
        )
    ordered_paths = sorted(Path(path).resolve() for path in image_paths)
    groups = [
        VibeGroup(
            group_id=f"fallback-{index + 1}",
            image_paths=[str(path)],
            representative_path=str(path),
            start_timestamp=None,
            end_timestamp=None,
            recognized_person_ids=(),
            recognized_person_names=(),
            label=None,
            cohesion_score=1.0,
            metadata={"fallback": True},
        )
        for index, path in enumerate(ordered_paths)
    ]
    return VibeGroupingResult(
        groups=groups,
        ungrouped_paths=[],
        errors=[],
        config_snapshot={"fallback": "simple_chronological"},
        model_fingerprint="automatic_fallback_v1",
        provider="CPUExecutionProvider",
        used_fallback_embedder=True,
        diagnostics={"fallback": "simple_chronological"},
    )
