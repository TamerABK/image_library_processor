from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: F401  # preload runtime dependencies

from grouping.models import VibeGroupingResult
from grouping.vibe import VibeGroupingPreset, VibeGroupingProcessor, preset_config
from grouping.vibe.similarity import CombinedSimilarityComputer
from grouping.vibe.temporal_segments import segment_by_time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scene-oriented vibe grouping on a photo folder.",
    )
    parser.add_argument("--input", required=True, help="Folder containing photos.")
    parser.add_argument(
        "--preset",
        default="balanced_scene",
        choices=["session", "balanced_scene", "tight_scene"],
        help="Scene-grouping preset.",
    )
    parser.add_argument("--output", required=True, help="Write a JSON report to this path.")
    parser.add_argument(
        "--ground-truth",
        help="Optional JSON mapping of image path to scene id for evaluation.",
    )
    parser.add_argument(
        "--orientation",
        choices=["landscape", "portrait"],
        help="Optional orientation filter.",
    )
    parser.add_argument("--no-people", action="store_true", help="Disable known-people overlap.")
    parser.add_argument("--no-color", action="store_true", help="Disable color similarity.")
    parser.add_argument("--no-composition", action="store_true", help="Disable composition similarity.")
    return parser.parse_args()


def main_cli() -> None:
    args = _parse_args()
    preset = VibeGroupingPreset(args.preset)
    config = preset_config(
        preset,
        include_people=not args.no_people,
        include_color=not args.no_color,
        include_composition=not args.no_composition,
    )
    processor = VibeGroupingProcessor(config)
    folder = Path(args.input).resolve()
    result = processor.scan_folder(folder, orientation_filter=args.orientation)
    report = _build_report(folder, processor, result)

    if args.ground_truth:
        ground_truth = _load_ground_truth(Path(args.ground_truth), base_folder=folder)
        report["evaluation"] = _evaluate_against_ground_truth(folder, processor, result, ground_truth)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _build_report(
    folder: Path,
    processor: VibeGroupingProcessor,
    result: VibeGroupingResult,
) -> dict[str, object]:
    group_sizes = [len(group.image_paths) for group in result.groups]
    cohesion_scores = [group.cohesion_score for group in result.groups]
    return {
        "input": str(folder),
        "provider": result.provider,
        "used_fallback_embedder": result.used_fallback_embedder,
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "group_count": len(result.groups),
        "group_sizes": group_sizes,
        "average_group_size": 0.0 if not group_sizes else sum(group_sizes) / len(group_sizes),
        "ungrouped_count": len(result.ungrouped_paths),
        "ungrouped_rate": 0.0
        if not result.groups and not result.ungrouped_paths
        else len(result.ungrouped_paths) / max(sum(group_sizes) + len(result.ungrouped_paths), 1),
        "cohesion_distribution": {
            "min": 0.0 if not cohesion_scores else min(cohesion_scores),
            "median": 0.0 if not cohesion_scores else sorted(cohesion_scores)[len(cohesion_scores) // 2],
            "max": 0.0 if not cohesion_scores else max(cohesion_scores),
            "values": cohesion_scores,
        },
        "stage_timings": result.stage_timings,
        "model_fingerprint": result.model_fingerprint,
        "prototype_fingerprint": processor.prototype_fingerprint,
        "config_snapshot": result.config_snapshot,
        "error_count": len(result.errors),
        "diagnostics": result.diagnostics,
    }


def _load_ground_truth(path: Path, *, base_folder: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    resolved: dict[str, str] = {}
    for raw_path, scene_id in payload.items():
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (base_folder / candidate).resolve()
        resolved[str(candidate)] = str(scene_id)
    return resolved


def _evaluate_against_ground_truth(
    folder: Path,
    processor: VibeGroupingProcessor,
    result: VibeGroupingResult,
    ground_truth: dict[str, str],
) -> dict[str, object]:
    predicted_groups = [tuple(group.image_paths) for group in result.groups]
    predicted_pairs = _pair_set(predicted_groups)
    truth_groups: dict[str, list[str]] = defaultdict(list)
    for image_path, scene_id in ground_truth.items():
        truth_groups[scene_id].append(image_path)
    truth_pairs = _pair_set(truth_groups.values())

    pair_metrics = _precision_recall_f1(predicted_pairs, truth_pairs)

    ordered_features = processor._attach_prototype_scores(  # type: ignore[attr-defined]
        processor._extractor.extract(sorted(Path(path) for path in ground_truth.keys()))[0]  # type: ignore[attr-defined]
    )
    similarity = CombinedSimilarityComputer(processor.config, prototype_table=processor._prototype_table)  # type: ignore[attr-defined]
    segmentation = segment_by_time(ordered_features, config=processor.config, similarity=similarity)
    ordered_timed_paths = [
        feature.image_path
        for session in segmentation.sessions
        for feature in session.ordered_features
    ]
    predicted_boundaries = {
        (item["left_image"], item["right_image"])
        for item in result.diagnostics.get("transitions", [])
        if item.get("hard_boundary") or item.get("soft_boundary")
    }
    truth_boundaries = _ground_truth_boundaries(ordered_timed_paths, ground_truth)
    boundary_metrics = _precision_recall_f1(predicted_boundaries, truth_boundaries)

    over_merges = _count_over_merges(predicted_groups, ground_truth)
    over_splits = _count_over_splits(predicted_groups, truth_groups)
    return {
        "pairwise": pair_metrics,
        "boundaries": boundary_metrics,
        "over_merges": over_merges,
        "over_splits": over_splits,
        "average_group_size": 0.0
        if not result.groups
        else sum(len(group.image_paths) for group in result.groups) / len(result.groups),
        "ungrouped_rate": len(result.ungrouped_paths) / max(len(ground_truth), 1),
        "cohesion_distribution": {
            "values": [group.cohesion_score for group in result.groups],
        },
    }


def _pair_set(groups) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for group in groups:
        ordered = sorted(str(path) for path in group)
        for left, right in combinations(ordered, 2):
            pairs.add((left, right))
    return pairs


def _precision_recall_f1(
    predicted: set[tuple[str, str]],
    truth: set[tuple[str, str]],
) -> dict[str, float]:
    true_positive = len(predicted & truth)
    precision = true_positive / max(len(predicted), 1)
    recall = true_positive / max(len(truth), 1)
    if precision + recall <= 1e-12:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _ground_truth_boundaries(
    ordered_paths: list[str],
    ground_truth: dict[str, str],
) -> set[tuple[str, str]]:
    boundaries: set[tuple[str, str]] = set()
    for left, right in zip(ordered_paths, ordered_paths[1:]):
        if ground_truth.get(left) != ground_truth.get(right):
            boundaries.add((left, right))
    return boundaries


def _count_over_merges(
    predicted_groups: list[tuple[str, ...]],
    ground_truth: dict[str, str],
) -> int:
    count = 0
    for group in predicted_groups:
        scene_ids = {
            ground_truth[path]
            for path in group
            if path in ground_truth
        }
        if len(scene_ids) > 1:
            count += 1
    return count


def _count_over_splits(
    predicted_groups: list[tuple[str, ...]],
    truth_groups: dict[str, list[str]],
) -> int:
    membership: dict[str, int] = {}
    for index, group in enumerate(predicted_groups):
        for path in group:
            membership[path] = index
    over_split_count = 0
    for scene_id, paths in truth_groups.items():
        predicted_ids = {membership[path] for path in paths if path in membership}
        if len(predicted_ids) > 1:
            over_split_count += 1
    return over_split_count


if __name__ == "__main__":
    main_cli()
