from __future__ import annotations

import argparse
import json
from pathlib import Path

from grouping.vibe import VibeGroupingPreset, VibeGroupingProcessor, preset_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the vibe grouping pipeline on a folder of photos.",
    )
    parser.add_argument("--input", required=True, help="Folder containing photos.")
    parser.add_argument(
        "--preset",
        default="balanced_scene",
        choices=["session", "balanced_scene", "tight_scene"],
        help="Vibe grouping preset.",
    )
    parser.add_argument("--output", required=True, help="Write a JSON summary to this path.")
    parser.add_argument(
        "--debug-output",
        help="Optional JSON file with per-group members and metadata.",
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


def main() -> None:
    args = _parse_args()
    preset = VibeGroupingPreset(args.preset)
    config = preset_config(
        preset,
        include_people=not args.no_people,
        include_color=not args.no_color,
        include_composition=not args.no_composition,
    )
    processor = VibeGroupingProcessor(config)
    result = processor.scan_folder(
        Path(args.input),
        orientation_filter=args.orientation,
    )

    group_sizes = [len(group.image_paths) for group in result.groups]
    summary = {
        "input": str(Path(args.input).resolve()),
        "provider": result.provider,
        "used_fallback_embedder": result.used_fallback_embedder,
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "group_count": len(result.groups),
        "group_sizes": group_sizes,
        "ungrouped_count": len(result.ungrouped_paths),
        "average_cohesion": (
            0.0
            if not result.groups
            else sum(group.cohesion_score for group in result.groups) / len(result.groups)
        ),
        "stage_timings": result.stage_timings,
        "model_fingerprint": result.model_fingerprint,
        "config_snapshot": result.config_snapshot,
        "error_count": len(result.errors),
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.debug_output:
        debug_payload = {
            "groups": [
                {
                    "group_id": group.group_id,
                    "label": group.label,
                    "cohesion_score": group.cohesion_score,
                    "representative_path": group.representative_path,
                    "image_paths": group.image_paths,
                    "recognized_people": list(group.recognized_person_names),
                    "metadata": group.metadata,
                }
                for group in result.groups
            ],
            "ungrouped_paths": result.ungrouped_paths,
            "errors": [
                {
                    "path": error.path,
                    "message": error.message,
                    "fatal": error.fatal,
                }
                for error in result.errors
            ],
            "diagnostics": result.diagnostics,
        }
        Path(args.debug_output).write_text(
            json.dumps(debug_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
