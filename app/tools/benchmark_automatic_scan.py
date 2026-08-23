from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import tempfile
import time
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the automatic scan pipeline with an isolated cache namespace.",
    )
    parser.add_argument("--input", required=True, help="Folder containing photos to scan.")
    parser.add_argument(
        "--provider",
        default="automatic",
        choices=["automatic", "cpu", "cuda"],
        help="Execution provider to request for ONNX Runtime.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the provider-aware default batch size for embedding and vibe inference.",
    )
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="Clear the isolated cache directory before each run.",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Reuse the isolated cache directory between runs.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional JSON file for the full benchmark report.",
    )
    return parser.parse_args()


def _provider_enum(provider_name: str):
    from automatic_scan.config import ExecutionProvider

    if provider_name == "cpu":
        return ExecutionProvider.CPU
    if provider_name == "cuda":
        return ExecutionProvider.GPU
    return ExecutionProvider.AUTOMATIC


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.name == "posix":
        return round(value / 1024.0, 3)
    return round(value / (1024.0 * 1024.0), 3)


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_config(args: argparse.Namespace):
    from automatic_scan.config import AutomaticScanConfig, PerformanceConfig

    return AutomaticScanConfig(
        performance=PerformanceConfig(
            execution_provider=_provider_enum(args.provider),
            batch_size=args.batch_size,
        )
    )


def _summarize_run(payload: dict[str, Any]) -> dict[str, Any]:
    face_embedder = payload.get("models", {}).get("face_embedder", {})
    vibe_model = payload.get("models", {}).get("vibe_model", {})
    return {
        "total_seconds": payload.get("total_seconds", 0.0),
        "startup_seconds": payload.get("startup_seconds", 0.0),
        "scan_seconds": payload.get("scan_seconds", 0.0),
        "images_per_second": payload.get("images_per_second", 0.0),
        "cache_mode": payload.get("cache_mode", "unknown"),
        "stage_stats": payload.get("stage_stats", {}),
        "step_timings": payload.get("step_timings", {}),
        "image_io": payload.get("image_io", {}),
        "database": payload.get("database", {}),
        "models": payload.get("models", {}),
        "face_embedding_face_count": face_embedder.get("items", 0),
        "face_embedding_call_count": face_embedder.get("calls", 0),
        "face_embedding_average_batch_size": face_embedder.get("average_batch_size", 0.0),
        "face_embedding_max_batch_size": face_embedder.get("max_batch_size", 0),
        "vibe_call_count": vibe_model.get("calls", 0),
        "vibe_average_batch_size": vibe_model.get("average_batch_size", 0.0),
        "vibe_max_batch_size": vibe_model.get("max_batch_size", 0),
    }


def main() -> None:
    args = _parse_args()
    if args.cold_cache and args.warm_cache:
        raise SystemExit("Choose at most one of --cold-cache or --warm-cache.")

    cache_root = Path(tempfile.mkdtemp(prefix="automatic-scan-benchmark-")).resolve()
    os.environ["IMAGE_DEDUPLICATOR_APP_DATA_ROOT"] = str(cache_root)

    from automatic_scan.orchestrator import AutomaticScanOrchestrator

    config = _build_config(args)
    input_path = Path(args.input).expanduser().resolve()
    runs: list[dict[str, Any]] = []
    use_cold_cache = args.cold_cache or not args.warm_cache

    for run_index in range(1, max(args.runs, 1) + 1):
        if use_cold_cache:
            _clear_directory(cache_root)

        orchestrator = AutomaticScanOrchestrator()
        started = time.perf_counter()
        result = orchestrator.scan_folder(input_path, config)
        wall_seconds = time.perf_counter() - started
        payload = result.to_diagnostics_payload()
        payload["benchmark_wall_seconds"] = round(wall_seconds, 6)
        payload["peak_rss_mb"] = _peak_rss_mb()
        payload["run_index"] = run_index
        runs.append(payload)

        print(
            json.dumps(
                {
                    "run_index": run_index,
                    "wall_seconds": payload["benchmark_wall_seconds"],
                    "total_seconds": payload["total_seconds"],
                    "startup_seconds": payload["startup_seconds"],
                    "scan_seconds": payload["scan_seconds"],
                    "images_per_second": payload["images_per_second"],
                    "cache_mode": payload["cache_mode"],
                    "peak_rss_mb": payload["peak_rss_mb"],
                },
                sort_keys=True,
            )
        )

    summarized_runs = [_summarize_run(run) for run in runs]
    aggregate = {
        "input": str(input_path),
        "provider": args.provider,
        "batch_size": args.batch_size,
        "runs": len(runs),
        "isolated_cache_root": str(cache_root),
        "cache_mode": "cold" if use_cold_cache else "warm",
        "average_wall_seconds": round(
            statistics.fmean(run["benchmark_wall_seconds"] for run in runs),
            6,
        ),
        "average_total_seconds": round(
            statistics.fmean(run["total_seconds"] for run in runs),
            6,
        ),
        "average_images_per_second": round(
            statistics.fmean(run["images_per_second"] for run in runs),
            6,
        ),
        "peak_rss_mb": max(run["peak_rss_mb"] for run in runs),
        "runs_detail": summarized_runs,
        "full_runs": runs,
    }

    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(aggregate, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
