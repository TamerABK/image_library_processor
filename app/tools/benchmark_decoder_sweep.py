"""Isolated decoder sweep. Run --help; --plan-only never processes photographs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from app_paths import model_path

from image_file_utils import find_supported_files
from image_loader import default_image_loader
from automatic_scan.benchmark import _hardware, _run_pipeline
from automatic_scan.decode_pool import max_manual_decode_workers

BROAD_WORKERS = (2, 4, 6, 8, 10, 12)
BROAD_MEMORY = (256, 512, 768, 1024, 1536, 2048, 3072, 4096)
FINE_WORKERS = tuple(range(3, 13))
FINE_MEMORY = (384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096)
MIB = 1024 * 1024
RUNS = ("run_id timestamp git_commit dataset_id dataset_image_count detected_face_count benchmark_phase repetition run_order "
        "requested_workers effective_workers requested_memory_mib effective_memory_mib wall_seconds images_per_second "
        "faces_per_second avg_cpu_percent peak_cpu_percent avg_cpu_core_equivalents peak_rss_mib peak_working_set_mib "
        "avg_available_ram_mib peak_reserved_decoder_mib max_pending_decodes max_active_decodes total_decoder_tasks "
        "decoder_wait_seconds decoder_queue_wait_seconds avg_disk_read_mbps peak_disk_read_mbps gpu_provider cache_mode "
        "status error_message").split()
STAGES = ("run_id requested_workers effective_workers effective_memory_mib stage wall_seconds calls cache_hits computed_items "
          "file_errors decoder_tasks decoder_wait_seconds decoder_queue_wait_seconds max_pending_decodes peak_reserved_decoder_mib").split()
SUBSTAGES = "run_id operation stage calls inclusive_seconds exclusive_seconds average_seconds median_seconds p95_seconds p99_seconds sample_count".split()
RESOURCES = ("run_id elapsed_seconds active_stage process_cpu_percent system_cpu_percent process_rss_mib process_working_set_mib "
             "available_system_memory_mib disk_read_mbps disk_queue_depth gpu_utilization_percent gpu_memory_mib "
             "active_decoder_tasks pending_decoder_tasks reserved_decoder_mib").split()
SUMMARY = ("worker_count memory_budget_mib dataset_id repetitions mean_wall_seconds median_wall_seconds stddev_wall_seconds "
           "min_wall_seconds max_wall_seconds mean_images_per_second mean_peak_rss_mib mean_cpu_core_equivalents "
           "mean_decoder_wait_seconds mean_face_stage_seconds mean_embedding_stage_seconds speedup_vs_baseline "
           "percentage_improvement_vs_baseline efficiency_score confidence_interval_95_low confidence_interval_95_high").split()
HEADERS = {"runs.csv": RUNS, "stages.csv": STAGES, "substage_timings.csv": SUBSTAGES,
           "resources.csv": RESOURCES, "summary.csv": SUMMARY}


def _csv(path, header, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _model_fingerprints():
    result = {}
    for name in ("scrfd_10g_bnkps.onnx", "glintr100.onnx", "sixdrepnet.onnx", "open_closed_eye.onnx"):
        path = model_path(name)
        if not path.exists():
            result[name] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def _dataset(input_dir):
    paths = sorted((Path(p).resolve() for p in find_supported_files(input_dir,
        default_image_loader.supported_extensions())), key=lambda p: str(p).casefold())
    records = [(str(p.relative_to(input_dir)).replace("\\", "/"), p.stat().st_size, p.stat().st_mtime_ns)
               for p in paths]
    digest = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
    return paths, records, digest


def _subset(paths, root, size, seed):
    """Deterministic first-folder and size quartile stratification; face counts unavailable pre-scan."""
    groups = {}
    sizes = sorted(p.stat().st_size for p in paths)
    boundaries = [sizes[min(len(sizes)-1, len(sizes)*q//4)] for q in (1, 2, 3)]
    for path in paths:
        relative = path.relative_to(root)
        folder = relative.parts[0] if len(relative.parts) > 1 else "."
        size_band = sum(path.stat().st_size > boundary for boundary in boundaries)
        groups.setdefault((folder, size_band), []).append(path)
    rng = random.Random(seed)
    for group in groups.values(): rng.shuffle(group)
    selected = []
    while len(selected) < min(size, len(paths)):
        progressed = False
        for key in sorted(groups):
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) >= size: break
        if not progressed: break
    return sorted(selected, key=str)


def _plan(args, subset_id, full_id):
    if args.mode == "smoke":
        return [("smoke", 1, 4, 512, subset_id)]
    workers = tuple(args.workers or BROAD_WORKERS)
    memory = tuple(args.memory or BROAD_MEMORY)
    broad = [(w, m) for w in workers for m in memory]
    plan = []
    rng = random.Random(args.seed)
    for repetition in range(1, args.screen_repetitions + 1):
        order = broad.copy(); rng.shuffle(order)
        plan.extend(("A", repetition, w, m, subset_id) for w, m in order)
    # Fine selection happens after Phase A; a placeholder is recorded in manifest.
    return plan


def _fine_plan(args, summaries, full_id):
    if args.exhaustive:
        pairs = {(w, m) for w in FINE_WORKERS for m in FINE_MEMORY}
    else:
        ranked = sorted((r for r in summaries if r["dataset_id"] != full_id and r.get("median_wall_seconds")),
                        key=lambda r: float(r["median_wall_seconds"]))
        winners = [(int(r["worker_count"]), int(r["memory_budget_mib"])) for r in ranked[:3]]
        if not winners: return []
        pairs = set(winners)
        for w, m in winners[:2]:
            near_w = sorted(FINE_WORKERS, key=lambda x: abs(x-w))[:3]
            near_m = sorted(FINE_MEMORY, key=lambda x: abs(x-m))[:3]
            pairs.update((a, b) for a in near_w for b in near_m)
    pairs.add((4, 512))
    rng = random.Random(args.seed + 1)
    plan = []
    for repetition in range(1, args.full_repetitions + 1):
        order = sorted(pairs); rng.shuffle(order)
        plan.extend(("B", repetition, w, m, full_id) for w, m in order)
    return plan


def _mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return statistics.mean(values) if values else None


def _summary(runs, stages):
    groups = {}
    for row in runs:
        if row["status"] == "complete":
            groups.setdefault((row["dataset_id"], row["requested_workers"], row["requested_memory_mib"]), []).append(row)
    baseline = {dataset: statistics.median(float(r["wall_seconds"]) for r in rows)
                for (dataset, w, m), rows in groups.items() if (w, m) == ("4", "512")}
    result = []
    for (dataset, workers, memory), rows in sorted(groups.items()):
        walls = [float(r["wall_seconds"]) for r in rows]
        median = statistics.median(walls)
        base = baseline.get(dataset)
        related = [s for s in stages if s["run_id"] in {r["run_id"] for r in rows}]
        face = [s for s in related if s["stage"] == "faces"]
        embeddings = [s for s in related if s["stage"] == "embeddings"]
        result.append(dict(worker_count=workers, memory_budget_mib=memory, dataset_id=dataset,
            repetitions=len(rows), mean_wall_seconds=statistics.mean(walls), median_wall_seconds=median,
            stddev_wall_seconds=statistics.stdev(walls) if len(walls)>1 else None,
            min_wall_seconds=min(walls), max_wall_seconds=max(walls),
            mean_images_per_second=_mean(rows, "images_per_second"), mean_peak_rss_mib=_mean(rows, "peak_rss_mib"),
            mean_cpu_core_equivalents=_mean(rows, "avg_cpu_core_equivalents"),
            mean_decoder_wait_seconds=_mean(rows, "decoder_wait_seconds"),
            mean_face_stage_seconds=_mean(face, "wall_seconds"), mean_embedding_stage_seconds=_mean(embeddings, "wall_seconds"),
            speedup_vs_baseline=base/median if base else None,
            percentage_improvement_vs_baseline=(1-median/base)*100 if base else None,
            efficiency_score=(base/median)/int(workers) if base else None,
            confidence_interval_95_low=None, confidence_interval_95_high=None))
    return result


def _rows(run_id, phase, repetition, order, workers, memory, dataset_id, count, report, commit):
    profile, resource = report["profile"], report["resources"]
    samples = resource.get("samples", [])
    stage_data = report.get("stages", {})
    actual_workers = next((s.get("stats", {}).get("decode_effective_workers") for s in stage_data.values()
                           if s.get("stats", {}).get("decode_effective_workers")), None)
    actual_memory = next((s.get("stats", {}).get("decode_effective_memory_bytes")/MIB for s in stage_data.values()
                          if s.get("stats", {}).get("decode_effective_memory_bytes")), None)
    stage_rows = []
    for stage, info in stage_data.items():
        stats = info.get("stats", {})
        stage_rows.append(dict(run_id=run_id, requested_workers=workers, effective_workers=actual_workers,
            effective_memory_mib=actual_memory, stage=stage, wall_seconds=stats.get("seconds"),
            calls=1, cache_hits=stats.get("cache_hits"), computed_items=stats.get("computed"),
            file_errors=stats.get("file_errors"), decoder_tasks=stats.get("decode_tasks"),
            decoder_wait_seconds=stats.get("decode_wait_seconds"),
            decoder_queue_wait_seconds=stats.get("decode_queue_wait_seconds_sum"),
            max_pending_decodes=stats.get("decode_max_pending"),
            peak_reserved_decoder_mib=stats.get("decode_max_reserved_bytes", 0)/MIB))
    for operation, label in (("project.initialize_services", "project_initialization"),
                             ("project.metadata", "discovery_metadata")):
        timing = profile.get("timings", {}).get(operation)
        if timing:
            stage_rows.append(dict(run_id=run_id, requested_workers=workers,
                effective_workers=actual_workers, effective_memory_mib=actual_memory,
                stage=label, wall_seconds=timing["inclusive_seconds"], calls=timing["calls"]))
    timeline = []
    prior = None
    for sample in samples:
        elapsed = sample.get("seconds")
        project_elapsed = elapsed + (report.get("resource_start_offset_seconds") or 0) if elapsed is not None else None
        active_stage = next((item["stage"] for item in report.get("stage_intervals", [])
                             if project_elapsed is not None and item["start_seconds"] <= project_elapsed <= item["end_seconds"]), None)
        cpu_percent = None
        if prior and elapsed is not None and elapsed > prior.get("seconds", 0):
            cpu_percent = 100 * (sample.get("process_cpu_seconds", 0)-prior.get("process_cpu_seconds", 0))/(elapsed-prior["seconds"])
        timeline.append(dict(run_id=run_id, elapsed_seconds=elapsed, active_stage=active_stage,
            process_cpu_percent=cpu_percent, system_cpu_percent=sample.get("system_cpu_percent"),
            process_rss_mib=sample.get("rss_bytes", 0)/MIB if "rss_bytes" in sample else None,
            process_working_set_mib=sample.get("rss_bytes", 0)/MIB if "rss_bytes" in sample else None,
            available_system_memory_mib=sample.get("available_memory_bytes", 0)/MIB if "available_memory_bytes" in sample else None,
            disk_read_mbps=sample.get("physical_disk_read_mb_s"), disk_queue_depth=sample.get("physical_disk_queue_depth"),
            gpu_utilization_percent=sample.get("gpu_process_compute_engine_percent"),
            gpu_memory_mib=sample.get("gpu_process_dedicated_bytes", 0)/MIB if "gpu_process_dedicated_bytes" in sample else None,
            active_decoder_tasks=sample.get("active_decoder_tasks"), pending_decoder_tasks=sample.get("pending_decoder_tasks"),
            reserved_decoder_mib=sample.get("reserved_decoder_bytes", 0)/MIB if "reserved_decoder_bytes" in sample else None))
        prior = sample
    substage = [dict(run_id=run_id, operation=name, stage=name.split(".")[0], **values)
                for name, values in profile.get("timings", {}).items()]
    wall = profile["wall_seconds"]
    cpu = resource.get("process_cpu_core_equivalents")
    row = dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(), git_commit=commit,
        dataset_id=dataset_id, dataset_image_count=count, detected_face_count=profile.get("face_count"),
        benchmark_phase=phase, repetition=repetition, run_order=order, requested_workers=workers,
        effective_workers=actual_workers, requested_memory_mib=memory, effective_memory_mib=actual_memory,
        wall_seconds=wall, images_per_second=count/wall if wall else None,
        faces_per_second=profile.get("faces_per_second"), avg_cpu_percent=cpu*100 if cpu is not None else None,
        peak_cpu_percent=max((r["process_cpu_percent"] for r in timeline if r["process_cpu_percent"] is not None), default=None),
        avg_cpu_core_equivalents=cpu,
        peak_rss_mib=resource.get("peak_rss_bytes", 0)/MIB if resource.get("peak_rss_bytes") is not None else None,
        peak_working_set_mib=resource.get("peak_rss_bytes", 0)/MIB if resource.get("peak_rss_bytes") is not None else None,
        avg_available_ram_mib=_mean(timeline, "available_system_memory_mib"),
        peak_reserved_decoder_mib=max((float(s.get("peak_reserved_decoder_mib") or 0) for s in stage_rows), default=None),
        max_pending_decodes=max((int(s.get("max_pending_decodes") or 0) for s in stage_rows), default=None),
        max_active_decodes=max((s.get("stats", {}).get("decode_max_active", 0) for s in stage_data.values()), default=None),
        total_decoder_tasks=sum(int(s.get("decoder_tasks") or 0) for s in stage_rows),
        decoder_wait_seconds=sum(float(s.get("decoder_wait_seconds") or 0) for s in stage_rows),
        decoder_queue_wait_seconds=sum(float(s.get("decoder_queue_wait_seconds") or 0) for s in stage_rows),
        avg_disk_read_mbps=resource.get("physical_disk", {}).get("metrics", {}).get("read_mb_s", {}).get("mean") if resource.get("physical_disk", {}).get("metrics", {}).get("read_mb_s") else None,
        peak_disk_read_mbps=resource.get("physical_disk", {}).get("metrics", {}).get("read_mb_s", {}).get("peak") if resource.get("physical_disk", {}).get("metrics", {}).get("read_mb_s") else None,
        gpu_provider=(report.get("runtime_info") or {}).get("face_detector", {}).get("active_provider"),
        cache_mode="cold", status="complete", error_message="")
    return row, stage_rows, substage, timeline


def _child(spec):
    state = Path(spec["state"])
    paths = [Path(p) for p in spec["paths"]]
    try:
        run = _run_pipeline(paths, state, folder=Path(spec["input"]), scenario="full", detail="normal",
            batch_size=None, ort_profile=False, decoder_workers=spec["workers"],
            decoder_memory_mib=spec["memory"])
        (state / "report.json").write_text(json.dumps({k:v for k,v in run.items() if k != "trace"}, default=str), encoding="utf-8")
        (state / "trace.json").write_text(json.dumps(run["trace"]), encoding="utf-8")
        return 0
    except Exception as exc:
        (state / "error.txt").write_text(repr(exc), encoding="utf-8")
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("smoke", "comprehensive"), default="smoke")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--screen-repetitions", type=int, default=2)
    parser.add_argument("--full-repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+")
    parser.add_argument("--memory", type=int, nargs="+")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true", help="Confirm the multi-hour comprehensive matrix")
    parser.add_argument("--child-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_spec:
        return _child(json.loads(args.child_spec.read_text(encoding="utf-8")))
    if not args.input: parser.error("--input is required")
    if args.subset_size < 1 or args.screen_repetitions < 2 and args.mode == "comprehensive" or args.full_repetitions < 3 and args.mode == "comprehensive":
        parser.error("subset size and repetitions below required minimum")
    if args.workers and any(w < 1 or w > 128 for w in args.workers): parser.error("workers must be 1..128")
    if args.memory and any(m < 64 or m > 4096 for m in args.memory): parser.error("memory must be 64..4096 MiB")
    source = args.input.resolve(strict=True)
    if not source.is_dir(): parser.error("input must be a directory")
    output = (args.output or Path("benchmark_results") / ("decoder_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if source == output or source in output.parents: parser.error("output must be outside input")
    paths, records, dataset_id = _dataset(source)
    if not paths: parser.error("no supported photographs")
    subset = _subset(paths, source, args.subset_size if args.mode == "comprehensive" else min(args.subset_size, 2), args.seed)
    subset_id = hashlib.sha256("\n".join(str(p) for p in subset).encode()).hexdigest()
    phase_a = _plan(args, subset_id, dataset_id)
    print(f"Dataset: {len(paths)} images; screening subset: {len(subset)}. Phase A: {len(phase_a)} executions.", flush=True)
    for phase, repetition, workers, memory, _ in phase_a:
        print(f"{phase} rep={repetition} workers={workers} memory={memory} MiB")
    print("Phase B is selected after screening; baseline 4/512 is always included.", flush=True)
    if args.plan_only: return 0
    if args.mode == "comprehensive" and not args.allow_long_run:
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(json.dumps({"status": "not_started",
            "reason": "The 96-run screening plus full-dataset validation is a multi-hour experiment. "
                      "Re-run with --allow-long-run after reviewing the printed matrix.",
            "dataset_id": dataset_id, "planned_phase_a": len(phase_a)}, indent=2), encoding="utf-8")
        print("Stopped before processing: add --allow-long-run after reviewing the matrix.", flush=True)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "manifest.json").exists(): parser.error("output already contains a sweep; use --resume")
    previous_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8")) if args.resume and (output / "manifest.json").exists() else None
    if previous_manifest and previous_manifest["dataset_id"] != dataset_id: parser.error("dataset changed since previous run")
    if previous_manifest and previous_manifest.get("status") == "not_started": previous_manifest = None
    commit = _revision()
    manifest = previous_manifest or dict(schema=1, input=str(source), output=str(output), dataset_id=dataset_id,
        dataset_image_count=len(paths), dataset_records_signature=dataset_id, subset_id=subset_id,
        subset_selection="fixed-seed round-robin across first-level folder and file-size quartile; face-count strata unavailable before analysis",
        subset_paths=[str(p.relative_to(source)) for p in subset], seed=args.seed,
        git_commit=commit, hardware=_hardware(), model_fingerprints=_model_fingerprints(),
        processor_affinity=list(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        python=sys.version, platform=platform.platform(),
        cache_policy="new isolated child process and temporary state directory per execution; OS filesystem cache unchanged",
        sample_interval_seconds=0.25, mode=args.mode, screen_repetitions=args.screen_repetitions,
        full_repetitions=args.full_repetitions, planned_phase_a=len(phase_a),
        production_worker_limit=max_manual_decode_workers(), benchmark_worker_limit=max(args.workers or BROAD_WORKERS),
        effective_settings_note="requested values are passed unchanged to an isolated child; app resource-pressure sampler may lower active memory budget",
        limitations=["subset stratification does not know face counts before screening", "OS filesystem cache is not reset"], failed_runs=[])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, header in HEADERS.items():
        if not (output / name).exists(): _csv(output / name, header, [])
    def execute(plan):
        for order, (phase, repetition, workers, memory, did) in enumerate(plan, 1):
            run_id = f"{phase}-{repetition}-{workers}-{memory}"
            runs = _read(output / "runs.csv")
            if any(r["run_id"] == run_id and r["status"] == "complete" for r in runs): continue
            runs = [r for r in runs if r["run_id"] != run_id]
            selected = paths if phase == "B" else subset
            run_dir = output / "raw" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="room36-decoder-") as temporary:
                state = Path(temporary)
                spec = dict(state=str(state), paths=list(map(str, selected)), input=str(source), workers=workers, memory=memory)
                spec_path = run_dir / "spec.json"
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                env = os.environ.copy()
                env.pop("ROOM36_DECODE_WORKERS", None); env.pop("ROOM36_DECODE_MEMORY_MIB", None)
                env["ROOM36_BENCHMARK_DECODE_LIMIT"] = str(max(workers, max_manual_decode_workers()))
                try:
                    child = subprocess.run([sys.executable, "-B", "-m", "app.tools.benchmark_decoder_sweep",
                        "--child-spec", str(spec_path)], env=env, timeout=args.timeout,
                        capture_output=True, text=True)
                    for name in ("report.json", "trace.json", "error.txt"):
                        if (state / name).exists(): (run_dir / name).write_bytes((state / name).read_bytes())
                    (run_dir / "stdout.txt").write_text(child.stdout, encoding="utf-8")
                    (run_dir / "stderr.txt").write_text(child.stderr, encoding="utf-8")
                    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8")) if child.returncode == 0 else None
                    error = "" if report else ((run_dir / "error.txt").read_text(encoding="utf-8") if (run_dir / "error.txt").exists() else child.stderr[-1000:])
                except subprocess.TimeoutExpired:
                    report, error = None, f"timed out after {args.timeout} seconds"
            if report:
                row, stage_rows, sub_rows, resources = _rows(run_id, phase, repetition, order, workers, memory,
                    did, len(selected), report, commit)
                if (any(info["state"] == "FAILED" for info in report.get("stages", {}).values())
                    or len(report.get("stages", {})) < 7 or row["effective_workers"] is None):
                    row["status"] = "failed"; row["error_message"] = "stage failure"
                if _dataset(source)[2] != dataset_id:
                    row["status"] = "failed"; row["error_message"] = "source dataset metadata changed"
                signatures = report.get("result_hashes", {})
                reference = manifest.setdefault("reference_hashes", {}).setdefault(did, signatures)
                if signatures != reference:
                    row["status"] = "failed"; row["error_message"] = "output signatures differ from reference run"
                provider = row.get("gpu_provider")
                expected_provider = manifest.setdefault("provider_by_dataset", {}).setdefault(did, provider)
                if provider != expected_provider:
                    row["status"] = "failed"; row["error_message"] = "inference provider changed"
                for filename, extra in (("stages.csv", stage_rows), ("substage_timings.csv", sub_rows), ("resources.csv", resources)):
                    _csv(output / filename, HEADERS[filename], _read(output / filename) + extra)
            else:
                row = dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(), git_commit=commit,
                    dataset_id=did, dataset_image_count=len(selected), benchmark_phase=phase,
                    repetition=repetition, run_order=order, requested_workers=workers,
                    effective_workers=None, requested_memory_mib=memory, effective_memory_mib=None,
                    cache_mode="cold", status="failed", error_message=error)
            _csv(output / "runs.csv", RUNS, runs + [row])
            if row["status"] != "complete": manifest["failed_runs"].append(dict(run_id=run_id, error=row["error_message"]))
            _csv(output / "summary.csv", SUMMARY, _summary(_read(output / "runs.csv"), _read(output / "stages.csv")))
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"{run_id}: {row['status']} {row.get('wall_seconds', '')}", flush=True)
    try:
        execute(phase_a)
        if args.mode == "comprehensive":
            fine = manifest.get("phase_b_plan")
            if fine is None:
                fine = _fine_plan(args, _read(output / "summary.csv"), dataset_id)
                manifest["phase_b_plan"] = fine
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Phase B: {len(fine)} executions", flush=True)
            for phase, repetition, workers, memory, _ in fine:
                print(f"{phase} rep={repetition} workers={workers} memory={memory} MiB")
            execute(fine)
    except KeyboardInterrupt:
        manifest["interrupted"] = True
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return 130
    report = output / "report.md"
    finished = _read(output / "runs.csv")
    complete = [r for r in finished if r["status"] == "complete"]
    failed = [r for r in finished if r["status"] != "complete"]
    lines = ["# Decoder sweep", "", f"Dataset: {len(paths)} images; mode: {args.mode}.",
        f"Completed runs: {len(complete)}; failed runs: {len(failed)}.", "",
        "A decoder budget is permission to reserve memory, not preallocated RAM. "
        "Application caches were isolated; the OS filesystem cache was not purged.", ""]
    if args.mode == "smoke":
        lines.append("This two-image smoke run validates the harness only. It cannot identify optimal settings.")
    else:
        lines.append("See summary.csv for configuration comparisons. Stage and nested-operation durations "
                     "must not be added to estimate wall time.")
    if failed:
        lines.extend(["", "## Failed configurations", ""])
        lines.extend(f"- {r['run_id']}: {r['error_message']}" for r in failed)
    lines.extend(["", "Raw JSON reports and traces are in raw/. Unsupported telemetry is blank in the CSVs.", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    with zipfile.ZipFile(output / "decoder_sweep.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (*HEADERS, "manifest.json", "report.md"):
            archive.write(output / name, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
