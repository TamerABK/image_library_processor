"""Read-only source-photo benchmarks of Room36's progressive project pipeline.

Usage: python -B -m automatic_scan.benchmark --dataset PATH --size 50 --out DIR
All application state and analysis caches are created in TemporaryDirectory.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import platform
import pstats
from queue import Queue
import shutil
import statistics
import subprocess
import threading
import tempfile
import time
import tracemalloc
from pathlib import Path

from image_file_utils import find_supported_files
from image_loader import ImageLoader, default_image_loader
from scan_controls import CancellationToken

from .config import AutomaticScanConfig, BlurPolicy, PerformanceConfig
from .orchestrator import AutomaticScanOrchestrator
from .perf import active_profiler
from .profiling import HierarchicalProfiler, profile_scope
from .profiling_resources import ResourceSampler
from .project_pipeline import STAGES, process_project
from .services import AutomaticScanServices
from .state_store import AutomaticStateStore


FOCUS_STAGE = {"full": None, "blur": "blur", "duplicates": "duplicates",
               "faces": "faces", "embeddings": "embeddings", "clustering": "people",
               "cache-db": None, "image": None}
LOCATIONS = {"stage.blur": "automatic_scan/project_pipeline.py", "stage.duplicates": "automatic_scan/project_pipeline.py",
    "stage.faces": "automatic_scan/project_pipeline.py", "stage.embeddings": "automatic_scan/project_pipeline.py",
    "stage.people": "automatic_scan/project_pipeline.py", "stage.ranking": "automatic_scan/project_pipeline.py",
    "blur.decode_and_score": "automatic_scan/services.py", "faces.scrfd_detection": "automatic_scan/services.py",
    "embeddings.arcface_batch": "automatic_scan/services.py", "people.cluster_unknown": "automatic_scan/project_pipeline.py",
    "duplicates.orb_and_union": "automatic_scan/project_pipeline.py", "duplicates.candidates": "automatic_scan/project_pipeline.py",
    "project.metadata": "automatic_scan/project_pipeline.py"}


def _distribution(paths: list[Path]) -> dict:
    loader = ImageLoader()
    pixels = []
    unreadable = 0
    source_bytes = 0
    for path in paths:
        source_bytes += path.stat().st_size
        info = loader.read_metadata(path)
        if info is None:
            unreadable += 1
        else:
            pixels.append(info.width * info.height)
    pixels.sort()
    def at(fraction):
        return pixels[min(len(pixels) - 1, int((len(pixels) - 1) * fraction))] if pixels else None
    return {"readable": len(pixels), "unreadable": unreadable,
            "source_bytes": source_bytes,
            "pixels_min": at(0), "pixels_median": at(.5),
            "pixels_p95": at(.95), "pixels_max": at(1)}


def _select_paths(dataset: Path, size: int) -> list[Path]:
    """Deterministic round-robin across first-level corpus/source folders."""
    from collections import defaultdict
    groups = defaultdict(list)
    for path in find_supported_files(dataset, default_image_loader.supported_extensions()):
        relative = path.relative_to(dataset)
        key = relative.parts[0] if len(relative.parts) > 1 else "."
        groups[key].append(path)
    for group in groups.values():
        group.sort(key=lambda path: str(path).casefold())
    selected = []
    positions = {key: 0 for key in groups}
    while len(selected) < size:
        progress = False
        for key in sorted(groups):
            if positions[key] < len(groups[key]) and len(selected) < size:
                selected.append(groups[key][positions[key]])
                positions[key] += 1
                progress = True
        if not progress:
            break
    return selected


def _ort_summary(files: list[str]) -> dict:
    providers: dict[str, dict[str, int]] = {}
    operators: dict[tuple[str, str], dict[str, int]] = {}
    nodes = 0
    by_file = []
    for index, filename in enumerate(files, 1):
        try:
            events = json.loads(Path(filename).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        file_nodes = 0
        for event in events:
            if event.get("cat") != "Node":
                continue
            nodes += 1
            file_nodes += 1
            provider = str(event.get("args", {}).get("provider", "unknown"))
            entry = providers.setdefault(provider, {"calls": 0, "operator_duration_us": 0})
            entry["calls"] += 1
            entry["operator_duration_us"] += int(event.get("dur", 0))
            operation = str(event.get("args", {}).get("op_name", "unknown"))
            operator = operators.setdefault((operation, provider), {"calls": 0, "duration_us": 0})
            operator["calls"] += 1
            operator["duration_us"] += int(event.get("dur", 0))
        by_file.append({"index": index, "node_events": file_nodes})
    return {"profile_files": len(files), "node_events": nodes, "providers": providers,
            "by_file": by_file,
            "top_operators": [{"operation": operation, "provider": provider, **values}
                for (operation, provider), values in sorted(operators.items(),
                    key=lambda item: -item[1]["duration_us"])[:20]],
            "note": "ORT operator durations are not interchangeable with host-side run wall time."}


def _hardware() -> dict:
    from face_detector.onnx_runtime import startup_diagnostics
    from app_paths import model_path
    import cv2
    result = {"os": platform.platform(), "logical_cpus": os.cpu_count(),
              "cpu_model": platform.processor(), "opencv_threads": cv2.getNumThreads(),
              "onnx_runtime": startup_diagnostics(), "gpu_names": None}
    result["model_file_bytes"] = {}
    for name in ("scrfd_10g_bnkps.onnx", "glintr100.onnx", "sixdrepnet.onnx", "open_closed_eye.onnx"):
        try:
            result["model_file_bytes"][name] = model_path(name).stat().st_size
        except OSError:
            result["model_file_bytes"][name] = None
    if os.name == "nt" and shutil.which("powershell.exe"):
        try:
            query = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=5, check=True)
            result["gpu_names"] = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        from threadpoolctl import threadpool_info
        result["native_threadpools"] = [{"library": item.get("internal_api"),
            "num_threads": item.get("num_threads")} for item in threadpool_info()]
    except ImportError:
        result["native_threadpools"] = None
    return result


def _run_pipeline(paths: list[Path], state_dir: Path, *, folder: Path | None = None, scenario: str,
                  detail: str, batch_size: int | None, ort_profile: bool,
                  cprofile_file: Path | None = None, measured: bool = True,
                  concurrency: int = 1, python_allocations: bool = False,
                  native_trace_dir: Path | None = None) -> dict:
    """Keep benchmark logs and native profiles out of application data."""
    state_dir.mkdir(parents=True, exist_ok=True)
    keys = ("ROOM36_SCAN_LOG_PATH", "ROOM36_ORT_PROFILE_PREFIX")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["ROOM36_SCAN_LOG_PATH"] = str(state_dir / "scan.log")
    try:
        return _run_pipeline_inner(paths, state_dir, folder=folder, scenario=scenario,
            detail=detail, batch_size=batch_size, ort_profile=ort_profile,
            cprofile_file=cprofile_file, measured=measured, concurrency=concurrency,
            python_allocations=python_allocations, native_trace_dir=native_trace_dir)
    finally:
        from .scan_logging import close_scan_logger_for_path
        close_scan_logger_for_path(state_dir / "scan.log")
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_pipeline_inner(paths: list[Path], state_dir: Path, *, folder: Path | None = None, scenario: str,
                  detail: str, batch_size: int | None, ort_profile: bool,
                  cprofile_file: Path | None = None, measured: bool = True,
                  concurrency: int = 1, python_allocations: bool = False,
                  native_trace_dir: Path | None = None) -> dict:
    try:
        event_limit = max(1000, min(500000, int(os.environ.get("ROOM36_PROFILE_MAX_EVENTS", "100000"))))
        sample_interval = max(.1, min(5.0, float(os.environ.get("ROOM36_PROFILE_SAMPLE_INTERVAL", ".25"))))
    except ValueError:
        event_limit, sample_interval = 100000, .25
    profiler = HierarchicalProfiler(detail=detail, max_events=event_limit) if measured else None
    sampler = ResourceSampler(dataset=str(folder or paths[0].parent),
                              interval=sample_interval) if measured else None
    emitted: dict[str, dict] = {}
    result_hashes: dict[str, str] = {}
    face_count = 0
    allocations = None
    if ort_profile and measured:
        os.environ["ROOM36_ORT_PROFILE_PREFIX"] = str(state_dir / "ort")
    else:
        os.environ.pop("ROOM36_ORT_PROFILE_PREFIX", None)
    config = AutomaticScanConfig(blur_policy=BlurPolicy.KEEP_ALL,
        performance=PerformanceConfig(worker_count=1, batch_size=batch_size))
    token = CancellationToken()
    with active_profiler(profiler):
        with profile_scope("benchmark.total", detail="stages"):
            store = AutomaticStateStore(state_dir / "state")
            orchestrator = AutomaticScanOrchestrator(state_store=store)
            services = AutomaticScanServices(config, lazy_models=True,
                cache_path=state_dir / "analysis.sqlite3",
                face_database_path=state_dir / "faces.sqlite3")
            def emit(stage, state, done, total, error, data, stats):
                nonlocal face_count
                if str(state) in {"COMPLETE", "FAILED"}:
                    emitted[stage] = {"state": str(state), "error": error, "stats": stats}
                    if str(state) == "COMPLETE" and data is not None:
                        result_hashes[stage] = hashlib.sha256(json.dumps(data,
                            sort_keys=True, default=str).encode("utf-8")).hexdigest()
                    if stage == "faces" and isinstance(data, dict):
                        face_count = int(data.get("face_count", 0))
            if sampler:
                sampler.start()
            if python_allocations:
                tracemalloc.start(10)
            py_profile = cProfile.Profile() if cprofile_file else None
            try:
                if py_profile:
                    py_profile.enable()
                if scenario == "image":
                    loader = ImageLoader()
                    if concurrency == 1:
                        for path in paths:
                            with profile_scope("image.decode_and_preprocess", units=1):
                                loader.load_for_scan(path)
                    else:
                        tasks = Queue(maxsize=concurrency * 2)
                        errors = []
                        def worker():
                            with active_profiler(profiler):
                                while True:
                                    item = tasks.get()
                                    if item is None:
                                        return
                                    path, queued_ns = item
                                    if profiler:
                                        profiler.add_count("image.scheduling_latency_including_backpressure_ns",
                                            time.perf_counter_ns() - queued_ns)
                                    try:
                                        with profile_scope("image.decode_and_preprocess", units=1):
                                            loader.load_for_scan(path)
                                    except Exception as exc:
                                        errors.append(type(exc).__name__)
                        with profile_scope("image.worker_creation"):
                            workers = [threading.Thread(target=worker, name=f"room36-benchmark-image-{i}")
                                       for i in range(concurrency)]
                            for thread in workers:
                                thread.start()
                        for path in paths:
                            with profile_scope("image.queue_backpressure", detail="deep"):
                                tasks.put((path, time.perf_counter_ns()))
                        for _ in workers:
                            tasks.put(None)
                        for thread in workers:
                            thread.join()
                        if errors:
                            raise RuntimeError(f"Image loading failed for {len(errors)} photos")
                else:
                    process_project(orchestrator, str(folder or paths[0].parent), paths, emit=emit,
                        cancellation_token=token, cache_path=state_dir / "analysis.sqlite3",
                        config=config, services=services,
                        benchmark_stop_after=FOCUS_STAGE[scenario])
            finally:
                if py_profile:
                    py_profile.disable()
                    py_profile.dump_stats(str(cprofile_file))
                if python_allocations:
                    current, peak = tracemalloc.get_traced_memory()
                    snapshot = tracemalloc.take_snapshot()
                    allocations = {"current_bytes": current, "peak_bytes": peak,
                        "top_python_sites": [{"site": str(stat.traceback[0]), "bytes": stat.size,
                            "allocations": stat.count}
                            for stat in snapshot.statistics("lineno")[:20]],
                        "note": "Python-traced allocations exclude most NumPy/OpenCV/ORT native buffers."}
                    tracemalloc.stop()
                if sampler:
                    sampler.stop()
    if profiler:
        profiler.finish()
    native_files = []
    if ort_profile and measured:
        from face_detector.onnx_runtime import finish_native_profiling
        native_files = finish_native_profiling()
        native_files = sorted({*native_files,
            *(str(path) for path in state_dir.glob("ort-*.json"))})
        if native_trace_dir is not None:
            native_trace_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(native_files, 1):
                shutil.copyfile(source, native_trace_dir / f"session-{index}.json")
    os.environ.pop("ROOM36_ORT_PROFILE_PREFIX", None)
    measured_faces = face_count if emitted.get("faces", {}).get("state") == "COMPLETE" else None
    cache_disk_bytes = sum(path.stat().st_size for path in state_dir.rglob("*.sqlite3*") if path.is_file())
    trace = profiler.trace() if profiler else None
    if trace and sampler:
        trace["traceEvents"].extend(sampler.trace_events(profiler.started_ns))
    return {"profile": profiler.report(image_count=len(paths), face_count=measured_faces) if profiler else None,
            "trace": trace,
            "resources": sampler.report() if sampler else None,
            "stages": emitted, "ort": _ort_summary(native_files) if ort_profile else None,
            "result_hashes": result_hashes,
            "cache_database_bytes": cache_disk_bytes,
            "python_allocations": allocations}


def _markdown(result: dict) -> str:
    profile = result["profile"]
    resources = result["resources"]
    timings = profile["timings"]
    wall = profile["wall_seconds"]
    faces_text = (f"{profile['face_count']} ({profile['faces_per_second']:.2f} faces/s)"
                  if profile["face_count"] is not None else "not measured in this scenario")
    lines = ["# Room36 performance profile", "",
        f"Scenario: **{result['scenario']}**; cache: **{result['cache_mode']}**; images: **{profile['image_count']}**.",
        f"Measured wall time: **{wall:.3f} s**; throughput: **{profile['images_per_second']:.2f} images/s**; "
        f"faces: **{faces_text}**.", "",
        "Stage durations can overlap; their sum is not necessarily project wall time. "
        "Exclusive time subtracts only nested scopes on the same thread.", "",
        "## Execution-time breakdown", "",
        "| Scope | Calls | Inclusive s | Exclusive s | Mean ms | P95 ms | % wall |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, data in sorted(timings.items(), key=lambda item: -item[1]["inclusive_seconds"]):
        if name == "benchmark.total":
            continue
        lines.append(f"| `{name}` | {data['calls']} | {data['inclusive_seconds']:.3f} | "
            f"{data['exclusive_seconds']:.3f} | {1000*data['average_seconds']:.2f} | "
            f"{1000*data['p95_seconds']:.2f} | {100*data['inclusive_seconds']/wall if wall else 0:.1f}% |")
    lines.extend(["", "## Top 20 timed scopes by exclusive time", ""])
    for index, (name, data) in enumerate(sorted(
            ((name, data) for name, data in timings.items() if name != "benchmark.total"),
            key=lambda item: -item[1]["exclusive_seconds"])[:20], 1):
        lines.append(f"{index}. `{name}` — {data['exclusive_seconds']:.3f} s exclusive; "
                     f"{LOCATIONS.get(name, 'see named call boundary')}.")
    cpu = resources["process_cpu_core_equivalents"]
    core_samples = [value for sample in resources["samples"]
                    for value in sample.get("per_core_cpu_percent", [])]
    lines += ["", "## Resource and concurrency evidence", "",
        f"Hardware: {result['hardware']['cpu_model']}; {result['hardware']['logical_cpus']} logical CPUs; "
        f"GPU(s): {result['hardware']['gpu_names']}; ONNX providers: "
        f"{result['hardware']['onnx_runtime']['available_providers']}.",
        f"Process CPU core equivalents over sampled wall time: {cpu:.2f}." if cpu is not None else
        "Process CPU saturation unavailable.",
        f"Peak sampled per-core utilization: {max(core_samples):.1f}%." if core_samples else
        "Per-core utilization unavailable or run shorter than sample interval.",
        f"Peak sampled RSS: {resources['peak_rss_bytes']} bytes.",
        f"Process logical I/O reads/writes: {resources['disk_read_bytes_delta']} / "
        f"{resources['disk_write_bytes_delta']} bytes (may hit OS cache).",
        (f"Dataset PhysicalDisk instance {resources['physical_disk']['instance']}: "
         f"mean/peak read {resources['physical_disk']['metrics']['read_mb_s']['mean']:.2f}/"
         f"{resources['physical_disk']['metrics']['read_mb_s']['peak']:.2f} MB/s; "
         "device-wide activity includes other processes."
         if resources.get('physical_disk', {}).get('metrics', {}).get('read_mb_s') else
         "PhysicalDisk dataset-volume telemetry unavailable; logical process I/O cannot establish SSD saturation."),
        (f"Windows PDH process GPU engines: {resources['gpu_process_engines']}; "
         f"peak dedicated process GPU memory: {resources['gpu_process_dedicated_bytes_peak']} bytes. "
         "These are process-engine counters, not whole-device utilization."
         if resources.get("gpu_process_engines") else
         "Process GPU engine utilization / dedicated memory unavailable; no reliable device total is inferred."),
        "The current Qt production pipeline runs its seven main stages sequentially on one project worker. "
        "ONNX Runtime and OpenCV may use native threads inside those calls. Per-core samples or native traces "
        "are needed before attributing low parallelism to the GIL.", ""]
    if result.get("ort"):
        lines.append(f"ORT operator providers: `{result['ort']['providers']}`. Host inference-call time and "
                     "native operator time use different boundaries.")
    lines += ["", "## Cache and image characteristics", "",
        f"Resolution pixels min/median/P95/max: {result['resolution']}",
        f"Application cache hit rate: {result['cache_hit_rate']:.1%} "
        "(stage cache hits / cache hits plus computed; OS file cache is separate).",
        f"Temporary benchmark database files: {result['cache_database_bytes']} bytes. "
        "Model file sizes are in the JSON hardware section; file size is not resident model memory.",
        f"Image opens: {profile['image_io']['disk_reads']}; full decodes: {profile['image_io']['full_decodes']}; "
        f"bounded decodes: {profile['image_io']['bounded_decodes']}.",
        "", "## Bottleneck ranking (timed boundaries)", "",
        "| Scope | Exclusive s | % wall | Likely limiting subsystem | Confidence |",
        "|---|---:|---:|---|---|"]
    leaves = [(name, data) for name, data in timings.items()
              if name != "benchmark.total" and not name.startswith("stage.") and
              not name.endswith(".cache_and_compute") and not name.endswith(".compute")]
    for name, data in sorted(leaves, key=lambda item: -item[1]["exclusive_seconds"])[:8]:
        if "cache_put" in name or "cache_mark" in name:
            category, confidence = "SQLite/cache boundary; physical storage cause unmeasured", "high time / medium cause"
        elif "imread" in name or "decode" in name:
            category, confidence = "image decode + logical file I/O", "high time / medium cause"
        elif name.startswith("startup."):
            category, confidence = "one-time model/database startup", "high"
        elif name.startswith("onnx.") or "arcface" in name or "scrfd" in name:
            category, confidence = "host inference / model scheduling; GPU occupancy unknown", "high time / low cause"
        else:
            category, confidence = "CPU/native work or synchronization; inspect trace", "medium"
        lines.append(f"| `{name}` | {data['exclusive_seconds']:.3f} | "
            f"{100*data['exclusive_seconds']/wall if wall else 0:.1f}% | {category} | {confidence} |")
    lines += ["", "## Candidate improvements to test after profiling", "",
        "- Low risk: investigate whether cache rows can be persisted in fewer transactions; "
        "compare correctness and write latency before changing the cache.",
        "- Pipeline/concurrency: test bounded decode/preparation overlap with inference; the "
        "production stages are currently sequential. Measure memory and queue wait at each worker count.",
        "- Algorithmic: only investigate candidate generation or quadratic People clustering on "
        "a dataset where those scopes become significant; preserve safety limits.",
        "- GPU/inference: compare existing ArcFace batch sizes with isolated native ORT traces; "
        "operator time is not host wall time and DirectML session variants have startup cost.",
        "- Architecture: stage pipelining would be a larger change and requires a controlled "
        "critical-path experiment. No gain estimate is justified yet.", "",
        "These measurements are observational. CPU/GPU saturation and scaling claims require "
        "controlled repeated runs on the same dataset with hardware-specific telemetry. "
        "No optimization has been applied.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Existing local image directory; read only")
    parser.add_argument("--out", type=Path, required=True, help="Report directory")
    parser.add_argument("--size", type=int, choices=(50, 200, 1000, 6000), default=50)
    parser.add_argument("--scenario", choices=tuple(FOCUS_STAGE), default="full")
    parser.add_argument("--cache-mode", choices=("cold", "warm"), default="cold")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--detail", choices=("stages", "normal", "deep"), default="normal")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 6, 8), default=1,
                        help="Bounded image-loading workers; full Qt pipeline remains one worker")
    parser.add_argument("--ort-profile", action="store_true", help="Separate deep native ONNX operator trace")
    parser.add_argument("--cprofile", action="store_true", help="Write Python function-level pstats")
    parser.add_argument("--python-allocations", action="store_true",
                        help="Separate high-overhead Python tracemalloc run; native arrays excluded")
    parser.add_argument("--compare-overhead", action="store_true", help="Compare 3 disabled/profiled warm-cache pairs")
    args = parser.parse_args(argv)
    if args.repeat < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("repeat and batch-size must be positive")
    if args.concurrency != 1 and args.scenario != "image":
        parser.error("concurrency experiments are supported only for the real ImageLoader scenario")
    dataset = args.dataset.resolve(strict=True)
    if not dataset.is_dir():
        parser.error("dataset must be a directory")
    output = args.out.resolve()
    if output == dataset or dataset in output.parents:
        parser.error("report output must be outside the image dataset")
    paths = _select_paths(dataset, args.size)
    if not paths:
        parser.error("no supported images found")
    output.mkdir(parents=True, exist_ok=True)
    resolution = _distribution(paths)
    summary = []
    hardware = _hardware()
    with tempfile.TemporaryDirectory(prefix="room36-profile-") as temporary:
        root = Path(temporary)
        if args.cache_mode == "warm" and args.scenario != "image":
            seed_scenario = "full" if args.scenario == "cache-db" else args.scenario
            _run_pipeline(paths, root / "warm", folder=dataset, scenario=seed_scenario,
                          detail="stages", batch_size=args.batch_size, ort_profile=False, measured=False)
        if args.compare_overhead:
            if args.cache_mode != "warm":
                parser.error("--compare-overhead requires --cache-mode warm")
            pairs = []
            for pair in range(3):
                times = {}
                for enabled in ((False, True) if pair % 2 == 0 else (True, False)):
                    started = time.perf_counter_ns()
                    _run_pipeline(paths, root / "warm", folder=dataset, scenario=args.scenario,
                        detail=args.detail, batch_size=args.batch_size, ort_profile=False,
                        measured=enabled, concurrency=args.concurrency)
                    times["profiled" if enabled else "disabled"] = (time.perf_counter_ns() - started) / 1e9
                pairs.append(times)
            comparison = {"pairs": pairs,
                "median_disabled_seconds": statistics.median(p["disabled"] for p in pairs),
                "median_profiled_seconds": statistics.median(p["profiled"] for p in pairs),
                "note": "Warm-cache A/B pairs include sampler startup and report aggregation; "
                        "short runs may exaggerate fixed overhead."}
            comparison["relative_overhead"] = (comparison["median_profiled_seconds"] /
                comparison["median_disabled_seconds"] - 1) if comparison["median_disabled_seconds"] else None
            (output / "overhead.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        for index in range(args.repeat):
            state_dir = root / ("warm" if args.cache_mode == "warm" else f"cold-{index}")
            run = _run_pipeline(paths, state_dir, folder=dataset, scenario=args.scenario, detail=args.detail,
                batch_size=args.batch_size, ort_profile=args.ort_profile,
                cprofile_file=output / f"run-{index+1}.pstats" if args.cprofile else None,
                concurrency=args.concurrency, python_allocations=args.python_allocations,
                native_trace_dir=output / f"run-{index+1}.ort" if args.ort_profile else None)
            hits = sum(s.get("stats", {}).get("cache_hits", 0) for s in run["stages"].values())
            computed = sum(s.get("stats", {}).get("computed", 0) for s in run["stages"].values())
            run.update(scenario=args.scenario, cache_mode=args.cache_mode, concurrency=args.concurrency,
                       batch_size=args.batch_size, resolution=resolution,
                       decode_workers=os.environ.get("ROOM36_DECODE_WORKERS", "adaptive"),
                       decode_memory_mib=os.environ.get("ROOM36_DECODE_MEMORY_MIB", "512"),
                       hardware=hardware, cache_hit_rate=hits / (hits + computed) if hits + computed else 0)
            report_path = output / f"run-{index+1}.json"
            report_path.write_text(json.dumps({k: v for k, v in run.items() if k != "trace"},
                indent=2, default=str), encoding="utf-8")
            (output / f"run-{index+1}.trace.json").write_text(json.dumps(run["trace"]), encoding="utf-8")
            (output / f"run-{index+1}.md").write_text(_markdown(run), encoding="utf-8")
            if args.cprofile:
                stream = io.StringIO()
                pstats.Stats(str(output / f"run-{index+1}.pstats"), stream=stream).sort_stats(
                    "cumulative").print_stats(20)
                (output / f"run-{index+1}.cprofile-top20.txt").write_text(stream.getvalue(), encoding="utf-8")
            summary.append({"run": index + 1, "wall_seconds": run["profile"]["wall_seconds"],
                "stage_states": {k: v["state"] for k, v in run["stages"].items()}})
            print(f"run {index+1}: {summary[-1]['wall_seconds']:.3f}s; {summary[-1]['stage_states']}")
    (output / "summary.json").write_text(json.dumps({"runs": summary,
        "median_wall_seconds": statistics.median(r["wall_seconds"] for r in summary),
        "min_wall_seconds": min(r["wall_seconds"] for r in summary),
        "max_wall_seconds": max(r["wall_seconds"] for r in summary)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
