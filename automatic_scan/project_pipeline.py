"""Bounded progressive execution for AutomaticScanOrchestrator's Qt entry point.

No project discovery, Qt objects, review execution or additional worker pool.
The caller supplies the discovered identities and one cancellation token.
"""
from dataclasses import asdict
from contextlib import nullcontext
import ctypes
from enum import StrEnum
from pathlib import Path
import os
import time

from app_paths import model_path
from duplicate_detector.config import DetectorConfig
from image_loader import ImageLoader, default_image_loader
from .decode_pool import (BoundedDecodePool, decode_workers_from_environment,
                          decode_memory_budget_from_environment)
from image_cache_batch import (CacheBatchSession, CacheBatchWriteError,
    batch_size_from_environment, max_delay_from_environment)
from scan_controls import ScanCancelledError
from .config import AutomaticScanConfig, BlurPolicy, PerformanceConfig
from .duplicate_selection import select_duplicate_keepers
from .eligibility import assess_face_quality, assess_people_picture
from .models import DiscoveryRecord, PipelineImageState
from .project_cache import ProjectStageCache, signature
from .services import AutomaticScanServices
from .scan_logging import get_scan_logger
from .profiling import profile_scope, profile_count
from face_detector.connected_face_clusterer import ClusterCapacityError


class StageState(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


STAGES = ("blur", "duplicates", "faces", "face_quality", "embeddings", "people", "ranking")
MAX_CANDIDATE_PAIRS = 50000
MAX_UNKNOWN_FACES = 2000 * 8  # Clustering allocates an N x N float32 similarity matrix.


def unknown_face_limit() -> int:
    """Configurable downward from the 16,000-face hard safety ceiling."""
    try:
        value = int(os.environ.get("ROOM36_MAX_UNKNOWN_FACES", str(MAX_UNKNOWN_FACES)))
    except ValueError:
        return MAX_UNKNOWN_FACES
    return value if 1 <= value <= MAX_UNKNOWN_FACES else MAX_UNKNOWN_FACES


def _available_memory_bytes() -> int | None:
    if os.name == "nt":
        from ctypes import wintypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_physical", "available_physical", "total_pagefile",
                    "available_pagefile", "total_virtual", "available_virtual", "available_extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        return int(status.available_physical) if ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)) else None
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return None


class StageUnavailable(RuntimeError):
    """User-safe resource/model failure, distinct from internal exceptions."""


class _BenchmarkStop(BaseException):
    """Stop after a completed stage without changing any production stage result."""


def process_project(orchestrator, folder, paths, *, emit, cancellation_token,
                    cache_path=None, config=None, services=None, benchmark_stop_after=None):
    logger = get_scan_logger()
    logger.info("project_started folder=%s discovered_paths=%d", folder, len(paths))
    config = config or AutomaticScanConfig(blur_policy=BlurPolicy.KEEP_ALL,
        performance=PerformanceConfig(worker_count=1, batch_size=32))
    folder = str(folder)
    token = cancellation_token
    with profile_scope("project.initialize_services", detail="stages"):
        cache = ProjectStageCache(cache_path)
        services = services or AutomaticScanServices(config, lazy_models=True,
            cache_path=cache_path, duplicate_config=DetectorConfig(max_workers=1, cache_size=8))
    records, missing = [], []
    loader = ImageLoader()
    with profile_scope("project.metadata", detail="stages", units=len(paths)):
        for path in paths:
            token.raise_if_canceled()
            path = Path(path)
            try:
                with profile_scope("project.metadata.stat_and_header", detail="deep"):
                    stat = path.stat()
                    metadata = loader.read_metadata(path)
                capture_timestamp = metadata.capture_timestamp if metadata else None
                records.append(DiscoveryRecord(path, stat.st_size, stat.st_mtime_ns,
                    path.suffix.lower(), metadata.orientation if metadata else None,
                    capture_timestamp if capture_timestamp is not None else stat.st_mtime_ns / 1_000_000_000,
                    metadata.width if metadata else None, metadata.height if metadata else None))
            except Exception:
                logger.exception("metadata_failed folder=%s photo=%s", folder, path)
                missing.append(str(path))
    states = {str(r.path): PipelineImageState(**asdict(r)) for r in records}
    for state in states.values():
        state.path = str(state.path)
    fingerprint = [(str(r.path), r.file_size, r.mtime_ns) for r in records]
    blur_key = signature(["blur-v2", vars(services._blur_detector).get("grid_size"),
        {k: getattr(services._blur_detector, k) for k in
         ("lap_max", "sobel_max", "contrast_max", "review_threshold", "sharp_threshold", "max_dimension")}])
    duplicate_key = signature(["hash-orb-v2", asdict(services._duplicate_config)])
    models = []
    for name in ("scrfd_10g_bnkps.onnx", "glintr100.onnx", "sixdrepnet.onnx", "open_closed_eye.onnx"):
        path = model_path(name)
        try:
            stat = path.stat()
            models.append((name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            models.append((name, "missing"))
    face_key = signature(["faces-v1", models, config.cache_signature()])
    diagnostics = {}
    completed = set()
    try:
        with profile_scope("project.load_reviews"):
            decisions = orchestrator._state_store.load(folder).image_decisions
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("review_state_load_failed folder=%s", folder)
        decisions = {}
    cache_batch_size = batch_size_from_environment()
    cache_session = (CacheBatchSession(cache.storage.db_path, cache.storage._open_connection,
        batch_size=cache_batch_size, max_delay_seconds=max_delay_from_environment())
        if cache_batch_size is not None else None)
    if cache_session is not None:
        cache_session.__enter__()
    decode_workers = decode_workers_from_environment()
    decode_pool = None
    if decode_workers and type(services) is AutomaticScanServices:
        try:
            decode_pool = BoundedDecodePool(workers=decode_workers,
                memory_budget_bytes=decode_memory_budget_from_environment())
        except (OSError, RuntimeError):
            logger.exception("decode_pool_unavailable folder=%s; using sequential decoding", folder)

    def stage(name, work):
        token.raise_if_canceled()
        logger.info("stage_started folder=%s stage=%s", folder, name)
        emit(name, StageState.RUNNING, 0, 0, "", None, {})
        started = time.perf_counter()
        stats = {"cache_hits": 0, "computed": 0, "file_errors": 0}
        def progress(done, total):
            token.raise_if_canceled()
            emit(name, StageState.RUNNING, done, total, "", None, {})
        try:
            with profile_scope("stage." + name, detail="stages"):
                data = work(stats, progress)
                if cache_session is not None:
                    cache_session.flush()  # Stage results are durable before COMPLETE is emitted.
            token.raise_if_canceled()
            stats["seconds"] = round(time.perf_counter() - started, 4)
            diagnostics[name] = stats
            completed.add(name)
            logger.info("stage_completed folder=%s stage=%s seconds=%.4f cache_hits=%d computed=%d file_errors=%d aggregate_reused=%d",
                        folder, name, stats["seconds"], stats["cache_hits"], stats["computed"],
                        stats["file_errors"], stats.get("aggregate_reused", 0))
            message = (f"{stats['file_errors']} photos could not be analyzed." if stats["file_errors"] else "")
            emit(name, StageState.COMPLETE, 0, 0, message, data, stats)
            if name == benchmark_stop_after:
                raise _BenchmarkStop
            return data
        except ScanCancelledError:
            logger.info("stage_cancelled folder=%s stage=%s", folder, name)
            raise
        except CacheBatchWriteError:
            stats["error_type"] = "CacheBatchWriteError"
            diagnostics[name] = stats
            emit(name, StageState.FAILED, 0, 0,
                 "The analysis cache could not be saved. Check application data permissions, then retry.",
                 None, stats)
            raise
        except Exception as exc:
            stats["seconds"] = round(time.perf_counter() - started, 4)
            stats["error_type"] = type(exc).__name__
            diagnostics[name] = stats
            logger.exception("stage_failed folder=%s stage=%s seconds=%.4f cache_hits=%d computed=%d file_errors=%d",
                             folder, name, stats["seconds"], stats["cache_hits"],
                             stats["computed"], stats["file_errors"])
            emit(name, StageState.FAILED, 0, 0,
                 str(exc) if isinstance(exc, StageUnavailable) else
                 "This stage could not finish. Check the files and installed models, then retry processing.", None, stats)
            return None

    def per_file(name, key, get, compute, stats, progress, source_records):
        results = {}
        base_getter = {
            "blur": AutomaticScanServices.get_blur_cached,
            "duplicates": AutomaticScanServices.get_duplicate_photo_cached,
            "faces": AutomaticScanServices.get_face_analysis_cached,
        }.get(name)
        shared_getter = base_getter is not None and getattr(get, "__func__", None) is base_getter
        with (
            profile_scope(name + ".cache_and_compute", units=len(source_records)),
            cache.storage.connect() as read_db,
        ):
            try:
                with profile_scope(name + ".cache_prefetch"):
                    compatible = cache.compatible_paths((r.path for r in source_records),
                                                        name, key, connection=read_db)
            except Exception:
                logger.exception("compatibility_prefetch_failed folder=%s stage=%s; using individual lookups",
                                 folder, name)
                compatible = None
            parallel = (decode_pool is not None and shared_getter and
                        (name in {"duplicates", "faces"} or
                         getattr(compute, "__func__", None) in {
                             AutomaticScanServices.analyze_blur,
                             AutomaticScanServices.analyze_faces}))
            if parallel:
                params = {
                    "blur": (default_image_loader, services._blur_detector.max_dimension, True),
                    "duplicates": (loader, services._duplicate_config.hash_decode_dimension, False),
                    "faces": (default_image_loader, None, False),
                }
                image_loader, max_dimension, grayscale = params[name]
                entries = []
                for record in source_records:
                    token.raise_if_canceled()
                    try:
                        with profile_scope(name + ".cache_lookup", detail="deep"):
                            valid_signature = (str(record.path) in compatible if compatible is not None else
                                               cache.compatible(record.path, name, key, connection=read_db))
                            value = (get(record, connection=read_db) if shared_getter else get(record)) if valid_signature else None
                    except Exception:
                        logger.exception("cache_read_failed folder=%s stage=%s photo=%s; recomputing",
                                         folder, name, record.path)
                        value = None
                    entries.append((record, value))
                misses = [record for record, value in entries if value is None]
                before_submitted, before_wait = decode_pool.submitted, decode_pool.wait_ns
                before_task, before_queue = decode_pool.task_ns, decode_pool.queue_wait_ns
                decoded = decode_pool.ordered(misses, image_loader, max_dimension=max_dimension,
                                              grayscale=grayscale, token=token)
                try:
                    for index, (record, value) in enumerate(entries, 1):
                        token.raise_if_canceled()
                        operation = "compute" if value is None else "cache_lookup"
                        try:
                            if value is None:
                                decoded_record, outcome = next(decoded)
                                assert decoded_record is record
                                if outcome is not None and outcome.error is not None:
                                    raise outcome.error
                                context = (image_loader.use_prefetched(record.path, outcome.image,
                                    max_dimension=max_dimension, grayscale=grayscale)
                                    if outcome is not None else nullcontext())
                                with context, profile_scope(name + ".compute", units=1,
                                    width=record.width or 0, height=record.height or 0):
                                    value = compute(record)
                                if value is None:
                                    raise ValueError("No result")
                                token.raise_if_canceled()
                                operation = "cache_write"
                                with profile_scope(name + ".cache_mark", detail="deep"):
                                    cache.mark(record.path, name, key)
                                stats["computed"] += 1
                            else:
                                stats["cache_hits"] += 1
                                profile_count(name + ".cache_hits")
                            results[str(record.path)] = value
                        except ScanCancelledError:
                            raise
                        except (CacheBatchWriteError, StageUnavailable):
                            raise
                        except Exception:
                            stats["file_errors"] += 1
                            logger.exception("photo_failed folder=%s stage=%s operation=%s photo=%s",
                                             folder, name, operation, record.path)
                        progress(index, len(source_records))
                finally:
                    decoded.close()
                stats["decode_tasks"] = decode_pool.submitted - before_submitted
                stats["decode_wait_seconds"] = (decode_pool.wait_ns - before_wait) / 1e9
                stats["decode_task_seconds_sum"] = (decode_pool.task_ns - before_task) / 1e9
                stats["decode_queue_wait_seconds_sum"] = (decode_pool.queue_wait_ns - before_queue) / 1e9
                stats["decode_max_pending"] = decode_pool.last_max_pending if stats["decode_tasks"] else 0
                stats["decode_max_reserved_bytes"] = (decode_pool.last_max_reserved_bytes
                    if stats["decode_tasks"] else 0)
                return results
            for index, record in enumerate(source_records, 1):
                token.raise_if_canceled()
                operation = "cache_lookup"
                try:
                    try:
                        with profile_scope(name + ".cache_lookup", detail="deep"):
                            valid_signature = (str(record.path) in compatible if compatible is not None else
                                               cache.compatible(record.path, name, key, connection=read_db))
                            if valid_signature:
                                value = (get(record, connection=read_db) if shared_getter else
                                         get(record))
                            else:
                                value = None
                    except Exception:
                        logger.exception("cache_read_failed folder=%s stage=%s photo=%s; recomputing",
                                         folder, name, record.path)
                        value = None  # A corrupt cache is a miss, not a corrupt photo.
                    if value is None:
                        operation = "compute"
                        with profile_scope(name + ".compute", units=1,
                                           width=record.width or 0, height=record.height or 0):
                            value = compute(record)
                        if value is None:
                            raise ValueError("No result")
                        token.raise_if_canceled()
                        operation = "cache_write"
                        with profile_scope(name + ".cache_mark", detail="deep"):
                            cache.mark(record.path, name, key)
                        stats["computed"] += 1
                    else:
                        stats["cache_hits"] += 1
                        profile_count(name + ".cache_hits")
                    results[str(record.path)] = value
                except ScanCancelledError:
                    raise
                except CacheBatchWriteError:
                    raise
                except StageUnavailable:
                    raise
                except Exception:
                    stats["file_errors"] += 1
                    logger.exception("photo_failed folder=%s stage=%s operation=%s photo=%s",
                                     folder, name, operation, record.path)
                progress(index, len(source_records))
        return results

    blur_results = {}
    def blur(stats, progress):
        nonlocal blur_results
        blur_results = per_file("blur", blur_key, services.get_blur_cached, services.analyze_blur,
            stats, progress, records)
        stats["file_errors"] += len(missing)
        rows = [dict(photoId=str(r.path), filename=r.path.name, file_size=r.file_size,
                     mtime_ns=r.mtime_ns, **{k: v if isinstance(v, str) else float(v)
                         for k, v in asdict(blur_results[str(r.path)]).items()})
                for r in records if str(r.path) in blur_results]
        for row in rows:
            decision = decisions.get(row["photoId"])
            intent = "UNDECIDED"
            if decision and (decision.file_size, decision.mtime_ns) == (row["file_size"], row["mtime_ns"]):
                intent = decision.review_state or ("KEEP" if decision.manually_kept else
                         "REMOVE" if decision.selected_for_deletion else "UNDECIDED")
            row["reviewState"] = intent if intent in {"UNDECIDED", "KEEP", "REVIEW", "REMOVE"} else "UNDECIDED"
        if not rows and paths:
            raise ValueError("No readable photos")
        return rows

    try:
        stage("blur", blur)
        valid = [r for r in records if str(r.path) in blur_results]
        photos = {}
        def duplicates(stats, progress):
            nonlocal photos
            def compute(record):
                with profile_scope("duplicates.decode", detail="deep"):
                    image = loader.load_for_scan(record.path, max_dimension=services._duplicate_config.hash_decode_dimension)
                if image is None:
                    raise ValueError("Unreadable image")
                with profile_scope("duplicates.hash_index", detail="deep"):
                    return services.index_duplicate_photo_from_image(record, image)
            photos = per_file("duplicates", duplicate_key, services.get_duplicate_photo_cached,
                              compute, stats, progress, valid)
            key = signature([fingerprint, duplicate_key, sorted(photos)])
            groups = cache.get(folder, "duplicates", key)
            if groups is not None:
                stats["aggregate_reused"] = 1
                return groups
            # Keep existing candidate/ORB/union-find semantics, bound pair storage.
            pairs = []
            with profile_scope("duplicates.candidates", units=len(photos)):
                for pair in services._duplicate_candidates.generate(list(photos.values()), cancellation_token=token):
                    if len(pairs) >= MAX_CANDIDATE_PAIRS:
                        raise StageUnavailable("Duplicate analysis exceeded its safe candidate limit. Image quality results remain available.")
                    pairs.append(pair)
            profile_count("duplicates.candidate_pairs", len(pairs))
            progress(0, 0)  # ORB has no percentage counter.
            with profile_scope("duplicates.orb_and_union", units=len(pairs)):
                groups = services.verify_duplicate_groups(list(photos.values()), pairs, cancellation_token=token)
            payload = [[str(p.path) for p in group] for group in groups]
            cache.put(folder, "duplicates", key, payload)
            return payload
        groups = stage("duplicates", duplicates)
        face_results = {}
        def faces(stats, progress):
            nonlocal face_results
            def compute(record):
                with profile_scope("faces.initialize_models"):
                    services.initialize_faces()
                token.raise_if_canceled()
                return services.analyze_faces(record, cancellation_token=token)
            face_results = per_file("faces", face_key, services.get_face_analysis_cached,
                compute, stats, progress, valid)
            if valid and not face_results:
                raise StageUnavailable("People analysis is unavailable. Check that the required face models are installed.")
            return {"photo_count": len(face_results), "face_count": sum(map(len, face_results.values()))}
        stage("faces", faces)
        def quality(stats, progress):
            for i, record in enumerate(valid, 1):
                token.raise_if_canceled()
                found = face_results.get(str(record.path))
                if found is None or any(f.analysis is None for f in found):
                    stats["file_errors"] += 1
                    progress(i, len(valid))
                    continue
                state = states[str(record.path)]
                state.people_picture_assessment = assess_people_picture(record.width, record.height, found, config)
                state.face_quality_assessment = assess_face_quality(found, state.people_picture_assessment, config)
                progress(i, len(valid))
            if valid and stats["file_errors"] == len(valid):
                raise ValueError("Quality unavailable")
            return {"photo_count": len(valid) - stats["file_errors"]}
        if "faces" in completed:
            stage("face_quality", quality)
        embedded = {}
        def embeddings(stats, progress):
            def compute(record):
                if not face_results[str(record.path)]:
                    services._face_scan_cache.put(record.path, record.file_size, record.mtime_ns, [],
                        coverage=services._face_scan_cache.COVERAGE_ALL_FACES)
                    return []
                services.initialize_faces()
                token.raise_if_canceled()
                return services.embed_faces(record, face_results[str(record.path)],
                    minimum_embedding_utility=config.faces.minimum_embedding_utility, cancellation_token=token)
            source_records = [r for r in valid if str(r.path) in face_results]
            # Keep the per-photo cache boundary, but group cache misses across
            # photos so DirectML sees a stable batch size instead of thousands
            # of different one-photo batch sizes/session variants.
            pending = []
            embedded_getter = services.get_embedded_faces_cached
            shared_getter = (getattr(embedded_getter, "__func__", None) is
                             AutomaticScanServices.get_embedded_faces_cached)
            with cache.storage.connect() as read_db:
                try:
                    with profile_scope("embeddings.cache_prefetch"):
                        compatible = cache.compatible_paths((r.path for r in source_records),
                                                            "embeddings", face_key, connection=read_db)
                except Exception:
                    logger.exception("compatibility_prefetch_failed folder=%s stage=embeddings; using individual lookups",
                                     folder)
                    compatible = None
                for record in source_records:
                    token.raise_if_canceled()
                    try:
                        with profile_scope("embeddings.cache_lookup", detail="deep"):
                            valid_signature = (str(record.path) in compatible if compatible is not None else
                                               cache.compatible(record.path, "embeddings", face_key,
                                                                connection=read_db))
                            if valid_signature:
                                value = (embedded_getter(record, connection=read_db) if shared_getter else
                                         embedded_getter(record))
                            else:
                                value = None
                    except Exception:
                        logger.exception("cache_read_failed folder=%s stage=embeddings photo=%s; recomputing",
                                         folder, record.path)
                        value = None
                    if value is None:
                        pending.append(record)
                    else:
                        embedded[str(record.path)] = value
                        stats["cache_hits"] += 1
                        profile_count("embeddings.cache_hits")
            progress(stats["cache_hits"], len(source_records))

            def store_result(record, value):
                token.raise_if_canceled()
                if value is None:
                    raise ValueError("No embedding result")
                with profile_scope("embeddings.cache_mark", detail="deep"):
                    cache.mark(record.path, "embeddings", face_key)
                embedded[str(record.path)] = value
                stats["computed"] += 1

            def photo_error(record, exc, operation):
                stats["file_errors"] += 1
                logger.error("photo_failed folder=%s stage=embeddings operation=%s photo=%s",
                             folder, operation, record.path,
                             exc_info=(type(exc), exc, exc.__traceback__))

            # Test adapters and custom service implementations may override the
            # single-photo embed method; retain their contract unchanged.
            use_batch = (type(services).embed_faces is AutomaticScanServices.embed_faces
                         and "embed_faces" not in vars(services))
            stats["batch_limit"] = services.embedding_batch_size() if use_batch else 0
            logger.info("embedding_plan folder=%s cached_photos=%d pending_photos=%d batch_limit=%d cross_photo=%s",
                        folder, stats["cache_hits"], len(pending), stats["batch_limit"], use_batch)
            if pending and use_batch:
                if any(face_results[str(r.path)] for r in pending):
                    services.initialize_faces()
                before_decode_tasks = decode_pool.submitted if decode_pool is not None else 0
                before_decode_wait = decode_pool.wait_ns if decode_pool is not None else 0
                before_decode_task = decode_pool.task_ns if decode_pool is not None else 0
                before_decode_queue = decode_pool.queue_wait_ns if decode_pool is not None else 0
                try:
                    with profile_scope("embeddings.cross_photo_batch", units=len(pending)):
                        batch_results = services.embed_faces_batched(
                            pending, face_results,
                            minimum_embedding_utility=config.faces.minimum_embedding_utility,
                            cancellation_token=token,
                            on_error=lambda record, exc: photo_error(record, exc, "compute"),
                            on_progress=lambda done, total: progress(
                                stats["cache_hits"] + done, len(source_records)),
                            **({"decode_pool": decode_pool} if decode_pool is not None else {}),
                        )
                except ScanCancelledError:
                    raise
                except Exception:
                    logger.exception("embedding_batch_failed folder=%s; retrying individual photos", folder)
                    # A native batch failure is exceptional. Retry per photo
                    # to isolate it; ordinary operation stays cross-photo.
                    batch_results = None
                if decode_pool is not None:
                    stats["decode_tasks"] = decode_pool.submitted - before_decode_tasks
                    stats["decode_wait_seconds"] = (decode_pool.wait_ns - before_decode_wait) / 1e9
                    stats["decode_task_seconds_sum"] = (decode_pool.task_ns - before_decode_task) / 1e9
                    stats["decode_queue_wait_seconds_sum"] = (decode_pool.queue_wait_ns - before_decode_queue) / 1e9
                    stats["decode_max_pending"] = (decode_pool.last_max_pending
                        if stats["decode_tasks"] else 0)
                    stats["decode_max_reserved_bytes"] = (decode_pool.last_max_reserved_bytes
                        if stats["decode_tasks"] else 0)
                if batch_results is not None:
                    for record in pending:
                        path = str(record.path)
                        if path not in batch_results:
                            continue  # The batch service already logged a per-photo error.
                        try:
                            store_result(record, batch_results[path])
                        except ScanCancelledError:
                            raise
                        except Exception as exc:
                            photo_error(record, exc, "cache_write")
            if pending and (not use_batch or batch_results is None):
                for record in pending:
                    token.raise_if_canceled()
                    try:
                        store_result(record, compute(record))
                    except ScanCancelledError:
                        raise
                    except Exception as exc:
                        photo_error(record, exc, "compute")
                    progress(stats["cache_hits"] + stats["computed"] + stats["file_errors"],
                             len(source_records))
            progress(len(source_records), len(source_records))
            if face_results and not embedded:
                raise ValueError("Embedding unavailable")
            return {"face_count": sum(map(len, embedded.values()))}
        if "face_quality" in completed:
            stage("embeddings", embeddings)
        def people(stats, progress):
            # Entire aggregate can be reused without opening ONNX sessions.
            from face_detector.cosine_similarity import CosineEmbeddingSimilarity
            from face_detector.face_database_sqlite import SQLiteFaceDatabase
            database = SQLiteFaceDatabase(services._face_database_path, CosineEmbeddingSimilarity())
            key = signature([fingerprint, face_key, database.cache_signature()])
            key = signature([key, sorted(embedded)])
            with profile_scope("people.aggregate_cache_get"):
                cached = cache.get(folder, "people", key)
            if cached is not None:
                stats["aggregate_reused"] = 1
                return cached
            known, unknown = [], []
            face_limit = unknown_face_limit()
            if any(embedded.values()):
                services.initialize_faces()
            for i, found in enumerate(embedded.values(), 1):
                token.raise_if_canceled()
                if found:
                    with profile_scope("people.recognize", units=len(found), face_count=len(found)):
                        recognized, other = services.recognize_faces(found) if config.faces.recognize_known_people else ([], found)
                    known.extend(dict(path=str(f.path), person_id=f.person_id, bbox=f.bbox) for f in recognized)
                    unknown.extend(other)
                    if len(unknown) > face_limit and config.faces.cluster_unknown_people:
                        raise StageUnavailable(
                            f"People matching has too many unknown faces ({len(unknown):,}); "
                            f"the safe limit is {face_limit:,}. Other analysis remains available."
                        )
                progress(i, len(embedded))
            logger.info("people_candidates folder=%s known_faces=%d unknown_faces=%d limit=%d",
                        folder, len(known), len(unknown), face_limit)
            stats["known_face_count"] = len(known)
            stats["unknown_face_count"] = len(unknown)
            stats["unknown_face_limit"] = face_limit
            if unknown and config.faces.cluster_unknown_people:
                progress(0, 0)  # Clustering has no per-face progress counter.
                available = _available_memory_bytes()
                matrix_bytes = len(unknown) ** 2 * 4
                stats["cluster_matrix_bytes"] = matrix_bytes
                if available is not None and matrix_bytes * 3 > available:
                    raise StageUnavailable(
                        "People matching needs more available memory to group these faces safely. "
                        "Other analysis remains available.")
                cluster_started = time.perf_counter()
                try:
                    with profile_scope("people.cluster_unknown", units=len(unknown), face_count=len(unknown)):
                        clusters = (services.cluster_unknown_faces(unknown, cancellation_token=token)
                                    if type(services) is AutomaticScanServices else
                                    services.cluster_unknown_faces(unknown))
                except ClusterCapacityError as exc:
                    raise StageUnavailable(
                        "People matching reached the face-grouping memory safety limit. "
                        "Other analysis remains available.") from exc
                finally:
                    stats["cluster_seconds"] = time.perf_counter() - cluster_started
            else:
                clusters = []
            stats["unknown_cluster_count"] = len(clusters)
            stats["largest_unknown_cluster"] = max((len(c.faces) for c in clusters), default=0)
            payload = {"known": known, "unknown": [dict(id=c.id,
                paths=sorted({str(f.path) for f in c.faces}), representative=str(c.representative.path),
                representative_bbox=c.representative.bbox, face_count=len(c.faces),
                faces=[dict(path=str(f.path), bbox=f.bbox) for f in c.faces]) for c in clusters]}
            cache.put(folder, "people", key, payload)
            return payload
        if "embeddings" in completed:
            stage("people", people)
        def ranking(stats, progress):
            if groups is None:
                raise ValueError("Duplicate groups unavailable")
            key = signature(["ranking-v1", fingerprint, face_key, duplicate_key, config.cache_signature(),
                             "face_quality" in completed, sorted(face_results)])
            value = cache.get(folder, "ranking", key)
            if value is None:
                grouped = [[photos[p] for p in group] for group in groups]
                with profile_scope("ranking.select_keepers", units=len(grouped)):
                    value = [asdict(g) for g in select_duplicate_keepers(grouped, image_states=states, config=config)]
                cache.put(folder, "ranking", key, value)
            else:
                stats["aggregate_reused"] = 1
            return value
        stage("ranking", ranking)
        for name in STAGES:
            if name not in diagnostics:
                logger.info("stage_skipped folder=%s stage=%s reason=earlier_required_stage", folder, name)
                emit(name, StageState.CANCELLED, 0, 0, "An earlier required stage could not finish.", None, {})
    except _BenchmarkStop:
        pass
    except ScanCancelledError:
        logger.info("project_cancelled folder=%s completed_stages=%s", folder, ",".join(sorted(completed)))
        for name in STAGES:
            if name not in completed:
                emit(name, StageState.CANCELLED, 0, 0, "", None, {})
    except CacheBatchWriteError:
        logger.exception("project_cache_batch_failed folder=%s", folder)
        for name in STAGES:
            if name not in diagnostics:
                emit(name, StageState.CANCELLED, 0, 0,
                     "Analysis stopped because the cache could not be saved.", None, {})
    finally:
        if decode_pool is not None:
            decode_pool.close()
        if cache_session is not None:
            cache_session.__exit__(None, None, None)
    logger.info("project_finished folder=%s completed_stages=%s failed_stages=%s missing_photos=%d",
                folder, ",".join(sorted(completed)),
                ",".join(name for name, stats in diagnostics.items() if "error_type" in stats), len(missing))
    return diagnostics
