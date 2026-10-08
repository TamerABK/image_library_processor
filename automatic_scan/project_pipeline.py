"""Bounded progressive execution for AutomaticScanOrchestrator's Qt entry point.

No project discovery, Qt objects, review execution or additional worker pool.
The caller supplies the discovered identities and one cancellation token.
"""
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
import time

from app_paths import model_path
from duplicate_detector.config import DetectorConfig
from image_loader import ImageLoader
from scan_controls import ScanCancelledError
from .config import AutomaticScanConfig, BlurPolicy, PerformanceConfig
from .duplicate_selection import select_duplicate_keepers
from .eligibility import assess_face_quality, assess_people_picture
from .models import DiscoveryRecord, PipelineImageState
from .project_cache import ProjectStageCache, signature
from .services import AutomaticScanServices
from .scan_logging import get_scan_logger


class StageState(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


STAGES = ("blur", "duplicates", "faces", "face_quality", "embeddings", "people", "ranking")
MAX_CANDIDATE_PAIRS = 50000
MAX_UNKNOWN_FACES = 20000  # Temporary test cap; clustering still builds a quadratic similarity matrix.
MAX_PROJECT_FACES = 20000


class StageUnavailable(RuntimeError):
    """User-safe resource/model failure, distinct from internal exceptions."""


def process_project(orchestrator, folder, paths, *, emit, cancellation_token,
                    cache_path=None, config=None, services=None):
    logger = get_scan_logger()
    logger.info("project_started folder=%s discovered_paths=%d", folder, len(paths))
    config = config or AutomaticScanConfig(blur_policy=BlurPolicy.KEEP_ALL,
        performance=PerformanceConfig(worker_count=1, batch_size=8))
    folder = str(folder)
    token = cancellation_token
    cache = ProjectStageCache(cache_path)
    services = services or AutomaticScanServices(config, lazy_models=True,
        cache_path=cache_path, duplicate_config=DetectorConfig(max_workers=1, cache_size=8))
    records, missing = [], []
    loader = ImageLoader()
    for path in paths:
        token.raise_if_canceled()
        path = Path(path)
        try:
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
        decisions = orchestrator._state_store.load(folder).image_decisions
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("review_state_load_failed folder=%s", folder)
        decisions = {}

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
            data = work(stats, progress)
            token.raise_if_canceled()
            stats["seconds"] = round(time.perf_counter() - started, 4)
            diagnostics[name] = stats
            completed.add(name)
            logger.info("stage_completed folder=%s stage=%s seconds=%.4f cache_hits=%d computed=%d file_errors=%d aggregate_reused=%d",
                        folder, name, stats["seconds"], stats["cache_hits"], stats["computed"],
                        stats["file_errors"], stats.get("aggregate_reused", 0))
            message = (f"{stats['file_errors']} photos could not be analyzed." if stats["file_errors"] else "")
            emit(name, StageState.COMPLETE, 0, 0, message, data, stats)
            return data
        except ScanCancelledError:
            logger.info("stage_cancelled folder=%s stage=%s", folder, name)
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
        face_count = 0
        for index, record in enumerate(source_records, 1):
            token.raise_if_canceled()
            operation = "cache_lookup"
            try:
                try:
                    value = get(record) if cache.compatible(record.path, name, key) else None
                except Exception:
                    logger.exception("cache_read_failed folder=%s stage=%s photo=%s; recomputing",
                                     folder, name, record.path)
                    value = None  # A corrupt cache is a miss, not a corrupt photo.
                if value is None:
                    operation = "compute"
                    value = compute(record)
                    if value is None:
                        raise ValueError("No result")
                    token.raise_if_canceled()
                    operation = "cache_write"
                    cache.mark(record.path, name, key)
                    stats["computed"] += 1
                else:
                    stats["cache_hits"] += 1
                if name in ("faces", "embeddings"):
                    face_count += len(value)
                    if face_count > MAX_PROJECT_FACES:
                        raise StageUnavailable("People analysis exceeded its safe face limit. Image quality results remain available.")
                results[str(record.path)] = value
            except ScanCancelledError:
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
                image = loader.load_for_scan(record.path, max_dimension=services._duplicate_config.hash_decode_dimension)
                if image is None:
                    raise ValueError("Unreadable image")
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
            for pair in services._duplicate_candidates.generate(list(photos.values()), cancellation_token=token):
                if len(pairs) >= MAX_CANDIDATE_PAIRS:
                    raise StageUnavailable("Duplicate analysis exceeded its safe candidate limit. Image quality results remain available.")
                pairs.append(pair)
            progress(0, 0)  # ORB has no percentage counter.
            groups = services.verify_duplicate_groups(list(photos.values()), pairs, cancellation_token=token)
            payload = [[str(p.path) for p in group] for group in groups]
            cache.put(folder, "duplicates", key, payload)
            return payload
        groups = stage("duplicates", duplicates)
        face_results = {}
        def faces(stats, progress):
            nonlocal face_results
            def compute(record):
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
            embedded.update(per_file("embeddings", face_key, services.get_embedded_faces_cached,
                compute, stats, progress, [r for r in valid if str(r.path) in face_results]))
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
            cached = cache.get(folder, "people", key)
            if cached is not None:
                stats["aggregate_reused"] = 1
                return cached
            known, unknown = [], []
            if any(embedded.values()):
                services.initialize_faces()
            for i, found in enumerate(embedded.values(), 1):
                token.raise_if_canceled()
                if found:
                    recognized, other = services.recognize_faces(found) if config.faces.recognize_known_people else ([], found)
                    known.extend(dict(path=str(f.path), person_id=f.person_id, bbox=f.bbox) for f in recognized)
                    unknown.extend(other)
                progress(i, len(embedded))
            logger.info("people_candidates folder=%s known_faces=%d unknown_faces=%d limit=%d",
                        folder, len(known), len(unknown), MAX_UNKNOWN_FACES)
            if len(unknown) > MAX_UNKNOWN_FACES:
                raise StageUnavailable("People grouping exceeded its safe face limit. Image quality results remain available.")
            progress(0, 0)
            clusters = services.cluster_unknown_faces(unknown) if unknown and config.faces.cluster_unknown_people else []
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
    except ScanCancelledError:
        logger.info("project_cancelled folder=%s completed_stages=%s", folder, ",".join(sorted(completed)))
        for name in STAGES:
            if name not in completed:
                emit(name, StageState.CANCELLED, 0, 0, "", None, {})
    logger.info("project_finished folder=%s completed_stages=%s failed_stages=%s missing_photos=%d",
                folder, ",".join(sorted(completed)),
                ",".join(name for name, stats in diagnostics.items() if "error_type" in stats), len(missing))
    return diagnostics
