"""Read-only source-photo audit of the production face and keeper ranking path.

The only databases opened for writing live below --output. A copied analysis
cache is an optional input; source photographs and application state are never
written. See docs/PHASE_8G_AUDIT.md for the CSV contract and labeling workflow.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from automatic_scan.config import AutomaticScanConfig, ExecutionProvider, PerformanceConfig
from automatic_scan.duplicate_selection import _ranking_sort_key, select_duplicate_keepers
from automatic_scan.eligibility import assess_face_quality, assess_people_picture
from automatic_scan.models import DiscoveryRecord, PipelineImageState
from automatic_scan.services import AutomaticScanServices
from automatic_scan.keeper_ranker_v2 import (FACE_WEIGHTS, VERSION as V2_VERSION,
    Face as V2Face, Feature as V2Feature, Photo as V2Photo,
    Preferences as V2Preferences, rank_group as rank_v2_group)
from automatic_scan.keeper_ranker_v2_adapter import photo_input as v2_photo_input, extract_faces, technical_feature, METRICS
from duplicate_detector.models import PhotoInfo
from face_analyzer.config import FaceAnalyzerConfig
from face_analyzer.ranking import describe_distribution
from image_loader import ImageLoader


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RANK_FIELDS = (
    "acceptable_primary_face_count", "worst_primary_score", "weighted_primary_average",
    "eye_open_score", "pose_visibility_score", "image_sharpness_score",
    "exposure_contrast_score", "overall_quality_score", "group_visual_centrality",
    "capture_timestamp", "path_key",
)
PHOTO_FIELDS = (
    "image_path", "filename", "file_size", "width", "height", "capture_timestamp",
    "face_count", "analyzed_face_count", "is_people_picture", "people_confidence",
    "primary_face_count", "acceptable_primary_face_count", "closed_eye_primary_count",
    "unusable_primary_count", "worst_primary_score", "weighted_primary_average",
    "best_primary_score", "terminal_failure", "quality_warnings", "quality_reasons",
    "face_cache_hit", "duplicate_index_available", "error",
    "technical_blur_score", "technical_raw_laplacian", "technical_raw_sobel",
    "technical_raw_local_contrast", "technical_v2_score", "technical_v2_confidence",
)
FACE_EXTRA_FIELDS = (
    "image_path", "face_index", "is_primary", "bbox_x", "bbox_y", "bbox_width",
    "bbox_height", "detector_confidence_observed", "analysis_missing", "quality_warnings",
)
RANK_EXTRA_FIELDS = (
    "group_id", "candidate_image_id", "final_position", "selected_keeper",
    "first_deciding_field_vs_winner", "first_field_margin_vs_winner",
    "missing_face_analysis", "human_preference_agreement", "scenario",
)
PAIR_FIELDS = (
    "group_id", "image_a", "image_b", "system_preference", "human_preference",
    "agreement", "first_deciding_field", "first_field_margin", "face_count_difference",
    "worst_primary_score_difference", "weighted_primary_average_difference", "scenario",
)
SUMMARY_FIELDS = ("scenario", "group_count", "candidate_count", "pair_count",
    "labeled_pair_count", "pairwise_accuracy", "top1_labeled_groups", "top1_agreement",
    "top2_labeled_groups", "top2_agreement", "missing_face_analysis_rate",
    "terminal_failure_rate", "missing_eye_measurement_rate", "missing_pose_measurement_rate",
    "low_reliability_face_rate", "mean_global_face_score", "mean_worst_primary_score",
    "p10_worst_primary_score", "median_worst_primary_score", "p90_worst_primary_score")


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, (list, tuple)):
            result[name] = "|".join(str(part) for part in item)
        else:
            result[name] = getattr(item, "value", item)
    return result


def _csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _dataset_paths(root: Path, limit: int | None) -> list[Path]:
    paths = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    return paths if limit is None else paths[:limit]


def _signature(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def _copy_cache(source: Path, destination: Path) -> None:
    uri = source.resolve().as_uri().replace("file://", "file:") + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as read_db, sqlite3.connect(destination) as write_db:
        read_db.backup(write_db)


def _model_fingerprints() -> dict[str, dict[str, Any]]:
    from app_paths import model_path
    result = {}
    for name in ("scrfd_10g_bnkps.onnx", "sixdrepnet.onnx", "open_closed_eye.onnx"):
        path = model_path(name)
        if not path.is_file():
            result[name] = {"available": False}
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = {"available": True, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return result


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                   cwd=Path(__file__).resolve().parents[2]).strip()


def _production_unchanged_since(source_commit: str) -> bool:
    repository = Path(__file__).resolve().parents[2]
    try:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", source_commit, "HEAD"], text=True,
            cwd=repository).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    diagnostic_paths = {"app/tools/audit_face_quality.py", "app/tools/test_audit_face_quality.py",
                        "docs/PHASE_8G_AUDIT.md"}
    return set(changed).issubset(diagnostic_paths)


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "opencv-python", "Pillow", "onnxruntime", "onnxruntime-directml", "PySide6"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    physical = total_ram = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        total_ram = psutil.virtual_memory().total
    except (ImportError, OSError):
        pass
    return dict(operating_system=platform.platform(), processor=platform.processor() or
                os.environ.get("PROCESSOR_IDENTIFIER", ""), logical_processors=os.cpu_count(),
                physical_cores=physical, system_ram_bytes=total_ram, dependency_versions=packages)


def _record(path: Path, loader: ImageLoader) -> DiscoveryRecord:
    stat = path.stat()
    metadata = loader.read_metadata(path)
    return DiscoveryRecord(path, stat.st_size, stat.st_mtime_ns, path.suffix.lower(),
                           metadata.orientation if metadata else None,
                           metadata.capture_timestamp if metadata and metadata.capture_timestamp else stat.st_mtime_ns / 1e9,
                           metadata.width if metadata else None, metadata.height if metadata else None)


def _face_rows(path: Path, found: list, primary_indices: tuple[int, ...]) -> list[dict]:
    rows = []
    for index, face in enumerate(found):
        row = {key: "" for key in FACE_EXTRA_FIELDS}
        row.update(image_path=str(path), face_index=index, is_primary=int(index in primary_indices),
                   bbox_x=face.bbox[0], bbox_y=face.bbox[1], bbox_width=face.bbox[2],
                   bbox_height=face.bbox[3], detector_confidence_observed=face.confidence,
                   analysis_missing=int(face.analysis is None))
        if face.analysis is not None:
            row.update(_flatten(face.analysis))
            row["quality_warnings"] = "|".join(face.analysis.warnings)
        rows.append(row)
    return rows


def _photo_row(record: DiscoveryRecord, found: list, people, quality, *, cache_hit: bool,
               duplicate_available: bool, error: str = "") -> dict:
    return dict(image_path=str(record.path), filename=record.path.name, file_size=record.file_size,
                width=record.width, height=record.height, capture_timestamp=record.capture_timestamp,
                face_count=len(found), analyzed_face_count=sum(face.analysis is not None for face in found),
                is_people_picture=int(people.is_people_picture) if people else "",
                people_confidence=people.confidence if people else "",
                primary_face_count=quality.primary_face_count if quality else "",
                acceptable_primary_face_count=quality.acceptable_primary_face_count if quality else "",
                closed_eye_primary_count=quality.closed_eye_primary_count if quality else "",
                unusable_primary_count=quality.unusable_primary_count if quality else "",
                worst_primary_score=quality.worst_primary_score if quality else "",
                weighted_primary_average=quality.weighted_primary_average if quality else "",
                best_primary_score=quality.best_primary_score if quality else "",
                terminal_failure=int(quality.terminal_failure) if quality else "",
                quality_warnings="|".join(quality.warnings) if quality else "",
                quality_reasons="|".join(quality.reasons) if quality else "",
                face_cache_hit=int(cache_hit), duplicate_index_available=int(duplicate_available), error=error)


def _first_difference(winner, other) -> tuple[str, Any]:
    a, b = _ranking_sort_key(winner), _ranking_sort_key(other)
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return RANK_FIELDS[index], left - right if isinstance(left, (int, float)) else ""
    return "tie", 0


def _group_map(rows: list[dict[str, str]], root: Path, available: dict[str, PhotoInfo]) -> list[list[PhotoInfo]]:
    grouped: dict[str, list[PhotoInfo]] = {}
    for row in rows:
        image = Path(row["image_path"])
        if not image.is_absolute():
            image = root / image
        path = str(image.resolve())
        if path not in available:
            raise ValueError(f"Group map references an unavailable photograph: {path}")
        grouped.setdefault(row["group_id"], []).append(available[path])
    return [list(dict.fromkeys(group)) for _, group in sorted(grouped.items()) if len(set(group)) > 1]


def _labels(path: Path | None, root: Path) -> dict[tuple[str, str], str]:
    labels = {}
    for row in _read_csv(path):
        a, b = (Path(row[key]) for key in ("image_a", "image_b"))
        preferred = row["preferred_image"]
        a = str((root / a).resolve()) if not a.is_absolute() else str(a.resolve())
        b = str((root / b).resolve()) if not b.is_absolute() else str(b.resolve())
        if preferred.lower() in {"tie", "equal", "uncertain"}:
            choice = preferred.lower()
        else:
            p = Path(preferred)
            choice = str((root / p).resolve()) if not p.is_absolute() else str(p.resolve())
            if choice not in {a, b}:
                raise ValueError(f"Invalid preferred_image for pair {a}, {b}")
        labels[tuple(sorted((a, b)))] = choice
    return labels


def _contact_sheet(group_id: str, paths: list[str], directory: Path) -> None:
    tile_w, tile_h = 360, 280
    cols = min(3, len(paths))
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        x, y = (index % cols) * tile_w, (index // cols) * tile_h
        try:
            with Image.open(path) as source:
                photo = ImageOps.exif_transpose(source).convert("RGB")
                photo.thumbnail((tile_w - 16, tile_h - 56))
                canvas.paste(photo, (x + (tile_w - photo.width) // 2, y + 8))
        except (OSError, ValueError):
            draw.text((x + 8, y + 8), "Preview unavailable", fill="black")
        draw.text((x + 8, y + tile_h - 40), f"{index + 1}: {Path(path).name[:40]}", fill="black")
    canvas.save(directory / f"{group_id}.jpg", quality=85)


def _summaries(photo_rows: list[dict], ranking_rows: list[dict], pair_rows: list[dict],
               face_rows: list[dict] = ()) -> list[dict]:
    result = []
    for scenario in ("all", "single_person", "group_photo", "no_primary_face"):
        candidates = [r for r in ranking_rows if scenario == "all" or r["scenario"] == scenario]
        group_ids = {r["group_id"] for r in candidates}
        pairs = [r for r in pair_rows if r["group_id"] in group_ids]
        labeled = [r for r in pairs if r["agreement"] != ""]
        photos = [r for r in photo_rows if scenario == "all" or
                  (scenario == "single_person" and r["primary_face_count"] == 1) or
                  (scenario == "group_photo" and isinstance(r["primary_face_count"], int) and r["primary_face_count"] > 1) or
                  (scenario == "no_primary_face" and r["primary_face_count"] == 0)]
        scores = [float(r["worst_primary_score"]) for r in photos if r["worst_primary_score"] not in ("", None)]
        photo_paths = {r["image_path"] for r in photos}
        faces = [r for r in face_rows if r["image_path"] in photo_paths]
        face_scores = [float(r["global_selection_score"]) for r in faces
                       if r.get("global_selection_score") not in ("", None)]
        group_labels = {}
        for row in pairs:
            if row["human_preference"] not in ("", "tie", "equal", "uncertain"):
                group_labels.setdefault(row["group_id"], []).append(row)
        top1 = []
        top2 = []
        for group_id, decisions in group_labels.items():
            ranked = sorted((r for r in candidates if r["group_id"] == group_id), key=lambda r: r["final_position"])
            votes: dict[str, int] = {r["candidate_image_id"]: 0 for r in ranked}
            for decision in decisions:
                preferred = decision["human_preference"]
                votes[preferred] = votes.get(preferred, 0) + 1
            if len(votes) < 2 or list(votes.values()).count(max(votes.values())) != 1:
                continue
            preferred = max(votes, key=votes.get)
            if ranked:
                top1.append(preferred == ranked[0]["candidate_image_id"])
                top2.append(preferred in {r["candidate_image_id"] for r in ranked[:2]})
        result.append(dict(scenario=scenario, group_count=len(group_ids), candidate_count=len(candidates),
            pair_count=len(pairs), labeled_pair_count=len(labeled),
            pairwise_accuracy=sum(int(r["agreement"]) for r in labeled) / len(labeled) if labeled else "",
            top1_labeled_groups=len(top1), top1_agreement=sum(top1) / len(top1) if top1 else "",
            top2_labeled_groups=len(top2), top2_agreement=sum(top2) / len(top2) if top2 else "",
            missing_face_analysis_rate=sum(bool(r["error"]) or r["face_count"] != r["analyzed_face_count"] for r in photos) / len(photos) if photos else "",
            terminal_failure_rate=sum(r["terminal_failure"] == 1 for r in photos) / len(photos) if photos else "",
            missing_eye_measurement_rate=sum(r.get("eye_state_left_open_probability") in ("", None)
                or r.get("eye_state_right_open_probability") in ("", None) for r in faces) / len(faces) if faces else "",
            missing_pose_measurement_rate=sum(r.get("head_pose_yaw_degrees") in ("", None)
                for r in faces) / len(faces) if faces else "",
            low_reliability_face_rate=sum(float(r.get("measurement_reliability_quality_score") or 0) < .36
                for r in faces) / len(faces) if faces else "",
            mean_global_face_score=sum(face_scores) / len(face_scores) if face_scores else "",
            mean_worst_primary_score=sum(scores) / len(scores) if scores else "",
            p10_worst_primary_score=describe_distribution(scores)["p10"] if scores else "",
            median_worst_primary_score=describe_distribution(scores)["median"] if scores else "",
            p90_worst_primary_score=describe_distribution(scores)["p90"] if scores else ""))
    return result


def run(args: argparse.Namespace) -> Path:
    root, output = Path(args.input).resolve(), Path(args.output).resolve()
    if not root.is_dir():
        raise ValueError("--input must name an existing photo directory")
    if output == root or root in output.parents:
        raise ValueError("--output must be outside the source-photo directory")
    output.mkdir(parents=True, exist_ok=True)
    paths = _dataset_paths(root, args.max_images)
    if not paths:
        raise ValueError("No supported photographs found")
    signature = _signature(paths)
    config = AutomaticScanConfig(performance=PerformanceConfig(execution_provider=ExecutionProvider(args.provider)))
    analyzer_config = FaceAnalyzerConfig()
    model_fingerprints = _model_fingerprints()
    cache_path = output / "audit_cache.sqlite3"
    if args.cache_source and not cache_path.exists():
        _copy_cache(Path(args.cache_source), cache_path)
    tool_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (prior["dataset_signature"] != signature or prior["git_commit"] != _git_sha()
                or prior["provider_requested"] != args.provider
                or prior.get("audit_tool_sha256") != tool_sha):
            raise ValueError("Existing audit output belongs to a different dataset, revision, provider, or audit tool")
    cache_source_manifest = None
    if args.cache_source:
        source_manifest_path = Path(args.cache_source).resolve().parent / "manifest.json"
        if source_manifest_path.is_file():
            try:
                cache_source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
    checkpoint_source_manifest = None
    checkpoint = output / "image_checkpoint.jsonl"
    if args.checkpoint_source:
        source_checkpoint = Path(args.checkpoint_source).resolve()
        source_manifest_path = source_checkpoint.parent / "manifest.json"
        checkpoint_source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_commit = checkpoint_source_manifest.get("git_commit", "")
        if (checkpoint_source_manifest.get("dataset_signature") != signature
                or not (source_commit == _git_sha() or _production_unchanged_since(source_commit))
                or checkpoint_source_manifest.get("models") != model_fingerprints):
            raise ValueError("Checkpoint source has a different dataset, code revision, or model set")
        if not checkpoint.exists():
            shutil.copy2(source_checkpoint, checkpoint)
    manifest = dict(dataset_root=str(root), dataset_signature=signature, dataset_image_count=len(paths),
        git_commit=_git_sha(), audit_tool_sha256=tool_sha,
        started_at=datetime.now(timezone.utc).isoformat(),
        provider_requested=args.provider, mode=args.mode, python=sys.version,
        environment=_environment(), config=_flatten(config),
        face_calibration_profile=analyzer_config.calibration_profile.to_dict(),
        face_ranking_weights=analyzer_config.ranking_weights.as_dict(),
        face_metric_priors=asdict(analyzer_config.metric_priors),
        models=model_fingerprints, cache_source=str(args.cache_source or ""),
        checkpoint_source=str(args.checkpoint_source or ""),
        measurement_origin=("imported fresh checkpoint" if checkpoint_source_manifest and
            checkpoint_source_manifest.get("mode") == "full" and checkpoint_source_manifest.get("cache_hits") == 0
            else "imported cached checkpoint" if checkpoint_source_manifest else args.mode),
        checkpoint_source_detector_runtime=checkpoint_source_manifest.get("detector_runtime") if checkpoint_source_manifest else None,
        cache_source_git_commit=cache_source_manifest.get("git_commit") if cache_source_manifest else None,
        cache_source_detector_runtime=cache_source_manifest.get("detector_runtime") if cache_source_manifest else None,
        cache_isolation=("SQLite backup into output directory" if args.cache_source else "fresh output-local SQLite database"),
        cache_compatibility=("face cache checks source size/mtime; model provenance is unverified"
                             if args.cache_source and not cache_source_manifest else
                             "source manifest supplied; inspect model hashes and revision before comparing" if args.cache_source else
                             "fresh analysis"),
        group_method="explicit CSV" if args.groups_csv else "production hash candidate and ORB verification",
        label_method="optional pairwise CSV; no labels are assumed", status="running")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    saved = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            saved[item["image_path"]] = item
    loader = ImageLoader()
    services = AutomaticScanServices(config, lazy_models=True, cache_path=cache_path,
                                     face_database_path=output / "audit_faces.sqlite3")
    photo_rows, face_rows, photo_info, states = [], [], {}, {}
    with checkpoint.open("a", encoding="utf-8") as progress:
        for number, path in enumerate(paths, 1):
            item = saved.get(str(path))
            if item is None:
                record = _record(path, loader)
                found, indexed, error = [], None, ""
                cache_hit = False
                try:
                    found = services.get_face_analysis_cached(record)
                    cache_hit = found is not None
                    if found is None:
                        if args.mode == "cached":
                            raise ValueError("No compatible cached face analysis")
                        services.initialize_faces()
                        found = services.analyze_faces(record)
                    indexed = services.get_duplicate_photo_cached(record)
                    if indexed is None:
                        indexed = services.index_duplicate_photo(record)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                people = quality = None
                if found and all(face.analysis is not None for face in found):
                    people = assess_people_picture(record.width, record.height, found, config)
                    quality = assess_face_quality(found, people, config)
                elif not found and not error:
                    people = assess_people_picture(record.width, record.height, [], config)
                    quality = assess_face_quality([], people, config)
                face_data = _face_rows(path, found, people.primary_face_indices if people else ())
                photo_data = _photo_row(record, found, people, quality, cache_hit=cache_hit,
                                        duplicate_available=indexed is not None, error=error)
                blur_getter = getattr(services, "get_blur_cached", None)
                cached_blur = blur_getter(record) if blur_getter else None
                if cached_blur is None and getattr(args, "include_blur", False) and args.mode == "full":
                    cached_blur = services.analyze_blur(record)
                photo_data["technical_blur_score"] = (cached_blur.final_score if cached_blur else "")
                for key in ("laplacian", "sobel", "local_contrast"):
                    photo_data["technical_raw_" + key] = getattr(cached_blur, key) if cached_blur else ""
                item = dict(image_path=str(path), face_rows=face_data, photo_row=photo_data,
                            photo_info=({key: str(value) if key == "path" else value for key, value in asdict(indexed).items()}
                                        if indexed else None))
                progress.write(json.dumps(item, default=str) + "\n")
                progress.flush()
            photo_rows.append(item["photo_row"])
            face_rows.extend(item["face_rows"])
            if item["photo_info"]:
                info = dict(item["photo_info"])
                info["path"] = Path(info["path"])
                indexed = PhotoInfo(**info)
                photo_info[str(path)] = indexed
            record = _record(path, loader)
            state = PipelineImageState(path=str(path), file_size=record.file_size, mtime_ns=record.mtime_ns,
                extension=record.extension, orientation=record.orientation,
                capture_timestamp=record.capture_timestamp, width=record.width, height=record.height)
            # Reconstruct only the aggregate assessment needed by the production sorter.
            from automatic_scan.models import ImageFaceQualityAssessment
            p = item["photo_row"]
            if p["is_people_picture"] != "":
                state.face_quality_assessment = ImageFaceQualityAssessment(
                    bool(p["is_people_picture"]), int(p["primary_face_count"]),
                    int(p["acceptable_primary_face_count"]), p["weighted_primary_average"],
                    p["worst_primary_score"], p["best_primary_score"],
                    int(p["closed_eye_primary_count"]), int(p["unusable_primary_count"]),
                    bool(p["terminal_failure"]), tuple(filter(None, p["quality_warnings"].split("|"))),
                    tuple(filter(None, p["quality_reasons"].split("|"))))
            states[str(path)] = state
            if number % 25 == 0 or number == len(paths):
                print(f"Analyzed {number}/{len(paths)} photographs", flush=True)
    if args.groups_csv:
        groups = _group_map(_read_csv(Path(args.groups_csv)), root, photo_info)
    elif photo_info:
        indexed = list(photo_info.values())
        groups = services.verify_duplicate_groups(indexed, services.generate_duplicate_candidates(indexed))
    else:
        groups = []
    decisions = select_duplicate_keepers(groups, image_states=states, config=config)
    face_by_path = {}
    for row in face_rows:
        face_by_path.setdefault(row["image_path"], []).append(row)
    photo_by_path = {row["image_path"]: row for row in photo_rows}
    v2_features, v2_comparison, v2_summary = [], [], {}
    metric_columns = METRICS
    def number(value):
        try:
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None
    for prefer_faces, prefer_eyes in itertools.product((True, False), repeat=2):
        prefs = V2Preferences(prefer_faces, prefer_eyes)
        preference_id = f"face_{int(prefer_faces)}_eyes_{int(prefer_eyes)}"
        disagreements = uncertain = near_ties = 0
        for group in decisions:
            candidates = []
            for path in group.image_paths:
                rows = face_by_path.get(path, [])
                photo_row = photo_by_path[path]
                raw = {key: number(photo_row.get("technical_raw_" + key))
                       for key in ("laplacian", "sobel", "local_contrast")}
                if any(value is None for value in raw.values()):
                    record = _record(Path(path), loader)
                    blur_getter = getattr(services, "get_blur_cached", None)
                    cached_blur = blur_getter(record) if blur_getter else None
                    if cached_blur is not None:
                        raw = {key: getattr(cached_blur, key) for key in raw}
                for key, value in raw.items():
                    photo_row["technical_raw_" + key] = value if value is not None else ""
                technical_v2 = technical_feature(raw)
                photo_row["technical_v2_score"] = (technical_v2.value if technical_v2.value is not None else "")
                photo_row["technical_v2_confidence"] = technical_v2.confidence
                candidates.append(v2_photo_input(path, rows, raw,
                    number(photo_row.get("width")), number(photo_row.get("height")), rows=True))
            ranked = rank_v2_group(tuple(candidates), prefs)
            uncertain += int(any(score.uncertain for score in ranked))
            disagreements += int(ranked[0].image_id != group.image_paths[0])
            margin = (ranked[0].score - ranked[1].score
                      if len(ranked) > 1 and ranked[0].score is not None
                      and ranked[1].score is not None else None)
            near_ties += int(ranked[0].tie_class in ("exact", "below_resolution", "near"))
            for position, score in enumerate(ranked, 1):
                v2_features.append({**asdict(score), "group_id": group.group_id,
                                    "preference_id": preference_id, "v2_position": position,
                                    "eye_included": int("eyes" in score.used_face_features),
                                    "moment_available": 0, "composition_available": 0,
                                    "ranking_version": V2_VERSION})
            v2_comparison.append(dict(group_id=group.group_id, preference_id=preference_id,
                v1_winner=group.image_paths[0], v2_winner=ranked[0].image_id,
                agreement=int(group.image_paths[0] == ranked[0].image_id),
                candidate_ids="|".join(group.image_paths),
                v1_positions="|".join(str(group.image_paths.index(path) + 1) for path in group.image_paths),
                v2_positions="|".join(str(next(i for i, s in enumerate(ranked, 1)
                                             if s.image_id == path)) for path in group.image_paths),
                v2_margin=margin if margin is not None else "",
                near_tie=int(ranked[0].tie_class in ("exact", "below_resolution", "near")),
                tie_class=ranked[0].tie_class,
                quality_preference=int(ranked[0].quality_preference),
                technical_discriminative=int(ranked[0].technical_discriminative),
                face_evidence_incomplete=int(any(s.face_evidence_incomplete for s in ranked)),
                quality_comparison_available=int(all(s.score is not None for s in ranked)),
                key_features="|".join(ranked[0].used_face_features) + ("|technical" if ranked[0].technical_score is not None else ""),
                uncertain=int(any(score.uncertain for score in ranked))))
        v2_summary[preference_id] = dict(groups=len(decisions), winner_disagreements=disagreements,
                                         uncertain_groups=uncertain, near_ties=near_ties)
    _csv(output / "ranking_v2_features.csv", ["group_id", "preference_id", "v2_position",
        "eye_included", "moment_available", "composition_available", "ranking_version"]
        + list(asdict(rank_v2_group((V2Photo("schema", (), V2Feature(None)),))[0]).keys()), v2_features)
    _csv(output / "ranking_v1_v2_comparison.csv", ["group_id", "preference_id", "v1_winner",
        "v2_winner", "agreement", "candidate_ids", "v1_positions", "v2_positions",
        "v2_margin", "near_tie", "tie_class", "quality_preference", "technical_discriminative",
        "face_evidence_incomplete", "quality_comparison_available", "key_features", "uncertain"], v2_comparison)
    (output / "ranking_v2_summary.json").write_text(json.dumps({"version": V2_VERSION,
        "preferences": v2_summary, "technical_source": (
            "existing blur stage in isolated full audit" if getattr(args, "include_blur", False)
            else "compatible blur cache only"),
        "moment_available": False, "composition_available": False}, indent=2), encoding="utf-8")
    human_labels = _labels(Path(args.labels_csv) if args.labels_csv else None, root)
    photos_by_path = {r["image_path"]: r for r in photo_rows}
    ranking_rows, pair_rows, annotation_rows = [], [], []
    sheet_dir = output / "contact_sheets"
    if args.contact_sheets:
        sheet_dir.mkdir(exist_ok=True)
    for group in decisions:
        paths_in_group = group.image_paths
        scenario = "group_photo" if any(photos_by_path[p]["primary_face_count"] not in ("", 0, 1) for p in paths_in_group) else (
            "single_person" if any(photos_by_path[p]["primary_face_count"] == 1 for p in paths_in_group) else "no_primary_face")
        winner = group.ranking[paths_in_group[0]]
        if args.contact_sheets:
            _contact_sheet(group.group_id, paths_in_group, sheet_dir)
        for position, path in enumerate(paths_in_group, 1):
            breakdown = group.ranking[path]
            field, margin = _first_difference(winner, breakdown)
            matched_labels = [
                (pair[0] if paths_in_group.index(pair[0]) < paths_in_group.index(pair[1]) else pair[1]) == value
                for pair, value in human_labels.items()
                if path in pair and pair[0] in paths_in_group and pair[1] in paths_in_group
                and value not in {"tie", "equal", "uncertain"}
            ]
            ranking_rows.append({**asdict(breakdown), "group_id": group.group_id,
                "candidate_image_id": path, "final_position": position,
                "selected_keeper": int(path in group.keeper_paths),
                "first_deciding_field_vs_winner": field if position > 1 else "",
                "first_field_margin_vs_winner": margin if position > 1 else "",
                "missing_face_analysis": int(photos_by_path[path]["face_count"] != photos_by_path[path]["analyzed_face_count"]),
                "human_preference_agreement": sum(matched_labels) / len(matched_labels) if matched_labels else "",
                "scenario": scenario})
            annotation_rows.append(dict(group_id=group.group_id, image_path=path,
                preferred_keeper="", equally_acceptable="", unusable="", uncertain="", reasons=""))
        for a, b in itertools.combinations(paths_in_group, 2):
            preferred = human_labels.get(tuple(sorted((a, b))), "")
            field, margin = _first_difference(group.ranking[a], group.ranking[b])
            system = a if paths_in_group.index(a) < paths_in_group.index(b) else b
            pair_rows.append(dict(group_id=group.group_id, image_a=a, image_b=b,
                system_preference=system, human_preference=preferred,
                agreement=int(system == preferred) if preferred and preferred not in {"tie", "equal", "uncertain"} else "",
                first_deciding_field=field, first_field_margin=margin,
                face_count_difference=photos_by_path[a]["face_count"] - photos_by_path[b]["face_count"],
                worst_primary_score_difference=group.ranking[a].worst_primary_score - group.ranking[b].worst_primary_score,
                weighted_primary_average_difference=group.ranking[a].weighted_primary_average - group.ranking[b].weighted_primary_average,
                scenario=scenario))
    face_keys = sorted({key for row in face_rows for key in row if key not in FACE_EXTRA_FIELDS})
    _csv(output / "face_metrics.csv", list(FACE_EXTRA_FIELDS) + face_keys, face_rows)
    _csv(output / "photo_metrics.csv", PHOTO_FIELDS, photo_rows)
    v2_face_diagnostics = []
    for path, rows in face_by_path.items():
        photo = photo_by_path[path]
        _, diagnostics = extract_faces(rows, number(photo.get("width")),
                                       number(photo.get("height")), rows=True)
        v2_face_diagnostics.extend(dict(image_path=path, **item) for item in diagnostics)
    _csv(output / "ranking_v2_face_diagnostics.csv",
         ["image_path", "face_index", "relative_area", "detector_confidence",
          "eligible", "primary", "prominence", "reason"], v2_face_diagnostics)
    _csv(output / "ranking_decisions.csv", list(RANK_EXTRA_FIELDS) + list(RANK_FIELDS) + ["explanation"], ranking_rows)
    _csv(output / "pairwise_comparisons.csv", PAIR_FIELDS, pair_rows)
    _csv(output / "audit_summary.csv", SUMMARY_FIELDS, _summaries(photo_rows, ranking_rows, pair_rows, face_rows))
    if not (output / "review_annotations_template.csv").exists():
        _csv(output / "review_annotations_template.csv", ["group_id", "image_path", "preferred_keeper",
            "equally_acceptable", "unusable", "uncertain", "reasons"], annotation_rows)
    failed_count = sum(bool(row["error"]) for row in photo_rows)
    signature_after = _signature(paths)
    detector_runtime = (services._face_detector.runtime_info()
                        if getattr(services, "_faces_initialized", False) else None)
    manifest.update(status="partial" if failed_count or signature_after != signature else "complete",
        completed_at=datetime.now(timezone.utc).isoformat(),
        dataset_signature_after=signature_after,
        detector_runtime=detector_runtime,
        detected_face_count=len(face_rows), comparison_group_count=len(decisions),
        labeled_pair_count=sum(row["agreement"] != "" for row in pair_rows),
        failed_photo_count=failed_count,
        cache_hits=sum(row["face_cache_hit"] for row in photo_rows),
        csv_columns={name: list(keys) for name, keys in (("face_metrics", list(FACE_EXTRA_FIELDS) + face_keys),
            ("photo_metrics", PHOTO_FIELDS), ("ranking_decisions", list(RANK_EXTRA_FIELDS) + list(RANK_FIELDS) + ["explanation"]),
            ("pairwise_comparisons", PAIR_FIELDS), ("audit_summary", SUMMARY_FIELDS))})
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Read-only source photo directory")
    parser.add_argument("--output", required=True, help="Isolated output directory outside --input")
    parser.add_argument("--mode", choices=("full", "cached"), default="full")
    parser.add_argument("--cache-source", help="Existing cache DB to copy by SQLite backup; never opened for writing")
    parser.add_argument("--checkpoint-source", help="Import a prior verified image_checkpoint.jsonl without remeasuring faces")
    parser.add_argument("--groups-csv", help="Optional CSV with group_id,image_path; otherwise use production duplicate grouping")
    parser.add_argument("--labels-csv", help="Optional pairwise CSV with image_a,image_b,preferred_image")
    parser.add_argument("--provider", choices=("automatic", "cpu", "directml", "gpu"), default="automatic")
    parser.add_argument("--include-blur", action="store_true",
                        help="Compute the existing whole-image blur stage in full mode for V2 technical evidence")
    parser.add_argument("--max-images", type=int, help="Deterministic first N images for a smoke run")
    parser.add_argument("--contact-sheets", action="store_true")
    args = parser.parse_args(argv)
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    try:
        output = run(args)
        print(output)
        if json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] != "complete":
            return 1
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
