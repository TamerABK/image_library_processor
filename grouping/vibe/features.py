from __future__ import annotations

import logging
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_detector.face_database_sqlite import SQLiteFaceDatabase
from face_processing.cache import FaceScanCache, ImageFaceAnalysisCache
from face_processing.models import DetectedFace, EmbeddedFace, Match
from image_loader import RAW_EXTENSIONS, default_image_loader
from scan_controls import CancellationToken

from grouping.models import ScanError, VibeImageFeatures

from .cache import VibeFeatureCache
from .config import VibeGroupingConfig
from .embedder import VibeEmbedder


LOGGER = logging.getLogger(__name__)

_EXIF_DATETIME_TAGS = (
    36867,  # DateTimeOriginal
    36868,  # DateTimeDigitized
    306,  # DateTime
)


@dataclass(frozen=True, slots=True)
class ExtractionSummary:
    cache_hits: int
    cache_misses: int
    people_signature: str | None


class VibeFeatureExtractor:
    def __init__(
        self,
        *,
        config: VibeGroupingConfig,
        embedder: VibeEmbedder,
        feature_cache: VibeFeatureCache | None = None,
        face_cache: FaceScanCache | None = None,
        analysis_cache: ImageFaceAnalysisCache | None = None,
        face_database: SQLiteFaceDatabase | None = None,
        similarity: CosineEmbeddingSimilarity | None = None,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._feature_cache = feature_cache or VibeFeatureCache()
        self._face_cache = face_cache or FaceScanCache()
        self._analysis_cache = analysis_cache or ImageFaceAnalysisCache()
        self._face_database = face_database
        self._similarity = similarity or CosineEmbeddingSimilarity()
        self._decode_dimension = 320

    def extract(
        self,
        image_paths: Sequence[Path],
        *,
        progress_callback: callable | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> tuple[list[VibeImageFeatures], list[ScanError], ExtractionSummary]:
        ordered_paths = sorted(Path(path).resolve() for path in image_paths)
        features: list[VibeImageFeatures] = []
        errors: list[ScanError] = []
        cache_hits = 0
        cache_misses = 0
        people_signature = None

        if self._config.include_people and self._face_database is not None:
            people_signature = self._face_database.cache_signature()
        dependency_signature = self._feature_dependency_signature(people_signature)

        pending: list[tuple[Path, int, int]] = []
        pending_subject_scene: list[tuple[int, Path, int, int]] = []
        total = len(ordered_paths)

        for index, path in enumerate(ordered_paths):
            if cancellation_token is not None:
                cancellation_token.raise_if_canceled()

            try:
                stat = path.stat()
            except OSError as exc:
                errors.append(ScanError(path=str(path), message=str(exc)))
                if progress_callback is not None:
                    progress_callback("loading_visual_features", index + 1, total)
                continue

            cached = self._feature_cache.get(
                path,
                stat.st_size,
                stat.st_mtime_ns,
                model_fingerprint=self._embedder.model_fingerprint,
                preprocessing_version=self._config.preprocessing_version,
                feature_version=self._config.feature_version,
                people_signature=dependency_signature,
                subject_scene_preprocessing_version=(
                    self._config.subject_scene_preprocessing_version
                    if self._config.include_subject_scene_embedding
                    else None
                ),
            )
            if cached is not None:
                features.append(cached)
                cache_hits += 1
                if (
                    self._config.include_subject_scene_embedding
                    and cached.subject_scene_embedding is None
                ):
                    pending_subject_scene.append(
                        (len(features) - 1, path, stat.st_size, stat.st_mtime_ns)
                    )
            else:
                pending.append((path, stat.st_size, stat.st_mtime_ns))
                cache_misses += 1

            if progress_callback is not None:
                progress_callback("loading_visual_features", index + 1, total)

        if pending:
            batch_size = max(1, self._config.batch_size)
            total_pending = len(pending)
            processed_pending = 0

            for start in range(0, total_pending, batch_size):
                if cancellation_token is not None:
                    cancellation_token.raise_if_canceled()

                batch = pending[start:start + batch_size]
                decoded_images: list[np.ndarray] = []
                batch_metadata: list[dict[str, Any]] = []

                for path, file_size, mtime_ns in batch:
                    try:
                        decoded = self._load_preview_image(path)
                        if decoded is None:
                            raise ValueError("Preview decode failed.")
                        decoded_images.append(decoded)
                        batch_metadata.append(
                            self._collect_metadata(
                                path,
                                file_size=file_size,
                                mtime_ns=mtime_ns,
                                decoded_image=decoded,
                                people_signature=people_signature if self._config.include_people else None,
                            )
                        )
                    except Exception as exc:
                        errors.append(ScanError(path=str(path), message=str(exc)))
                        processed_pending += 1
                        if progress_callback is not None:
                            progress_callback("loading_visual_features", processed_pending, total_pending)

                if not decoded_images:
                    continue

                embeddings = self._embedder.encode_images(decoded_images)
                subject_scene_embeddings: Sequence[np.ndarray | None]
                if self._config.include_subject_scene_embedding:
                    subject_scene_crops = [
                        _crop_subject_scene(
                            decoded,
                            metadata.get("subject_box_norm"),
                        )
                        for decoded, metadata in zip(decoded_images, batch_metadata)
                    ]
                    subject_scene_embeddings = self._embedder.encode_images(subject_scene_crops)
                else:
                    subject_scene_embeddings = [None] * len(decoded_images)
                background_embeddings: Sequence[np.ndarray | None]
                if self._config.include_background_embedding:
                    masked_images = [
                        _mask_face_regions(
                            decoded,
                            metadata["analysis_faces"],
                            original_width=metadata["width"],
                            original_height=metadata["height"],
                        )
                        for decoded, metadata in zip(decoded_images, batch_metadata)
                    ]
                    background_embeddings = self._embedder.encode_images(masked_images)
                else:
                    background_embeddings = [None] * len(decoded_images)

                for decoded, metadata, embedding, subject_scene_embedding, background_embedding in zip(
                    decoded_images,
                    batch_metadata,
                    embeddings,
                    subject_scene_embeddings,
                    background_embeddings,
                ):
                    if cancellation_token is not None:
                        cancellation_token.raise_if_canceled()

                    feature = self._build_feature(
                        metadata=metadata,
                        embedding=np.asarray(embedding, dtype=np.float32),
                        decoded_image=decoded,
                        subject_scene_embedding=(
                            None
                            if subject_scene_embedding is None
                            else np.asarray(subject_scene_embedding, dtype=np.float32)
                        ),
                        background_embedding=(
                            None
                            if background_embedding is None
                            else np.asarray(background_embedding, dtype=np.float32)
                        ),
                    )
                    features.append(feature)
                    self._feature_cache.put(
                        feature,
                        model_fingerprint=self._embedder.model_fingerprint,
                        preprocessing_version=self._config.preprocessing_version,
                        feature_version=self._config.feature_version,
                        people_signature=dependency_signature,
                        subject_scene_preprocessing_version=(
                            self._config.subject_scene_preprocessing_version
                            if self._config.include_subject_scene_embedding
                            else None
                        ),
                    )
                    processed_pending += 1
                    if progress_callback is not None:
                        progress_callback("loading_visual_features", processed_pending, total_pending)

        if pending_subject_scene:
            batch_size = max(1, self._config.batch_size)
            for start in range(0, len(pending_subject_scene), batch_size):
                batch = pending_subject_scene[start:start + batch_size]
                decoded_images: list[np.ndarray] = []
                feature_indices: list[int] = []
                cached_features: list[VibeImageFeatures] = []
                cache_stats: list[tuple[int, int]] = []
                for feature_index, path, file_size, mtime_ns in batch:
                    decoded = self._load_preview_image(path)
                    if decoded is None:
                        continue
                    decoded_images.append(decoded)
                    feature_indices.append(feature_index)
                    cached_features.append(features[feature_index])
                    cache_stats.append((file_size, mtime_ns))
                if not decoded_images:
                    continue
                subject_scene_crops = [
                    _crop_subject_scene(decoded, feature.metadata.get("subject_box_norm"))
                    for decoded, feature in zip(decoded_images, cached_features)
                ]
                subject_scene_embeddings = self._embedder.encode_images(subject_scene_crops)
                for feature_index, cached_feature, embedding in zip(
                    feature_indices,
                    cached_features,
                    subject_scene_embeddings,
                ):
                    updated_feature = replace(
                        cached_feature,
                        subject_scene_embedding=_l2_normalize_vector(
                            np.asarray(embedding, dtype=np.float32)
                        ),
                    )
                    features[feature_index] = updated_feature
                    self._feature_cache.put(
                        updated_feature,
                        model_fingerprint=self._embedder.model_fingerprint,
                        preprocessing_version=self._config.preprocessing_version,
                        feature_version=self._config.feature_version,
                        people_signature=dependency_signature,
                        subject_scene_preprocessing_version=self._config.subject_scene_preprocessing_version,
                    )

        features.sort(key=lambda item: item.image_path)
        return (
            features,
            errors,
            ExtractionSummary(
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                people_signature=people_signature if self._config.include_people else None,
            ),
        )

    def _build_feature(
        self,
        *,
        metadata: dict[str, Any],
        embedding: np.ndarray,
        decoded_image: np.ndarray,
        subject_scene_embedding: np.ndarray | None,
        background_embedding: np.ndarray | None,
    ) -> VibeImageFeatures:
        color_features, color_metadata = _compute_color_features(decoded_image)
        composition_features, composition_metadata = _compute_composition_features(
            metadata["width"],
            metadata["height"],
            metadata["analysis_faces"],
            brightness=float(color_metadata["brightness"]),
        )
        face_layout, face_layout_metadata = _compute_face_layout_features(
            metadata["width"],
            metadata["height"],
            metadata["analysis_faces"],
            recognized_person_ids=tuple(metadata["recognized_person_ids"]),
        )
        extra_metadata = {
            **color_metadata,
            **composition_metadata,
            **face_layout_metadata,
            "timestamp_confidence": metadata["timestamp_source"] != "filesystem",
            "timestamp_source": metadata["timestamp_source"],
            "orientation": composition_metadata["orientation"],
            "subject_box_norm": metadata.get("subject_box_norm"),
        }
        return VibeImageFeatures(
            image_path=str(metadata["path"]),
            semantic_embedding=_l2_normalize_vector(embedding),
            capture_timestamp=metadata["capture_timestamp"],
            timestamp_source=metadata["timestamp_source"],
            recognized_person_ids=tuple(metadata["recognized_person_ids"]),
            color_features=color_features,
            composition_features=composition_features,
            face_layout=face_layout,
            face_scale_summary=_extract_face_scale_summary(face_layout),
            subject_scene_embedding=(
                None
                if subject_scene_embedding is None
                else _l2_normalize_vector(subject_scene_embedding)
            ),
            background_embedding=(
                None
                if background_embedding is None
                else _l2_normalize_vector(background_embedding)
            ),
            action_scores=None,
            scene_scores=None,
            shot_type_scores=None,
            width=metadata["width"],
            height=metadata["height"],
            file_mtime_ns=metadata["mtime_ns"],
            file_size=metadata["file_size"],
            quality_score=metadata["quality_score"],
            brightness=float(color_metadata["brightness"]),
            face_count=int(composition_metadata["face_count"]),
            face_area_ratio=float(composition_metadata["face_area_ratio"]),
            dominant_people_names=tuple(metadata["recognized_person_names"]),
            metadata=extra_metadata,
        )

    def _feature_dependency_signature(self, people_signature: str | None) -> str:
        return json.dumps(
            {
                "people_signature": people_signature if self._config.include_people else None,
                "include_background_embedding": self._config.include_background_embedding,
                "background_preprocessing_version": self._config.background_preprocessing_version,
                "composition_feature_version": self._config.composition_feature_version,
            },
            sort_keys=True,
        )

    def _collect_metadata(
        self,
        path: Path,
        *,
        file_size: int,
        mtime_ns: int,
        decoded_image: np.ndarray,
        people_signature: str | None,
    ) -> dict[str, Any]:
        metadata = default_image_loader.read_metadata(path)
        width = metadata.width if metadata is not None else int(decoded_image.shape[1])
        height = metadata.height if metadata is not None else int(decoded_image.shape[0])

        capture_timestamp, timestamp_source = _read_capture_timestamp(path, fallback_mtime_ns=mtime_ns)
        recognized_person_ids: list[str] = []
        recognized_person_names: list[str] = []
        analysis_faces = self._analysis_cache.get(path, file_size, mtime_ns) or []
        quality_score = _best_quality_score(analysis_faces)

        if self._config.include_people and self._face_database is not None:
            embedded_faces = self._face_cache.get(
                path,
                file_size,
                mtime_ns,
                include_unknown_faces=True,
                database_signature=people_signature,
                require_analysis=False,
            ) or []
            recognized_person_ids, recognized_person_names, quality_score = self._recognize_people(
                embedded_faces,
                analysis_faces,
                default_quality_score=quality_score,
            )
            if embedded_faces and not analysis_faces:
                analysis_faces = [
                    DetectedFace(
                        bbox=face.bbox,
                        confidence=face.confidence,
                        landmarks=face.landmarks,
                        path=face.path,
                        analysis=face.analysis,
                    )
                    for face in embedded_faces
                ]
                quality_score = _best_quality_score(analysis_faces)

        return {
            "path": path,
            "width": width,
            "height": height,
            "file_size": file_size,
            "mtime_ns": mtime_ns,
            "capture_timestamp": capture_timestamp,
            "timestamp_source": timestamp_source,
            "recognized_person_ids": recognized_person_ids,
            "recognized_person_names": recognized_person_names,
            "analysis_faces": analysis_faces,
            "subject_box_norm": _compute_subject_box_norm(width, height, analysis_faces),
            "quality_score": quality_score,
        }

    def _recognize_people(
        self,
        embedded_faces: Sequence[EmbeddedFace],
        analysis_faces: Sequence[DetectedFace],
        *,
        default_quality_score: float | None,
    ) -> tuple[list[str], list[str], float | None]:
        if not embedded_faces or self._face_database is None:
            return [], [], default_quality_score

        matches: list[tuple[str, str]] = []
        quality_score = default_quality_score
        for face in embedded_faces:
            match = self._face_database.find_nearest_embedding(face.embedding)
            if not _is_recognized_match(match, self._similarity.default_threshold):
                continue
            person = self._face_database.get_person(match.person_id)
            if person is None:
                continue
            matches.append((f"person:{person.id}", person.name))
            if face.analysis is not None:
                face_score = face.analysis.selection_score
                quality_score = face_score if quality_score is None else max(quality_score, face_score)

        if not matches:
            return [], [], quality_score

        counts = Counter(matches)
        ordered_matches = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][1].casefold(), item[0][0]),
        )
        ids = [item[0][0] for item in ordered_matches]
        names = [item[0][1] for item in ordered_matches]
        return ids, names, quality_score

    def _load_preview_image(self, path: Path) -> np.ndarray | None:
        if path.suffix.lower() in RAW_EXTENSIONS:
            return default_image_loader.load_for_scan(path, max_dimension=self._decode_dimension)

        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.thumbnail((self._decode_dimension, self._decode_dimension), Image.Resampling.LANCZOS)
                rgb = np.asarray(image, dtype=np.uint8)
        except Exception:
            return None
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _read_capture_timestamp(path: Path, *, fallback_mtime_ns: int) -> tuple[float | None, str]:
    if path.suffix.lower() not in RAW_EXTENSIONS:
        try:
            with Image.open(path) as image:
                image_exif = image.getexif()
                for tag_id in _EXIF_DATETIME_TAGS:
                    value = image_exif.get(tag_id)
                    parsed = _parse_exif_datetime(value)
                    if parsed is not None:
                        return parsed, "exif"
        except Exception:
            pass

    fallback_seconds = fallback_mtime_ns / 1_000_000_000
    return fallback_seconds, "filesystem"


def _parse_exif_datetime(raw_value: object) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bytes):
        try:
            raw_value = raw_value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if not isinstance(raw_value, str):
        return None

    value = raw_value.strip()
    if not value:
        return None

    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc).timestamp()
    return None


def _compute_color_features(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    resized = cv2.resize(image_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    std_hsv = hsv.reshape(-1, 3).std(axis=0)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    luminance = float(lab[..., 0].mean())
    saturation = float(hsv[..., 1].mean())
    warmth = float(np.clip(rgb[..., 0].mean() - rgb[..., 2].mean() + 0.5, 0.0, 1.0))
    bw_likelihood = float(
        np.clip(1.0 - np.mean(np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])), 0.0, 1.0)
    )

    hist_h = cv2.calcHist([hsv], [0], None, [8], [0.0, 1.0]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [4], [0.0, 1.0]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [4], [0.0, 1.0]).flatten()
    descriptor = np.concatenate(
        [
            mean_hsv,
            std_hsv,
            mean_lab,
            hist_h,
            hist_s,
            hist_v,
            np.asarray([luminance, saturation, warmth, bw_likelihood], dtype=np.float32),
        ]
    ).astype(np.float32)
    descriptor = _l2_normalize_vector(descriptor)
    return descriptor, {
        "brightness": luminance,
        "saturation": saturation,
        "warmth": warmth,
        "black_and_white_likelihood": bw_likelihood,
    }


def _compute_composition_features(
    width: int | None,
    height: int | None,
    faces: Sequence[DetectedFace],
    *,
    brightness: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    safe_width = max(int(width or 1), 1)
    safe_height = max(int(height or 1), 1)
    aspect_ratio = safe_width / safe_height
    orientation = "square"
    if safe_width > safe_height:
        orientation = "landscape"
    elif safe_height > safe_width:
        orientation = "portrait"

    total_area_ratio = 0.0
    largest_area_ratio = 0.0
    centrality = 0.0
    if faces:
        image_area = safe_width * safe_height
        face_ratios: list[float] = []
        face_centralities: list[float] = []
        centers_x: list[float] = []
        centers_y: list[float] = []
        for face in faces:
            x, y, w, h = face.bbox
            area_ratio = max(0.0, min((w * h) / image_area, 1.0))
            face_ratios.append(area_ratio)
            center_x = (x + (w / 2.0)) / safe_width
            center_y = (y + (h / 2.0)) / safe_height
            face_centralities.append(1.0 - min(abs(center_x - 0.5) + abs(center_y - 0.5), 1.0))
            centers_x.append(float(center_x))
            centers_y.append(float(center_y))
        total_area_ratio = float(sum(face_ratios))
        largest_area_ratio = float(max(face_ratios))
        centrality = float(sum(face_centralities) / len(face_centralities))
    else:
        centers_x = []
        centers_y = []

    close_up_score = float(np.clip(largest_area_ratio * 4.0, 0.0, 1.0))
    subject_count_proxy = min(len(faces), 8) / 8.0
    background_subject_ratio = float(max(0.0, 1.0 - total_area_ratio))
    shot_scale_category = _shot_scale_category(largest_area_ratio, face_count=len(faces))
    face_count_bucket = _face_count_bucket(len(faces))
    centroid_x = 0.5 if not centers_x else float(np.mean(centers_x))
    centroid_y = 0.5 if not centers_y else float(np.mean(centers_y))
    horizontal_spread = 0.0 if len(centers_x) <= 1 else float(np.std(centers_x))
    vertical_spread = 0.0 if len(centers_y) <= 1 else float(np.std(centers_y))
    descriptor = np.asarray(
        [
            min(aspect_ratio / 2.5, 1.5),
            1.0 if orientation == "portrait" else 0.0,
            1.0 if orientation == "landscape" else 0.0,
            1.0 if orientation == "square" else 0.0,
            min(len(faces), 8) / 8.0,
            largest_area_ratio,
            total_area_ratio,
            centrality,
            close_up_score,
            brightness,
            subject_count_proxy,
            background_subject_ratio,
            centroid_x,
            centroid_y,
            horizontal_spread,
            vertical_spread,
        ],
        dtype=np.float32,
    )
    descriptor = _l2_normalize_vector(descriptor)
    return descriptor, {
        "orientation": orientation,
        "face_count": len(faces),
        "face_area_ratio": total_area_ratio,
        "largest_face_area_ratio": largest_area_ratio,
        "face_centrality": centrality,
        "close_up_score": close_up_score,
        "subject_count_proxy": subject_count_proxy,
        "background_subject_ratio": background_subject_ratio,
        "shot_scale_category": shot_scale_category,
        "face_count_bucket": face_count_bucket,
        "subject_centroid_x": centroid_x,
        "subject_centroid_y": centroid_y,
        "subject_horizontal_spread": horizontal_spread,
        "subject_vertical_spread": vertical_spread,
    }


def _compute_face_layout_features(
    width: int | None,
    height: int | None,
    faces: Sequence[DetectedFace],
    *,
    recognized_person_ids: tuple[str, ...],
) -> tuple[np.ndarray | None, dict[str, float | int | str]]:
    if not faces:
        return None, {
            "participant_mode": "none",
            "recognized_face_count": 0,
            "unknown_face_count": 0,
            "face_horizontal_spread": 0.0,
            "face_vertical_spread": 0.0,
        }

    safe_width = max(int(width or 1), 1)
    safe_height = max(int(height or 1), 1)
    rows: list[list[float]] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    scales: list[float] = []

    for face in faces:
        x, y, w, h = face.bbox
        center_x = np.clip((x + (w / 2.0)) / safe_width, 0.0, 1.0)
        center_y = np.clip((y + (h / 2.0)) / safe_height, 0.0, 1.0)
        width_ratio = np.clip(w / safe_width, 0.0, 1.0)
        height_ratio = np.clip(h / safe_height, 0.0, 1.0)
        area_ratio = np.clip((w * h) / max(safe_width * safe_height, 1), 0.0, 1.0)
        rows.append([center_x, center_y, width_ratio, height_ratio, area_ratio])
        centers_x.append(float(center_x))
        centers_y.append(float(center_y))
        scales.append(float(area_ratio))

    array = np.asarray(rows, dtype=np.float32)
    order = np.argsort(-array[:, 4], kind="stable")
    ordered = array[order]
    top = np.zeros((4, 4), dtype=np.float32)
    selected = ordered[:4, :4]
    top[:selected.shape[0], :] = selected
    summary = np.concatenate(
        [
            top.reshape(-1),
            np.asarray(
                [
                    len(faces) / 8.0,
                    max(scales),
                    float(sum(scales)),
                    float(np.mean(centers_x)),
                    float(np.mean(centers_y)),
                    float(np.std(centers_x)) if len(centers_x) > 1 else 0.0,
                    float(np.std(centers_y)) if len(centers_y) > 1 else 0.0,
                    min(len(recognized_person_ids), len(faces)) / max(len(faces), 1),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    mode = "solo"
    if len(faces) == 2:
        mode = "couple"
    elif len(faces) <= 4:
        mode = "small_group"
    elif len(faces) > 4:
        mode = "crowd"
    if len(recognized_person_ids) >= 3 and len(faces) >= 3:
        mode = "family_group"
    return _l2_normalize_vector(summary), {
        "participant_mode": mode,
        "recognized_face_count": min(len(recognized_person_ids), len(faces)),
        "unknown_face_count": max(len(faces) - len(recognized_person_ids), 0),
        "primary_subject_scale": max(scales),
        "total_face_area_ratio": float(sum(scales)),
        "subject_centroid_x": float(np.mean(centers_x)),
        "subject_centroid_y": float(np.mean(centers_y)),
        "face_horizontal_spread": float(np.std(centers_x)) if len(centers_x) > 1 else 0.0,
        "face_vertical_spread": float(np.std(centers_y)) if len(centers_y) > 1 else 0.0,
    }


def _extract_face_scale_summary(face_layout: np.ndarray | None) -> np.ndarray | None:
    if face_layout is None:
        return None
    vector = np.asarray(face_layout, dtype=np.float32)
    if vector.size < 16:
        return None
    tail = vector[-8:]
    return _l2_normalize_vector(tail.copy())


def _mask_face_regions(
    image_bgr: np.ndarray,
    faces: Sequence[DetectedFace],
    *,
    original_width: int | None,
    original_height: int | None,
) -> np.ndarray:
    if not faces:
        return image_bgr

    preview_height, preview_width = image_bgr.shape[:2]
    scaled = image_bgr.copy()
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=21, sigmaY=21)
    scale_x = preview_width / max(int(original_width or preview_width), 1)
    scale_y = preview_height / max(int(original_height or preview_height), 1)

    for face in faces:
        x, y, w, h = face.bbox
        left = max(0, int(round(x * scale_x)))
        top = max(0, int(round(y * scale_y)))
        right = min(preview_width, int(round((x + w) * scale_x)))
        bottom = min(preview_height, int(round((y + h) * scale_y)))
        if right <= left or bottom <= top:
            continue
        pad_x = max(1, (right - left) // 4)
        pad_y = max(1, (bottom - top) // 4)
        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(preview_width, right + pad_x)
        bottom = min(preview_height, bottom + pad_y)
        scaled[top:bottom, left:right] = blurred[top:bottom, left:right]
    return scaled


def _compute_subject_box_norm(
    width: int | None,
    height: int | None,
    faces: Sequence[DetectedFace],
) -> list[float]:
    if not faces:
        return [0.2, 0.2, 0.8, 0.8]

    safe_width = max(int(width or 1), 1)
    safe_height = max(int(height or 1), 1)
    left = min(face.bbox[0] for face in faces)
    top = min(face.bbox[1] for face in faces)
    right = max(face.bbox[0] + face.bbox[2] for face in faces)
    bottom = max(face.bbox[1] + face.bbox[3] for face in faces)

    expand_x = (right - left) * 0.35
    expand_y = (bottom - top) * 0.35
    norm_left = max(0.0, (left - expand_x) / safe_width)
    norm_top = max(0.0, (top - expand_y) / safe_height)
    norm_right = min(1.0, (right + expand_x) / safe_width)
    norm_bottom = min(1.0, (bottom + expand_y) / safe_height)
    if norm_right - norm_left < 0.1:
        center_x = (norm_left + norm_right) / 2.0
        norm_left = max(0.0, center_x - 0.15)
        norm_right = min(1.0, center_x + 0.15)
    if norm_bottom - norm_top < 0.1:
        center_y = (norm_top + norm_bottom) / 2.0
        norm_top = max(0.0, center_y - 0.15)
        norm_bottom = min(1.0, center_y + 0.15)
    return [
        round(float(norm_left), 4),
        round(float(norm_top), 4),
        round(float(norm_right), 4),
        round(float(norm_bottom), 4),
    ]


def _crop_subject_scene(
    image_bgr: np.ndarray,
    subject_box_norm: object,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if (
        not isinstance(subject_box_norm, (list, tuple))
        or len(subject_box_norm) != 4
    ):
        left = int(width * 0.2)
        top = int(height * 0.2)
        right = int(width * 0.8)
        bottom = int(height * 0.8)
    else:
        left = max(0, min(width - 1, int(round(float(subject_box_norm[0]) * width))))
        top = max(0, min(height - 1, int(round(float(subject_box_norm[1]) * height))))
        right = max(left + 1, min(width, int(round(float(subject_box_norm[2]) * width))))
        bottom = max(top + 1, min(height, int(round(float(subject_box_norm[3]) * height))))
    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        return image_bgr
    return crop


def _shot_scale_category(
    largest_face_area_ratio: float,
    *,
    face_count: int,
) -> str:
    if face_count <= 0:
        return "wide"
    if largest_face_area_ratio >= 0.18:
        return "close"
    if largest_face_area_ratio >= 0.08:
        return "medium"
    if largest_face_area_ratio >= 0.03:
        return "full_body"
    return "wide"


def _face_count_bucket(face_count: int) -> str:
    if face_count <= 0:
        return "zero"
    if face_count == 1:
        return "one"
    if face_count == 2:
        return "couple"
    if face_count <= 4:
        return "small_group"
    return "crowd"


def _best_quality_score(faces: Sequence[DetectedFace]) -> float | None:
    scores = [
        face.analysis.selection_score
        for face in faces
        if face.analysis is not None
    ]
    if not scores:
        return None
    return max(float(score) for score in scores)


def _is_recognized_match(match: Match | None, threshold: float) -> bool:
    return match is not None and float(match.score) >= threshold


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm
