"""Experimental, user-requested expression observations; never keeper scores."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np

from app.tools.acquire_expression_models import MODELS, acquire, verify
from face_processing.cache import ImageFaceAnalysisCache
from image_loader import ImageLoader

LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
VERSION = "opencv-zoo-fer-fp32-alignment-v1"
NAME, SIZE, SHA256 = MODELS["fp32"]


def model_directory():
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Room36" / "experimental_models"


def align_official(image, landmarks):
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return None
    x, y = TEMPLATE[:, 0:1], TEMPLATE[:, 1:2]
    design = np.vstack((np.hstack((x, y, np.ones((5, 1)), np.zeros((5, 1)))),
                        np.hstack((y, -x, np.zeros((5, 1)), np.ones((5, 1))))))
    if np.linalg.matrix_rank(design) < 4:
        return None
    a, b, tx, ty = np.linalg.lstsq(design, np.r_[points[:, 0], points[:, 1]], rcond=-1)[0]
    inverse = np.array([[a, -b, 0], [b, a, 0], [tx, ty, 1]])
    return cv2.warpAffine(image, np.linalg.inv(inverse)[:, 0:2].T, (112, 112))


def preprocess(crop):
    if crop is None or crop.shape != (112, 112, 3):
        raise ValueError("Expected a 112x112 BGR aligned face")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return np.ascontiguousarray(((rgb - .5) / .5).transpose(2, 0, 1)[None])


def probabilities(output):
    vector = np.asarray(output, dtype=np.float64).reshape(-1)
    if vector.size != 7 or not np.isfinite(vector).all():
        raise ValueError("Invalid seven-class model output")
    if np.min(vector) >= 0 and abs(float(vector.sum()) - 1) < 1e-3:
        return (vector / vector.sum()).tolist()
    shifted = np.exp(vector - vector.max())
    return (shifted / shifted.sum()).tolist()


class ExpressionService:
    def __init__(self, model_path, *, providers=None, session_factory=None):
        if not verify(Path(model_path), SIZE, SHA256):
            raise ValueError("Expression model is missing or its SHA-256 does not match OpenCV Zoo.")
        if session_factory is None:
            import onnxruntime as ort
            session_factory = ort.InferenceSession
            available = ort.get_available_providers()
        else:
            available = providers or ["CPUExecutionProvider"]
        requested = providers or (["DmlExecutionProvider", "CPUExecutionProvider"]
                                  if "DmlExecutionProvider" in available else ["CPUExecutionProvider"])
        selected = [provider for provider in requested if provider in available]
        if not selected:
            raise ValueError("No supported ONNX execution provider is available.")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.log_severity_level = 3
        if "DmlExecutionProvider" in selected:
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_mem_pattern = False
        try:
            self.session = session_factory(str(model_path), sess_options=options, providers=selected)
        except Exception:
            if selected == ["CPUExecutionProvider"] or "CPUExecutionProvider" not in available:
                raise
            options = ort.SessionOptions()
            options.log_severity_level = 3
            self.session = session_factory(str(model_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 112, 112]:
            raise ValueError("Unexpected expression model input signature.")
        self.provider = self.session.get_providers()[0]

    def predict(self, image, landmarks):
        crop = align_official(image, landmarks)
        if crop is None:
            return dict(status="unavailable", reason="Missing or invalid face landmarks")
        values = probabilities(self.session.run(None, {self.session.get_inputs()[0].name: preprocess(crop)})[0])
        index = int(np.argmax(values))
        confidence = values[index]
        return dict(status="uncertain" if confidence < .5 else "estimated",
                    label=LABELS[index], confidence=confidence, probabilities=values,
                    reason="Low prediction confidence" if confidence < .5 else "")


def analyze_group(paths, cache_path, model_path, canceled, *, install=False, progress=None,
                  service_factory=ExpressionService, loader=None):
    """One bounded group, one decode per uncached photo, no detector or ArcFace calls."""
    if len(paths) > 64:
        raise ValueError("Expression analysis is limited to 64 photos per group.")
    if install:
        acquire(Path(model_path).parent, ("fp32",))
    service = None
    face_cache = ImageFaceAnalysisCache(cache_path)
    db_path = Path(cache_path) if cache_path else face_cache._db_path
    loader = loader or ImageLoader(max_decode_dimension=None)
    connection = sqlite3.connect(db_path)
    connection.execute("""CREATE TABLE IF NOT EXISTS experimental_expression_predictions (
        cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)""")
    output = {}
    metrics = dict(decodes=0, cache_hits=0, inference=0, elapsed=0.0)
    started = time.perf_counter()
    try:
        for number, name in enumerate(paths):
            if canceled.is_set():
                return None, metrics
            path = Path(name)
            try:
                stat = path.stat()
                faces = face_cache.get(path, stat.st_size, stat.st_mtime_ns)
            except OSError:
                faces = None
            if faces is None:
                output[name] = [dict(status="unavailable", reason="Compatible cached face detections unavailable")]
                continue
            records, pending = [None] * len(faces), []
            for index, face in enumerate(faces):
                if min(face.bbox[2:]) < 24:
                    records[index] = dict(status="unavailable", reason="Face too small for reliable expression display")
                    continue
                landmarks = face.landmarks
                if landmarks is None or np.asarray(landmarks).shape != (5, 2):
                    records[index] = dict(status="unavailable", reason="Face landmarks unavailable")
                    continue
                identity = hashlib.sha256(np.asarray(landmarks, dtype=np.float32).tobytes()
                    + np.asarray(face.bbox, dtype=np.int32).tobytes()).hexdigest()
                key = hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{index}|{identity}|{SHA256}|{VERSION}".encode()).hexdigest()
                hit = connection.execute("SELECT payload FROM experimental_expression_predictions WHERE cache_key=?", (key,)).fetchone()
                if hit:
                    records[index] = json.loads(hit[0])
                    metrics["cache_hits"] += 1
                else:
                    pending.append((index, face, key))
            if pending:
                if service is None:
                    service = service_factory(model_path)
                metadata = loader.read_metadata(path)
                if metadata is None or metadata.width * metadata.height * 3 > 256 * 1024 * 1024:
                    for index, _, _ in pending:
                        records[index] = dict(status="unavailable", reason="Image dimensions unavailable or exceed analysis memory limit")
                    output[name] = records
                    continue
                image = loader.load_for_scan(path, max_dimension=None)
                metrics["decodes"] += 1
                if image is None:
                    for index, _, _ in pending:
                        records[index] = dict(status="unavailable", reason="Image could not be decoded")
                else:
                    if image.nbytes > 256 * 1024 * 1024:
                        for index, _, _ in pending:
                            records[index] = dict(status="unavailable", reason="Image exceeds analysis memory limit")
                    else:
                        for index, face, key in pending:
                            if canceled.is_set():
                                return None, metrics
                            try:
                                result = service.predict(image, face.landmarks)
                                metrics["inference"] += 1
                            except Exception:
                                result = dict(status="unavailable", reason="Expression inference failed")
                            records[index] = result
                            if result["status"] != "unavailable":
                                connection.execute("INSERT OR REPLACE INTO experimental_expression_predictions VALUES (?,?)",
                                                   (key, json.dumps(result)))
                        connection.commit()
                    del image
            output[name] = records
            if progress:
                progress(number + 1, len(paths))
    finally:
        metrics["elapsed"] = time.perf_counter() - started
        connection.close()
    return output, metrics
