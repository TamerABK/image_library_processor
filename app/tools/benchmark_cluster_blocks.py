"""Isolated dense-reference versus blockwise clustering memory benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import numpy as np

from face_detector.connected_face_clusterer import ConnectedComponentFaceClusterer
from face_detector.cosine_similarity import CosineEmbeddingSimilarity
from face_processing.models import EmbeddedFace


def _vectors(cache: Path, dataset: str, count: int | None):
    connection = sqlite3.connect(cache.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT entry_path, face_index, embedding FROM face_scan_cache_faces "
            "WHERE lower(entry_path) LIKE ? ORDER BY entry_path, face_index",
            (f"%{dataset.lower()}%",)).fetchall()
    finally:
        connection.close()
    if count is not None:
        rows = rows[:count]
    return [(f"{path}#{index}", np.frombuffer(blob, dtype=np.float32).copy())
            for path, index, blob in rows]


def _synthetic(count: int):
    rng = np.random.default_rng(371)
    vectors = rng.normal(size=(count, 512)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [(str(i), vector) for i, vector in enumerate(vectors)]


def _sample_peak(stop, samples):
    try:
        import psutil
        process = psutil.Process()
        while True:
            memory = process.memory_info()
            samples.append((memory.rss, getattr(memory, "private", None)))
            if stop.wait(.005):
                break
    except (ImportError, OSError):
        return


def _signature(groups, positions):
    membership = [tuple(positions[id(face)] for face in group.faces) for group in groups]
    representatives = [positions[id(group.representative)] for group in groups]
    payload = json.dumps([membership, representatives], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _child(args):
    data = _vectors(args.cache, args.dataset, None) if args.kind == "real" else _synthetic(args.count)
    faces = [EmbeddedFace((0, 0, 1, 1), 1., np.empty((5, 2), np.float32), vector,
                          Path(name)) for name, vector in data]
    positions = {id(face): i for i, face in enumerate(faces)}
    clusterer = ConnectedComponentFaceClusterer(CosineEmbeddingSimilarity(),
        strong_threshold=.63, weak_threshold=.55, block_rows=args.rows,
        block_budget_bytes=args.budget_mib * 1024 * 1024)
    stop = threading.Event()
    samples = []
    sampler = threading.Thread(target=_sample_peak, args=(stop, samples), daemon=True)
    sampler.start()
    started = time.perf_counter()
    if args.mode == "dense":
        embeddings = np.asarray([face.embedding for face in faces], dtype=np.float32)
        matrix = embeddings @ embeddings.T
        similarity_seconds = time.perf_counter() - started
        tick = time.perf_counter()
        from collections import defaultdict
        strong, weak = defaultdict(list), []
        for i in range(len(faces) - 1):
            scores = matrix[i, i + 1:]
            strong_offsets = np.flatnonzero(scores >= clusterer._strong)
            weak_offsets = np.flatnonzero((scores >= clusterer._weak) & (scores < clusterer._strong))
            for offset in strong_offsets:
                j = i + 1 + int(offset)
                strong[i].append(j)
                strong[j].append(i)
            for offset in weak_offsets:
                weak.append((i, i + 1 + int(offset)))
        graph_seconds = time.perf_counter() - tick
        labels = clusterer._attach_weak_faces(
            clusterer._strong_components(strong, len(faces)), strong, weak)
        by_label = {}
        for i, label in enumerate(labels):
            by_label.setdefault(label, []).append(i)
        tick = time.perf_counter()
        result = []
        for label, indices in by_label.items():
            scores = matrix[np.ix_(indices, indices)].sum(axis=1)
            result.append((label, indices, indices[int(np.argmax(scores))]))
        representative_seconds = time.perf_counter() - tick
        result.sort(key=lambda item: len(item[1]), reverse=True)
        membership = [tuple(indices) for _, indices, _ in result]
        representatives = [representative for _, _, representative in result]
        digest = hashlib.sha256(json.dumps([membership, representatives],
            separators=(",", ":")).encode()).hexdigest()
        stats = dict(face_count=len(faces), embedding_dimension=embeddings.shape[1],
            embedding_bytes=embeddings.nbytes, block_rows=len(faces),
            block_budget_bytes=0, max_similarity_block_bytes=matrix.nbytes,
            block_count=1, strong_edges=sum(map(len, strong.values())) // 2,
            weak_edges=len(weak), total_edges=sum(map(len, strong.values())) // 2 + len(weak),
            largest_component=max(map(len, by_label.values()), default=0),
            max_adjacency_length=max(map(len, strong.values()), default=0),
            similarity_seconds=similarity_seconds, graph_seconds=graph_seconds,
            representative_seconds=representative_seconds)
    else:
        groups = clusterer.cluster(faces)
        digest = _signature(groups, positions)
        stats = clusterer.last_diagnostics
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=1)
    stats.update(kind=args.kind, mode=args.mode, requested_rows=args.rows,
                 requested_budget_mib=args.budget_mib, elapsed_seconds=elapsed,
                 cluster_count=len(result) if args.mode == "dense" else len(groups),
                 result_sha256=digest,
                 sampled_peak_rss_bytes=max((sample[0] for sample in samples), default=None),
                 sampled_peak_private_bytes=max((sample[1] for sample in samples
                                                  if sample[1] is not None), default=None))
    print(json.dumps(stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("image_analysis_cache.sqlite3"))
    parser.add_argument("--dataset", default="nsc_meeting")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=("real", "synthetic"), default="real")
    parser.add_argument("--count", type=int, default=8219)
    parser.add_argument("--mode", choices=("dense", "block"), default="block")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--budget-mib", type=int, default=16)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--real-repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.child:
        _child(args)
        return
    if args.real_repetitions < 1:
        parser.error("--real-repetitions must be positive")
    tasks = [("real", 0, "dense", 256, 16)]
    tasks += [("real", 0, "block", rows, budget)
              for rows, budget in ((64, 16), (128, 16), (256, 16), (256, 4))]
    tasks += [("synthetic", count, "block", 256, 16)
              for count in (1000, 4000, 8000, 12000, 16000)]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for kind, count, mode, rows, budget in tasks:
        for repetition in range(1, (args.real_repetitions if kind == "real" else 1) + 1):
            command = [sys.executable, "-B", "-m", "app.tools.benchmark_cluster_blocks",
                       "--cache", str(args.cache), "--dataset", args.dataset,
                       "--out", str(args.out), "--kind", kind, "--count", str(count),
                       "--mode", mode, "--rows", str(rows), "--budget-mib", str(budget), "--child"]
            result = subprocess.run(command, text=True, capture_output=True, timeout=300)
            if result.returncode:
                raise RuntimeError(f"{kind}/{count}/{mode}/{rows}: {result.stderr[-1000:]}")
            record = json.loads(result.stdout)
            record["repetition"] = repetition
            records.append(record)
            with (args.out / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted({key for row in records for key in row}))
                writer.writeheader(); writer.writerows(records)
            print(kind, count or record["face_count"], mode, rows, repetition,
                  round(record["elapsed_seconds"], 3), flush=True)
    (args.out / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
