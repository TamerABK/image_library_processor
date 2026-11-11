"""Private, score-blind review sheets and gated pairwise calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from automatic_scan.technical_pairwise_calibration import LABELS, calibrate, split_events, validate_label

DIMENSIONS = ("focus_detail", "noise", "exposure_clipping", "processing_artifacts",
              "technical_preference", "overall_keeper_preference")


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def sample_pairs(groups, seed=3605):
    rng = np.random.default_rng(seed)
    rows = []
    for group in sorted(groups, key=lambda g: g["group_id"]):
        photos = group["candidate_ids"].split("|")
        event = Path(photos[0]).parent.parent.name or "unknown_event"
        for left, right in itertools.combinations(photos, 2):
            if rng.integers(2):
                left, right = right, left
            pair_id = hashlib.sha256((group["group_id"]+"|"+"|".join(sorted((left, right)))).encode()).hexdigest()[:16]
            row = dict(pair_id=pair_id, event_id=event, group_id=group["group_id"],
                       photo_a=left, photo_b=right, reviewer_id="", reason_tags="", region_notes="")
            row.update({key: "" for key in DIMENSIONS})
            rows.append(row)
    return rows


def sheet(pair, output):
    canvas = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 20), f"Pair {pair['pair_id']} - score-blind technical review", fill="black")
    for side, x in (("a", 40), ("b", 820)):
        path = Path(pair[f"photo_{side}"])
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((740, 850))
            canvas.paste(image, (x, 90))
        draw.text((x, 950), f"{side.upper()}: {path.name}", fill="black")
    canvas.save(output)


def export(audit, output, seed=3605, sheets=True):
    output.mkdir(parents=True, exist_ok=True)
    groups = [r for r in read(audit/"ranking_v1_v2_comparison.csv")
              if r["preference_id"] == "face_1_eyes_1"]
    rows = sample_pairs(groups, seed)
    write(output/"review_labels_template.csv", rows)
    split = split_events((r["event_id"] for r in rows), seed)
    (output/"split_manifest.json").write_text(json.dumps(dict(seed=seed, event_split=split,
        pairs=len(rows), groups=len(groups), dimensions=DIMENSIONS), indent=2), encoding="utf-8")
    if sheets:
        directory = output/"review_sheets"; directory.mkdir(exist_ok=True)
        for row in rows:
            sheet(row, directory/(row["pair_id"]+".jpg"))
    return dict(groups=len(groups), pairs=len(rows), sheets=sheets, seed=seed)


def evaluate(labels, metrics, output, seed=3605):
    rows = read(labels)
    for row in rows:
        for dimension in DIMENSIONS:
            if row.get(dimension) and row[dimension] not in LABELS:
                raise ValueError(f"Invalid {dimension} label: {row[dimension]}")
    labeled = [validate_label(row) for row in rows if row["technical_preference"]]
    metric_rows = {row["source_path"]: row for row in read(metrics)}
    for row in metric_rows.values():
        for key in ("noise_aware_detail", "chroma_noise", "artifact_measurement"):
            row[key] = float(row[key]) if row.get(key) not in ("", None) else None
    result = calibrate(labeled, metric_rows, seed=seed)
    counts = {choice: sum(r["technical_preference"] == choice for r in labeled)
              for choice in ("A", "B", "equal", "both_unacceptable", "cannot_determine")}
    pair_votes = {}
    for row in labeled:
        pair_votes.setdefault(row["pair_id"], []).append(row["technical_preference"])
    multiply_reviewed = [votes for votes in pair_votes.values() if len(votes) > 1]
    summary = dict(result.__dict__, label_counts=counts,
                   reviewers=len({r.get("reviewer_id") for r in labeled if r.get("reviewer_id")}),
                   multiply_reviewed_pairs=len(multiply_reviewed),
                   reviewer_disagreement_pairs=sum(len(set(votes)) > 1 for votes in multiply_reviewed))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "evaluate"))
    parser.add_argument("--audit", type=Path, default=Path("audit_results/nsc_fresh"))
    parser.add_argument("--output", type=Path, default=Path("audit_results/phase8g5_final_calibration"))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--metrics", type=Path,
                        default=Path("audit_results/phase8g5_technical_calibration/corrected_native_final/photo_metrics.csv"))
    parser.add_argument("--seed", type=int, default=3605)
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        result = export(args.audit, args.output, args.seed, not args.no_sheets)
    else:
        result = evaluate(args.labels or args.output/"review_labels_template.csv", args.metrics,
                          args.output/"calibration_result.json", args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
