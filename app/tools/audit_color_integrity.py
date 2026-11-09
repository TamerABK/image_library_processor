"""Run isolated color diagnostics on confirmed duplicate candidates.

Example: python -m app.tools.audit_color_integrity --audit-dir audit_results/nsc_fresh
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil

from automatic_scan.color_integrity_experiment import VERSION, measure_color, compare_white_balance
from automatic_scan.keeper_ranker_v2_adapter import technical_feature
from image_loader import ImageLoader


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path, entries):
    if not entries:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, sorted({k for row in entries for k in row}), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)


def key(path, dimension):
    stat = path.stat()
    return hashlib.sha256(json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                                      dimension, VERSION, "ImageLoader-BGR-v1"]).encode()).hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    groups = rows(args.groups or args.audit_dir / "ranking_v1_v2_comparison.csv")
    groups = [g for g in groups if g.get("preference_id") == "face_1_eyes_1"]
    photos = {p["image_path"]: p for p in rows(args.audit_dir / "photo_metrics.csv")}
    loader = ImageLoader(max_decode_dimension=args.max_dimension)
    cache_dir = args.output / "cache"
    cache_dir.mkdir(exist_ok=True)
    observations, performance = {}, []
    start = time.perf_counter()
    process = psutil.Process()
    peak = process.memory_info().rss
    hits = decodes = 0
    for group in groups:
        for name in group["candidate_ids"].split("|"):
            if name in observations:
                continue
            path = Path(name)
            if args.dataset and not path.is_file():
                matches = list(args.dataset.rglob(path.name))
                if len(matches) == 1:
                    path = matches[0]
            try:
                signature = key(path, args.max_dimension)
                cached = cache_dir / (signature + ".json")
                if cached.is_file():
                    item = json.loads(cached.read_text(encoding="utf-8"))
                    hits += 1
                    elapsed = 0.0
                else:
                    t0 = time.perf_counter()
                    image = loader.load_for_scan(path, max_dimension=args.max_dimension)
                    decodes += 1
                    item = measure_color(image)
                    del image
                    elapsed = time.perf_counter() - t0
                    cached.write_text(json.dumps(item), encoding="utf-8")
                observations[name] = item
                performance.append(dict(image_id=path.name, status=item["status"], seconds=elapsed,
                                        cache_hit=int(elapsed == 0), width=item.get("width", ""),
                                        height=item.get("height", "")))
            except (OSError, ValueError, cv2.error) as exc:
                observations[name] = dict(status="unavailable", reason=type(exc).__name__, version=VERSION)
            peak = max(peak, process.memory_info().rss)
    measurements, comparisons, variants, labels, group_summaries = [], [], [], [], []
    for group in groups:
        gid = group["group_id"]
        paths = group["candidate_ids"].split("|")
        observed = [observations[p] for p in paths]
        def span(field):
            values = [item[field] for item in observed if item.get(field) is not None]
            return max(values) - min(values) if len(values) == len(paths) else None
        group_summaries.append(dict(group_id=gid, candidate_count=len(paths),
            clipping_range=span("clipping_max"), noise_range=span("chroma_noise"),
            fringing_range=span("fringing"),
            noise_eligible=int(all(item.get("chroma_noise") is not None for item in observed)),
            color_winner_change="not_evaluated_no_validated_direction"))
        for name in paths:
            obs = observations[name]
            measurements.append(dict(group_id=gid, image_id=Path(name).name, **{
                k: json.dumps(v) if isinstance(v, list) else v for k, v in obs.items()}))
            p = photos.get(name, {})
            raw = {field: float(p["technical_raw_" + field]) for field in
                   ("laplacian", "sobel", "local_contrast") if p.get("technical_raw_" + field)}
            baseline = technical_feature(raw).value if len(raw) == 3 else None
            # Each variant is a diagnostic candidate. No color term is activated
            # without an independently validated defect direction.
            for variant in ("A_cached_baseline", "B_color_diagnostics", "C_clipping",
                            "D_chroma_noise", "E_fringing", "F_white_balance", "G_selected"):
                variants.append(dict(group_id=gid, image_id=Path(name).name, variant=variant,
                                     baseline_technical=baseline, experimental_score="",
                                     status="diagnostic_only_no_validated_direction"))
        for i, a in enumerate(paths):
            for b in paths[i+1:]:
                delta, state = compare_white_balance(observations[a], observations[b])
                comparisons.append(dict(group_id=gid, image_a=Path(a).name, image_b=Path(b).name,
                                        wb_chroma_delta=delta, wb_status=state,
                                        clipped_fraction_delta=(abs(observations[a].get("clipping_max", 0) -
                                                                    observations[b].get("clipping_max", 0))
                                                                if observations[a].get("status") == "measured" and
                                                                   observations[b].get("status") == "measured" else "")))
                labels.append(dict(group_id=gid, image_a=Path(a).name, image_b=Path(b).name,
                                   technical_preference="", reason="", confidence=""))
    write(args.output / "color_metrics.csv", measurements)
    write(args.output / "color_group_comparisons.csv", comparisons)
    write(args.output / "technical_variants.csv", variants)
    write(args.output / "review_labels_template.csv", labels)
    write(args.output / "performance.csv", performance)
    write(args.output / "color_group_summary.csv", group_summaries)
    correlations = []
    for field in ("clipping_max", "chroma_noise", "fringing", "mean_luminance"):
        paired = [(observations[name].get(field), photos.get(name, {}).get("technical_raw_local_contrast"))
                  for name in observations]
        paired = [(float(a), float(b)) for a, b in paired if a is not None and b not in (None, "")]
        valid = (len(paired) >= 4 and np.std([a for a, _ in paired]) > 0
                 and np.std([b for _, b in paired]) > 0)
        correlation = float(np.corrcoef(np.asarray(paired).T)[0, 1]) if valid else None
        correlations.append(dict(metric=field, comparison="cached_local_contrast", pairs=len(paired),
                                 pearson=correlation))
    write(args.output / "metric_correlations.csv", correlations)
    if args.contact_sheets:
        from PIL import Image, ImageDraw, ImageOps
        sheet_dir = args.output / "contact_sheets"
        sheet_dir.mkdir(exist_ok=True)
        for group in groups:
            paths = group["candidate_ids"].split("|")
            canvas = Image.new("RGB", (520 * len(paths), 400), "white")
            draw = ImageDraw.Draw(canvas)
            for index, name in enumerate(paths):
                try:
                    with Image.open(name) as source:
                        thumb = ImageOps.exif_transpose(source).convert("RGB")
                        thumb.thumbnail((500, 300))
                    canvas.paste(thumb, (index * 520 + (520-thumb.width)//2, 8))
                except OSError:
                    pass
                item = observations[name]
                draw.text((index*520+10, 315), Path(name).name, fill="black")
                draw.text((index*520+10, 335), f"clipping {item.get('clipping_max', 'NA')}", fill="black")
                draw.text((index*520+10, 355), f"chroma residual {item.get('chroma_noise', 'NA')}", fill="black")
            canvas.save(sheet_dir / (group["group_id"] + ".jpg"), quality=85)
    summary = dict(version=VERSION, groups=len(groups), candidates=len(observations),
                   additional_decodes=decodes, cache_hits=hits,
                   elapsed_seconds=time.perf_counter()-start, peak_process_rss=peak,
                   directional_color_metrics_adopted=0,
                   human_accuracy="unavailable_without_labels")
    (args.output / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.controlled:
        controlled(args.output, args.seed)
    return summary


def controlled(output, seed):
    """Synthetic directions and ambiguous styles, never photographic accuracy."""
    rng = np.random.default_rng(seed)
    base = np.full((256, 256, 3), 80, np.uint8)
    cv2.circle(base, (128, 128), 55, (130, 130, 130), -1)
    records = []
    for severity in (0, 4, 8, 16, 32):
        noisy = np.clip(base.astype(np.int16) + rng.normal(0, severity, base.shape), 0, 255).astype(np.uint8)
        red = base.copy()
        red[:, :severity*4, 2] = 255
        for kind, image in (("chroma_noise", noisy), ("red_channel_clip", red)):
            obs = measure_color(image)
            records.append(dict(kind=kind, severity=severity,
                                clipping_max=obs.get("high_r"), chroma_noise=obs.get("chroma_noise"),
                                noise_coverage=obs.get("noise_coverage")))
    warm = base.copy()
    warm[:, :, 2] = 170
    saturated = np.full_like(base, (0, 0, 255))
    for kind, image in (("warm_grade_ambiguous", warm), ("saturated_scene_ambiguous", saturated)):
        obs = measure_color(image)
        records.append(dict(kind=kind, severity=0, clipping_max=obs.get("high_r"),
                            chroma_noise=obs.get("chroma_noise"), noise_coverage=obs.get("noise_coverage")))
    write(output / "controlled_perturbations.csv", records)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    for kind, field, filename in (("chroma_noise", "chroma_noise", "noise_response.svg"),
                                  ("red_channel_clip", "clipping_max", "clipping_response.svg")):
        selected = [r for r in records if r["kind"] == kind and r[field] is not None]
        ceiling = max(float(r[field]) for r in selected) or 1
        points = " ".join(f"{50+r['severity']*12:.1f},{250-200*float(r[field])/ceiling:.1f}" for r in selected)
        title = kind.replace("_", " ")
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="520" height="320" viewBox="0 0 520 320">'
               f'<rect width="520" height="320" fill="white"/><text x="50" y="30" font-size="18">{title}</text>'
               f'<path d="M50 50 V250 H460" fill="none" stroke="#333"/>'
               f'<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="3"/>'
               f'<text x="50" y="290" font-size="12">Synthetic severity; omitted points = unavailable</text>'
               f'</svg>')
        (plots / filename).write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--audit-dir", type=Path, default=Path("audit_results/nsc_fresh"))
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--output", type=Path, default=Path("audit_results/phase8g3b_color_integrity"))
    parser.add_argument("--max-dimension", type=int, default=1024)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--contact-sheets", action="store_true")
    parser.add_argument("--seed", type=int, default=3603)
    args = parser.parse_args()
    if not 256 <= args.max_dimension <= 2048:
        parser.error("max-dimension must be 256..2048")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
