"""Private, real-photo PDF for the isolated Phase 8G.5 calibration audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.tools.report_subject_quality import draw_photo, magnified_region


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def independent_face_observations(rows, image_id):
    """Image-local observations; face indexes never certify cross-photo identity."""
    selected = sorted((r for r in rows if r["image_id"] == image_id and r["status"] == "measured"),
                      key=lambda r: int(r["face_index"]))[:2]
    return [dict(row, region_label=f"{image_id}:ROI{row['face_index']}") for row in selected]


def font(size, bold=False):
    path = Path("C:/Windows/Fonts")/("segoeuib.ttf" if bold else "segoeui.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def label(draw, xy, value, size=30, color="#25394e", bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=color)


def plot(draw, rows, keys, box, title):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=20, fill="#e9f0f5")
    label(draw, (x0+30, y0+20), title, 30, bold=True)
    selected = [r for r in rows if r["variant"] in keys]
    values = [float(r["noise_aware_detail"]) for r in selected]
    base = values[0] if values else 1
    scale = max(values+[.01])
    for index, row in enumerate(selected):
        y = y0+90+index*70
        label(draw, (x0+35, y), row["variant"], 23)
        draw.rectangle((x0+280, y+5, x0+280+min(420, 420*float(row["noise_aware_detail"])/scale), y+34),
                       fill="#206d9a" if index == 0 else "#b48624")
        label(draw, (x0+730, y), f"{float(row['noise_aware_detail']):.3f}", 23)
    return base


def line_chart(draw, rows, keys, field, box, title):
    """Plot only measured values; unavailable data are explicitly named."""
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill="#e9f0f5")
    label(draw, (x0+20, y0+15), title, 22, bold=True)
    points = []
    for key in keys:
        row = next((item for item in rows if item["variant"] == key), None)
        if row is not None and row.get(field) not in (None, ""):
            points.append((key, float(row[field])))
    if not points:
        label(draw, (x0+20, y0+110), "Unavailable / no calibrated score", 19, "#9d6817")
        return
    low, high = min(v for _, v in points), max(v for _, v in points)
    if high == low:
        high += max(.01, abs(high)*.01)
        low -= max(.01, abs(low)*.01)
    left, right, top, bottom = x0+70, x1-35, y0+80, y1-85
    draw.line((left, bottom, right, bottom), fill="#90a3b4", width=3)
    draw.line((left, top, left, bottom), fill="#90a3b4", width=3)
    coords = []
    for i, (name, value) in enumerate(points):
        x = left + (right-left)*i/max(1, len(points)-1)
        y = bottom - (bottom-top)*(value-low)/(high-low)
        coords.append((x, y))
        draw.ellipse((x-6, y-6, x+6, y+6), fill="#206d9a")
        label(draw, (x-55, bottom+14), name.replace("_", " "), 15)
        label(draw, (x-35, y-27), f"{value:.3f}", 15)
    if len(coords) > 1:
        draw.line(coords, fill="#206d9a", width=4)


def controlled_chart_pages(rows, comparisons, cold, warm):
    pages = []
    panels = [
        (["baseline", "luma_noise_5", "luma_noise_12", "luma_noise_25"], "baseline_laplacian", "Noise / raw Laplacian"),
        (["baseline", "luma_noise_5", "luma_noise_12", "luma_noise_25"], "noise_aware_detail", "Noise / adjusted detail"),
        (["baseline", "gaussian_1", "gaussian_2", "gaussian_3"], "noise_aware_detail", "Gaussian blur / adjusted detail"),
        (["baseline", "gaussian_2", "blur_plus_noise_5", "blur_plus_noise_12"], "noise_aware_detail", "Blur + noise / adjusted detail"),
        (["baseline", "chroma_noise_5", "chroma_noise_12", "chroma_noise_25"], "chroma_noise", "Chroma injection / float-Lab residual"),
        (["baseline", "chroma_noise_5", "chroma_noise_12", "chroma_noise_25"], "chroma_coverage", "Chroma injection / eligible coverage"),
        (["baseline", "unsharp_0.5", "unsharp_1", "unsharp_2"], "noise_aware_detail", "Unsharp masking / false reward"),
        (["baseline", "jpeg_90", "jpeg_60", "jpeg_25"], "noise_aware_detail", "JPEG re-encode / unstable response"),
        (["baseline", "luma_noise_5", "luma_noise_12", "luma_noise_25"], "baseline_raw_log", "Noise / baseline raw-log score"),
        (["baseline", "chroma_noise_5", "chroma_noise_12", "chroma_noise_25"], "noise_aware_detail", "Chroma injection / adjusted detail"),
    ]
    for page_index in range(2):
        page = Image.new("RGB", (1800, 2500), "#f8fafc")
        draw = ImageDraw.Draw(page)
        label(draw, (80, 65), f"Controlled responses / {page_index+1} of 2", 50, bold=True)
        label(draw, (80, 135), "Source: real NSC face crop; deterministic in-memory perturbations", 25)
        for j, (keys, field, title) in enumerate(panels[page_index*5:(page_index+1)*5]):
            col, row = j % 2, j // 2
            line_chart(draw, rows, keys, field,
                       (80+col*850, 230+row*650, 900+col*850, 790+row*650), title)
        if page_index == 1:
            label(draw, (80, 2180), "These curves demonstrate response, not photographic accuracy.", 25)
        pages.append(page)
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    draw = ImageDraw.Draw(page)
    label(draw, (80, 70), "Group margins and cache performance", 51, bold=True)
    y = 210
    for row in comparisons:
        margin = float(row["baseline_margin"])
        label(draw, (90, y), f"{row['v1_winner']} / {row['v2_winner']}", 23)
        draw.rectangle((800, y+5, 800+min(500, margin*150000), y+32), fill="#206d9a")
        label(draw, (1350, y), f"{margin:.6f}", 23)
        y += 110
    label(draw, (90, 1190), "Candidate margins unavailable: no validated 0-1 transformation.", 28, "#9d6817")
    label(draw, (90, 1330), "Cold and warm local audit runs", 37, bold=True)
    for name, result, ypos in (("Cold", cold, 1440), ("Warm", warm, 1630)):
        label(draw, (100, ypos), f"{name}: {result['elapsed_seconds']:.3f} s; {result['additional_decodes']} decodes; "
              f"{result['cache_hits']} hits; peak RSS {result['peak_process_rss']/1048576:.1f} MiB", 27)
    label(draw, (90, 1870), "Peak RSS is whole-process RSS sampled after each candidate.", 25)
    label(draw, (90, 1930), "It is not incremental allocation or a high-frequency memory peak.", 25)
    label(draw, (90, 2050), "No V1, default V2, or experimental keeper decision was changed.", 27)
    pages.append(page)
    return pages


def build(audit, output):
    if not audit.is_dir():
        raise FileNotFoundError(f"Real audit directory is missing: {audit}")
    summary = json.loads((audit/"summary.json").read_text(encoding="utf-8"))
    cold_path = audit/"cold_summary.json"
    cold = json.loads(cold_path.read_text(encoding="utf-8")) if cold_path.is_file() else summary
    photos = read(audit/"photo_metrics.csv")
    faces = read(audit/"face_metrics.csv")
    all_controls = read(audit/"controlled.csv")
    controls = [r for r in all_controls if r["group_id"] == all_controls[0]["group_id"]] if all_controls else []
    comparisons = read(audit/"group_comparisons.csv")
    # Phase 8G.4 geometry-only links are tentative without independently
    # verified identity. This report never pairs faces by index.
    eligible = []
    for group in comparisons:
        members = [p for p in photos if p["group_id"] == group["group_id"]]
        if len(members) != 2:
            continue
        if all(sum(f["status"] == "measured" for f in faces if f["image_id"] == p["image_id"]) >= 2 for p in members):
            eligible.append((float(group["baseline_margin"]), group, members))
    if not eligible:
        raise ValueError("No real pair with two measurable face regions in both photos")
    _, comparison, members = min(eligible, key=lambda item: item[0])
    left, right = members
    lf = independent_face_observations(faces, left["image_id"])
    rf = independent_face_observations(faces, right["image_id"])
    pages = []
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    label(d, (90, 100), "Room36 / Technical Calibration", 69, bold=True)
    label(d, (90, 205), "Phase 8G.5 corrective pass | Native ROI audit", 36)
    d.rounded_rectangle((90, 350, 1710, 810), radius=30, fill="#e5eef4")
    label(d, (130, 405), "Decision: diagnostic only", 52, "#9d6817", True)
    label(d, (130, 515), "Noise-aware detail resists tested noise and blur,", 35)
    label(d, (130, 575), "but oversharpening and JPEG artifacts remain confounds.", 35)
    label(d, (130, 665), "No candidate score or keeper change was activated.", 35)
    for i, line in enumerate((
        "8 NSC groups; 17 real candidates; cached faces reused.",
        "Float-Lab chroma residual stays available at tested high noise.",
        f"Native ROI detail measured: {sum(f['status'] == 'measured' for f in faces)}/{len(faces)}; no upsampling.",
        "A decoded native ROI is not a calibrated sensor-noise measurement.",
        "V1 production and default shadow V2 are unchanged.")):
        label(d, (110, 930+i*105), line, 32)
    label(d, (90, 1700), "Go / no-go for 8G.6", 48, bold=True)
    label(d, (105, 1795), "CONDITIONAL: float-Lab chroma diagnostic for controlled review.", 31)
    label(d, (105, 1860), "DIAGNOSTIC: noise-aware detail and face clipping.", 31)
    label(d, (105, 1925), "NOT READY: any directional technical score.", 31, "#9d6817", True)
    pages.append(page)
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    label(d, (80, 75), "Controlled degradation: real face crop", 53, bold=True)
    label(d, (90, 155), f"Source: {controls[0]['source_image']} | copies transformed in memory", 29)
    plot(d, controls, ["baseline", "gaussian_1", "gaussian_2", "gaussian_3"],
         (80, 260, 1710, 680), "Gaussian blur: provisional adjusted detail")
    plot(d, controls, ["baseline", "luma_noise_5", "luma_noise_12", "luma_noise_25"],
         (80, 740, 1710, 1160), "Luminance noise: adjusted detail")
    plot(d, controls, ["baseline", "unsharp_0.5", "unsharp_1", "unsharp_2"],
         (80, 1220, 1710, 1640), "Sharpening: known failure")
    baseline = next(r for r in controls if r["variant"] == "baseline")
    noisy = next(r for r in controls if r["variant"] == "luma_noise_25")
    sharpened = next(r for r in controls if r["variant"] == "unsharp_2")
    label(d, (90, 1740), f"Raw-log: {float(baseline['baseline_raw_log']):.3f} to {float(noisy['baseline_raw_log']):.3f} at luma-noise 25.", 29)
    label(d, (90, 1800), f"Laplacian: {float(baseline['baseline_laplacian']):.1f} to {float(noisy['baseline_laplacian']):.1f}.", 29)
    label(d, (90, 1860), f"Adjusted detail: {float(baseline['noise_aware_detail']):.3f} to {float(noisy['noise_aware_detail']):.3f}.", 29)
    label(d, (90, 1920), f"Strong sharpening: {float(sharpened['noise_aware_detail']):.3f}; score withheld.", 29, "#9d6817")
    pages.append(page)
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    label(d, (80, 70), "Real NSC pair / diagnostic comparison", 54, bold=True)
    label(d, (80, 145), f"{left['image_id']} vs {right['image_id']} | INCONCLUSIVE", 32)
    draw_photo(page, left["source_path"], (60, 245), (810, 740),
               [json.loads(f["box"]) for f in lf],
               (int(left["decoded_width"]), int(left["decoded_height"])))
    draw_photo(page, right["source_path"], (930, 245), (810, 740),
               [json.loads(f["box"]) for f in rf],
               (int(right["decoded_width"]), int(right["decoded_height"])))
    d = ImageDraw.Draw(page)
    label(d, (80, 1010), "Circles mark measured ROIs. Identity correspondence is UNVERIFIED.", 28)
    label(d, (80, 1100), f"Existing technical: {float(left['baseline_score']):.6f} vs {float(right['baseline_score']):.6f}", 30)
    label(d, (80, 1160), f"Whole-image adjusted detail: {float(left['noise_aware_detail']):.4f} vs {float(right['noise_aware_detail']):.4f}", 30)
    for i, face in enumerate(lf):
        label(d, (80, 1260+i*65), f"A ROI {face['face_index']}: {float(face['noise_aware_detail']):.4f} (independent)", 25)
    for i, face in enumerate(rf):
        label(d, (900, 1260+i*65), f"B ROI {face['face_index']}: {float(face['noise_aware_detail']):.4f} (independent)", 25)
    label(d, (80, 1400), f"A: {float(left['detail']):.4f} x {float(left['edge_persistence']):.4f} - {float(left['noise_floor']):.4f} = {float(left['noise_aware_detail']):.4f}", 27)
    label(d, (80, 1450), f"B: {float(right['detail']):.4f} x {float(right['edge_persistence']):.4f} - {float(right['noise_floor']):.4f} = {float(right['noise_aware_detail']):.4f}", 27)
    label(d, (80, 1500), "Calculation: median fine gradient × cross-scale persistence − noise floor.", 28)
    label(d, (80, 1550), "That uncalibrated value is not a 0–1 quality score.", 28)
    label(d, (80, 1600), "Face clipping does not prove lost information; no directional penalty.", 28)
    if lf and rf:
        page.paste(magnified_region(left["source_path"], json.loads(lf[0]["box"]),
                                    (int(left["decoded_width"]), int(left["decoded_height"]))), (90, 1780))
        page.paste(magnified_region(right["source_path"], json.loads(rf[0]["box"]),
                                    (int(right["decoded_width"]), int(right["decoded_height"]))), (930, 1780))
        d = ImageDraw.Draw(page)
        label(d, (90, 2110), "Independent ROI crops; no verified same-person comparison.", 27)
    pages.append(page)
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    label(d, (80, 75), "Calibration, availability and performance", 54, bold=True)
    label(d, (90, 170), "Algorithm v5 / separate evidence fields", 38, bold=True)
    for i, line in enumerate((
        "Fine: median Sobel magnitude after Gaussian sigma 0.65.",
        "Coarse: median gradient after Gaussian sigma 2.0 on strong-edge mask.",
        "Persistence: median coarse/(fine+0.002), clipped to [0,1].",
        "Noise floor: median high-pass luminance residual in low-coarse-edge pixels.",
        "Diagnostic adjusted detail: max(0, fine*persistence-noise floor).",
        "Float-Lab noise: median chroma residual plus 1.4826 MAD on 35% smoothest luminance.",
        "Artifact and chroma reliability are separate; technical score = unavailable.")):
        label(d, (100, 255+i*95), line, 26)
    label(d, (100, 940), "Artifact suspicion: persistence < 0.65 or 8-pixel boundary ratio > 1.5 (provisional).", 25)
    label(d, (90, 1050), "All NSC groups", 38, bold=True)
    for i, row in enumerate(comparisons):
        label(d, (105, 1125+i*90), f"{row['group_id']}  baseline range {float(row['baseline_margin']):.5f}  |  new score withheld", 26)
    label(d, (90, 1910), f"Cold: {cold['additional_decodes']} decodes; {cold['elapsed_seconds']:.3f} s; RSS {cold['peak_process_rss']/1048576:.1f} MiB.", 27)
    label(d, (90, 1970), f"Warm: {summary['cache_hits']} hits; {summary['elapsed_seconds']:.3f} s; RSS {summary['peak_process_rss']/1048576:.1f} MiB.", 27)
    label(d, (90, 2080), "Next: validate against sharpening, JPEG, texture and labeled technical preferences.", 27)
    pages.append(page)
    pages.extend(controlled_chart_pages(controls, comparisons, cold, summary))
    separation = Image.new("RGB", (1800, 2500), "#f8fafc")
    sd = ImageDraw.Draw(separation)
    label(sd, (80, 70), "Isolated luminance and chroma perturbations", 47, bold=True)
    line_chart(sd, controls, ["baseline", "luma_noise_5", "luma_noise_12", "luma_noise_25"],
               "chroma_noise", (80, 210, 900, 820), "Chroma estimator / luminance-only")
    line_chart(sd, controls, ["baseline", "chroma_noise_5", "chroma_noise_12", "chroma_noise_25"],
               "chroma_noise", (930, 210, 1750, 820), "Chroma estimator / chroma-only")
    line_chart(sd, controls, ["baseline", "luma_noise_5", "luma_noise_12", "luma_noise_25"],
               "observed_chroma_std", (80, 900, 900, 1510), "Luminance injection / chroma leakage")
    line_chart(sd, controls, ["baseline", "chroma_noise_5", "chroma_noise_12", "chroma_noise_25"],
               "observed_luma_std", (930, 900, 1750, 1510), "Chroma injection / luminance leakage")
    label(sd, (80, 1640), "Lab conversion and RGB gamut clipping introduce measured cross-channel leakage.", 27)
    label(sd, (80, 1710), "Neither curve is calibrated recognition accuracy or a keeper preference.", 27)
    pages.append(separation)
    original = Image.new("RGB", (1800, 2500), "#f8fafc")
    label(ImageDraw.Draw(original), (80, 80), "Original photographs / no annotations", 49, bold=True)
    draw_photo(original, left["source_path"], (60, 220), (810, 740), [],
               (int(left["decoded_width"]), int(left["decoded_height"])))
    draw_photo(original, right["source_path"], (930, 220), (810, 740), [],
               (int(right["decoded_width"]), int(right["decoded_height"])))
    label(ImageDraw.Draw(original), (90, 1050), "Annotation page circles measured regions; no identity claim.", 29)
    pages.append(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(pages, 1):
        page.save(output.parent/f"report_page_{index}.png")
    pages[0].save(output, save_all=True, append_images=pages[1:], resolution=180)
    return dict(pages=len(pages), featured=[left["image_id"], right["image_id"]],
                outcome="inconclusive", candidate_score_active=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("audit_results/phase8g5_technical_calibration"))
    parser.add_argument("--output", type=Path, default=Path("audit_results/phase8g5_technical_calibration/Room36_Phase_8G5_Technical_Calibration_Revised.pdf"))
    args = parser.parse_args()
    print(json.dumps(build(args.audit, args.output), indent=2))


if __name__ == "__main__":
    main()
