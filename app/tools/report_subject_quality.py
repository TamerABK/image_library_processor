"""Create a private real-photo PDF from the Phase 8G.4 audit outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text(draw, xy, content, size=30, color="#263b50", bold=False):
    draw.text(xy, str(content), fill=color, font=font(size, bold))


def pair_score(rows, scene):
    valid = [r for r in rows if r["laplacian_difference"] != ""]
    if not valid:
        return -1
    # Select interpretable local evidence, not the largest raw score margin.
    return len(valid) * float(scene["inlier_ratio"]) * sum(
        abs(float(r["laplacian_difference"]))/(float(r["laplacian_a"])+float(r["laplacian_b"])+1)
        for r in valid)/len(valid)


def draw_photo(canvas, source, offset, box_size, circles, decoded_size):
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    image.thumbnail(box_size)
    canvas.paste(image, offset)
    draw = ImageDraw.Draw(canvas)
    sx, sy = image.width/decoded_size[0], image.height/decoded_size[1]
    for index, box in enumerate(circles):
        x0, y0, x1, y1 = box
        x0, x1 = offset[0]+x0*sx, offset[0]+x1*sx
        y0, y1 = offset[1]+y0*sy, offset[1]+y1*sy
        radius = max(x1-x0, y1-y0)*.7
        center = ((x0+x1)/2, (y0+y1)/2)
        color = "#d89719" if index == 0 else "#2077a8"
        draw.ellipse((center[0]-radius, center[1]-radius,
                      center[0]+radius, center[1]+radius), outline=color, width=7)
    return image


def magnified_region(source, decoded_box, decoded_size, size=(320, 250)):
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    x0, y0, x1, y1 = decoded_box
    sx, sy = image.width/decoded_size[0], image.height/decoded_size[1]
    pad_x, pad_y = (x1-x0)*.35, (y1-y0)*.35
    box = (max(0, int((x0-pad_x)*sx)), max(0, int((y0-pad_y)*sy)),
           min(image.width, int((x1+pad_x)*sx)), min(image.height, int((y1+pad_y)*sy)))
    return image.crop(box).resize(size, Image.Resampling.LANCZOS)


def build(audit, output):
    summary = json.loads((audit/"summary.json").read_text(encoding="utf-8"))
    scenes, links = read(audit/"scene_regions.csv"), read(audit/"matched_faces.csv")
    photos = read(audit/"photo_metrics.csv")
    rankings = {r["group_id"]: r for r in read(audit/"ranking_comparison.csv")}
    by_name = {r["image_id"]: r for r in photos}
    options = []
    for scene in scenes:
        related = [r for r in links if r["group_id"] == scene["group_id"]
                   and r["image_b"] == scene["image_b"]]
        options.append((pair_score(related, scene), scene, related))
    _, chosen, related = max(options, key=lambda entry: entry[0])
    valid = sorted((r for r in related if r["laplacian_difference"] != ""),
                   key=lambda r: abs(float(r["laplacian_difference"])), reverse=True)
    negative = next((s for s in scenes if not any(r["group_id"] == s["group_id"] for r in links)), scenes[0])
    a, b = chosen["image_a"], chosen["image_b"]
    pa, pb = by_name[a], by_name[b]
    ranking = rankings[chosen["group_id"]]
    pages = []
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    text(d, (100, 110), "Room36 / Subject Quality", 74, bold=True)
    text(d, (100, 215), "Phase 8G.4   |   NSC Meeting real-photo audit", 35)
    d.rounded_rectangle((100, 350, 1700, 750), radius=35, fill="#e7eff5")
    text(d, (145, 390), "Evaluation outcome: INCONCLUSIVE", 50, bold=True)
    text(d, (145, 475), f"{summary['groups']} duplicate groups  /  {summary['candidates']} candidates", 34)
    text(d, (145, 535), f"{summary['tentative_links']} tentative geometric face links", 34)
    text(d, (145, 595), "0 identity-verified links; no validated keeper changes", 34)
    text(d, (100, 870), "Why we abstain", 48, bold=True)
    for i, line in enumerate(("Cached face analyses were available, but ArcFace embeddings were absent.",
                               "Registration links facial regions; it cannot prove person identity.",
                               "Laplacian energy can reflect detail, noise, processing, or pose.",
                               "No human keeper labels exist for these eight groups.")):
        text(d, (110, 960+i*78), line, 32)
    text(d, (100, 1420), "Methods", 48, bold=True)
    for i, line in enumerate(("ORB + RANSAC homography aligns nearby frames.",
                               "One-to-one face links require spatial fit and ambiguity margin.",
                               "Face crops at <=1024 px are measured for detail and clipping.",
                               "All added observations remain diagnostic; V1/V2 are unchanged.")):
        text(d, (110, 1510+i*78), line, 32)
    text(d, (100, 2240), "Data: local NSC Meeting photographs and cached Room36 audit.", 26)
    pages.append(page)
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    text(d, (80, 75), "All groups and controlled checks", 55, bold=True)
    text(d, (80, 160), "Group-local links are tentative; zero identities are verified.", 30)
    table = read(audit/"ranking_comparison.csv")
    text(d, (90, 270), "Group", 29, bold=True)
    text(d, (650, 270), "V1 / V2", 29, bold=True)
    text(d, (1270, 270), "Links", 29, bold=True)
    for index, row in enumerate(table):
        y = 340 + index*95
        d.line((80, y-15, 1720, y-15), fill="#cbd5df", width=2)
        text(d, (90, y), row["group_id"], 25)
        text(d, (650, y), f"{row['v1_winner']} / {row['v2_winner']}", 25)
        text(d, (1290, y), row["tentative_face_links"], 25)
    text(d, (80, 1170), "Controlled face-crop perturbations", 42, bold=True)
    controls = read(audit/"controlled_perturbations.csv")
    for index, row in enumerate(controls):
        y = 1260 + index*112
        value = float(row["laplacian"])
        text(d, (85, y), row["perturbation"].replace("_", " "), 27)
        d.rectangle((470, y+5, 470+min(1000, value*.55), y+39), fill="#24749e")
        text(d, (1480, y), f"{value:.1f}", 26)
    text(d, (85, 2100), "Laplacian variance rises under added noise and synthetic occlusion.", 27)
    text(d, (85, 2150), "Thus higher edge energy cannot be used as a quality reward.", 27)
    text(d, (85, 2240), f"17 decodes; {summary['elapsed_seconds']:.3f} s audit; peak RSS {summary['peak_process_rss']/1048576:.1f} MiB.", 27)
    pages.append(page)
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    text(d, (80, 70), "Featured real-photo comparison", 56, bold=True)
    text(d, (80, 145), f"{a}  vs  {b}   |   INCONCLUSIVE", 32)
    circles_a = [json.loads(r["box_a"]) for r in valid[:2]]
    circles_b = [json.loads(r["box_b"]) for r in valid[:2]]
    draw_photo(page, pa["source_path"], (60, 245), (810, 750), circles_a,
               (int(pa["decoded_width"]), int(pa["decoded_height"])))
    draw_photo(page, pb["source_path"], (930, 245), (810, 750), circles_b,
               (int(pb["decoded_width"]), int(pb["decoded_height"])))
    d = ImageDraw.Draw(page)
    text(d, (70, 1010), f"A: {a}   V1 winner={bool(int(pa['v1_winner']))}   V2={float(pa['v2_score']):.3f}", 29)
    text(d, (930, 1010), f"B: {b}   V1 winner={bool(int(pb['v1_winner']))}   V2={float(pb['v2_score']):.3f}", 29)
    text(d, (70, 1095), "Amber circles: tentative measured face pair", 29, "#a76e10")
    text(d, (70, 1140), "Blue circles: second tentative reference pair", 29, "#2077a8")
    text(d, (70, 1230), "Measured calculation (diagnostic only)", 40, bold=True)
    text(d, (75, 1300), f"ORB inliers {chosen['inliers']}; ratio {float(chosen['inlier_ratio']):.3f};", 30)
    text(d, (75, 1345), f"median reprojection error {float(chosen['reprojection_error']):.2f} px.", 30)
    for i, row in enumerate(valid[:2]):
        la, lb = float(row["laplacian_a"]), float(row["laplacian_b"])
        text(d, (75, 1440+i*95), f"Face pair {i+1}: Laplacian variance A={la:.1f}, B={lb:.1f},", 29)
        text(d, (75, 1480+i*95), f"raw difference B-A={lb-la:+.1f}; match confidence={float(row['confidence']):.2f}.", 29)
    text(d, (75, 1715), "Laplacian = Var(4-neighbor grayscale second derivative).", 28)
    text(d, (75, 1760), "No calibrated repeatability threshold or artifact separation is available.", 28)
    text(d, (75, 1805), "Contribution to experimental score: 0 (measurement not validated).", 28)
    text(d, (75, 1890), "Baseline and result", 40, bold=True)
    text(d, (75, 1960), f"V1: {ranking['v1_winner']}   |   V2: {ranking['v2_winner']}", 30)
    text(d, (75, 2010), "New local evidence does not justify changing either decision.", 30)
    text(d, (75, 2060), "Experimental keeper preference: abstain.", 32, "#a76e10", True)
    if valid:
        page.paste(magnified_region(pa["source_path"], json.loads(valid[0]["box_a"]),
                                    (int(pa["decoded_width"]), int(pa["decoded_height"]))), (90, 2160))
        page.paste(magnified_region(pb["source_path"], json.loads(valid[0]["box_b"]),
                                    (int(pb["decoded_width"]), int(pb["decoded_height"]))), (940, 2160))
        d = ImageDraw.Draw(page)
        text(d, (430, 2230), "Same tentative region; magnified crops", 27)
    pages.append(page)
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    text(d, (80, 70), "Negative example / no forced winner", 53, bold=True)
    text(d, (80, 150), f"{negative['image_a']} vs {negative['image_b']}", 34)
    text(d, (80, 225), f"Scene registration: {negative['registration']} ({negative['inliers']} inliers)", 32)
    text(d, (80, 280), f"Face correspondence: {negative['correspondence_status']}", 32)
    text(d, (80, 340), "No defensible face-local difference can be attributed to the same person.", 28)
    na, nb = by_name[negative["image_a"]], by_name[negative["image_b"]]
    draw_photo(page, na["source_path"], (60, 490), (810, 800), [],
               (int(na["decoded_width"]), int(na["decoded_height"])))
    draw_photo(page, nb["source_path"], (930, 490), (810, 800), [],
               (int(nb["decoded_width"]), int(nb["decoded_height"])))
    d = ImageDraw.Draw(page)
    text(d, (80, 1450), "No circles are drawn: no matched measured face region passed the gate.", 29)
    text(d, (80, 1530), "Recommendation", 43, bold=True)
    text(d, (80, 1600), "Collect identity-verified correspondences and human technical labels.", 30)
    text(d, (80, 1660), "Calibrate ROI detail against blur, noise, sharpening and JPEG artifacts.", 30)
    text(d, (80, 1720), "Only then test a controlled shadow V2 contribution.", 30)
    pages.append(page)
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    text(d, (80, 80), "Method, equations and evidence gates", 54, bold=True)
    sections = [
        ("Correspondence", ["ORB descriptors: Hamming nearest-neighbor ratio < 0.72.",
          "RANSAC homography: >=12 inliers, inlier ratio >=0.45, error <=3 px.",
          "Hungarian one-to-one links: normalized center distance <0.35; margin >0.15.",
          "No ArcFace embeddings were cached; geometry links are tentative."]),
        ("Regional technical measurements", ["Face boxes are mapped from 2048-pixel detection space.",
          "Laplacian variance and Tenengrad measure image gradients, not quality.",
          "Anisotropy = |E_x - E_y| / (E_x + E_y + 1).",
          "Clipping: per-channel fraction <=1 or >=254 in the decoded face ROI.",
          "JPEG boundary proxy: 8-pixel grid edge difference; scene edges confound it."]),
        ("Component boundaries", ["Face exposure/detail would belong in F after validation.",
          "Image-wide defects would belong in T; expression observations in M.",
          "Completeness and framing may belong in C after correspondence review.",
          "No new measurement changed V1 or the default shadow V2 score."]),
    ]
    y = 260
    for title, lines in sections:
        text(d, (90, y), title, 39, bold=True)
        y += 72
        for line in lines:
            text(d, (105, y), line, 26)
            y += 57
        y += 85
    text(d, (90, 2210), "Difference significance requires repeatability and defect-direction evidence.", 27, "#a76e10")
    pages.append(page)
    page = Image.new("RGB", (1800, 2500), "#f8fafc")
    d = ImageDraw.Draw(page)
    text(d, (80, 80), "Limits, performance and next experiment", 52, bold=True)
    lines = [
        "17 real candidates; 8 groups; 53 tentative region links; 0 verified identities.",
        "Face-count differences cannot distinguish occlusion from missed detection.",
        "Pose and expression changes confound direct gradient comparison.",
        "Chroma noise at 1024 px cannot establish full-resolution subject noise.",
        "Motion anisotropy cannot identify a motion-blur kernel on its own.",
        "White-balance differences have no correct direction without a reference.",
        "No independent human labels or repeatability calibration are available.",
        "A–I variants keep C–I diagnostic; no winner change was asserted.",
    ]
    for i, line in enumerate(lines):
        y = 240+i*100
        d.rounded_rectangle((75, y-14, 1725, y+65), radius=12,
                            fill="#e7eff5" if i % 2 == 0 else "#f8fafc")
        text(d, (100, y), line, 26)
    text(d, (85, 1180), "Measured runtime", 42, bold=True)
    text(d, (100, 1260), f"Audit cold: {summary['elapsed_seconds']:.3f} s; 17 added decodes.", 30)
    text(d, (100, 1320), f"Peak sampled process RSS: {summary['peak_process_rss']/1048576:.1f} MiB.", 30)
    text(d, (100, 1380), "Persistent warm cache and GPU memory: not measured.", 30)
    text(d, (85, 1520), "Reproduce locally", 42, bold=True)
    for i, line in enumerate(("python -m app.tools.audit_subject_quality --audit audit_results/nsc_fresh",
                              "python -m app.tools.report_subject_quality")):
        text(d, (100, 1600+i*75), line, 25)
    text(d, (85, 1850), "Go / no-go", 42, bold=True)
    text(d, (100, 1930), "NO GO: gather reviewed correspondences and labeled preferences first.", 29, "#a76e10", True)
    text(d, (100, 2000), "Then calibrate local detail against noise, blur, sharpening and JPEG artifacts.", 27)
    pages.append(page)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(pages, 1):
        page.save(output.parent / f"report_page_{index}.png")
    pages[0].save(output, save_all=True, append_images=pages[1:], resolution=180.0)
    return dict(featured_group=chosen["group_id"], featured_pair=[a, b],
                negative_group=negative["group_id"], pages=len(pages), outcome="inconclusive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("audit_results/phase8g4_subject_quality"))
    parser.add_argument("--output", type=Path, default=Path("audit_results/phase8g4_subject_quality/Room36_Phase_8G4_Visual_Evaluation.pdf"))
    args = parser.parse_args()
    print(json.dumps(build(args.audit, args.output), indent=2))


if __name__ == "__main__":
    main()
