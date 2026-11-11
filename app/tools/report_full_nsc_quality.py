"""Private full-NSC technical audit charts, review sheets and real-photo PDF."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from automatic_scan.technical_quality_v2_calibration import _blocking_ratio, measure_technical
from app.tools.audit_technical_calibration import read, write
from app.tools.audit_full_nsc_quality import flag_parts
from image_loader import ImageLoader


def _font(size):
    path=Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path),size) if path.is_file() else ImageFont.load_default()


def _text(draw,xy,value,size=28,color="#233b51"):
    draw.text(xy,str(value),font=_font(size),fill=color)


def _photo(path, size=(740,520)):
    with Image.open(path) as source:
        image=ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail(size)
        return image.copy()


def phase_stress(root, photos, output, sample=32, faces=None):
    """Measure block-grid phase sensitivity; no visual-defect conclusion."""
    flagged=[r for r in photos if str(r.get("any_flag")).lower()=="true"]
    clear=[r for r in photos if str(r.get("any_flag")).lower()=="false"]
    rng=random.Random(8365)
    selected=rng.sample(flagged,min(len(flagged),sample//2))+rng.sample(clear,min(len(clear),sample//2))
    loader=ImageLoader(max_decode_dimension=1024)
    rows=[];face_rows=[]
    by_photo={}
    for face in faces or []:
        by_photo.setdefault(face["relative_path"],[]).append(face)
    for row in selected:
        image=loader.load_for_scan(root/row["relative_path"],max_dimension=1024)
        if image is None:continue
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        for offset in range(8):
            crop=gray[offset:,offset:]
            ratio=_blocking_ratio(crop)
            rows.append(dict(relative_path=row["relative_path"],baseline_flag=row["any_flag"],
                             offset=offset,blocking_ratio=ratio,phase_flag=ratio is not None and ratio>1.5))
        available=by_photo.get(row["relative_path"],[])
        if available:
            face=max(available,key=lambda f:int(f["roi_width"])*int(f["roi_height"]))
            x0,y0,x1,y1=json.loads(face["box"])
            sx=image.shape[1]/float(row["decoded_width"]);sy=image.shape[0]/float(row["decoded_height"])
            roi=gray[int(y0*sy):int(y1*sy),int(x0*sx):int(x1*sx)]
            if min(roi.shape[:2])>=40:
                for offset in range(8):
                    ratio=_blocking_ratio(roi[offset:,offset:])
                    face_rows.append(dict(relative_path=row["relative_path"],face_index=face["face_index"],
                                          offset=offset,blocking_ratio=ratio,
                                          phase_flag=ratio is not None and ratio>1.5))
    write(output/"phase_sensitivity.csv",rows)
    write(output/"face_phase_sensitivity.csv",face_rows)
    return rows


def perturbation_stress(root, paths, output):
    """Controlled measurements on real photos; no assumption of keeper accuracy."""
    rng=np.random.default_rng(8365)
    loader=ImageLoader(max_decode_dimension=1024)
    rows=[]
    for relative in sorted(set(paths)):
        image=loader.load_for_scan(root/relative,max_dimension=1024)
        if image is None:
            continue
        noisy=np.clip(image.astype(np.int16)+rng.normal(0,12,image.shape).astype(np.int16),0,255).astype(np.uint8)
        variants={"original":image,
                  "gaussian_blur_1p5":cv2.GaussianBlur(image,(0,0),1.5),
                  "rgb_noise_12":noisy,
                  "unsharp_mild":cv2.addWeighted(image,1.4,cv2.GaussianBlur(image,(0,0),1),-.4,0),
                  "unsharp_strong":cv2.addWeighted(image,2.5,cv2.GaussianBlur(image,(0,0),1),-1.5,0)}
        for quality in (90,60):
            ok,encoded=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,quality])
            if ok:
                variants[f"jpeg_{quality}"]=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
        for treatment,changed in variants.items():
            result=measure_technical(changed)
            fields=result.to_dict()
            rows.append(dict(relative_path=relative,treatment=treatment,
                             detail_score=fields.get("detail_score"),
                             noise_aware_detail=fields.get("noise_aware_detail"),
                             edge_persistence=fields.get("edge_persistence"),
                             blocking_ratio=fields.get("artifact_measurement"),
                             chroma_noise=fields.get("chroma_noise"),
                             **flag_parts(fields)))
    write(output/"perturbation_results.csv",rows)
    return rows


def _hist(draw, values, box, title, color="#2478aa"):
    x0,y0,x1,y1=box
    draw.rounded_rectangle(box,20,fill="#e8eff4")
    _text(draw,(x0+20,y0+12),title,27)
    data=np.array([float(v) for v in values if v not in (None,"")],float)
    if not data.size:
        _text(draw,(x0+20,y0+100),"Unavailable",23);return
    bins=np.histogram(data,bins=15)
    counts=bins[0]
    width=(x1-x0-80)/len(counts)
    for i,count in enumerate(counts):
        height=(y1-y0-120)*count/max(1,counts.max())
        draw.rectangle((x0+40+i*width,y1-50-height,x0+38+(i+1)*width,y1-50),fill=color)
    _text(draw,(x0+30,y1-42),f"n={len(data)}  median={np.median(data):.3f}  range={data.min():.3f}..{data.max():.3f}",18)


def _scatter(draw, rows, xkey, ykey, box, title):
    x0,y0,x1,y1=box;draw.rounded_rectangle(box,20,fill="#e8eff4")
    _text(draw,(x0+20,y0+12),title,25)
    points=[]
    for row in rows:
        if row.get(xkey) not in (None,"") and row.get(ykey) not in (None,""):
            points.append((float(row[xkey]),float(row[ykey])))
    if not points:
        _text(draw,(x0+20,y0+100),"Unavailable",23);return
    data=np.asarray(points);lo=data.min(axis=0);span=np.maximum(data.max(axis=0)-lo,1e-9)
    for px,py in data[::max(1,len(data)//500)]:
        x=x0+55+(x1-x0-95)*(px-lo[0])/span[0]
        y=y1-70-(y1-y0-140)*(py-lo[1])/span[1]
        draw.ellipse((x-2,y-2,x+2,y+2),fill="#2b77a2")
    _text(draw,(x0+35,y1-46),f"{xkey}  /  {ykey}  (n={len(data)})",18)


def build(root,audit,output):
    root=root.resolve();audit=audit.resolve();output=output.resolve()
    summary=json.loads((audit/"summary.json").read_text(encoding="utf-8"))
    cold=json.loads((audit/"cold_summary.json").read_text(encoding="utf-8")) if (audit/"cold_summary.json").is_file() else summary
    photos=read(audit/"photo_metrics.csv");faces=read(audit/"face_metrics.csv")
    face_flags=[]
    for face in faces:
        result=flag_parts(dict(edge_persistence=float(face["edge_persistence"]) if face.get("edge_persistence") else None,
                                artifact_measurement=float(face["artifact_measurement"]) if face.get("artifact_measurement") else None))
        face_flags.append(dict(relative_path=face["relative_path"],face_index=face["face_index"],
                               native=face["native"],status=face["status"],**result))
    write(audit/"face_artifact_flags.csv",face_flags)
    by_path={str((root/p["relative_path"]).resolve()):p for p in photos}
    baseline=audit.parent/"nsc_fresh"/"ranking_v1_v2_comparison.csv"
    comparisons=[]
    if baseline.is_file():
        for group in read(baseline):
            if group["preference_id"]!="face_1_eyes_1":continue
            members=[by_path.get(str(Path(name).resolve())) for name in group["candidate_ids"].split("|")]
            comparisons.append(dict(group_id=group["group_id"],v1_winner=Path(group["v1_winner"]).name,
                v2_winner=Path(group["v2_winner"]).name,
                candidate_ids="|".join(Path(name).name for name in group["candidate_ids"].split("|")),
                v2_margin=group.get("v2_margin"),v2_tie_class=group.get("tie_class"),
                measured_candidates=sum(p is not None for p in members),
                candidate_count=len(members),candidate_score="withheld",
                adjusted_detail=json.dumps([p.get("noise_aware_detail") if p else None for p in members]),
                chroma=json.dumps([p.get("chroma_noise") if p else None for p in members])))
    write(audit/"group_comparisons.csv",comparisons)
    coverage=[dict(cohort=cohort,photos=len(group),faces=sum(int(p["face_count"]) for p in group),
                   native_rois=sum(int(p["face_count"]) for p in group if str(p["roi_native"]).lower()=="true"),
                   any_flags=sum(str(p["any_flag"]).lower()=="true" for p in group))
              for cohort,group in (("all",photos),("confirmed_duplicate",[p for p in photos if p["cohort"]=="confirmed_duplicate"]))]
    write(audit/"coverage_summary.csv",coverage)
    phases=phase_stress(root,photos,audit,faces=faces)
    stress=perturbation_stress(root,[r["relative_path"] for r in phases if int(r["offset"])==0],audit)
    strata={}
    for p in photos:
        try: iso=float(p["iso"])
        except (TypeError,ValueError): iso=None
        key="unknown" if iso is None else ("ISO<400" if iso<400 else "ISO400-1599" if iso<1600 else "ISO>=1600")
        for label in (key,p["format"]):
            bucket=strata.setdefault(label,{"photos":0,"persistence":0,"blocking":0})
            bucket["photos"]+=1
            bucket["persistence"]+=str(p["persistence_flag"]).lower()=="true"
            bucket["blocking"]+=str(p["blocking_flag"]).lower()=="true"
    write(audit/"flag_strata.csv",[dict(stratum=k,**v) for k,v in sorted(strata.items())])
    face_phases=read(audit/"face_phase_sensitivity.csv") if (audit/"face_phase_sensitivity.csv").is_file() else []
    flags=[r for r in photos if str(r.get("any_flag")).lower()=="true"]
    clear=[r for r in photos if str(r.get("any_flag")).lower()=="false"]
    selected=([flags[0],flags[len(flags)//2]] if flags else [])+([clear[0],clear[len(clear)//2]] if clear else [])
    review=[];pages=[]
    page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
    _text(d,(80,85),"Room36 / Full NSC technical audit",58)
    _text(d,(80,190),"Population accounting and diagnostic incidence",34)
    for i,(name,value) in enumerate((
        ("Discovered files",summary["discovered"]),("Successfully measured",summary["measured"]),
        ("Cached detected faces",summary["cached_faces"]),("Native measured face ROIs",summary["native_face_rois"]),
        ("Face ROI artifact flags",sum(r["any_flag"] for r in face_flags)),
        ("Whole-image evaluable",summary["whole_evaluable"]),("Persistence flags",summary["persistence_flags"]),
        ("Blocking flags",summary["blocking_flags"]),("Both rules",summary["both_flags"]),
        ("Confirmed duplicate candidates",summary["confirmed_candidates"]),
        ("Errors",summary["errors"]),("Cold extra decodes",cold["additional_decodes"]),
        ("Cold / warm seconds",f"{cold['elapsed_s']:.1f} / {summary['elapsed_s']:.1f}"),
        ("Cold sampled RSS MiB",f"{cold['sampled_peak_rss']/1048576:.1f}"))):
        y=335+i*125
        _text(d,(100,y),name,28);_text(d,(1000,y),value,32)
    _text(d,(100,2120),"Artifact flags are unreviewed algorithmic suspicions, not defects.",28,"#966715")
    _text(d,(100,2200),"New candidate score withheld; V1 and default V2 unchanged.",28)
    pages.append(page)
    page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
    _text(d,(80,70),"Controlled perturbations on sampled NSC photos",44)
    treatments=("original","gaussian_blur_1p5","rgb_noise_12","unsharp_mild","unsharp_strong","jpeg_90","jpeg_60")
    for i,name in enumerate(treatments):
        subset=[r for r in stress if r["treatment"]==name]
        vals=[float(r["noise_aware_detail"]) for r in subset if r["noise_aware_detail"] is not None]
        flag_count=sum(r["any_flag"] for r in subset)
        _text(d,(95,200+i*290),name,29)
        _text(d,(700,200+i*290),f"median adjusted detail {np.median(vals):.4f}" if vals else "detail unavailable",27)
        _text(d,(700,250+i*290),f"artifact flags {flag_count}/{len(subset)}",25)
    original={r["relative_path"]:r for r in stress if r["treatment"]=="original"}
    strong=[r for r in stress if r["treatment"]=="unsharp_strong" and r["relative_path"] in original]
    raised=sum(float(r["noise_aware_detail"])>float(original[r["relative_path"]]["noise_aware_detail"]) for r in strong)
    _text(d,(90,2200),f"Strong unsharp raised adjusted detail in {raised}/{len(strong)} photos; ranking remains withheld.",23,"#966715")
    _text(d,(90,2290),"RGB noise changes luma and chroma; this is a stress test, not a quality label.",23)
    pages.append(page)
    metrics=("noise_aware_detail","chroma_noise","edge_persistence","artifact_measurement",
             "detail_coverage","chroma_noise_coverage")
    for offset in (0,3):
        page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
        _text(d,(80,65),"Population distributions / missing values excluded",45)
        for j,key in enumerate(metrics[offset:offset+3]):
            _hist(d,[r.get(key) for r in photos],(80,180+j*750,1720,850+j*750),key)
        pages.append(page)
    page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
    _text(d,(80,60),"Legacy versus new observations",45)
    _scatter(d,photos,"legacy_v2_technical","noise_aware_detail",(80,170,1720,1250),"Old raw-log versus adjusted detail")
    _scatter(d,photos,"noise_floor","noise_aware_detail",(80,1340,1720,2420),"Luminance residual versus adjusted detail")
    pages.append(page)
    page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
    _text(d,(80,60),"Face ROI versus whole-image evidence",45)
    _hist(d,[f.get("noise_aware_detail") for f in faces],(80,170,1720,820),"Face ROI adjusted detail")
    _hist(d,[p.get("noise_aware_detail") for p in photos],(80,900,1720,1550),"Whole-image adjusted detail")
    _hist(d,[f.get("chroma_noise") for f in faces],(80,1630,1720,2280),"Face ROI chroma residual")
    pages.append(page)
    page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
    _text(d,(80,70),"Block-grid crop-phase sensitivity",48)
    for i,offset in enumerate(range(8)):
        subset=[r for r in phases if int(r["offset"])==offset]
        flagged=sum(r["phase_flag"] for r in subset)
        _text(d,(100,210+i*175),f"Offset {offset}: {flagged}/{len(subset)} flagged",28)
        d.rectangle((650,215+i*175,650+700*flagged/max(1,len(subset)),255+i*175),fill="#c58b2d")
    _text(d,(100,1790),"Changing crop phase changes the measured 8-pixel boundary ratio.",28)
    _text(d,(100,1850),"These are not reviewer-confirmed false positives.",28)
    phase_groups={}
    for row in face_phases:
        phase_groups.setdefault(row["relative_path"],set()).add(row["phase_flag"])
    unstable=sum(len(states)>1 for states in phase_groups.values())
    _text(d,(100,1940),f"Face ROI flag state changed: {unstable}/{len(phase_groups)} sampled ROIs.",28)
    pages.append(page)
    for index,row in enumerate(selected):
        path=root/row["relative_path"]
        private=audit/"review_sheets";private.mkdir(exist_ok=True)
        blind=Image.new("RGB",(850,700),"white")
        blind_photo=_photo(path,(800,600));blind.paste(blind_photo,(25,40))
        _text(ImageDraw.Draw(blind),(25,655),f"Sample {index+1}: {path.name}",22)
        blind.save(private/f"sample_{index+1:02d}.jpg")
        page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
        _text(d,(80,75),f"Real NSC example {index+1} / {'flagged' if row in flags else 'unflagged'}",48)
        photo=_photo(path,(1500,1250));page.paste(photo,(100,170))
        related=[f for f in faces if f["relative_path"]==row["relative_path"]]
        if related:
            face=related[0];box=json.loads(face["box"])
            sx=photo.width/float(row["decoded_width"]);sy=photo.height/float(row["decoded_height"])
            x0,y0,x1,y1=box;cx=100+(x0+x1)*sx/2;cy=170+(y0+y1)*sy/2
            radius=max((x1-x0)*sx,(y1-y0)*sy)*.65
            d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline="#cf9025",width=7)
            with Image.open(path) as original:
                original=ImageOps.exif_transpose(original).convert("RGB")
                scale_x=original.width/float(row["decoded_width"]);scale_y=original.height/float(row["decoded_height"])
                crop=original.crop((int(x0*scale_x),int(y0*scale_y),int(x1*scale_x),int(y1*scale_y)))
                crop.thumbnail((450,360));page.paste(crop,(110,1570))
        _text(d,(100,1480),f"Photo: {Path(row['relative_path']).name} | ROI circle = measured region",27)
        _text(d,(100,1990),f"Persistence {row['edge_persistence']}  /  block ratio {row['artifact_measurement']}",27)
        _text(d,(100,2050),f"Noise-adjusted detail {row['noise_aware_detail']}  /  chroma {row['chroma_noise']}",27)
        _text(d,(100,2110),"Amber = unverified metric region. No identity or defect claim.",27,"#966715")
        pages.append(page)
        review.append(dict(sample_id=f"sample_{index+1:02d}",relative_path=row["relative_path"],
                           human_label="",reason="",reviewer_id=""))
    write(audit/"sample_review_manifest.csv",review)
    pair=[p for p in photos if p["image_id"] in ("IMG_4183.JPG","IMG_4184.JPG")]
    if len(pair)==2:
        page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
        _text(d,(80,75),"Confirmed duplicate pair / diagnostic abstention",45)
        for i,row in enumerate(sorted(pair,key=lambda p:p["image_id"])):
            x=50+i*880;photo=_photo(root/row["relative_path"],(810,760));page.paste(photo,(x,190))
            matched=[f for f in faces if f["relative_path"]==row["relative_path"]]
            if matched:
                x0,y0,x1,y1=json.loads(matched[0]["box"])
                sx=photo.width/float(row["decoded_width"]);sy=photo.height/float(row["decoded_height"])
                cx=x+(x0+x1)*sx/2;cy=190+(y0+y1)*sy/2
                radius=max((x1-x0)*sx,(y1-y0)*sy)*.65
                d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline="#cb8c23",width=6)
            _text(d,(x,1000),row["image_id"],31)
            _text(d,(x,1080),f"Legacy T: {row['legacy_v2_technical']}",26)
            _text(d,(x,1140),f"Adjusted detail: {row['noise_aware_detail']}",26)
            _text(d,(x,1200),f"Chroma: {row['chroma_noise']}",26)
            _text(d,(x,1260),f"Artifact flag: {row['any_flag']}",26)
        _text(d,(80,1540),"Face identity is not inferred from box order or proximity.",29)
        _text(d,(80,1580),"Amber circles = measured ROIs, not matched identities or defects.",27)
        _text(d,(80,1660),"No candidate score or quality winner: directional calibration unavailable.",29,"#966715")
        pages.append(page)
    if comparisons:
        page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
        _text(d,(80,75),"Confirmed duplicate groups / diagnostics only",45)
        for i,row in enumerate(comparisons):
            _text(d,(90,180+i*240),f"{row['group_id']}  {row['measured_candidates']}/{row['candidate_count']} measured",26)
            _text(d,(110,240+i*240),f"V1: {row['v1_winner']}  |  V2: {row['v2_winner']}",25)
            _text(d,(110,295+i*240),"New technical winner: withheld",25,"#966715")
        pages.append(page)
    page=Image.new("RGB",(1800,2500),"#f8fafc");d=ImageDraw.Draw(page)
    _text(d,(80,80),"Interpretation and decision",52)
    for i,line in enumerate((
        "Full-population measurements describe signal incidence, not keeper accuracy.",
        "Eight confirmed groups supply only 17 candidate photos.",
        "Current artifact thresholds remain unchanged: 0.65 / 1.5.",
        "Review flagged and unflagged samples before any false-positive claim.",
        "Native face ROIs are decoded image pixels, not camera RAW sensor samples.",
        "Technical Feature score remains unavailable; no winner is manufactured.",
        "GO/NO-GO for 8G.6 scoring integration: NO-GO without labels.")):
        _text(d,(100,300+i*180),line,29)
    pages.append(page)
    output.parent.mkdir(parents=True,exist_ok=True)
    for index,page in enumerate(pages,1):
        page.save(output.parent/f"report_page_{index:02d}.png")
    pages[0].save(output,save_all=True,append_images=pages[1:],resolution=170)
    shifted={}
    for row in phases:
        shifted.setdefault(row["relative_path"],set()).add(bool(row["phase_flag"]))
    variable=sum(len(states)>1 for states in shifted.values())
    findings=(f"# Full NSC technical audit\n\nDiscovered: {summary['discovered']}; measured: {summary['measured']}; "
              f"faces: {summary['cached_faces']}; native ROIs: {summary['native_face_rois']}.\n\n"
              f"Persistence flags: {summary['persistence_flags']}/{summary['whole_evaluable']}; "
              f"blocking flags: {summary['blocking_flags']}/{summary['whole_evaluable']}; "
              f"both: {summary['both_flags']}/{summary['whole_evaluable']}.\n\n"
              f"Crop-phase flag state changed in {variable}/{len(shifted)} sampled photos and "
              f"{unstable}/{len(phase_groups)} sampled face ROIs. "
              "This is measurement instability, not a reviewer-confirmed false positive.\n\n"
              "No human artifact labels were imported. No new technical score or keeper winner was computed.\n")
    (audit/"findings.md").write_text(findings,encoding="utf-8")
    return dict(pages=len(pages),real_photos=len(selected),phase_samples=len({r["relative_path"] for r in phases}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos",type=Path,default=Path("room36_test_set/NSC_Meeting"))
    parser.add_argument("--audit",type=Path,default=Path("audit_results/phase8g5_full_nsc"))
    parser.add_argument("--output",type=Path,default=Path("audit_results/phase8g5_full_nsc/Room36_Full_NSC_Meeting_Technical_Quality_Evaluation.pdf"))
    args=parser.parse_args();print(json.dumps(build(args.photos,args.audit,args.output),indent=2))


if __name__=="__main__":main()
