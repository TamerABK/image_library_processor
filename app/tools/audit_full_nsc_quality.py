"""Read-only, resumable full-population technical audit of a local photo set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import psutil
from PIL import Image

from automatic_scan.technical_quality_v2_calibration import VERSION, measure_technical, chroma_noise_multiscale
from automatic_scan.subject_quality_experiment import detection_space_size, face_region
from automatic_scan.keeper_ranker_v2_adapter import technical_feature, extract_faces
from face_processing.cache import ImageFaceAnalysisCache
from image_loader import ImageLoader
from app.tools.audit_technical_calibration import read, write

AUDIT_VERSION = "full-nsc-technical-v1"
SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".nef", ".cr2", ".cr3", ".arw"}


def source_key(path, stat, face_revision, max_pixels):
    payload = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, face_revision,
               max_pixels, VERSION, AUDIT_VERSION, "ImageLoader-BGR8-v1", "face-roi-adaptive-inset-v2")
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def summarize(values):
    data = np.asarray([float(v) for v in values if v not in (None, "") and np.isfinite(float(v))], dtype=float)
    if not data.size:
        return dict(count=0, mean=None, median=None, sd=None, mad=None, **{f"p{p}": None for p in (5,10,25,75,90,95,99)})
    med = float(np.median(data))
    return dict(count=int(data.size), mean=float(data.mean()), median=med, sd=float(data.std()),
                mad=float(np.median(np.abs(data-med))), **{f"p{p}": float(np.percentile(data,p)) for p in (5,10,25,75,90,95,99)})


def flag_parts(result):
    persistence = result.get("edge_persistence")
    blocking = result.get("artifact_measurement")
    p = persistence is not None and persistence < .65
    b = blocking is not None and blocking > 1.5
    return dict(persistence_flag=p, blocking_flag=b, any_flag=p or b,
                both_flag=p and b, flag_evaluable=persistence is not None or blocking is not None)


def _exif_iso(path):
    try:
        with Image.open(path) as image:
            return image.getexif().get(34855)
    except (OSError, ValueError):
        return None


def measure_one(path, root, known, cache, face_rows, loader, max_pixels, face_revision):
    stat = path.stat()
    metadata = loader.read_metadata(path)
    if metadata is None:
        raise ValueError("metadata_unavailable")
    faces = cache.get(path, stat.st_size, stat.st_mtime_ns)
    is_fallback = metadata.width*metadata.height > max_pixels
    start = time.perf_counter()
    image = loader.load_for_scan(path, max_dimension=2048 if is_fallback else None)
    decode_s = time.perf_counter()-start
    if image is None:
        raise ValueError("decode_failed")
    h,w = image.shape[:2]
    preview = (cv2.resize(image, (round(w*1024/max(w,h)), round(h*1024/max(w,h))),
                          interpolation=cv2.INTER_AREA) if max(w,h)>1024 else image)
    start=time.perf_counter(); whole=measure_technical(preview).to_dict(); whole_s=time.perf_counter()-start
    start=time.perf_counter(); alternative=chroma_noise_multiscale(preview); color_s=time.perf_counter()-start
    source=detection_space_size(metadata.width,metadata.height)
    diagnostics = extract_faces(face_rows.get(str(path), []), *source, rows=True)[1] if faces is not None else []
    per_face=[]; start=time.perf_counter()
    if faces is not None:
        for index, face in enumerate(faces):
            roi,box=face_region(image,face.bbox,source)
            if roi.size and min(roi.shape[:2])>=48:
                inset=max(1,min(3,int(min(roi.shape[:2])*.03)))
                roi=roi[inset:-inset,inset:-inset]
            result=measure_technical(roi).to_dict()
            if is_fallback:
                result["status"]="fallback_roi_not_native"
            per_face.append(dict(image_id=path.name, relative_path=str(path.relative_to(root)),
                face_index=index,box=list(box),roi_width=roi.shape[1],roi_height=roi.shape[0],
                native=not is_fallback, prominence=diagnostics[index]["prominence"] if index<len(diagnostics) else None,
                primary=diagnostics[index]["primary"] if index<len(diagnostics) else None,
                low_clip_bgr=np.mean(roi<=1,axis=(0,1)).tolist() if roi.size else None, **result))
    face_s=time.perf_counter()-start
    old=technical_feature({key:known.get("technical_raw_"+key) for key in ("laplacian","sobel","local_contrast")}) if known else None
    record=dict(image_id=path.name,relative_path=str(path.relative_to(root)),source_revision=source_key(path,stat,face_revision,max_pixels),
        format=path.suffix.lower(),bytes=stat.st_size,width=metadata.width,height=metadata.height,
        decoded_width=w,decoded_height=h,is_raw=metadata.is_raw,iso=_exif_iso(path),
        face_cache_state="compatible" if faces is not None else "missing_or_incompatible",
        face_count=len(faces) if faces is not None else 0,roi_native=not is_fallback,
        legacy_laplacian=known.get("technical_raw_laplacian") if known else None,
        legacy_sobel=known.get("technical_raw_sobel") if known else None,
        legacy_contrast=known.get("technical_raw_local_contrast") if known else None,
        legacy_v2_technical=old.value if old else None,
        chroma_multiscale=alternative, dark_clip_bgr=np.mean(preview<=1,axis=(0,1)).tolist(), **whole)
    record.update(flag_parts(whole))
    return dict(photo=record,faces=per_face,timing=dict(decode_s=decode_s,whole_s=whole_s,
        color_s=color_s,face_s=face_s,total_s=decode_s+whole_s+color_s+face_s))


def run(root, audit, output, *, resume=False, limit=None, max_workers=1, cancel_file=None, max_pixels=32_000_000):
    root=root.resolve();audit=audit.resolve();output=output.resolve()
    if max_workers!=1:
        raise ValueError("The full audit supports one bounded decode worker")
    output.mkdir(parents=True,exist_ok=True); checkpoints=output/"checkpoints";checkpoints.mkdir(exist_ok=True)
    paths=sorted(p for p in root.rglob("*") if p.is_file())
    if limit is not None: paths=paths[:limit]
    known={str(Path(r["image_path"]).resolve()):r for r in read(audit/"photo_metrics.csv")}
    face_rows={}
    for row in read(audit/"face_metrics.csv"):
        face_rows.setdefault(str(Path(row["image_path"]).resolve()),[]).append(row)
    candidates={str(Path(name).resolve()) for row in read(audit/"ranking_v1_v2_comparison.csv")
                if row["preference_id"]=="face_1_eyes_1" for name in row["candidate_ids"].split("|")}
    face_db=audit/"audit_cache.sqlite3"
    face_revision=str(face_db.stat().st_mtime_ns)+":"+str((audit/"face_metrics.csv").stat().st_mtime_ns)
    cache=ImageFaceAnalysisCache(face_db);loader=ImageLoader(max_decode_dimension=None)
    photos=[];faces=[];timings=[];errors=[];manifest=[];hits=decodes=0
    start=time.perf_counter();peak=psutil.Process().memory_info().rss
    for path in paths:
        item_start=time.perf_counter()
        relative=str(path.relative_to(root))
        entry=dict(relative_path=relative,image_id=path.name,format=path.suffix.lower(),
                   cohort="confirmed_duplicate" if str(path.resolve()) in candidates else "population")
        if cancel_file and cancel_file.exists():
            entry["status"]="cancelled_before_processing";manifest.append(entry);break
        if path.suffix.lower() not in SUFFIXES:
            entry["status"]="unsupported_format";manifest.append(entry);continue
        try:
            stat=path.stat();key=source_key(path,stat,face_revision,max_pixels)
            checkpoint=checkpoints/(key+".json")
            was_hit=bool(resume and checkpoint.is_file())
            if was_hit:
                payload=json.loads(checkpoint.read_text(encoding="utf-8"));hits+=1
            else:
                payload=measure_one(path,root,known.get(str(path.resolve())),cache,face_rows,loader,max_pixels,face_revision)
                temp=checkpoint.with_suffix(".tmp");temp.write_text(json.dumps(payload),encoding="utf-8");os.replace(temp,checkpoint);decodes+=1
            photo=payload["photo"];photo["cohort"]=entry["cohort"]
            photos.append(photo);faces.extend(payload["faces"])
            timing=(dict(decode_s=0.,whole_s=0.,color_s=0.,face_s=0.,
                         total_s=time.perf_counter()-item_start)
                    if was_hit else payload["timing"])
            timings.append(dict(relative_path=relative,cache_hit=was_hit,**timing))
            entry.update(status="measured",source_revision=key,bytes=stat.st_size,
                         width=photo["width"],height=photo["height"],face_cache_state=photo["face_cache_state"])
        except (OSError,ValueError,KeyError,cv2.error) as exc:
            entry["status"]="error";errors.append(dict(relative_path=relative,error=type(exc).__name__,message=str(exc)[:200]))
        manifest.append(entry);peak=max(peak,psutil.Process().memory_info().rss)
    for name,rows in (("manifest",manifest),("photo_metrics",photos),("face_metrics",faces),
                      ("timings",timings),("errors",errors)):
        write(output/(name+".csv"),[{k:json.dumps(v) if isinstance(v,(list,tuple,dict)) else v for k,v in row.items()} for row in rows])
    if not errors:
        (output/"errors.csv").write_text("relative_path,error,message\n",encoding="utf-8")
    flags=[dict(relative_path=p["relative_path"],cohort=p["cohort"],
                persistence_flag=p["persistence_flag"],blocking_flag=p["blocking_flag"],
                any_flag=p["any_flag"],both_flag=p["both_flag"],evaluable=p["flag_evaluable"])
           for p in photos]
    write(output/"artifact_flags.csv",flags)
    distributions=[]
    for cohort in ("all","confirmed_duplicate"):
        selected=photos if cohort=="all" else [p for p in photos if p["cohort"]==cohort]
        for key in ("detail_score","noise_aware_detail","chroma_noise","noise_floor","edge_persistence",
                    "artifact_measurement","detail_coverage","chroma_noise_coverage","luminance_noise_coverage"):
            distributions.append(dict(cohort=cohort,metric=key,**summarize(p.get(key) for p in selected)))
    write(output/"distribution_summary.csv",distributions)
    summary=dict(version=AUDIT_VERSION,metric_version=VERSION,discovered=len(paths),measured=len(photos),
                 errors=len(errors),unsupported=sum(r["status"]=="unsupported_format" for r in manifest),
                 cached_faces=sum(p["face_count"] for p in photos),face_rois=len(faces),
                 native_face_rois=sum(f["native"] for f in faces),
                 confirmed_candidates=sum(p["cohort"]=="confirmed_duplicate" for p in photos),
                 whole_evaluable=sum(p["flag_evaluable"] for p in photos),
                 persistence_flags=sum(p["persistence_flag"] for p in photos),
                 blocking_flags=sum(p["blocking_flag"] for p in photos),
                 both_flags=sum(p["both_flag"] for p in photos),
                 cache_hits=hits,additional_decodes=decodes,elapsed_s=time.perf_counter()-start,
                 sampled_peak_rss=peak,python=platform.python_version(),max_workers=max_workers,
                 max_pixels=max_pixels,limited=limit is not None)
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos",type=Path,default=Path("room36_test_set/NSC_Meeting"))
    parser.add_argument("--audit-cache",type=Path,default=Path("audit_results/nsc_fresh"))
    parser.add_argument("--output",type=Path,default=Path("audit_results/phase8g5_full_nsc"))
    parser.add_argument("--resume",action="store_true");parser.add_argument("--limit",type=int)
    parser.add_argument("--max-workers",type=int,default=1);parser.add_argument("--cancel-file",type=Path)
    args=parser.parse_args()
    if not args.photos.is_dir():parser.error(f"Source root unavailable: {args.photos}")
    print(json.dumps(run(args.photos,args.audit_cache,args.output,resume=args.resume,limit=args.limit,
                         max_workers=args.max_workers,cancel_file=args.cancel_file),indent=2))


if __name__=="__main__":main()
