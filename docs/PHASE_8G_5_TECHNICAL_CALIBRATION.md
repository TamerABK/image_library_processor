# Phase 8G.5: technical calibration

## Decision

The new measurements are **experimental diagnostics**. They are separately callable and cached; no production V1 code, default V2 code, blur classification, face inference, or keeper decision is changed. The tested detail measure resists added random noise but rewards strong unsharp masking and has an unstable JPEG response. It therefore has no normalized quality score and must not enter V2 in Phase 8G.6 yet. There are no human technical-preference labels for the eight NSC groups.

| Measurement | Decision | Evidence and next requirement |
|---|---|---|
| Cross-scale, noise-adjusted detail | **DIAGNOSTIC** | Falls under tested blur and random noise; rises under strong sharpening (0.358 to 0.410). Add halo/artifact rejection and test across independent scenes. |
| Float-Lab chroma residual | **CONDITIONAL** diagnostic | Responds monotonically to injected chroma noise (1.559, 6.393, 14.596, 29.956) without losing coverage. Scene texture, colored patterns and JPEG artifacts remain confounds; evaluate source-resolution ROIs and human labels before ranking. |
| Face-region channel clipping | **DIAGNOSTIC** | Correctly localized decoded-pixel saturation, but saturation alone does not prove lost facial detail. Require matched-face detail or another independent defect cue. |
| Candidate combined technical score | **REJECTED for this phase** | No defensible calibrated mapping or component weights. `score=None` by design. |

## Existing baseline and failure

`technical_feature` remains `raw-log-v1`. With raw Laplacian variance `L`, Sobel `S`, and local contrast `C`, it computes `l=log1p(L)/log1p(400)`, `s=log1p(S)/log1p(60)`, `c=log1p(C)/log1p(50)`, `a=exp(-abs(l-s))`, `e=sqrt(l*s)*a`, `u=e/(1+e/max(.01,c+.25))`, and `T=.55*u/(1+u)+.45*c/(1+c)`. These fixed anchors are provisional classification-scale values. The baseline is independent of the new implementation.

On a real IMG_4184 face crop, adding luminance noise with standard deviation 25 raised Laplacian from **488.536 to 12,730.391** and `raw-log-v1` from **0.385235 to 0.399269**. This is a false sharpness reward. Noise is not preserved optical information.

## Candidate measurements

`automatic_scan/technical_quality_v2_calibration.py` accepts a BGR8 decoded image. It returns `TechnicalQualityResult` version `technical-calibration-v3-noise-detail`, with raw observations, coverage, confidence, status and explanation. Unsupported pixels, images below 32 pixels on one side, and images with fewer than 64 structured edge pixels are unavailable. The diagnostic detail calculation is:

1. Convert BGR8 to gray float in `[0,1]`.
2. Compute Sobel magnitude after Gaussian smoothing with sigma 0.65 (`G_f`) and sigma 2.0 (`G_c`).
3. Select pixels where `G_c >= max(0.015, 80th percentile(G_c))`.
4. `D = median(G_f)` on those pixels.
5. `P = clip(median(G_c/(G_f+0.002)), 0, 1)` on those pixels.
6. `N = median(abs(gray - Gaussian(gray, 0.7)))` on pixels below the 30th percentile of `G_c`.
7. `D_adjusted = max(0, D*P-N)`.

`D_adjusted` is an uncalibrated diagnostic, **not** a probability or 0–1 quality score. Cross-scale persistence and a low-gradient residual reduce the random-noise reward in the tested examples. They do not prove robustness against correlated noise, processing halos, demosaicing, scene texture or camera differences. Confidence is bounded by 0.75 and indicates only measurement coverage, not perceptual correctness.

For chroma noise, BGR8 is converted to OpenCV float Lab, which avoids the previous 8-bit Lab quantization. The 35% lowest-gradient pixels in sigma-2.5-smoothed `L` are eligible, excluding `L<=3` and `L>=97`. Coverage below 8% returns unavailable. A 3x3 median-filtered BGR image provides a local comparison in the same float-Lab space. Let `r` be Euclidean `(a,b)` residual magnitude. The reported number is `median(r)+1.4826*MAD(r)`. These are OpenCV float-Lab chroma units; the value is **not** CIEDE2000 and is not a calibrated sensor-noise estimate. The estimator may measure legitimate chromatic microtexture. A minimum 64-pixel side is required.

## Subject regions, decoding and cache

The audit uses the Phase 8G.4 `detection_space_size` and `face_region` mapping to convert cached SCRFD coordinates from their 2048-pixel detection space into the analyzed image. It insets each mapped rectangle by 8% to reduce background contamination. This is an approximate interior ROI, not skin segmentation. It reuses `ImageFaceAnalysisCache`; no SCRFD or ArcFace call is made. V2 prominence diagnostics are reused; there is no new face-importance formula. Channel clipping is recorded as fractions of decoded BGR8 values at or above 254. It is not penalized. For non-people photographs the whole-image diagnostic still runs without face data.

The CLI decodes only confirmed duplicate candidates, using `ImageLoader` at a maximum dimension of 1024 by default. This resolution makes the image-wide chroma result a **preview-level** observation, not a source-resolution sensor-noise value. Small face crops are often below the 32/64-pixel gates. The experimental JSON cache key includes source path, byte length, modification time, analysis resolution, algorithm version, decoder/color representation and ROI version. Cached face analyses are read by their own revision-aware interface. Experimental cache files stay under the ignored local audit directory and do not touch production caches. A warm run avoids image decoding.

## Real NSC audit

The existing `audit_results/nsc_fresh` data supplied eight confirmed groups, 17 candidate photographs and cached face detections. All 17 images yielded image-wide measurements. Only **8 of 124** face ROIs met the structured-detail measurement gate at the 1024-pixel decode. This limits subject-level claims. The baseline technical score spans at most about 0.00235 within any group. The new observations differ numerically, but their direction is not reliably validated; candidate scores and experimental winners are deliberately withheld. The report selects IMG_4184 / IMG_4183 by an explicit rule: a two-photo group with at least two measurable face ROIs per photo and the smallest baseline technical margin. Baseline technical values are 0.419818 and 0.419762; adjusted image detail is 0.3618 and 0.3606. These tiny differences do not justify a preference. The report circles the measured face regions and labels the outcome **inconclusive**.

## Controlled benchmark

The audit applies deterministic in-memory changes to a real IMG_4184 face crop: Gaussian blur at sigma 1/2/3, luminance and chroma noise at 5/12/25, blur plus noise, unsharp masking at 0.5/1/2, and JPEG quality 90/60/25. The seed is 3605. Selected results:

| Condition | Baseline raw-log | Adjusted detail | Float-Lab chroma residual |
|---|---:|---:|---:|
| Original | 0.385235 | 0.357738 | 1.559 |
| Gaussian sigma 3 | — | 0.250054 | — |
| Luminance noise 25 | 0.399269 | 0.328115 | — |
| Chroma noise 25 | — | 0.330523 | 29.956 |
| Strong unsharp mask 2 | — | **0.409935** | — |
| JPEG quality 25 | — | **0.362987** | — |

The two highlighted increases are failures for a directional ranking score. One real crop and synthetic changes are not a population-level accuracy result. The scripted fixture test additionally checks blur plus noise and strong chroma-noise availability. Colored fabrics, low-light patterns, RAW-derived images, varied camera profiles, full-resolution ROIs and intentional stylization still need broader evaluation.

## Performance and validation

On the local Windows environment, a cold run processed 17 candidates with 17 additional decodes, zero hits, **2.361 s** end to end and **195.1 MiB** peak sampled process RSS. A warm run processed the same 17 with zero decodes, 17 hits, **0.0080 s**, and **179.5 MiB** peak sampled RSS. `timings.csv` records decoding, whole-image measurement, legacy color measurement, face-region measurement and per-candidate elapsed time. RSS is the whole process sampled after each candidate; it is not incremental memory use or a precise peak. No GPU work is added. The audit is synchronous CLI work only and does not affect scan scheduling or V1 latency.

The focused calibration tests cover noise, blur, blur-plus-noise, chroma response, saturation, coordinate mapping, cache revision/resolution and unavailable inputs. Full suite and hosted CI outcomes are recorded below when run.

## Reproduce locally

```powershell
.\.venv\Scripts\python.exe -B -m app.tools.audit_technical_calibration --audit audit_results/nsc_fresh --output audit_results/phase8g5_technical_calibration
.\.venv\Scripts\python.exe -B -m app.tools.report_technical_calibration --audit audit_results/phase8g5_technical_calibration
.\.venv\Scripts\python.exe -B -m unittest automatic_scan.tests.test_technical_calibration
```

The private PDF is `audit_results/phase8g5_technical_calibration/Room36_Phase_8G5_Technical_Calibration.pdf`. It and the source photographs are ignored by Git. The PDF includes real side-by-side photographs, measured face circles, original and magnified crops, the exact comparison calculations, controlled-response charts, group margins, and cold/warm performance. Annotation colors identify measured regions; no red/green defect claim is made because the real-pair evidence does not support one.

## Phase 8G.6 gate

**No-go for ranking integration now.** Next work should test independent, human-labeled technical comparisons; identify/reject halos and JPEG artifacts; check source-resolution chroma estimates on diverse textures and lighting; establish repeatability and group-level comparability; and then calibrate a 0–1 score only for defect cues with supported direction. V1 and default V2 remain the controls throughout.

## Build record

Full Windows suite: **583 tests, 10 skipped, OK (210.126 s)**. The immutable commit SHA and hosted CI run are recorded in the delivery summary; a commit cannot contain its own hash.
