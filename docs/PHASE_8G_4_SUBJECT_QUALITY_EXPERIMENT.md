# Phase 8G.4 — subject-aware quality audit

## Decision

**No go for V2 integration.** The audit processed all eight confirmed NSC
Meeting duplicate groups, 17 candidate photographs, and 53 *tentative*
face-region links. The available NSC cache has 8,453 face analyses but zero
stored face-scan embeddings. ORB registration and one-to-one spatial assignment
can link regions across frames; without embedding or reviewed identity evidence,
none is an identity-verified correspondence. No human keeper labels exist.
All experimental keeper preferences therefore abstain. No V1, default V2,
grouping, recognition, user setting, scan stage, or photograph changed.

The private six-page [visual PDF](../audit_results/phase8g4_subject_quality/Room36_Phase_8G4_Visual_Evaluation.pdf)
uses actual NSC photographs. Its featured pair is IMG_2928 / IMG_2929,
chosen from all groups by the number of usable localized face comparisons,
registration inlier ratio, and relative detail difference—not V2 margin alone.
Circles are drawn from the cached face boxes through Room36's 2048-pixel
detection coordinates into the report preview. Amber and blue mean tentative
measured regions, **not** supported defect and improvement. The report shows
magnified real face crops and a negative example (IMG_3437 / IMG_3436, no
accepted face link). Real photos and the PDF are ignored by Git.

## Method and measured result

The isolated module `automatic_scan/subject_quality_experiment.py` measures
only confirmed duplicate candidates. `ImageFaceAnalysisCache.get` validates
source size, modification time, and face-analysis cache version. ORB features
are matched by Hamming distance and ratio 0.72, then a RANSAC homography is
accepted at >=12 inliers, >=0.45 inlier ratio, and <=3-pixel median
reprojection error on <=1024-pixel images. A one-to-one Hungarian assignment
matches projected face centers only when distance/face-size <0.35 and the
second-best margin >0.15. These are provisional geometry gates; the output
status is `tentative_geometry` and confidence is capped at 0.7. A missing
face link does not mean a person is absent from the photo.

Face ROI boxes were initially mapped from full source dimensions, which
placed circles on background objects. Visual inspection caught the error.
Room36 caches boxes from its 2048-pixel detection image; the final version
uses that coordinate system, and a regression test checks the mapping. The
final PDF was regenerated and visually inspected.

Within each eligible face ROI the audit records per-channel high/low clipping,
median luminance, Laplacian variance, Tenengrad energy, gradient anisotropy,
and a JPEG-boundary proxy. All remain **diagnostic**. The compression proxy
can respond to genuine scene edges. Anisotropy cannot alone diagnose motion
blur. V2 already has face sharpness and exposure, so these ROI measures may
duplicate existing evidence. Expression outputs were not read from the
separate experimental cache; no moment score was activated. Group completeness
could not be established from face counts because missed detections and
occlusion cannot be distinguished reliably here.

In the featured pair, the two highlighted face ROIs had Laplacian variances
5689.8 vs 3793.5 and 1882.3 vs 3145.7 (A vs B). The differences have
opposite directions. ORB yielded 341 inliers, 0.907 inlier ratio, and 1.06
pixels median reprojection error. V1 chose IMG_2928; default V2 scored
IMG_2928 at 53.346 and IMG_2929 at 54.521. The new observations have
**zero contribution** to any experimental score because identity,
repeatability, noise separation and photographic direction are unvalidated.
No experimental winner changed; winner changes were not evaluated as an
accuracy claim.

## Controlled checks and performance

A real IMG_4184 face crop was perturbed in memory. Laplacian variance changed
from 488.5 baseline to 11.5 under Gaussian blur, 126.0 under horizontal
motion blur, **1776.0 after added noise**, and 704.1 with synthetic occlusion.
Thus edge energy cannot be treated as a monotonic quality reward. These
single-strength perturbations verify mechanics and a failure case; they do
not calibrate resolution or false-positive rates. JPEG, sharpening, skin
segmentation, full-resolution noise, stable-scene exposure, and moment
comparisons require further evaluation.

The local cold audit decoded 17 candidates once, took 1.56 seconds end-to-end,
and sampled peak process RSS at about 146 MiB. It used no additional face
inference. The output is not persistently cached yet, so a repeat still
decodes the 17 candidates; warm-cache speed and cancellation remain open
performance work. GPU memory was not measured.

## Answers and next phase

No new metric is presently reliable enough for a V2 score. Face-localized
comparisons are more interpretable than whole-image averages, but without
verified identities or human judgments they do not demonstrate better keeper
choices. Scene registration succeeded for the illustrated pairs, yet moving
subjects and unequal detected-face counts keep completeness uncertain.
Motion and artifact proxies remain diagnostic. Expression changes are
observations, not photographic-quality preferences.

Next: build a reviewed correspondence set with reliable embeddings or manual
face links; label at least 50–100 independent groups for technical, moment,
composition and overall preference; then calibrate ROI repeatability and
noise/blur/artifact separation before a controlled shadow-V2 experiment.
No Phase 8G.4 measurement should enter F, M, C or T yet.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -B -m app.tools.audit_subject_quality --audit audit_results/nsc_fresh --output audit_results/phase8g4_subject_quality
.\.venv\Scripts\python.exe -B -m app.tools.report_subject_quality --audit audit_results/phase8g4_subject_quality --output audit_results/phase8g4_subject_quality/Room36_Phase_8G4_Visual_Evaluation.pdf
.\.venv\Scripts\python.exe -B -m pytest automatic_scan/tests/test_subject_quality_experiment.py -q
```

Outputs: `photo_metrics.csv`, `face_metrics.csv`, `matched_faces.csv`,
`scene_regions.csv`, `ranking_comparison.csv`, `ablation_results.csv`,
`controlled_perturbations.csv`, `group_runtime.csv`, `summary.json`,
`human_labels_template.csv`, six PDF-page previews, and the PDF.
The focused tests pass 6/6. The full local Windows suite passed: 576 tests,
10 skipped, in 190.868 s. Hosted Windows CI is reported separately after
the commit is pushed.
