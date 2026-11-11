# Phase 8G.5 full NSC Meeting technical audit

## Scope and decision

This is a read-only experimental audit. It does not alter V1, default V2, duplicate grouping, manual review, expression scoring, image files, or the provisional artifact thresholds (`persistence < 0.65`, `blocking ratio > 1.5`). `TechnicalQualityResult.score` remains unavailable. **No-go for a calibrated technical Feature in 8G.6** until independent human review establishes defect direction and keeper relevance.

## Population accounting

The local `room36_test_set/NSC_Meeting` root contained **1,732 discovered unique JPEG paths**. The unlimited cold run measured **1,732/1,732**, with zero unsupported files and zero decode errors. It reused the existing `audit_results/nsc_fresh` cache: **8,453 compatible detected faces**, all represented by native decoded ROIs (not RAW sensor pixels). No new SCRFD or ArcFace inference ran. Exactly **17 photos in 8 confirmed duplicate groups** form the separate keeper-comparison cohort; the other 1,715 photos contribute only population-level descriptions. All 1,732 had an existing raw-log V2 technical feature available. The new whole-image chroma residual was available for **1,687/1,732**; 45 lacked eligible evidence. At face level, chroma was available for **8,399/8,453**. Face ROI availability is not a guarantee of perceptual validity.

| Cohort / rule | Count | Denominator | Rate |
|---|---:|---:|---:|
| Full photos, persistence `< 0.65` | 494 | 1,732 evaluable | 28.52% |
| Full photos, blocking ratio `> 1.5` | 0 | 1,732 evaluable | 0% |
| Confirmed candidate photos, either rule | 11 | 17 | 64.71% |
| Native face ROIs, persistence rule | 94 | 8,453 | 1.11% |
| Native face ROIs, blocking rule | 1 | 8,453 | 0.012% |

The single native-ROI blocking flag is on `IMG_3434.JPG`, face index 0; it has not been visually verified. Together the two face-level rules flag 95 distinct ROIs. The image-wide blocking rule flags none, which also shows that whole-image and ROI artifact results cannot be interchanged.

The whole-image result uses a bounded 1024-pixel preview, whereas face results use native decoded ROIs. Their flag rates must not be compared as if they used identical pixels or area. A 20% coarse-gradient selection makes `detail_coverage` nearly constant, so that coverage is **not** a confidence estimate. The same caution applies to the 30% luminance flat-region selection.

Selected full-population distributions (raw diagnostic units):

| Metric | Available | Median | P5 | P95 |
|---|---:|---:|---:|---:|
| Raw structured detail | 1,732 | 0.4812 | 0.2106 | 0.6906 |
| Noise-adjusted detail | 1,732 | 0.3222 | 0.1733 | 0.4336 |
| Chroma residual, float Lab a/b units | 1,687 | 1.2482 | 0.0000 | 2.3227 |
| Edge persistence | 1,732 | 0.6821 | 0.5958 | 0.8387 |
| 8-pixel boundary ratio | 1,732 | 0.9446 | 0.8521 | 0.9745 |

These are descriptions across one event, not technical-quality rankings of unrelated scenes. Within the eight confirmed groups, the new candidate score and winner remain withheld. The original V1 and V2 winner columns are preserved in `group_comparisons.csv`; disagreement with V1 is not ground truth. No human artifact or keeper labels were available.

## Artifact stability and limitations

On a seeded 32-photo sample split between flagged and unflagged records, shifting the 1024-pixel crop by 0–7 pixels changed the blocking-flag state for **0/32** whole-image previews. Only 13 selected face ROIs met the preview-size gate; **1/13** changed blocking-flag state across crop phases. This confirms a concrete face-crop phase sensitivity case, but it is not a measured false-positive rate. The 494 persistence flags also are **not** established defects or false positives. They require score-blind visual review. The review manifest and four private, score-blind sheets are stored locally for that purpose.

A seven-treatment stress battery on the same 32 seeded, real-photo samples showed the adjusted-detail diagnostic falling after Gaussian blur in 32/32 and RGB noise in 32/32. It **increased after strong unsharp masking in 32/32**, after JPEG quality 90 in 10/32, and after JPEG quality 60 in 13/32. The unsharp result is a concrete residual false-reward risk; mild sharpening is an ambiguous edit, and a numerical increase is not proof that any JPEG result looks better. The suspicion flag count rose from 17/32 originals to 25/32 strong-unsharp samples. These observations support withholding a new ranking score. The noise treatment changes RGB channels together and does not isolate chroma; the prior chroma/luma separation experiment remains the appropriate evidence for that distinction. The exact treatment values and photo lineage are in `perturbation_results.csv`.

The legacy raw-log technical score, new detail, chroma, artifact and clipping values overlap conceptually. Scatter plots and distribution tables show numerical relationships only; correlations do not prove independent photographic value. Native ROIs below the 32-million-pixel source budget are decoded once per photo. Larger sources use an explicitly marked 2048-pixel fallback, and the audit does not claim fallback ROIs are native. EXIF orientation follows `ImageLoader`; camera ISO is recorded from EXIF only where present. The audit never upscales a crop to manufacture source detail.

All 1,732 files were JPEG, and PIL returned no ISO for this collection; `flag_strata.csv` therefore reports an unknown-ISO denominator rather than inventing an ISO stratification. The eight confirmed groups are the only verified keeper-comparison cohort available in the prior audit. The PDF's circles locate measured ROIs; no person correspondence or visible artifact is asserted from an index or an algorithmic flag.

## Performance and reproducibility

Cold full-population run: **1,732 additional decodes, 459.941 s wall time**, and **226.2 MiB sampled peak process RSS**. Warm resumed run: **1,732 cache hits, zero decodes, 1.004 s**. Sampled RSS is whole-process RSS observed after each photo, not an exact peak allocation. The warm run writes fresh derived CSV/JSON summaries while reusing versioned per-photo checkpoints. The cache signature includes source size and modification time, face-cache revision, decoder/ROI and metric versions, and the native-resolution budget.

```powershell
.\.venv\Scripts\python.exe -B -m app.tools.audit_full_nsc_quality --photos room36_test_set/NSC_Meeting --audit-cache audit_results/nsc_fresh --output audit_results/phase8g5_full_nsc
.\.venv\Scripts\python.exe -B -m app.tools.audit_full_nsc_quality --resume --photos room36_test_set/NSC_Meeting --audit-cache audit_results/nsc_fresh --output audit_results/phase8g5_full_nsc
.\.venv\Scripts\python.exe -B -m app.tools.report_full_nsc_quality --photos room36_test_set/NSC_Meeting --audit audit_results/phase8g5_full_nsc
```

`--limit` is for development only. `--max-workers` currently accepts one bounded decode worker; other values fail explicitly. `--cancel-file PATH` stops between photos, and each successful photo has an atomically replaced checkpoint for resume. The private output includes `manifest.csv`, per-photo and per-face metrics, whole and face artifact flags, coverage and distribution summaries, confirmed-group comparisons, phase sensitivity, perturbation measurements, timings, errors, review manifest/sheets, `findings.md`, `summary.json`, and `Room36_Full_NSC_Meeting_Technical_Quality_Evaluation.pdf`. The private output directory is Git-ignored.

## Readiness

- **READY for further shadow measurement:** bounded full-population extraction and native face ROI provenance.
- **CONDITIONAL diagnostic:** float-Lab chroma residual where eligible coverage and source representation are known.
- **DIAGNOSTIC only:** edge persistence, blocking ratio, clipping and noise-adjusted detail. Preserve current thresholds for future comparison.
- **REJECTED for current ranking:** any learned or hand-weighted combined technical Feature. No calibrated direction, human review or held-out pairwise evidence exists.

The next evidence task is independent, score-blind review of flagged and unflagged professional images plus more labeled duplicate groups across different shoots. This audit does not start Phase 8G.6.
