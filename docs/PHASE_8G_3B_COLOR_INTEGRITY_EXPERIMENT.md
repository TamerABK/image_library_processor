# Phase 8G.3b — color-integrity experiment

## Decision

**Do not integrate color into V2 yet.** The audit measured all eight NSC
duplicate groups (17 candidates), but there are no human technical-quality
labels. The current clipping detector observes saturation, not proved
information loss. Chromatic noise responds to controlled mild noise yet is
non-discriminative in seven of eight real groups. Fringing and white-balance
observations lack reliable defect direction. No production selector, default
V2 formula, face analysis, grouping, or photo was changed.

| Metric | Reliable? | Independently useful? | NSC discrimination | Action |
| --- | --- | --- | --- | --- |
| RGB clipping | Reproducible decoded-pixel fractions; defect interpretation unresolved | Correlates 0.71 with cached local contrast across 17 candidates | Range 0.00006–0.01695 across groups | Keep diagnostic |
| Chroma noise | Responds to mild synthetic noise; smooth-region coverage collapses at high noise | Unproved on textured scenes | Nonzero range in 1/8 groups | Improve estimator and label |
| Color fringing | Colored-edge proxy confounded by scene color | Unproved | Numerical range in 4/8 groups | Reject as quality score |
| White-balance consistency | Non-directional Lab chroma delta only | Unproved | Differences measured, no correct reference | Keep conditional diagnostic |
| Color banding | Not evaluated | Unknown | Unknown | Defer |

## Implementation and interpretation

`automatic_scan/color_integrity_experiment.py` accepts only 8-bit three-channel
BGR arrays. The standalone CLI `app/tools/audit_color_integrity.py` reads only
confirmed duplicate candidates from the existing NSC audit. It never imports
the production ranker as a caller or writes to production caches. It reuses
the existing `ImageLoader`, caps the longest image dimension at 1024, and
caches each measurement by resolved path, source size and modification time,
analysis dimension, decoder signature, and metric version.

RGB clipping reports each RGB channel's fraction with values <=1 and >=254.
An 8×8 grid records spatial extent. The directional quality confidence is
zero because black/white backgrounds, stage lighting, and saturated colors
can produce identical measurements. These are clipping **in decoded output**,
not proof of lost RAW sensor data.

Chromatic noise is the median plus 1.4826 times the median absolute deviation
of OpenCV's 8-bit Lab chroma-code residuals against a 3×3 median-filtered image. Regions are
eligible only when grayscale Sobel gradient <12, local luminance residual <6,
and luminance lies between 8 and 245. Coverage must reach 3%. Confidence is
limited to 0.5; the value is provisional because real fabric, hair, LEDs, JPEG
artifacts, and demosaicing can create similar chroma residuals.

Fringing is a median Lab chroma magnitude near high-contrast edges. It is
**not** a validated chromatic-aberration detector: genuine colored edges have
the same signal. Its quality confidence is zero. White-balance comparison uses
median encoded Lab a/b in low-chroma, non-extreme regions and reports only the
Euclidean chroma difference. It is unavailable when mean luminance differs by
more than 20 units or neutral-region coverage is inadequate. Low chroma does
not establish a neutral reference. Neither metric says which photograph is
better. White balance is never an absolute warm/cool preference.

ImageLoader applies its current raster and RAW-preview decoding behavior;
the audit uses resized BGR8 output. OpenCV decoding does not establish
equivalent ICC-managed color across disparate profiles or bit depths, and
resizing changes noise and clipping fractions. The 1024-pixel measurements
are not interchangeable with full-resolution sensor measurements. Images
without a supported decoded array are unavailable. No color-management
engine was added.

## NSC Meeting and controlled results

The first cold run decoded 17 candidates once in 1.826 s; peak process RSS
sampled at 57.8 MB. A warm run used 17 cache hits, zero decodes, and about
0.010 s excluding optional contact-sheet generation. Group clipping ranges
spanned 0.00006–0.01695. Chroma-noise range was nonzero in one group
(0.8685 encoded Lab units), zero in seven. These figures indicate numerical
variation, not selection accuracy. Pairwise white-balance differences are
non-directional. Experimental variants A–G are exported, but C–G contain no
color-adjusted winner because no metric passed the defect-direction gate.
There are therefore **zero validated color winner changes**, not zero
possible winner changes. Contact sheets are local and ignored by Git.

In seeded synthetic tests (seed 3603), a red-channel clipped stripe occupying
0, 6.25, 12.5, 25 and 50% of the image yielded matching high-red fractions.
Injected RGB noise with sigma 0, 4 and 8 yielded chroma estimates 0, 4.89 and
9.79. At sigma 16 and 32, eligible coverage fell below 3%, so the result
became unavailable. A fully saturated red image yielded high-red fraction 1
but no defect penalty. A warm grade remained non-directional. These controlled
cases check mechanics, not photographic validity.

Outputs under `audit_results/phase8g3b_color_integrity/` include metrics,
pairwise comparisons, group summaries, correlations, variant status, controlled
perturbations, performance records, review-label template, two SVG response
plots, and eight private contact sheets. The directory is ignored by Git.
Review labels remain blank. The CLI does not yet compute labeled accuracy,
registration-based correspondence, or a regional sharpness plus color variant.
Its fringing proxy, ICC handling, high-texture behavior, cancellation, and
color-noise behavior under JPEG compression need further study.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -B -m app.tools.audit_color_integrity --audit-dir audit_results/nsc_fresh --output audit_results/phase8g3b_color_integrity --controlled --contact-sheets
.\.venv\Scripts\python.exe -B -m pytest automatic_scan/tests/test_color_integrity_experiment.py -q
```

The focused tests passed 7/7. The final full local Windows unittest run passed:
570 tests, 10 skipped, in 191.250 s. The hosted Windows Full Test Suite
[passed on the follow-up CI compatibility commit](https://github.com/TamerABK/image_library_processor/actions/runs/42661665165).
The initial push exposed a missing-ONNX-Runtime import in an existing mocked
expression-service test; commit `fd4370e` removed that unnecessary test
dependency. No color-ranking code changed in the fix.
This small unlabeled data set cannot justify integration into the 15%
technical component.
