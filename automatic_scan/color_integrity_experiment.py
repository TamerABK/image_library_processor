"""Audit-only encoded-color diagnostics for confirmed duplicate photographs.

These values are observations, not keeper penalties. No production importer
uses this module.
"""
from __future__ import annotations

import cv2
import numpy as np

VERSION = "color-integrity-experiment-v1"


def _fraction(mask):
    return float(np.mean(mask))


def measure_color(image):
    """Measure BGR uint8 decoder output at a fixed analysis resolution."""
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return {"status": "unsupported_pixels", "version": VERSION}
    h, w = image.shape[:2]
    if min(h, w) < 64:
        return {"status": "insufficient_resolution", "version": VERSION}
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    near_low = rgb <= 1
    near_high = rgb >= 254
    low = near_low.mean(axis=(0, 1)).tolist()
    high = near_high.mean(axis=(0, 1)).tolist()
    # Grid occupancy is diagnostic; broad white/black areas are not defects.
    occupied = []
    for c in range(3):
        mask = near_low[:, :, c] | near_high[:, :, c]
        grid = [float(np.mean(mask[y*h//8:(y+1)*h//8, x*w//8:(x+1)*w//8]))
                for y in range(8) for x in range(8)]
        occupied.append(sum(v > .05 for v in grid) / 64)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Flat luminance AND low chroma gradient excludes most detailed edges.
    luminance_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    luminance_residual = cv2.absdiff(gray, luminance_blur)
    chroma = lab[:, :, 1:3]
    med = cv2.medianBlur(image, 3)
    med_chroma = cv2.cvtColor(med, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 1:3]
    residual = np.linalg.norm(chroma - med_chroma, axis=2)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                             cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    eligible = (gradient < 12) & (luminance_residual < 6) & (gray > 8) & (gray < 245)
    coverage = _fraction(eligible)
    if coverage >= .03:
        values = residual[eligible]
        noise = float(np.median(values) + 1.4826 * np.median(np.abs(values - np.median(values))))
    else:
        noise = None
    # Candidate only: chroma at achromatic edges. Colored scene boundaries
    # remain a strong confound, so this is never a directional quality score.
    edge = (gradient > 60) & (gray > 20) & (gray < 235)
    edge_coverage = _fraction(edge)
    fringe = float(np.median(np.linalg.norm(chroma[edge] - 128, axis=1))) if edge_coverage > .002 else None
    # Comparative white-balance feature, without claiming which color is right.
    neutral = (np.linalg.norm(chroma - 128, axis=2) < 20) & (gray > 40) & (gray < 215)
    neutral_coverage = _fraction(neutral)
    wb = np.median(chroma[neutral], axis=0).tolist() if neutral_coverage > .02 else None
    return dict(status="measured", version=VERSION, width=w, height=h,
                low_r=low[0], low_g=low[1], low_b=low[2],
                high_r=high[0], high_g=high[1], high_b=high[2],
                clip_extent_r=occupied[0], clip_extent_g=occupied[1], clip_extent_b=occupied[2],
                clipping_max=max(low + high), clipping_confidence=0.0,
                chroma_noise=noise, noise_coverage=coverage,
                noise_confidence=min(.5, coverage) if noise is not None else 0.0,
                fringing=fringe, fringe_edge_coverage=edge_coverage,
                fringing_confidence=0.0, wb_lab_ab=wb,
                wb_coverage=neutral_coverage, wb_confidence=0.0,
                mean_luminance=float(gray.mean()))


def compare_white_balance(a, b):
    """Non-directional Lab chroma difference; no 'better' candidate."""
    if not a.get("wb_lab_ab") or not b.get("wb_lab_ab"):
        return None, "neutral_region_unavailable"
    if abs(a["mean_luminance"] - b["mean_luminance"]) > 20:
        return None, "lighting_or_exposure_changed"
    delta = float(np.linalg.norm(np.asarray(a["wb_lab_ab"]) - b["wb_lab_ab"]))
    return delta, "non_directional_difference"
