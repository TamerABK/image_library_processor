"""Experimental noise-aware image technical measurements, never a default V2 input."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import cv2
import numpy as np

VERSION = "technical-calibration-v3-noise-detail"


@dataclass(frozen=True)
class TechnicalQualityResult:
    version: str
    status: str
    score: float | None
    detail_score: float | None
    noise_aware_detail: float | None
    edge_persistence: float | None
    noise_floor: float | None
    chroma_noise: float | None
    chroma_coverage: float
    clipping_bgr: tuple[float, float, float] | None
    confidence: float
    explanation: str

    def to_dict(self):
        return asdict(self)


def _gradient(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def chroma_noise_float_lab(image):
    """Robust median residual in OpenCV float Lab a/b; not a DeltaE value."""
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or min(image.shape[:2]) < 64:
        return None, 0., "unsupported_pixels"
    lab = cv2.cvtColor(image.astype(np.float32)/255., cv2.COLOR_BGR2Lab)
    luma = lab[:, :, 0]
    chroma = lab[:, :, 1:3]
    # Eligibility uses blurred luminance, so increasing sensor noise does not
    # automatically remove every low-texture pixel.
    smooth_luma = cv2.GaussianBlur(luma, (0, 0), 2.5)
    texture = _gradient(smooth_luma)
    eligible = (texture <= np.percentile(texture, 35)) & (luma > 3) & (luma < 97)
    coverage = float(np.mean(eligible))
    if coverage < .08:
        return None, coverage, "insufficient_low_texture_regions"
    median_bgr = cv2.medianBlur(image, 3).astype(np.float32)/255.
    median_chroma = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2Lab)[:, :, 1:3]
    residual = np.linalg.norm(chroma-median_chroma, axis=2)[eligible]
    # Robust location plus dispersion. Report raw units without claiming
    # that colorful microtexture has been perfectly removed.
    center = np.median(residual)
    return float(center + 1.4826*np.median(np.abs(residual-center))), coverage, "measured"


def measure_technical(image):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
        return TechnicalQualityResult(VERSION, "unavailable", None, None, None, None, None, None, 0.,
                                      None, 0., "Unsupported or too-small decoded image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    fine = _gradient(cv2.GaussianBlur(gray, (0, 0), .65))
    coarse = _gradient(cv2.GaussianBlur(gray, (0, 0), 2.0))
    mask = coarse >= max(.015, float(np.percentile(coarse, 80)))
    if int(mask.sum()) < 64:
        return TechnicalQualityResult(VERSION, "unavailable", None, None, None, None, None, None, 0.,
                                      None, 0., "Insufficient structured edges")
    # Persistent coarse edges are less sensitive to uncorrelated sensor noise.
    persistence = float(np.median(coarse[mask] / (fine[mask]+.002)))
    persistence = min(1., max(0., persistence))
    detail = float(np.median(fine[mask]))
    flat = coarse < np.percentile(coarse, 30)
    residual = np.abs(gray-cv2.GaussianBlur(gray, (0, 0), .7))
    noise = float(np.median(residual[flat])) if flat.any() else None
    adjusted = max(0., detail*persistence-noise) if noise is not None else None
    chroma, coverage, chroma_status = chroma_noise_float_lab(image)
    clipping = tuple(float(v) for v in np.mean(image >= 254, axis=(0, 1)))
    # This score is deliberately uncalibrated. It cannot enter ranking until
    # controlled and labeled evidence support a mapping and noise gate.
    confidence = float(min(.75, mask.mean()*4) * (coverage if chroma is not None else .5))
    return TechnicalQualityResult(VERSION, "measured", None, detail, adjusted, persistence, noise,
                                  chroma, coverage, clipping, confidence,
                                  "Raw structured-detail and noise observations; score withheld")
