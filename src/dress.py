"""Dress photo analysis and fit scoring against a customer's body profile.

The CV stage extracts silhouette, length, colour and pattern. The fit score is
an explainable heuristic informed by the fashion knowledge base; Gemini adds
the natural-language judgement. This is *not* a trained fit classifier - see
README (Limitations) and the evaluation harness for how to validate it.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .knowledge import FashionKnowledgeBase
from .profile import BodyProfile

GOOD_SILHOUETTES = {
    "Pear": {"A-line", "fit-and-flare", "empire", "wrap"},
    "Rectangle": {"fit-and-flare", "ruched", "wrap", "belted"},
    "Inverted Triangle": {"A-line", "straight", "wrap", "v-neck"},
}
BAD_SILHOUETTES = {
    "Pear": {"bodycon", "pencil", "tight"},
    "Rectangle": {"shift", "shapeless", "straight"},
    "Inverted Triangle": {"bodycon", "halter", "puff-sleeve"},
}

SEASON_PALETTE = {
    "Summer": {"white", "blue", "beige", "pink", "yellow", "green"},
    "Winter": {"black", "navy", "grey", "burgundy", "purple", "brown"},
    "Autumn": {"brown", "beige", "olive", "orange", "burgundy"},
    "Spring": {"pink", "green", "white", "blue", "yellow"},
    "All": set(),
}


@dataclass
class DressAnalysis:
    category: str = "dress"
    silhouette: str = "unknown"
    length: str = "unknown"
    colors: list[str] = field(default_factory=list)
    pattern: str = "unknown"
    mask_coverage: float = 0.0
    source: str = "cv"

    def merge(self, other: dict[str, Any]) -> "DressAnalysis":
        """Overlay Gemini's fields without losing CV values."""
        for key in ("category", "silhouette", "length", "pattern"):
            value = other.get(key)
            if value and str(value).strip():
                setattr(self, key, str(value).strip().lower())
        colors = other.get("colors")
        if isinstance(colors, list) and colors:
            self.colors = [str(c).lower() for c in colors]
        self.source = "cv+gemini"
        return self

    def summary(self) -> str:
        return (
            f"{self.length} {self.silhouette} dress in {', '.join(self.colors) or 'unknown colour'} "
            f"({self.pattern})"
        )


@dataclass
class FitVerdict:
    score: int
    verdict: str
    reasons: list[str]
    analysis: DressAnalysis

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "dress": self.analysis.summary(),
        }


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Use JPG/PNG.")
    return image


def segment_garment(image_bgr: np.ndarray) -> np.ndarray:
    """Foreground mask for a garment on a plain background."""
    h, w = image_bgr.shape[:2]
    border = np.concatenate(
        [image_bgr[:10].reshape(-1, 3), image_bgr[-10:].reshape(-1, 3),
         image_bgr[:, :10].reshape(-1, 3), image_bgr[:, -10:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(image_bgr.astype(np.float32) - bg, axis=2)
    _, mask = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((max(3, h // 150), max(3, w // 150)), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    if np.count_nonzero(mask) < 0.01 * h * w:
        mask[:] = 255
    return mask


def _color_name(rgb: np.ndarray) -> str:
    r, g, b = [float(v) / 255 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.18:
        return "black"
    if s < 0.16:
        return "white" if v > 0.85 else "grey"
    hue = h * 360
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange" if v > 0.7 else "brown"
    if hue < 70:
        return "yellow"
    if hue < 160:
        return "green" if s > 0.35 else "olive"
    if hue < 200:
        return "blue" if v > 0.6 else "navy"
    if hue < 260:
        return "purple"
    return "pink" if s < 0.6 else "burgundy"


def analyze_dress_cv(image_bgr: np.ndarray) -> DressAnalysis:
    mask = segment_garment(image_bgr)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return DressAnalysis()
    cover = float(ys.size / mask.size)

    row_widths = np.count_nonzero(mask, axis=1).astype(float)
    k = max(1, len(row_widths) // 20)
    smooth = np.convolve(row_widths, np.ones(k) / k, mode="same")
    top = float(np.mean(smooth[: max(1, len(smooth) // 5)]))
    bottom = float(np.mean(smooth[-max(1, len(smooth) // 5):]))
    flare = bottom / max(top, 1.0)

    if flare < 1.12:
        silhouette = "bodycon"
    elif flare < 1.45:
        silhouette = "straight"
    elif flare < 2.0:
        silhouette = "A-line"
    else:
        silhouette = "fit-and-flare"

    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    aspect = bbox_h / max(bbox_w, 1)
    length = "maxi" if aspect > 3.2 else "midi" if aspect > 2.0 else "mini"

    pixels = image_bgr[mask > 0]
    sample = pixels[np.linspace(0, len(pixels) - 1, min(5000, len(pixels)), dtype=int)]
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            sample.astype(np.float32), min(3, len(sample)), None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        counts = np.bincount(labels.ravel())
        order = np.argsort(counts)[::-1]
        colors = []
        for idx in order:
            bgr = centers[idx]
            colors.append(_color_name(np.array([bgr[2], bgr[1], bgr[0]])))
        colors = list(dict.fromkeys(colors))[:2]
    except cv2.error:
        colors = []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.count_nonzero(edges[mask > 0]) / max(1, np.count_nonzero(mask)))
    pattern = "patterned" if edge_density > 0.14 else "solid"

    return DressAnalysis(
        silhouette=silhouette,
        length=length,
        colors=colors,
        pattern=pattern,
        mask_coverage=cover,
    )


def score_fit(
    profile: BodyProfile,
    dress: DressAnalysis,
    *,
    occasion: str = "Any",
    season: str = "All",
    kb: FashionKnowledgeBase | None = None,
) -> FitVerdict:
    reasons: list[str] = []
    score = 62
    shape = profile.shape.shape

    if dress.silhouette in GOOD_SILHOUETTES.get(shape, set()):
        score += 14
        reasons.append(f"(+14) {dress.silhouette} flatters a {shape} frame")
    if dress.silhouette in BAD_SILHOUETTES.get(shape, set()):
        score -= 18
        reasons.append(f"(-18) {dress.silhouette} tends to emphasise a {shape} frame")

    if dress.length == "midi":
        score += 4
        reasons.append("(+4) midi length is a safe, versatile choice")

    if season in SEASON_PALETTE and any(c in SEASON_PALETTE[season] for c in dress.colors):
        score += 4
        reasons.append(f"(+4) colours suit {season}")

    if kb is not None:
        rows = kb.filter([dress.category], occasion=occasion, season=season, shape=shape, limit=8)
        advice = " ".join(rows["recommendation"].astype(str).tolist()) if not rows.empty else ""
        avoid = " ".join(rows["avoid"].astype(str).tolist()) if not rows.empty else ""
        if advice:
            score += 4
            reasons.append("(+4) supported by the fashion knowledge base")
        if dress.silhouette and dress.silhouette in avoid.lower():
            score -= 6
            reasons.append("(-6) knowledge base explicitly warns against this cut")

    score -= int(round((1 - profile.shape.confidence) * 12))
    reasons.append(f"(-{int(round((1 - profile.shape.confidence) * 12))}) body-profile confidence adjustment")
    if profile.stability:
        worst = max(profile.stability.values())
        penalty = int(round(min(worst, 0.2) * 30))
        score -= penalty
        reasons.append(f"(-{penalty}) measurements varied across your photos")

    score = int(np.clip(score, 0, 100))
    verdict = "Flawless" if score >= 80 else "Good with tweaks" if score >= 60 else "Risky"
    return FitVerdict(score=score, verdict=verdict, reasons=reasons, analysis=dress)
