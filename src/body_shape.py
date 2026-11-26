"""Body-shape classification.

IMPORTANT: these thresholds are heuristics calibrated to the distribution of
the bundled DeepFashion2-derived dataset (median shoulder/hip ratio ~1.70,
which is biased high because hip landmarks are often partially occluded).
They are deliberately chosen to avoid the original project's failure mode
where 99.3% of users were labelled "Inverted Triangle". With labelled data,
replace this with a calibrated classifier; ``evaluation.body_shape_metrics``
measures how good the current rules actually are.
"""

from __future__ import annotations

from dataclasses import dataclass

PEAR_MAX = 1.58
INVERTED_MIN = 1.80
IQR = INVERTED_MIN - PEAR_MAX
SILHOUETTE_REFERENCE = 1.20
MAX_CONFIDENCE = 0.95


@dataclass(frozen=True)
class ShapeResult:
    shape: str
    confidence: float
    ratio: float
    explanation: str


def classify(ratio: float, *, agreement: float = 0.0) -> ShapeResult:
    """Classify a shoulder/hip ratio. ``agreement`` in [0,1] boosts confidence
    when several photos independently agree on the same shape."""
    if ratio < PEAR_MAX:
        shape = "Pear"
        margin = (PEAR_MAX - ratio) / IQR
    elif ratio > INVERTED_MIN:
        shape = "Inverted Triangle"
        margin = (ratio - INVERTED_MIN) / IQR
    else:
        shape = "Rectangle"
        margin = min(ratio - PEAR_MAX, INVERTED_MIN - ratio) / IQR

    confidence = 0.50 + 0.35 * min(margin / 2.0, 1.0) + 0.08 * max(0.0, min(agreement, 1.0))
    confidence = round(min(confidence, MAX_CONFIDENCE), 3)
    explanation = (
        f"Shoulder/hip ratio {ratio:.2f} vs dataset thresholds "
        f"{PEAR_MAX:.2f} / {INVERTED_MIN:.2f}. Confidence is a heuristic "
        "margin, not a calibrated probability."
    )
    return ShapeResult(shape=shape, confidence=confidence, ratio=ratio, explanation=explanation)


def refine_ratio_with_silhouette(
    landmark_ratio: float,
    silhouette_ratio: float | None,
    *,
    n_fitted: int,
) -> float:
    """Use fitted-clothes silhouette width to correct the landmark ratio.

    MediaPipe's person mask measures clothing + body width. A fitted outfit
    therefore gives a cleaner shoulder/hip signal. ``SILHOUETTE_REFERENCE``
    is the assumed ratio for an average build; deviations scale the landmark
    ratio. The correction is capped at 1.0 weight for two fitted photos.
    """
    if silhouette_ratio is None or silhouette_ratio <= 0:
        return float(landmark_ratio)
    weight = min(1.0, n_fitted / 2.0)
    correction = 1.0 + weight * (silhouette_ratio / SILHOUETTE_REFERENCE - 1.0)
    return float(landmark_ratio * max(0.5, min(correction, 1.5)))
