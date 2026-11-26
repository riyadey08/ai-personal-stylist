"""Multi-photo body profile: robust fusion of several poses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

import config
from .body_shape import ShapeResult, classify, refine_ratio_with_silhouette
from .features import FEATURE_ORDER, extract_features
from .pose import PoseObservation


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    cutoff = cum[-1] / 2.0
    return float(values[np.searchsorted(cum, cutoff)])


@dataclass
class BodyProfile:
    features: dict[str, float]
    shape: ShapeResult
    measurements: dict[str, float]
    stability: dict[str, float] = field(default_factory=dict)
    n_photos: int = 0
    n_fitted: int = 0
    silhouette_ratio: float | None = None
    agreement: float = 0.0

    def to_evidence_text(self) -> str:
        lines = [
            f"Body shape (heuristic): {self.shape.shape} (confidence {self.shape.confidence:.2f})",
            f"Shoulder/hip ratio used: {self.shape.ratio:.2f}",
            f"Photos analysed: {self.n_photos} (fitted: {self.n_fitted})",
            f"Independent photo agreement: {self.agreement:.0%}",
        ]
        lines += [f"{k}: {v:.3f}" for k, v in self.measurements.items()]
        if self.stability:
            lines.append(
                "Cross-photo variation (relative std): "
                + ", ".join(f"{k}={v:.1%}" for k, v in self.stability.items())
            )
        return "\n".join(lines)


def _human_measurements(features: dict[str, float]) -> dict[str, float]:
    return {
        "height_units (pose-normalised)": features["height"],
        "shoulder_width_pct_of_height": features["shoulder_width"] * 100,
        "hip_width_pct_of_height": features["hip_width"] * 100,
        "torso_length_pct_of_height": features["torso_length"] * 100,
        "shoulder_hip_ratio": features["shoulder_hip_ratio"],
        "torso_leg_ratio": features["torso_leg_ratio"],
        "arm_leg_ratio": features["arm_leg_ratio"],
        "avg_visibility": features["avg_visibility"],
    }


def build_profile(observations: Iterable[PoseObservation]) -> BodyProfile:
    observations = list(observations)
    if not observations:
        raise ValueError("No usable photos were provided.")

    weights = np.array(
        [o.quality * (config.FITTED_PHOTO_WEIGHT if o.fitted else 1.0) for o in observations],
        dtype=float,
    )
    per_photo = [extract_features(o.row) for o in observations]

    features = {
        key: weighted_median(np.array([f[key] for f in per_photo]), weights)
        for key in FEATURE_ORDER
    }

    fitted = [o for o in observations if o.fitted and o.silhouette_ratio]
    silhouette_ratio = (
        float(np.median([o.silhouette_ratio for o in fitted])) if fitted else None
    )
    features["shoulder_hip_ratio"] = refine_ratio_with_silhouette(
        features["shoulder_hip_ratio"],
        silhouette_ratio,
        n_fitted=len(fitted),
    )

    shapes = [classify(f["shoulder_hip_ratio"]).shape for f in per_photo]
    final_shape = classify(features["shoulder_hip_ratio"])
    agreement = shapes.count(final_shape.shape) / len(shapes)
    # Re-score confidence with cross-photo agreement.
    shape = classify(features["shoulder_hip_ratio"], agreement=agreement)

    tracked = ("shoulder_width", "hip_width", "shoulder_hip_ratio")
    stability: dict[str, float] = {}
    for key in tracked:
        vals = np.array([f[key] for f in per_photo], dtype=float)
        mean = float(np.mean(vals))
        if mean > 0:
            stability[key] = float(np.std(vals) / mean)

    return BodyProfile(
        features=features,
        shape=shape,
        measurements=_human_measurements(features),
        stability=stability,
        n_photos=len(observations),
        n_fitted=len(fitted),
        silhouette_ratio=silhouette_ratio,
        agreement=agreement,
    )
