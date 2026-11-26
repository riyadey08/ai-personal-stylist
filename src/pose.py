"""MediaPipe pose detection, quality scoring and fitted-clothes silhouette cues."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

import config
from .features import EPS


@dataclass
class PoseObservation:
    """One usable photo of the user."""

    row: dict[str, float]
    quality: float
    fitted: bool = False
    silhouette_shoulder_width: float | None = None
    silhouette_hip_width: float | None = None
    silhouette_ratio: float | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)


class PoseEstimator:
    def __init__(self) -> None:
        import mediapipe as mp

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
        )

    def analyze(self, image_bgr: np.ndarray, *, fitted: bool = False, source: str = "") -> PoseObservation | None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return None

        row = row_from_landmarks(result.pose_landmarks.landmark)
        quality = quality_score(row)
        if quality < config.MIN_QUALITY_SCORE:
            return None

        obs = PoseObservation(row=row, quality=float(quality), fitted=fitted, source=source)
        mask = getattr(result, "segmentation_mask", None)
        if mask is not None:
            sw, hw, ratio = silhouette_widths(mask, row)
            obs.silhouette_shoulder_width = sw
            obs.silhouette_hip_width = hw
            obs.silhouette_ratio = ratio
            if fitted and ratio is not None:
                obs.notes.append("Fitted-clothes silhouette used to refine shoulder/hip ratio")
        return obs

    def draw(self, image_bgr: np.ndarray, obs: PoseObservation) -> np.ndarray:
        """Draw landmarks on a copy using the saved landmark coordinates."""
        out = image_bgr.copy()
        h, w = out.shape[:2]
        for i in range(33):
            x = int(obs.row[f"x_{i}"] * w)
            y = int(obs.row[f"y_{i}"] * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(out, (x, y), 3, (0, 255, 0), -1)
        return out


def row_from_landmarks(landmarks: Any) -> dict[str, float]:
    row: dict[str, float] = {}
    for i, lm in enumerate(landmarks):
        row[f"x_{i}"] = float(lm.x)
        row[f"y_{i}"] = float(lm.y)
        row[f"z_{i}"] = float(lm.z)
        row[f"v_{i}"] = float(lm.visibility)
    return row


def quality_score(row: dict[str, float]) -> float:
    score = 0.0
    for idx in (11, 12, 23, 24):
        if row[f"v_{idx}"] > 0.7:
            score += 2
    for idx in (25, 26, 27, 28):
        if row[f"v_{idx}"] > 0.5:
            score += 1
    if row["v_0"] > 0.5:
        score += 1
    return score


def _width_at_y(mask_bin: np.ndarray, y: int, center_x: float) -> float:
    row = np.flatnonzero(mask_bin[y])
    if row.size == 0:
        return 0.0
    runs = np.split(row, np.where(np.diff(row) > 1)[0] + 1)
    run = min(runs, key=lambda r: abs(float(r.mean()) - center_x))
    return float(run[-1] - run[0] + 1)


def silhouette_widths(mask: np.ndarray, row: dict[str, float]) -> tuple[float, float, float | None]:
    """Widths at shoulder/hip height from MediaPipe's person mask.

    Tight clothes make this a much better proxy for true body width than
    loose clothing, which is why fitted photos are weighted more heavily.
    """
    binary = np.asarray(mask) > 0.5
    h, w = binary.shape
    ys, xs = np.nonzero(binary)
    if ys.size < 100:
        return 0.0, 0.0, None

    body_h = float(ys.max() - ys.min() + 1)
    if body_h < EPS:
        return 0.0, 0.0, None
    center_x = float(xs.mean())

    y_shoulder = int(np.clip(round((row["y_11"] + row["y_12"]) / 2 * h), 0, h - 1))
    y_hip = int(np.clip(round((row["y_23"] + row["y_24"]) / 2 * h), 0, h - 1))
    shoulder_w = _width_at_y(binary, y_shoulder, center_x) / body_h
    hip_w = _width_at_y(binary, y_hip, center_x) / body_h
    ratio = float(shoulder_w / (hip_w + EPS)) if hip_w > 0 else None
    return shoulder_w, hip_w, ratio
