"""The single feature extractor used by training, inference and evaluation.

Keeping one implementation is what prevents the train/serve skew that broke
the original project (``pose_score`` was computed differently in each notebook).
The 43 features below intentionally match the original feature order so the
published FAISS/scaler/PCA artifacts stay compatible.
"""

from __future__ import annotations

import numpy as np

FEATURE_ORDER = [
    "height",
    "shoulder_width", "hip_width", "torso_length",
    "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm",
    "left_arm", "right_arm",
    "left_thigh", "right_thigh", "left_calf", "right_calf",
    "left_leg", "right_leg",
    "shoulder_hip_ratio", "torso_leg_ratio",
    "shoulder_height_ratio", "hip_height_ratio", "torso_height_ratio",
    "arm_height_ratio", "leg_height_ratio", "arm_leg_ratio",
    "shoulder_arm_ratio", "hip_leg_ratio",
    "left_right_arm_ratio", "left_right_leg_ratio",
    "arm_symmetry", "leg_symmetry",
    "shoulder_level_diff", "hip_level_diff", "knee_level_diff", "ankle_level_diff",
    "left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle",
    "avg_visibility", "min_visibility", "max_visibility",
    "visible_landmarks", "pose_score",
]

EPS = 1e-6


def distance3d(row, p1: int, p2: int) -> float:
    a = np.array([row[f"x_{p1}"], row[f"y_{p1}"], row[f"z_{p1}"]], dtype=float)
    b = np.array([row[f"x_{p2}"], row[f"y_{p2}"], row[f"z_{p2}"]], dtype=float)
    return float(np.linalg.norm(a - b))


def midpoint3d(row, p1: int, p2: int) -> np.ndarray:
    return np.array(
        [
            (row[f"x_{p1}"] + row[f"x_{p2}"]) / 2,
            (row[f"y_{p1}"] + row[f"y_{p2}"]) / 2,
            (row[f"z_{p1}"] + row[f"z_{p2}"]) / 2,
        ],
        dtype=float,
    )


def point_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def estimated_height(row) -> float:
    head = np.array([row["x_0"], row["y_0"], row["z_0"]], dtype=float)
    return (
        point_distance(head, midpoint3d(row, 11, 12))
        + point_distance(midpoint3d(row, 11, 12), midpoint3d(row, 23, 24))
        + point_distance(midpoint3d(row, 23, 24), midpoint3d(row, 25, 26))
        + point_distance(midpoint3d(row, 25, 26), midpoint3d(row, 27, 28))
    )


def angle3d(row, a: int, b: int, c: int) -> float:
    A = np.array([row[f"x_{a}"], row[f"y_{a}"], row[f"z_{a}"]], dtype=float)
    B = np.array([row[f"x_{b}"], row[f"y_{b}"], row[f"z_{b}"]], dtype=float)
    C = np.array([row[f"x_{c}"], row[f"y_{c}"], row[f"z_{c}"]], dtype=float)
    ba, bc = A - B, C - B
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < EPS:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def extract_features(row) -> dict[str, float]:
    """Return exactly ``FEATURE_ORDER`` keys.

    ``row`` must contain ``x_i/y_i/z_i/v_i`` for i in 0..32. Extra keys
    (e.g. silhouette widths from fitted photos) are ignored here on purpose
    so the FAISS input space never changes silently.
    """
    height = estimated_height(row)
    if height <= 0:
        raise ValueError("Invalid pose: estimated height is zero.")

    def norm(value: float) -> float:
        return float(value / height)

    shoulder = distance3d(row, 11, 12)
    hip = distance3d(row, 23, 24)
    torso = point_distance(midpoint3d(row, 11, 12), midpoint3d(row, 23, 24))

    lu, ru = distance3d(row, 11, 13), distance3d(row, 12, 14)
    lf, rf = distance3d(row, 13, 15), distance3d(row, 14, 16)
    left_arm, right_arm = lu + lf, ru + rf
    avg_arm = (left_arm + right_arm) / 2

    lt, rt = distance3d(row, 23, 25), distance3d(row, 24, 26)
    lc, rc = distance3d(row, 25, 27), distance3d(row, 26, 28)
    left_leg, right_leg = lt + lc, rt + rc
    avg_leg = (left_leg + right_leg) / 2

    visibility = [float(row[f"v_{i}"]) for i in range(33)]
    avg_vis = float(np.mean(visibility))
    visible = int(np.sum(np.array(visibility) > 0.5))

    features = {
        "height": float(height),
        "shoulder_width": norm(shoulder),
        "hip_width": norm(hip),
        "torso_length": norm(torso),
        "left_upper_arm": norm(lu),
        "right_upper_arm": norm(ru),
        "left_forearm": norm(lf),
        "right_forearm": norm(rf),
        "left_arm": norm(left_arm),
        "right_arm": norm(right_arm),
        "left_thigh": norm(lt),
        "right_thigh": norm(rt),
        "left_calf": norm(lc),
        "right_calf": norm(rc),
        "left_leg": norm(left_leg),
        "right_leg": norm(right_leg),
        "shoulder_hip_ratio": float(shoulder / (hip + EPS)),
        "torso_leg_ratio": float(torso / (avg_leg + EPS)),
        "shoulder_height_ratio": norm(shoulder),
        "hip_height_ratio": norm(hip),
        "torso_height_ratio": norm(torso),
        "arm_height_ratio": float(avg_arm / height),
        "leg_height_ratio": float(avg_leg / height),
        "arm_leg_ratio": float(avg_arm / (avg_leg + EPS)),
        "shoulder_arm_ratio": float(shoulder / (avg_arm + EPS)),
        "hip_leg_ratio": float(hip / (avg_leg + EPS)),
        "left_right_arm_ratio": float(left_arm / (right_arm + EPS)),
        "left_right_leg_ratio": float(left_leg / (right_leg + EPS)),
        "arm_symmetry": float(abs(left_arm - right_arm)),
        "leg_symmetry": float(abs(left_leg - right_leg)),
        "shoulder_level_diff": float(abs(row["y_11"] - row["y_12"])),
        "hip_level_diff": float(abs(row["y_23"] - row["y_24"])),
        "knee_level_diff": float(abs(row["y_25"] - row["y_26"])),
        "ankle_level_diff": float(abs(row["y_27"] - row["y_28"])),
        "left_elbow_angle": angle3d(row, 11, 13, 15),
        "right_elbow_angle": angle3d(row, 12, 14, 16),
        "left_knee_angle": angle3d(row, 23, 25, 27),
        "right_knee_angle": angle3d(row, 24, 26, 28),
        "avg_visibility": avg_vis,
        "min_visibility": float(np.min(visibility)),
        "max_visibility": float(np.max(visibility)),
        "visible_landmarks": float(visible),
        # Same formula as the original training notebook - critical for parity.
        "pose_score": float(avg_vis * visible),
    }
    return {key: float(features[key]) for key in FEATURE_ORDER}


def feature_frame(features: dict[str, float]):
    """Return a one-row DataFrame in the exact training order (lazy import)."""
    import pandas as pd

    return pd.DataFrame([[features[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
