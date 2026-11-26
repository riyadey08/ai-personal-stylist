"""Collect human dress-fit ratings for evaluating the fit scorer.

Ratings are appended to ``data/dress_ratings.csv`` in exactly the schema that
``evaluation.dress_fit_metrics`` consumes. Candidates are generated from a
deterministic catalogue so repeated runs produce the same sample order.
"""

from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

import config
from .dress import DressAnalysis
from .profile import BodyProfile

RATING_COLUMNS = ["human_score", "category", "silhouette", "length", "colors", "occasion", "season"]
RATING_PATH = config.DATA_DIR / "dress_ratings.csv"
PROFILE_PATH = config.MODELS_DIR / "user_profile.joblib"

SILHOUETTES = ["bodycon", "straight", "A-line", "fit-and-flare", "empire", "wrap", "shift"]
LENGTHS = ["mini", "midi", "maxi"]
PALETTES = [
    ["blue"], ["black"], ["white"], ["beige"], ["red"],
    ["green"], ["pink"], ["navy", "grey"], ["brown", "orange"], ["purple"],
]
OCCASIONS = ["Any", "Casual", "College", "Office", "Party", "Travel"]
SEASONS = ["All", "Summer", "Winter", "Autumn", "Spring"]


def generate_candidates(n: int = 200, seed: int = 42) -> list[dict[str, str]]:
    """Return n unique dress descriptions in a reproducible order."""
    combos = list(itertools.product(SILHOUETTES, LENGTHS, PALETTES, OCCASIONS, SEASONS))
    random.Random(seed).shuffle(combos)
    return [
        {
            "category": "dress",
            "silhouette": silhouette,
            "length": length,
            "colors": "|".join(colors),
            "occasion": occasion,
            "season": season,
        }
        for silhouette, length, colors, occasion, season in combos[:n]
    ]


def candidate_from_analysis(analysis: DressAnalysis, occasion: str, season: str) -> dict[str, str]:
    return {
        "category": analysis.category,
        "silhouette": analysis.silhouette,
        "length": analysis.length,
        "colors": "|".join(analysis.colors),
        "occasion": occasion,
        "season": season,
    }


def load_ratings(path: Path = RATING_PATH) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RATING_COLUMNS)
    return pd.read_csv(path)


def append_rating(candidate: dict[str, Any], score: int, path: Path = RATING_PATH) -> Path:
    if not 0 <= score <= 100:
        raise ValueError("Rating must be between 0 and 100.")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {"human_score": int(score), **{key: candidate.get(key, "") for key in RATING_COLUMNS[1:]}}
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RATING_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path


def rating_summary(path: Path = RATING_PATH) -> dict[str, Any]:
    df = load_ratings(path)
    if df.empty:
        return {"n": 0, "message": "No ratings collected yet."}
    return {
        "n": int(len(df)),
        "mean_score": round(float(df["human_score"].mean()), 1),
        "min_score": int(df["human_score"].min()),
        "max_score": int(df["human_score"].max()),
        "per_silhouette": df["silhouette"].value_counts().to_dict(),
        "per_occasion": df["occasion"].value_counts().to_dict(),
    }


def save_profile(profile: BodyProfile, path: Path = PROFILE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(profile, path)
    return path


def load_profile(path: Path = PROFILE_PATH) -> BodyProfile:
    return joblib.load(path)
