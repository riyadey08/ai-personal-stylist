"""Offline smoke test of the non-LLM path (no API call)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src import labeling
from src.dress import analyze_dress_cv, score_fit
from src.evaluation import dress_fit_metrics
from src.knowledge import FashionKnowledgeBase
from src.pipeline import StylistPipeline
from src.pose import PoseObservation
from src.retrieval import StyleIndex


def main() -> None:
    row = {}
    for i in range(33):
        row[f"x_{i}"] = 0.4 + 0.001 * i
        row[f"y_{i}"] = 0.1 + 0.01 * i
        row[f"z_{i}"] = 0.0
        row[f"v_{i}"] = 0.9

    pipeline = StylistPipeline(None, StyleIndex.load(), FashionKnowledgeBase(), None)
    pipeline.observations = [PoseObservation(row=row, quality=10, fitted=True, silhouette_ratio=1.3)]
    profile = pipeline.make_profile()
    print("shape:", profile.shape.shape, "confidence:", profile.shape.confidence,
          "ratio:", round(profile.shape.ratio, 3))

    image = np.full((400, 300, 3), 255, np.uint8)
    cv2.rectangle(image, (100, 30), (200, 350), (60, 40, 200), -1)
    dress = analyze_dress_cv(image)
    print("dress:", dress.summary())
    verdict = score_fit(profile, dress, kb=pipeline.knowledge)
    print("verdict:", verdict.verdict, verdict.score, "reasons:", len(verdict.reasons))

    results = pipeline.style_index.deduplicate(pipeline.style_index.query(profile.features, k=5))
    print("retrieved unique:", len(results), "top:", results[0]["category_name"])

    with tempfile.TemporaryDirectory() as tmp:
        rating_path = Path(tmp) / "ratings.csv"
        for score, candidate in zip([80, 55, 70, 40], labeling.generate_candidates(n=4)):
            labeling.append_rating(candidate, score, rating_path)
        print("ratings:", labeling.rating_summary(rating_path))
        print("fit evaluation:", dress_fit_metrics(rating_path, profile, pipeline.knowledge))


if __name__ == "__main__":
    main()
