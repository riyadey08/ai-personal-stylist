import csv
from pathlib import Path

from src.body_shape import classify
from src.dress import DressAnalysis, score_fit
from src.features import FEATURE_ORDER, extract_features
from src.knowledge import FashionKnowledgeBase
from src.profile import BodyProfile, weighted_median

ROOT = Path(__file__).resolve().parents[1]


def fake_row():
    row = {}
    for i in range(33):
        row[f"x_{i}"] = 0.40 + 0.001 * i
        row[f"y_{i}"] = 0.10 + 0.01 * i
        row[f"z_{i}"] = 0.0
        row[f"v_{i}"] = 0.9
    return row


def fake_profile():
    return BodyProfile(
        features={"shoulder_hip_ratio": 1.7},
        shape=classify(1.7),
        measurements={},
        stability={},
        n_photos=1,
    )


def test_feature_extractor_matches_training_order():
    features = extract_features(fake_row())
    assert list(features) == FEATURE_ORDER
    assert all(abs(v) < float("inf") for v in features.values())
    with (ROOT / "data" / "body_features_with_metadata.csv").open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert all(column in header for column in FEATURE_ORDER)


def test_body_shape_thresholds_are_balanced():
    assert classify(1.30).shape == "Pear"
    assert classify(1.70).shape == "Rectangle"
    assert classify(2.10).shape == "Inverted Triangle"
    assert classify(2.10).confidence > classify(1.70).confidence


def test_weighted_median():
    import numpy as np
    assert weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])) == 2.0


def test_fit_score_prefers_flattering_silhouette():
    kb = FashionKnowledgeBase()
    profile = fake_profile()
    good = score_fit(profile, DressAnalysis(silhouette="fit-and-flare", length="midi", colors=["blue"]), kb=kb)
    bad = score_fit(profile, DressAnalysis(silhouette="shift", length="midi", colors=["blue"]), kb=kb)
    assert good.score > bad.score
    assert 0 <= good.score <= 100
    assert good.verdict in {"Flawless", "Good with tweaks", "Risky"}


def test_llm_json_extraction():
    from src.llm import _extract_json
    assert _extract_json('```json\n{"silhouette": "A-line"}\n```')["silhouette"] == "A-line"


def test_candidate_generation_is_reproducible():
    from src.labeling import generate_candidates
    first = generate_candidates(n=20, seed=7)
    second = generate_candidates(n=20, seed=7)
    assert first == second
    assert len({tuple(c.items()) for c in first}) == 20


def test_rating_append_and_summary(tmp_path):
    from src.labeling import append_rating, generate_candidates, load_ratings, rating_summary
    path = tmp_path / "ratings.csv"
    candidate = generate_candidates(n=1)[0]
    append_rating(candidate, 85, path)
    append_rating(candidate, 65, path)
    df = load_ratings(path)
    assert list(df.columns) == ["human_score", "category", "silhouette", "length", "colors", "occasion", "season"]
    assert df["human_score"].tolist() == [85, 65]
    assert rating_summary(path)["n"] == 2
