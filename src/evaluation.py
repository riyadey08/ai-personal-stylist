"""Evaluation metrics for pose, body shape, retrieval, dress fit and RAG.

Every number produced here is persisted to reports/evaluation.md so the
project can be defended with evidence instead of screenshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

import config
from .body_shape import classify
from .dress import DressAnalysis, score_fit
from .features import FEATURE_ORDER
from .knowledge import FashionKnowledgeBase
from .profile import BodyProfile
from .retrieval import StyleIndex


def feature_parity(feature_keys: list[str]) -> dict[str, Any]:
    missing = [c for c in FEATURE_ORDER if c not in feature_keys]
    extra = [c for c in feature_keys if c not in FEATURE_ORDER]
    return {"passed": not missing and not extra, "missing": missing, "extra": extra}


def retrieval_metrics(csv_path: Path, k: int = 5, test_size: float = 0.2) -> dict[str, Any]:
    """Group-split by image_name so one person's clothing items never leak."""
    df = pd.read_csv(csv_path).dropna(subset=FEATURE_ORDER)
    groups = df["image_name"]
    train_idx, test_idx = next(
        GroupShuffleSplit(test_size=test_size, random_state=42).split(df, groups=groups)
    )
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train[FEATURE_ORDER].values)
    pca = PCA(n_components=0.95, random_state=42).fit(scaler.transform(train[FEATURE_ORDER].values))
    Xtr = pca.transform(scaler.transform(train[FEATURE_ORDER].values)).astype("float32")
    Xte = pca.transform(scaler.transform(test[FEATURE_ORDER].values)).astype("float32")

    index = faiss.IndexFlatL2(Xtr.shape[1])
    index.add(Xtr)
    _, neighbours = index.search(Xte, min(k, len(Xtr)))

    relevant = neighbours != -1
    if "body_shape" in test:
        train_shape = train["body_shape"].astype(str).to_numpy()
        truth = test["body_shape"].astype(str).to_numpy()[:, None]
        relevant &= train_shape[neighbours.clip(0)] == truth

    hits = relevant.any(axis=1)
    recall = float(hits.mean())
    first_hit = np.argmax(relevant, axis=1) + 1
    mrr = float(np.mean(1.0 / first_hit[hits])) if hits.any() else 0.0
    discounts = 1.0 / np.log2(np.arange(relevant.shape[1]) + 2)
    dcg_per_row = np.sum(relevant * discounts, axis=1)
    n_rel = relevant.sum(axis=1)
    idcg_per_row = np.array(
        [np.sum(discounts[: min(int(n), len(discounts))]) for n in n_rel]
    )
    ndcg = float(np.mean(np.divide(dcg_per_row, idcg_per_row, out=np.zeros_like(dcg_per_row), where=idcg_per_row > 0)))
    unique = float(np.mean([len(set(train.iloc[row]["image_name"])) for row in neighbours]))
    duplicate_rate = 1.0 - unique / k
    self_match = float(np.mean([np.any(train.iloc[row]["image_name"].values == name)
                                for row, name in zip(neighbours, test["image_name"].values)]))

    return {
        "n_train": int(len(train)), "n_test": int(len(test)),
        f"recall@{k}": round(recall, 4), f"mrr@{k}": round(mrr, 4),
        f"ndcg@{k}": round(ndcg, 4), f"unique_identities@{k}": round(unique, 2),
        f"duplicate_rate@{k}": round(duplicate_rate, 4),
        "self_match_rate": round(self_match, 4),
        "note": "Relevance proxy = same body_shape label; label is 99.3% one class in bundled data.",
    }


def body_shape_metrics(labeled_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(labeled_csv)
    ratio = df["shoulder_hip_ratio"] if "shoulder_hip_ratio" in df else df["ratio"]
    y_true = df["body_shape"]
    y_pred = [classify(float(v)).shape for v in ratio]
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    majority = float(y_true.value_counts(normalize=True).max())
    return {
        "n": int(len(df)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "majority_baseline_accuracy": round(majority, 4),
        "per_class_recall": {label: round(float(v), 4) for label, v in zip(labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0))},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }


def dress_fit_metrics(ratings_csv: Path, profile: BodyProfile, kb: FashionKnowledgeBase) -> dict[str, Any]:
    df = pd.read_csv(ratings_csv)
    predictions = [
        score_fit(
            profile,
            DressAnalysis(
                category=row.get("category", "dress"),
                silhouette=row.get("silhouette", "unknown"),
                length=row.get("length", "unknown"),
                colors=str(row.get("colors", "")).split("|") if row.get("colors") else [],
            ),
            occasion=row.get("occasion", "Any"),
            season=row.get("season", "All"),
            kb=kb,
        ).score
        for _, row in df.iterrows()
    ]
    human = df["human_score"].astype(float).values
    rho = float(spearmanr(human, predictions).statistic)
    return {
        "n": int(len(df)),
        "spearman_rho": round(rho, 4),
        "mae": round(float(np.mean(np.abs(human - np.array(predictions)))), 2),
        "note": "Human score expected on the same 0-100 scale as the fit score.",
    }


def write_report(metrics: dict[str, Any], path: Path | None = None) -> Path:
    path = path or config.REPORTS_DIR / "evaluation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Evaluation report", ""]
    for section, values in metrics.items():
        lines += [f"## {section}", ""]
        if isinstance(values, dict):
            lines += [f"- `{k}`: {v}" for k, v in values.items()]
        else:
            lines.append(str(values))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
