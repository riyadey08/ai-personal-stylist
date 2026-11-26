"""FAISS similarity search over body profiles, with train/test-safe helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from .features import FEATURE_ORDER, feature_frame

META_COLUMNS = [
    "image_name", "category_name", "style", "viewpoint", "occlusion", "scale", "body_shape",
]


@dataclass
class StyleIndex:
    index: faiss.Index
    scaler: StandardScaler
    pca: PCA
    metadata: pd.DataFrame
    feature_order: list[str]

    @classmethod
    def build(cls, csv_path: Path, out_dir: Path | None = None) -> "StyleIndex":
        df = pd.read_csv(csv_path)
        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            raise ValueError(f"Training CSV is missing features: {missing[:5]}")

        X = df[FEATURE_ORDER].replace([np.inf, -np.inf], np.nan).dropna()
        df = df.loc[X.index].reset_index(drop=True)
        scaler = StandardScaler().fit(X.values)
        X_scaled = scaler.transform(X.values)
        pca = PCA(n_components=0.95, random_state=42).fit(X_scaled)
        X_pca = pca.transform(X_scaled).astype("float32")

        index = faiss.IndexFlatL2(X_pca.shape[1])
        index.add(X_pca)
        meta = df[[c for c in META_COLUMNS if c in df.columns]].reset_index(drop=True)

        style_index = cls(index, scaler, pca, meta, list(FEATURE_ORDER))
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, out_dir / "scaler.joblib")
            joblib.dump(pca, out_dir / "pca.joblib")
            meta.to_csv(out_dir / "retrieval_metadata.csv", index=False)
            joblib.dump(list(FEATURE_ORDER), out_dir / "feature_order.joblib")
            faiss.write_index(index, str(out_dir / "faiss_index.bin"))
        return style_index

    @classmethod
    def load(cls, models_dir: Path | None = None) -> "StyleIndex":
        models_dir = models_dir or config.MODELS_DIR
        try:
            return cls(
                faiss.read_index(str(models_dir / "faiss_index.bin")),
                joblib.load(models_dir / "scaler.joblib"),
                joblib.load(models_dir / "pca.joblib"),
                pd.read_csv(models_dir / "retrieval_metadata.csv"),
                joblib.load(models_dir / "feature_order.joblib"),
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "FAISS artifacts not found. Run: python scripts/build_index.py"
            ) from exc

    def embedding(self, features: dict[str, float]) -> np.ndarray:
        frame = feature_frame(features)[self.feature_order]
        return self.pca.transform(self.scaler.transform(frame.values)).astype("float32")

    def query(self, features: dict[str, float], k: int = config.DEFAULT_K) -> list[dict[str, Any]]:
        k = max(1, min(k, self.index.ntotal))
        distances, indices = self.index.search(self.embedding(features), k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            row = {"distance": float(dist), "index": int(idx)}
            row.update(self.metadata.iloc[idx].to_dict())
            results.append(row)
        return results

    @staticmethod
    def deduplicate(results: list[dict[str, Any]], key: str = "image_name") -> list[dict[str, Any]]:
        seen, unique = set(), []
        for row in results:
            value = row.get(key)
            if value in seen:
                continue
            seen.add(value)
            unique.append(row)
        return unique

    def to_context(self, results: list[dict[str, Any]], max_rows: int = config.MAX_CONTEXT_ROWS) -> str:
        lines = []
        for rank, row in enumerate(results[:max_rows], 1):
            lines.append(
                f"{rank}. {row.get('category_name')} | style={row.get('style')} | "
                f"view={row.get('viewpoint')} | distance={row.get('distance', 0):.3f}"
            )
        text = "\n".join(lines)
        return text[: config.MAX_CONTEXT_CHARS]
