"""End-to-end orchestration used by the Streamlit app."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .body_shape import classify
from .dress import DressAnalysis, analyze_dress_cv, decode_image, score_fit
from .features import extract_features
from .knowledge import FashionKnowledgeBase
from .llm import GeminiClient
from .pose import PoseEstimator, PoseObservation
from .profile import BodyProfile, build_profile
from .retrieval import StyleIndex


class StylistPipeline:
    def __init__(
        self,
        estimator: PoseEstimator,
        style_index: StyleIndex,
        knowledge: FashionKnowledgeBase,
        llm: GeminiClient,
    ) -> None:
        self.estimator = estimator
        self.style_index = style_index
        self.knowledge = knowledge
        self.llm = llm
        self.observations: list[PoseObservation] = []
        self.profile: BodyProfile | None = None

    def reset(self) -> None:
        self.observations.clear()
        self.profile = None

    def add_photo(self, data: bytes, *, fitted: bool, name: str = "photo") -> tuple[str | None, np.ndarray | None]:
        """Return (error_message, annotated_image). Exactly one is non-None."""
        try:
            image = decode_image(data)
        except ValueError as exc:
            return str(exc), None
        obs = self.estimator.analyze(image, fitted=fitted, source=name)
        if obs is None:
            return "No clear full-body pose detected (or landmark quality too low). Try another photo.", None
        self.observations.append(obs)
        return None, self.estimator.draw(image, obs)

    def make_profile(self) -> BodyProfile:
        self.profile = build_profile(self.observations)
        return self.profile

    def recommend(self, *, occasion: str, season: str) -> dict[str, Any]:
        if self.profile is None:
            raise RuntimeError("Build a body profile first.")
        results = self.style_index.deduplicate(self.style_index.query(self.profile.features, k=8))
        categories = [r.get("category_name", "") for r in results]
        kb_rows = self.knowledge.filter(
            categories, occasion=occasion, season=season, shape=self.profile.shape.shape
        )
        advice = self.llm.outfit_advice(
            self.profile,
            occasion=occasion,
            season=season,
            retrieved_context=self.style_index.to_context(results),
            kb_context=self.knowledge.to_context(kb_rows),
        )
        return {
            "profile": self.profile,
            "retrieved": results,
            "kb_rows": kb_rows,
            "advice": advice,
        }

    def check_dress(self, data: bytes, *, name: str = "dress", occasion: str = "Any", season: str = "All") -> dict[str, Any]:
        if self.profile is None:
            raise RuntimeError("Build a body profile first.")
        image = decode_image(data)
        cv_analysis = analyze_dress_cv(image)

        suffix = Path(name).suffix.lower() or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            vision = self.llm.dress_analysis(tmp_path, cv_analysis, self.profile)
        finally:
            tmp_path.unlink(missing_ok=True)

        analysis: DressAnalysis = vision["analysis"]
        verdict = score_fit(
            self.profile,
            analysis,
            occasion=occasion,
            season=season,
            kb=self.knowledge,
        )
        return {
            "verdict": verdict,
            "vision_notes": vision["fields"],
            "image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        }

    def per_photo_shapes(self) -> list[str]:
        return [
            classify(extract_features(o.row)["shoulder_hip_ratio"]).shape
            for o in self.observations
        ]
