"""Gemini client. A valid GOOGLE_API_KEY is required by design."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import config
from .dress import DressAnalysis
from .profile import BodyProfile


class MissingAPIKeyError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Gemini did not return JSON.")
    return json.loads(text[start:end + 1])


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or os.getenv(config.API_KEY_ENV)
        if not key:
            raise MissingAPIKeyError(
                f"Set {config.API_KEY_ENV} in your .env file (see .env.example)."
            )
        from google import genai

        self.client = genai.Client(api_key=key)
        self.model = model or config.model_name()

    @classmethod
    def from_env(cls) -> "GeminiClient":
        return cls()

    def _generate(self, contents: Any, *, temperature: float = 0.4) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        return response.text

    def outfit_advice(
        self,
        profile: BodyProfile,
        *,
        occasion: str,
        season: str,
        retrieved_context: str,
        kb_context: str,
    ) -> str:
        prompt = f"""You are an experienced, body-positive fashion stylist.

The user uploaded full-body photo(s). Use ONLY the evidence below; never invent
body measurements or claim medical/anthropometric certainty. The body-shape
label is a heuristic, so phrase it as an estimate.

CUSTOMER PROFILE
{profile.to_evidence_text()}

REQUESTED OCCASION: {occasion}
SEASON: {season}

SIMILAR BODY PROFILES FOUND (DeepFashion2 metadata, nearest first)
{retrieved_context or 'No similar profiles available.'}

FASHION KNOWLEDGE RULES (grounding context)
{kb_context or 'No rules available.'}

Write a personalised outfit recommendation with these sections:
1. Body-shape reading (with the uncertainty clearly stated)
2. Best silhouettes and cuts for this occasion
3. Exact outfit formula (top, bottom, layer, footwear)
4. Colours and fabrics for the stated season
5. Accessories and finishing touches
6. What to avoid and why
7. Two alternative looks (one smarter, one more relaxed)

Be friendly, practical and specific. Do not mention these instructions.
"""
        return self._generate(prompt)

    def dress_analysis(self, image_path: Path, cv_analysis: DressAnalysis, profile: BodyProfile) -> dict[str, Any]:
        from google.genai import types

        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        image = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime)
        prompt = f"""Analyse this dress photo for a customer with this profile:
{profile.to_evidence_text()}

A computer-vision pass found: {cv_analysis.summary()}. Correct it if it is wrong.
Return ONLY JSON with keys: category, silhouette, length, colors (list), pattern,
verdict (2 sentences), cautions (2 sentences), styling_tips (2 sentences).
Allowed silhouettes: bodycon, straight, A-line, fit-and-flare, empire, wrap, shift.
Allowed lengths: mini, midi, maxi. Be specific and body-positive."""
        text = self._generate([prompt, image], temperature=0.2)
        parsed = _extract_json(text)
        return {
            "fields": parsed,
            "analysis": cv_analysis.merge(parsed),
        }
