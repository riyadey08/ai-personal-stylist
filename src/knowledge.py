"""Fashion rule knowledge base and grounded context building."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config

DEEPFASHION_TO_KB = {
    "short sleeve top": "blouse",
    "long sleeve top": "blouse",
    "vest top": "blouse",
    "shirt": "blouse",
    "trousers": "trousers",
    "shorts": "trousers",
    "jeans": "jeans",
    "skirt": "skirt",
    "vest dress": "dress",
    "short sleeve dress": "dress",
    "long sleeve dress": "dress",
    "jacket": "jacket",
    "coat": "coat",
    "blazer": "blazer",
}

BODY_FOCUS = {
    "Pear": {"Curvy hips", "Short legs"},
    "Rectangle": {"Balanced body", "Petite"},
    "Inverted Triangle": {"Broad shoulders", "Long legs"},
}


class FashionKnowledgeBase:
    def __init__(self, path: Path | None = None) -> None:
        self.df = pd.read_csv(path or config.KB_PATH).fillna("")

    @staticmethod
    def map_categories(retrieved_categories: list[str]) -> list[str]:
        mapped = []
        for value in retrieved_categories:
            key = str(value).strip().lower()
            mapped.append(DEEPFASHION_TO_KB.get(key, key))
        return sorted(set(mapped))

    def filter(
        self,
        categories: list[str],
        *,
        occasion: str = "Any",
        season: str = "All",
        shape: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        kb = self.df.copy()
        mapped = self.map_categories(categories)
        if mapped:
            kb = kb[kb["category"].isin(mapped)]
        if occasion != "Any":
            kb = kb[kb["occasion"].isin([occasion, "All"])]
        if season != "All":
            kb = kb[kb["season"].isin([season, "All"])]

        if kb.empty:  # graceful fallback: keep rules relevant to the body shape
            kb = self.df.copy()

        focus = BODY_FOCUS.get(shape or "", set())
        kb["_score"] = (
            kb["category"].isin(mapped).astype(int) * 3
            + kb["body_focus"].isin(focus).astype(int) * 2
            + (kb["occasion"] == occasion).astype(int)
            + (kb["season"] == season).astype(int)
        )
        kb = kb.sort_values("_score", ascending=False).drop(columns="_score")
        return kb.head(limit or config.MAX_CONTEXT_ROWS).reset_index(drop=True)

    @staticmethod
    def to_context(rows: pd.DataFrame, max_chars: int = config.MAX_CONTEXT_CHARS) -> str:
        lines = []
        for _, row in rows.iterrows():
            lines.append(
                f"- [{row['category']} / {row['occasion']} / {row['season']} / {row['body_focus']}] "
                f"DO: {row['recommendation']} AVOID: {row['avoid']}"
            )
        return "\n".join(lines)[:max_chars]
