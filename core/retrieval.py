"""
Lightweight retrieval layer over the curated JSON datasets.

Design choices:
- No heavy vector DB dependency (FAISS/Chroma) needed at this dataset size.
  Embeddings are computed once at startup and cached in memory as numpy
  arrays; similarity search is a plain cosine-similarity argsort.
- Falls back to keyword overlap scoring if the embedding API is unavailable
  (e.g. no API key during unit tests), so retrieval logic stays testable
  without a live network call.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_dataset(name: str) -> list[dict]:
    path = os.path.join(DATA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _record_text(record: dict) -> str:
    """Flatten a dataset record into a single string for embedding/matching."""
    parts = [
        record.get("name") or record.get("title", ""),
        record.get("state", ""),
        record.get("location", ""),
        record.get("description", ""),
        " ".join(record.get("type", [])) if isinstance(record.get("type"), list) else "",
        " ".join(record.get("tags", [])) if isinstance(record.get("tags"), list) else "",
    ]
    return " ".join(p for p in parts if p)


def _keyword_score(query: str, text: str) -> float:
    """Simple fallback scorer: fraction of query words present in the text."""
    query_words = set(query.lower().split())
    text_lower = text.lower()
    if not query_words:
        return 0.0
    hits = sum(1 for w in query_words if w in text_lower)
    return hits / len(query_words)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class DatasetIndex:
    """Holds a dataset plus (optionally) its precomputed embeddings."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.texts = [_record_text(r) for r in records]
        self._embeddings: list[list[float]] | None = None

    def build_embeddings(self) -> None:
        """
        Attempt to compute embeddings via the Gemini embedding API.
        Silently no-ops on failure — search then falls back to keyword
        scoring, which keeps the app usable even without network access.
        """
        from core.gemini_client import embed, GeminiClientError

        try:
            self._embeddings = embed(self.texts)
        except GeminiClientError:
            self._embeddings = None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._embeddings is not None:
            try:
                from core.gemini_client import embed

                query_vec = np.array(embed([query])[0])
                scored = [
                    (i, _cosine_sim(query_vec, np.array(vec)))
                    for i, vec in enumerate(self._embeddings)
                ]
            except Exception:
                scored = [(i, _keyword_score(query, t)) for i, t in enumerate(self.texts)]
        else:
            scored = [(i, _keyword_score(query, t)) for i, t in enumerate(self.texts)]

        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scored[:top_k]]
        return [self.records[i] for i in top_indices]


@lru_cache(maxsize=1)
def get_destinations_index() -> DatasetIndex:
    index = DatasetIndex(load_dataset("destinations.json"))
    index.build_embeddings()
    return index


@lru_cache(maxsize=1)
def get_experiences_index() -> DatasetIndex:
    index = DatasetIndex(load_dataset("experiences.json"))
    index.build_embeddings()
    return index


def search_destinations(query: str, top_k: int = 5) -> list[dict]:
    return get_destinations_index().search(query, top_k=top_k)


def search_experiences(query: str, top_k: int = 5) -> list[dict]:
    return get_experiences_index().search(query, top_k=top_k)
