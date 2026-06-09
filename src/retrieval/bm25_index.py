from __future__ import annotations

import json
from pathlib import Path

import joblib
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []

    def build(self, documents: list[str], ids: list[str]) -> None:
        if len(documents) != len(ids):
            raise ValueError("documents and ids must have the same length")
        self._ids = list(ids)
        self._corpus_tokens = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def query(self, text: str, top_k: int = 10) -> list[str]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index has not been built or loaded")
        query_tokens = _tokenize(text)
        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)), key=lambda idx: scores[idx], reverse=True
        )
        return [self._ids[idx] for idx in ranked_indices[:top_k]]

    def save(self, directory: Path) -> None:
        if self._bm25 is None:
            raise RuntimeError("BM25 index has not been built or loaded")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "bm25.pkl.tmp"
        joblib.dump(
            {
                "bm25": self._bm25,
                "ids": self._ids,
                "corpus_tokens": self._corpus_tokens,
            },
            tmp,
        )
        tmp.replace(directory / "bm25.pkl")

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        payload = joblib.load(directory / "bm25.pkl")
        self._bm25 = payload["bm25"]
        self._ids = payload["ids"]
        self._corpus_tokens = payload["corpus_tokens"]
