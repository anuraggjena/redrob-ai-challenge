from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


def build_hnsw_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    normalized = np.asarray(embeddings, dtype=np.float32).copy()
    faiss.normalize_L2(normalized)
    index.add(normalized)
    return index


class DenseIndex:
    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._ids: list[str] = []

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        if len(ids) != embeddings.shape[0]:
            raise ValueError("ids length must match embeddings row count")
        self._ids = list(ids)
        self._index = build_hnsw_index(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[str]:
        if self._index is None:
            raise RuntimeError("Dense index has not been built or loaded")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(query)
        _, indices = self._index.search(query, top_k)
        return [self._ids[idx] for idx in indices[0] if idx >= 0]

    def save(self, directory: Path) -> None:
        if self._index is None:
            raise RuntimeError("Dense index has not been built or loaded")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "faiss.index"
        ids_path = directory / "candidate_ids.json"
        tmp_index = directory / "faiss.index.tmp"
        tmp_ids = directory / "candidate_ids.json.tmp"
        faiss.write_index(self._index, str(tmp_index))
        tmp_ids.write_text(json.dumps(self._ids), encoding="utf-8")
        tmp_index.replace(index_path)
        tmp_ids.replace(ids_path)

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        self._index = faiss.read_index(str(directory / "faiss.index"))
        payload = json.loads((directory / "candidate_ids.json").read_text(encoding="utf-8"))
        if isinstance(payload, list):
            self._ids = payload
        else:
            self._ids = payload["candidate_ids"]
