from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex
from src.retrieval.query_builder import build_query_from_jd
from src.understanding.parser import load_candidates
from src.utils.artifacts import atomic_write_json, atomic_write_numpy, candidate_set_hash, log_progress

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _load_embedding_ids(ids_path: Path) -> list[str]:
    payload = json.loads(ids_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload["candidate_ids"]


def build_indexes(candidates_path: Path, artifacts_dir: Path, config_dir: Path) -> None:
    start = time.perf_counter()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = artifacts_dir / "embeddings.npy"
    ids_path = artifacts_dir / "candidate_ids.json"
    if not embeddings_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            "Expected embeddings.npy and candidate_ids.json in artifacts directory"
        )

    embeddings = np.load(embeddings_path)
    embedding_ids = _load_embedding_ids(ids_path)

    candidates = load_candidates(candidates_path)
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if embedding_ids != candidate_ids:
        raise ValueError(
            "Embedding candidate ID order does not match candidates file. "
            "Rebuild embeddings for this candidate pool."
        )

    dense_index = DenseIndex()
    dense_index.build(embeddings, embedding_ids)
    dense_index.save(artifacts_dir)

    documents = [candidate.document_text for candidate in candidates]
    bm25_index = BM25Index()
    bm25_index.build(documents, candidate_ids)
    bm25_index.save(artifacts_dir / "bm25")

    jd_path = config_dir / "job_requirements.yaml"
    if jd_path.exists():
        dense_query, _ = build_query_from_jd(jd_path)
        model = SentenceTransformer(MODEL_NAME)
        query_embedding = np.asarray(model.encode(dense_query), dtype=np.float32)
        atomic_write_numpy(artifacts_dir / "query_embedding.npy", query_embedding)

    atomic_write_json(
        artifacts_dir / "index_meta.json",
        {
            "candidate_hash": candidate_set_hash(candidate_ids),
            "candidate_count": len(candidate_ids),
        },
    )
    log_progress("indexes", len(candidate_ids), len(candidate_ids), start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS and BM25 indexes.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing embeddings.npy and candidate_ids.json",
    )
    parser.add_argument("--config", type=Path, default=Path("config"))
    args = parser.parse_args()
    build_indexes(args.candidates, args.artifacts, args.config)


if __name__ == "__main__":
    main()
