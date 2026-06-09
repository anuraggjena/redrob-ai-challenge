from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.understanding.parser import load_candidates
from src.utils.artifacts import (
    atomic_write_json,
    atomic_write_numpy,
    candidate_set_hash,
    log_progress,
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build_embeddings(candidates_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    candidates = load_candidates(candidates_path)
    texts = [candidate.document_text for candidate in candidates]
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    total = len(candidates)
    print(f"Encoding {total} candidates with {MODEL_NAME}...")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    log_progress("embeddings", total, total, start)

    atomic_write_numpy(out_dir / "embeddings.npy", embeddings)
    atomic_write_json(
        out_dir / "candidate_ids.json",
        {
            "candidate_ids": candidate_ids,
            "candidate_hash": candidate_set_hash(candidate_ids),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate embeddings.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_embeddings(args.candidates, args.out)


if __name__ == "__main__":
    main()
