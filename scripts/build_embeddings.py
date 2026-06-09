from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_BATCH_SIZE = 64


def _checkpoint_dir(out_dir: Path) -> Path:
    return out_dir / "checkpoints" / "embeddings"


def _manifest_path(out_dir: Path) -> Path:
    return _checkpoint_dir(out_dir) / "manifest.json"


def _load_manifest(out_dir: Path) -> dict:
    path = _manifest_path(out_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    atomic_write_json(_manifest_path(out_dir), manifest)


def _chunk_embedding_path(out_dir: Path, chunk_index: int) -> Path:
    return _checkpoint_dir(out_dir) / f"chunk_{chunk_index:05d}.npy"


def _chunk_ids_path(out_dir: Path, chunk_index: int) -> Path:
    return _checkpoint_dir(out_dir) / f"chunk_{chunk_index:05d}_ids.json"


def build_embeddings(
    candidates_path: Path,
    out_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = True,
) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_dir(out_dir)
    checkpoint.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    candidates = load_candidates(candidates_path)
    texts = [candidate.document_text for candidate in candidates]
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    total = len(candidates)
    dataset_hash = candidate_set_hash(candidate_ids)

    manifest = _load_manifest(out_dir) if resume else {}
    if manifest.get("candidate_hash") not in (None, dataset_hash):
        raise ValueError(
            "Embedding checkpoint candidate hash mismatch. "
            "Remove artifacts/checkpoints/embeddings or pass --no-resume."
        )

    completed: set[int] = set(manifest.get("completed_chunks", []))
    num_chunks = (total + chunk_size - 1) // chunk_size
    print(
        f"Encoding {total} candidates with {MODEL_NAME} "
        f"(chunk_size={chunk_size}, batch_size={batch_size}, "
        f"resume={resume}, completed_chunks={len(completed)})..."
    )

    model = SentenceTransformer(MODEL_NAME)

    for chunk_index in range(num_chunks):
        if chunk_index in completed:
            continue

        start_idx = chunk_index * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        chunk_texts = texts[start_idx:end_idx]
        chunk_ids = candidate_ids[start_idx:end_idx]
        chunk_start = time.perf_counter()
        print(
            f"[embeddings] starting chunk {chunk_index} "
            f"({start_idx + 1}-{end_idx} of {total})...",
            flush=True,
        )

        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=False,
        )
        chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)
        atomic_write_numpy(_chunk_embedding_path(out_dir, chunk_index), chunk_embeddings)
        atomic_write_json(
            _chunk_ids_path(out_dir, chunk_index),
            {"candidate_ids": chunk_ids},
        )

        completed.add(chunk_index)
        manifest = {
            "candidate_hash": dataset_hash,
            "candidate_count": total,
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "model_name": MODEL_NAME,
            "completed_chunks": sorted(completed),
        }
        _save_manifest(out_dir, manifest)
        print(
            f"[embeddings] finished chunk {chunk_index} in "
            f"{time.perf_counter() - chunk_start:.1f}s "
            f"({len(chunk_ids)} rows)",
            flush=True,
        )
        log_progress("embeddings", end_idx, total, start)

    chunk_arrays = [
        np.load(_chunk_embedding_path(out_dir, idx))
        for idx in range(num_chunks)
    ]
    embeddings = np.vstack(chunk_arrays)
    if embeddings.shape[0] != total:
        raise RuntimeError(
            f"Embedding build incomplete: expected {total} rows, got {embeddings.shape[0]}"
        )

    atomic_write_numpy(out_dir / "embeddings.npy", embeddings)
    atomic_write_json(
        out_dir / "candidate_ids.json",
        {
            "candidate_ids": candidate_ids,
            "candidate_hash": dataset_hash,
        },
    )
    log_progress("embeddings", total, total, start)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate embeddings.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    build_embeddings(
        args.candidates,
        args.out,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
