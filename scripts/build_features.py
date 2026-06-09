from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from joblib import Parallel, delayed

from src.features.registry import compute_all_features
from src.understanding.enrich import enrich_profile
from src.understanding.models import CandidateProfile
from src.understanding.parser import load_candidates
from src.utils.artifacts import (
    atomic_write_json,
    atomic_write_parquet,
    cap_n_jobs,
    candidate_set_hash,
    log_progress,
)

DEFAULT_CHUNK_SIZE = 5000


def _process_profile(profile: CandidateProfile) -> dict:
    enrich_profile(profile)
    features = compute_all_features(profile)
    return {"candidate_id": profile.candidate_id, **features}


def _checkpoint_dir(out_path: Path) -> Path:
    return out_path.parent / "checkpoints" / "features"


def _manifest_path(out_path: Path) -> Path:
    return _checkpoint_dir(out_path) / "manifest.json"


def _load_manifest(out_path: Path) -> dict:
    path = _manifest_path(out_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(out_path: Path, manifest: dict) -> None:
    atomic_write_json(_manifest_path(out_path), manifest)


def _chunk_path(out_path: Path, chunk_index: int) -> Path:
    return _checkpoint_dir(out_path) / f"chunk_{chunk_index:05d}.parquet"


def build_features(
    candidates_path: Path,
    out_path: Path,
    *,
    n_jobs: int = -1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    resume: bool = True,
) -> pd.DataFrame:
    candidates = load_candidates(candidates_path)
    candidate_ids = [profile.candidate_id for profile in candidates]
    total = len(candidates)
    dataset_hash = candidate_set_hash(candidate_ids)
    workers = cap_n_jobs(n_jobs)

    checkpoint = _checkpoint_dir(out_path)
    checkpoint.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(out_path) if resume else {}
    if manifest.get("candidate_hash") not in (None, dataset_hash):
        raise ValueError(
            "Checkpoint candidate hash mismatch. "
            "Remove artifacts/checkpoints/features or pass --no-resume."
        )

    completed: set[int] = set(manifest.get("completed_chunks", []))
    start_time = time.perf_counter()
    print(f"Building features for {total} candidates (chunk_size={chunk_size}, n_jobs={workers})...")

    num_chunks = (total + chunk_size - 1) // chunk_size
    for chunk_index in range(num_chunks):
        if chunk_index in completed:
            continue

        start = chunk_index * chunk_size
        end = min(start + chunk_size, total)
        chunk_profiles = candidates[start:end]
        chunk_start = time.perf_counter()
        print(
            f"[features] starting chunk {chunk_index} "
            f"({start + 1}-{end} of {total}, n_jobs={workers})...",
            flush=True,
        )

        rows = Parallel(n_jobs=workers, backend="loky", max_nbytes=None)(
            delayed(_process_profile)(profile) for profile in chunk_profiles
        )
        chunk_frame = pd.DataFrame(rows)
        atomic_write_parquet(_chunk_path(out_path, chunk_index), chunk_frame)
        print(
            f"[features] finished chunk {chunk_index} in "
            f"{time.perf_counter() - chunk_start:.1f}s "
            f"({len(chunk_frame)} rows)",
            flush=True,
        )

        completed.add(chunk_index)
        manifest = {
            "candidate_hash": dataset_hash,
            "candidate_count": total,
            "chunk_size": chunk_size,
            "completed_chunks": sorted(completed),
        }
        _save_manifest(out_path, manifest)
        log_progress("features", end, total, start_time)

    chunk_frames = [
        pd.read_parquet(_chunk_path(out_path, idx))
        for idx in range(num_chunks)
    ]
    frame = pd.concat(chunk_frames, ignore_index=True)
    if len(frame) != total:
        raise RuntimeError(f"Feature build incomplete: expected {total} rows, got {len(frame)}")

    atomic_write_parquet(out_path, frame)
    atomic_write_json(
        out_path.parent / "features_meta.json",
        {
            "candidate_hash": dataset_hash,
            "candidate_count": total,
        },
    )
    log_progress("features", total, total, start_time)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate feature parquet.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/features.parquet"),
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    frame = build_features(
        args.candidates,
        args.out,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        resume=not args.no_resume,
    )
    print(f"Wrote {len(frame)} rows x {len(frame.columns)} columns to {args.out}")


if __name__ == "__main__":
    main()
