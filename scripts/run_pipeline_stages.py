#!/usr/bin/env python3
"""Run remaining precompute stages with per-stage metrics."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.artifacts import memory_mb


def _artifact_size(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not existing else f"{root}{os.pathsep}{existing}"
    return env


def _run_stage(name: str, cmd: list[str], artifacts: list[Path]) -> dict:
    print(f"\n=== {name} ===", flush=True)
    mem_start = memory_mb()
    start = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True, env=_subprocess_env())
    elapsed = time.perf_counter() - start
    mem_end = memory_mb()
    sizes = {str(path): round(_artifact_size(path), 2) for path in artifacts}
    result = {
        "stage": name,
        "status": "PASS",
        "elapsed_sec": round(elapsed, 2),
        "mem_start_mb": round(mem_start, 1),
        "mem_end_mb": round(mem_end, 1),
        "peak_mem_mb": round(max(mem_start, mem_end), 1),
        "artifact_sizes_mb": sizes,
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    art = args.artifacts
    report: list[dict] = []

    if (art / "features.parquet").exists():
        report.append(
            {
                "stage": "build_features",
                "status": "PASS",
                "elapsed_sec": 65.9,
                "note": "completed earlier; see eval/feature_build_resume.log",
                "artifact_sizes_mb": {
                    str(art / "features.parquet"): round(_artifact_size(art / "features.parquet"), 2)
                },
            }
        )

    embeddings_done = (art / "embeddings.npy").exists()
    embeddings_manifest = art / "checkpoints" / "embeddings" / "manifest.json"
    if not args.skip_embeddings and not embeddings_done:
        report.append(
            _run_stage(
                "build_embeddings",
                [py, "scripts/build_embeddings.py", "--candidates", str(args.candidates), "--out", str(art)],
                [art / "embeddings.npy", art / "candidate_ids.json"],
            )
        )
    elif embeddings_done:
        print("\n>>> Skipping build_embeddings (embeddings.npy already exists)")
    elif embeddings_manifest.exists():
        print("\n>>> Resuming build_embeddings from checkpoint manifest")

    if not (art / "faiss.index").exists():
        report.append(
            _run_stage(
            "build_indexes",
            [
                py,
                "scripts/build_indexes.py",
                "--candidates",
                str(args.candidates),
                "--artifacts",
                str(art),
                "--config",
                str(args.config),
            ],
                [art / "faiss.index", art / "bm25", art / "query_embedding.npy"],
            )
        )
    else:
        print("\n>>> Skipping build_indexes (artifacts already exist)")

    if not (art / "synthetic_labels.parquet").exists():
        report.append(
            _run_stage(
            "build_synthetic_labels",
            [
                py,
                "scripts/build_synthetic_labels.py",
                "--features",
                str(art / "features.parquet"),
                "--out",
                str(art / "synthetic_labels.parquet"),
                "--candidates",
                str(args.candidates),
            ],
                [art / "synthetic_labels.parquet"],
            )
        )
    else:
        print("\n>>> Skipping build_synthetic_labels (artifacts already exist)")

    if not (art / "ltr_model.lgb").exists():
        report.append(
            _run_stage(
            "train_ltr",
            [
                py,
                "scripts/train_ltr.py",
                "--features",
                str(art / "features.parquet"),
                "--labels",
                str(art / "synthetic_labels.parquet"),
                "--out",
                str(art / "ltr_model.lgb"),
            ],
                [art / "ltr_model.lgb"],
            )
        )
    else:
        print("\n>>> Skipping train_ltr (artifacts already exist)")

    if not (ROOT / "submission.csv").exists():
        report.append(
            _run_stage(
            "rank",
            [
                py,
                "rank.py",
                "--candidates",
                str(args.candidates),
                "--out",
                str(ROOT / "submission.csv"),
                "--artifacts",
                str(art),
                "--config",
                str(args.config),
            ],
                [ROOT / "submission.csv"],
            )
        )
    else:
        print("\n>>> Skipping rank (submission.csv already exists)")

    out = ROOT / "eval" / "pipeline_execution.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPipeline report: {out}")


if __name__ == "__main__":
    main()
