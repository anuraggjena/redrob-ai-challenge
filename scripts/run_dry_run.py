#!/usr/bin/env python3
"""End-to-end precompute + rank timing report."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.artifacts import memory_mb


def _run(stage: str, cmd: list[str]) -> dict:
    print(f"\n=== {stage} ===", flush=True)
    start = time.perf_counter()
    mem_start = memory_mb()
    subprocess.run(cmd, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - start
    mem_peak = memory_mb()
    result = {
        "stage": stage,
        "elapsed_sec": round(elapsed, 2),
        "mem_start_mb": round(mem_start, 1),
        "mem_end_mb": round(mem_peak, 1),
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "India_runs_data_and_ai_challenge" / "candidates.jsonl",
    )
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--config", type=Path, default=ROOT / "config")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--fresh", action="store_true", help="Ignore feature checkpoints")
    args = parser.parse_args()

    py = sys.executable
    report: list[dict] = []

    feature_cmd = [
        py,
        "scripts/build_features.py",
        "--candidates",
        str(args.candidates),
        "--out",
        str(args.artifacts / "features.parquet"),
        "--n-jobs",
        str(args.n_jobs),
    ]
    if args.fresh:
        feature_cmd.append("--no-resume")

    report.append(_run("build_features", feature_cmd))

    if not args.skip_embeddings:
        report.append(
            _run(
                "build_embeddings",
                [
                    py,
                    "scripts/build_embeddings.py",
                    "--candidates",
                    str(args.candidates),
                    "--out",
                    str(args.artifacts),
                ],
            )
        )
        report.append(
            _run(
                "build_indexes",
                [
                    py,
                    "scripts/build_indexes.py",
                    "--candidates",
                    str(args.candidates),
                    "--artifacts",
                    str(args.artifacts),
                    "--config",
                    str(args.config),
                ],
            )
        )

    report.append(
        _run(
            "build_synthetic_labels",
            [
                py,
                "scripts/build_synthetic_labels.py",
                "--features",
                str(args.artifacts / "features.parquet"),
                "--out",
                str(args.artifacts / "synthetic_labels.parquet"),
                "--candidates",
                str(args.candidates),
            ],
        )
    )

    report.append(
        _run(
            "train_ltr",
            [
                py,
                "scripts/train_ltr.py",
                "--features",
                str(args.artifacts / "features.parquet"),
                "--labels",
                str(args.artifacts / "synthetic_labels.parquet"),
                "--out",
                str(args.artifacts / "ltr_model.lgb"),
            ],
        )
    )

    out_csv = ROOT / "submission.csv"
    report.append(
        _run(
            "rank",
            [
                py,
                "rank.py",
                "--candidates",
                str(args.candidates),
                "--out",
                str(out_csv),
                "--artifacts",
                str(args.artifacts),
                "--config",
                str(args.config),
            ],
        )
    )

    report_path = ROOT / "eval" / "dry_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDry run report written to {report_path}")


if __name__ == "__main__":
    main()
