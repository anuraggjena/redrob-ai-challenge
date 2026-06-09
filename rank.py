#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.config_loader import load_job_requirements
from src.pipeline.orchestrator import RankingPipeline
from src.submission.csv_writer import write_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--config", default="config")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    config_dir = Path(args.config)
    pipeline = RankingPipeline(Path(args.artifacts), config_dir)
    rows, profiles_by_id = pipeline.run(Path(args.candidates), top_n=args.top_n)
    job_requirements = load_job_requirements(config_dir / "job_requirements.yaml")
    write_submission(
        rows,
        Path(args.out),
        profiles_by_id=profiles_by_id,
        job_requirements=job_requirements,
    )
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
