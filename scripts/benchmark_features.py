#!/usr/bin/env python3
"""Benchmark per-candidate feature generation throughput."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.understanding.enrich import enrich_profile
from src.understanding.parser import parse_candidate
from scripts.build_features import _process_profile


def _expand_sample(n: int) -> list:
    raw = json.loads(
        (ROOT / "India_runs_data_and_ai_challenge" / "sample_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    profiles = []
    base_len = len(raw)
    for index in range(n):
        item = copy.deepcopy(raw[index % base_len])
        item["candidate_id"] = f"BENCH_{index:06d}"
        profiles.append(parse_candidate(item))
    return profiles


def _feature_fingerprint(profiles: list, count: int = 5) -> str:
    digest = hashlib.sha256()
    for profile in profiles[:count]:
        row = _process_profile(profile)
        digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _bench(n: int, *, warmup: int = 3) -> dict:
    profiles = _expand_sample(n)
    for profile in profiles[:warmup]:
        _process_profile(profile)

    profiles = _expand_sample(n)
    start = time.perf_counter()
    for profile in profiles:
        _process_profile(profile)
    elapsed = time.perf_counter() - start
    return {
        "n": n,
        "elapsed_sec": round(elapsed, 3),
        "ms_per_candidate": round(1000 * elapsed / n, 3),
        "candidates_per_sec": round(n / elapsed, 3),
    }


def main() -> None:
    fingerprint = _feature_fingerprint(_expand_sample(5))
    results = {
        "feature_fingerprint_first_5": fingerprint,
        "benchmarks": [_bench(200), _bench(1000)],
    }
    for row in results["benchmarks"]:
        hours_100k = row["elapsed_sec"] / row["n"] * 100_000 / 3600
        row["estimated_100k_hours_sequential"] = round(hours_100k, 2)
        row["estimated_100k_hours_n_jobs_2"] = round(hours_100k / 2, 2)
        row["estimated_100k_hours_n_jobs_4"] = round(hours_100k / 4, 2)

    out = ROOT / "eval" / "benchmark_features.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
