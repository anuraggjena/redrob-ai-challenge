#!/usr/bin/env python3
"""Profile build_features pipeline components — measurement only."""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import orjson

from src.features.registry import COMPUTERS, compute_all_features
from src.graph.evidence_graph import build_evidence_graph
from src.graph.graph_scorer import score_graph
from src.honeypot.run import run_honeypot_detection
from src.pipeline.config_loader import load_job_requirements
from src.understanding.enrich import enrich_profile
from src.understanding.parser import parse_candidate
from src.utils.artifacts import cap_n_jobs

SAMPLE_N = 200
CANDIDATES_PATH = ROOT / "India_runs_data_and_ai_challenge" / "candidates.jsonl"


def load_raw_sample(n: int) -> list[dict]:
    rows: list[dict] = []
    with CANDIDATES_PATH.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(orjson.loads(line))
            if len(rows) >= n:
                break
    return rows


def time_stage(name: str, fn, *args, **kwargs) -> tuple[float, object]:
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, result, peak / (1024 * 1024)


def main() -> None:
    print(f"=== Profiling sample N={SAMPLE_N} from {CANDIDATES_PATH.name} ===\n")

    # 1. JSON parse only
    raw_rows: list[dict] = []
    t0 = time.perf_counter()
    with CANDIDATES_PATH.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw_rows.append(orjson.loads(line))
            if len(raw_rows) >= SAMPLE_N:
                break
    parse_json_sec = time.perf_counter() - t0
    print(f"orjson.loads x{SAMPLE_N}: {parse_json_sec*1000/SAMPLE_N:.2f} ms/cand  total={parse_json_sec:.3f}s")

    # 2. CandidateProfile creation
    profiles = []
    t0 = time.perf_counter()
    for raw in raw_rows:
        profiles.append(parse_candidate(raw))
    parse_profile_sec = time.perf_counter() - t0
    print(f"parse_candidate x{SAMPLE_N}: {parse_profile_sec*1000/SAMPLE_N:.2f} ms/cand  total={parse_profile_sec:.3f}s")

    # Per-stage enrich timing (single pass, cumulative)
    stage_times: dict[str, float] = {}
    stage_mem: dict[str, float] = {}

    from src.understanding.experience_extractor import enrich_experience
    from src.understanding.career_extractor import enrich_career
    from src.understanding.timeline_validator import validate_timeline
    from src.understanding.technical_extractor import enrich_technical
    from src.understanding.behavioral_extractor import enrich_behavioral
    from src.understanding.availability_extractor import enrich_availability

    enrich_stages = [
        ("enrich_experience", enrich_experience),
        ("enrich_career", enrich_career),
        ("validate_timeline", validate_timeline),
        ("enrich_technical", enrich_technical),
        ("enrich_behavioral", enrich_behavioral),
        ("enrich_availability", enrich_availability),
    ]

    for name, fn in enrich_stages:
        t0 = time.perf_counter()
        tracemalloc.start()
        for p in profiles:
            fn(p)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stage_times[name] = elapsed
        stage_mem[name] = peak / (1024 * 1024)

    # Feature groups (fresh profiles copy via re-parse for isolation)
    fresh = [parse_candidate(r) for r in raw_rows]
    for p in fresh:
        enrich_profile(p)

    job_req = load_job_requirements(ROOT / "config" / "job_requirements.yaml")

    feature_group_times: dict[str, float] = {}
    for compute_fn in COMPUTERS:
        name = compute_fn.__module__.split(".")[-1]
        t0 = time.perf_counter()
        for p in fresh:
            if compute_fn.__name__ == "compute" and "trust_features" in compute_fn.__module__:
                run_honeypot_detection(p)
            if compute_fn.__name__ == "compute" and "graph_features" in compute_fn.__module__:
                p.evidence_graph = score_graph(build_evidence_graph(p, job_req))
            compute_fn(p)
        feature_group_times[name] = time.perf_counter() - t0

    # Isolated heavy components
    iso_profiles = [parse_candidate(r) for r in raw_rows]
    for p in iso_profiles:
        enrich_profile(p)

    t0 = time.perf_counter()
    for p in iso_profiles:
        run_honeypot_detection(p)
    honeypot_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    for p in iso_profiles:
        p.evidence_graph = score_graph(build_evidence_graph(p, job_req))
    graph_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    for p in iso_profiles:
        enrich_profile(p)
        compute_all_features(p)
    full_process_sec = time.perf_counter() - t0

    # Sequential vs parallel overhead estimate (small batch)
    from joblib import Parallel, delayed
    from scripts.build_features import _process_profile

    reprofiles = [parse_candidate(r) for r in raw_rows[:50]]
    t0 = time.perf_counter()
    for p in reprofiles:
        _process_profile(p)
    seq50 = time.perf_counter() - t0

    t0 = time.perf_counter()
    Parallel(n_jobs=4, backend="loky")(delayed(_process_profile)(p) for p in reprofiles)
    par50 = time.perf_counter() - t0

    # cProfile full pipeline on 50
    prof = cProfile.Profile()
    reprofiles2 = [parse_candidate(r) for r in raw_rows[:50]]
    prof.enable()
    for p in reprofiles2:
        _process_profile(p)
    prof.disable()

    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    report = {
        "sample_n": SAMPLE_N,
        "per_candidate_ms": {
            "orjson_parse": parse_json_sec * 1000 / SAMPLE_N,
            "parse_candidate": parse_profile_sec * 1000 / SAMPLE_N,
            "enrich_total": sum(stage_times.values()) * 1000 / SAMPLE_N,
            "honeypot": honeypot_sec * 1000 / SAMPLE_N,
            "evidence_graph": graph_sec * 1000 / SAMPLE_N,
            "full_process_profile": full_process_sec * 1000 / SAMPLE_N,
        },
        "enrich_stages_ms_per_cand": {k: v * 1000 / SAMPLE_N for k, v in stage_times.items()},
        "feature_groups_ms_per_cand": {k: v * 1000 / SAMPLE_N for k, v in feature_group_times.items()},
        "parallel_overhead_50": {"sequential_sec": seq50, "parallel_4_workers_sec": par50},
        "estimate_100k_hours": full_process_sec / SAMPLE_N * 100_000 / 3600,
        "estimate_100k_hours_parallel_4": full_process_sec / SAMPLE_N * 100_000 / 3600 / 4,
    }

    out = ROOT / "eval" / "profile_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Enrich stages (ms/candidate) ---")
    for k, v in sorted(stage_times.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v*1000/SAMPLE_N:.2f} ms  peak_mem={stage_mem[k]:.1f}MB")

    print("\n--- Feature groups (ms/candidate) ---")
    for k, v in sorted(feature_group_times.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v*1000/SAMPLE_N:.2f} ms")

    print("\n--- Isolated components (ms/candidate) ---")
    print(f"  honeypot_detection: {honeypot_sec*1000/SAMPLE_N:.2f} ms")
    print(f"  evidence_graph:     {graph_sec*1000/SAMPLE_N:.2f} ms")
    print(f"  full _process_profile: {full_process_sec*1000/SAMPLE_N:.2f} ms")

    print("\n--- Parallel overhead (50 candidates) ---")
    print(f"  sequential: {seq50:.2f}s  parallel(4): {par50:.2f}s  speedup: {seq50/par50:.2f}x")

    print("\n--- 100K runtime estimates ---")
    print(f"  sequential: {report['estimate_100k_hours']:.2f} hours")
    print(f"  ideal 4-worker: {report['estimate_100k_hours_parallel_4']:.2f} hours")

    print("\n--- Top 30 cProfile (50 candidates, cumulative) ---")
    print(stream.getvalue())

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
