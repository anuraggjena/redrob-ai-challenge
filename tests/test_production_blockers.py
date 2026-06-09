from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.orchestrator import RankingPipeline, score_from_rank
from src.submission.csv_writer import _finalize_rows, write_submission
from src.utils.artifacts import (
    atomic_write_json,
    atomic_write_parquet,
    cap_n_jobs,
    candidate_set_hash,
)


def test_cap_n_jobs_never_exceeds_maximum():
    assert cap_n_jobs(-1) <= 4
    assert cap_n_jobs(128) <= 4


def test_cache_rejects_count_only_match(tmp_path: Path):
    features_path = tmp_path / "features.parquet"
    meta_path = tmp_path / "features_meta.json"

    ids_a = ["CAND_0000001", "CAND_0000002"]
    ids_b = ["CAND_0000099", "CAND_0000100"]

    frame = pd.DataFrame(
        [
            {"candidate_id": ids_a[0], "feature_a": 1.0},
            {"candidate_id": ids_a[1], "feature_a": 2.0},
        ]
    )
    atomic_write_parquet(features_path, frame)
    atomic_write_json(
        meta_path,
        {"candidate_hash": candidate_set_hash(ids_a), "candidate_count": 2},
    )

    pipeline = RankingPipeline(tmp_path, Path("config"))
    with pytest.raises(ValueError, match="hash mismatch"):
        pipeline._load_features_from_artifacts(ids_b)


def test_cache_accepts_hash_match(tmp_path: Path):
    ids = ["CAND_0000001", "CAND_0000002"]
    frame = pd.DataFrame(
        [
            {"candidate_id": ids[0], "feature_a": 1.0},
            {"candidate_id": ids[1], "feature_a": 2.0},
        ]
    )
    atomic_write_parquet(tmp_path / "features.parquet", frame)
    atomic_write_json(
        tmp_path / "features_meta.json",
        {"candidate_hash": candidate_set_hash(ids), "candidate_count": 2},
    )

    pipeline = RankingPipeline(tmp_path, Path("config"))
    loaded = pipeline._load_features_from_artifacts(ids)
    assert set(loaded) == set(ids)


def test_honeypot_filter_backfills_to_top_n():
    ranked = [(f"CAND_{i:07d}", float(100 - i)) for i in range(120)]
    features = {
        f"CAND_{i:07d}": {"honeypot_probability": 0.95 if i < 30 else 0.0}
        for i in range(120)
    }
    selected = RankingPipeline._apply_honeypot_filter(ranked, features, top_n=100)
    assert len(selected) == 100
    assert all(features[cid]["honeypot_probability"] <= 0.95 for cid, _ in selected[:70])


def test_reasoning_regenerated_after_final_rank_assignment():
    rows = [
        {"candidate_id": "CAND_0000002", "raw_score": 0.9},
        {"candidate_id": "CAND_0000001", "raw_score": 0.9},
    ]
    finalized = _finalize_rows(
        rows,
        profiles_by_id=None,
        job_requirements=None,
    )
    assert finalized[0]["candidate_id"] == "CAND_0000001"
    assert finalized[0]["rank"] == 1
    assert finalized[0]["score"] == score_from_rank(1)


def test_write_submission_atomic(tmp_path: Path):
    rows = [{"candidate_id": "CAND_0000001", "raw_score": 0.5}]
    out = tmp_path / "submission.csv"
    write_submission(rows, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "candidate_id,rank,score,reasoning" in content


def test_build_features_resume_smoke(tmp_path: Path):
    from scripts.build_features import build_features

    sample = Path("India_runs_data_and_ai_challenge/sample_candidates.json")
    if not sample.exists():
        pytest.skip("sample candidates missing")

    out = tmp_path / "features.parquet"
    frame = build_features(sample, out, n_jobs=2, chunk_size=10, resume=True)
    assert len(frame) >= 1
    manifest = json.loads((tmp_path / "checkpoints" / "features" / "manifest.json").read_text())
    assert manifest["candidate_count"] == len(frame)


def test_runtime_requires_precomputed_features(tmp_path: Path):
    pipeline = RankingPipeline(tmp_path, Path("config"))
    sample = Path("India_runs_data_and_ai_challenge/sample_candidates.json")
    if not sample.exists():
        pytest.skip("sample candidates missing")
    with pytest.raises(FileNotFoundError, match="Missing precomputed features"):
        pipeline.run(sample, top_n=5)
