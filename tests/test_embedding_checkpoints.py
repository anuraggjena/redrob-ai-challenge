from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_embeddings import build_embeddings


@pytest.fixture
def sample_candidates_path() -> Path:
    return Path("India_runs_data_and_ai_challenge/sample_candidates.json")


def test_embedding_checkpoint_resume(tmp_path: Path, sample_candidates_path: Path):
    if not sample_candidates_path.exists():
        pytest.skip("sample candidates not available")

    out_dir = tmp_path / "artifacts"
    chunk_size = 25

    first = build_embeddings(
        sample_candidates_path,
        out_dir,
        chunk_size=chunk_size,
        batch_size=16,
        resume=False,
    )
    manifest_path = out_dir / "checkpoints" / "embeddings" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["completed_chunks"] == [0, 1]
    assert first.shape[0] == 50

    final_path = out_dir / "embeddings.npy"
    final_path.unlink()
    (out_dir / "candidate_ids.json").unlink()
    manifest["completed_chunks"] = [0]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    second = build_embeddings(
        sample_candidates_path,
        out_dir,
        chunk_size=chunk_size,
        batch_size=16,
        resume=True,
    )

    assert second.shape == first.shape
    assert np.allclose(second, first)
