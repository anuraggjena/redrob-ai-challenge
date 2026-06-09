from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CANDIDATES = ROOT / "India_runs_data_and_ai_challenge" / "sample_candidates.json"
VALIDATOR = ROOT / "India_runs_data_and_ai_challenge" / "validate_submission.py"


def test_write_submission_tie_break_by_candidate_id():
    from src.submission.csv_writer import write_submission

    rows = [
        {"candidate_id": "CAND_0000002", "rank": 1, "score": 0.9, "reasoning": "a"},
        {"candidate_id": "CAND_0000001", "rank": 2, "score": 0.9, "reasoning": "b"},
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sub.csv"
        write_submission(rows, out)
        with out.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            written = list(reader)
    assert written[0]["candidate_id"] == "CAND_0000001"
    assert written[1]["candidate_id"] == "CAND_0000002"
    assert written[0]["rank"] == "1"
    assert written[1]["rank"] == "2"


def test_write_submission_header_and_utf8():
    from src.submission.csv_writer import write_submission

    rows = [
        {
            "candidate_id": "CAND_0000001",
            "rank": 1,
            "score": 1.0,
            "reasoning": "Strong fit — retrieval & ranking.",
        }
    ]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sub.csv"
        write_submission(rows, out)
        raw = out.read_text(encoding="utf-8")
        assert raw.startswith("candidate_id,rank,score,reasoning")
        assert "—" in raw


def test_rank_sample_produces_valid_csv_top_n_10():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sub.csv"
        subprocess.run(
            [
                "python",
                "rank.py",
                "--candidates",
                str(SAMPLE_CANDIDATES),
                "--out",
                str(out),
                "--top-n",
                "10",
            ],
            cwd=ROOT,
            check=True,
        )
        assert out.exists()
        with out.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

    assert len(rows) == 10
    assert list(rows[0].keys()) == ["candidate_id", "rank", "score", "reasoning"]
    ranks = [int(row["rank"]) for row in rows]
    assert ranks == list(range(1, 11))

    scores = [float(row["score"]) for row in rows]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]

    for row in rows:
        assert row["candidate_id"].startswith("CAND_")
        assert row["reasoning"].strip()


def test_rank_scores_non_increasing_formula():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sub.csv"
        subprocess.run(
            [
                "python",
                "rank.py",
                "--candidates",
                str(SAMPLE_CANDIDATES),
                "--out",
                str(out),
                "--top-n",
                "10",
            ],
            cwd=ROOT,
            check=True,
        )
        with out.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

    for row in rows:
        rank = int(row["rank"])
        score = float(row["score"])
        expected = round(1.0 - (rank - 1) * 0.008, 4)
        assert score == pytest.approx(expected, abs=1e-4)


@pytest.mark.skipif(
    os.environ.get("RUN_FULL_VALIDATION") != "1",
    reason="optional full validation; set RUN_FULL_VALIDATION=1",
)
def test_validate_submission_when_100_rows():
    """Only run bundled validator when output has exactly 100 rows."""
    full_candidates = ROOT / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
    if not full_candidates.exists():
        pytest.skip("full candidates.jsonl not available")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "team_test.csv"
        subprocess.run(
            [
                "python",
                "rank.py",
                "--candidates",
                str(full_candidates),
                "--out",
                str(out),
                "--top-n",
                "100",
            ],
            cwd=ROOT,
            check=True,
            timeout=600,
        )
        result = subprocess.run(
            ["python", str(VALIDATOR), str(out)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, result.stdout + result.stderr
