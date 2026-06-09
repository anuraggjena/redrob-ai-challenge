from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from src.graph.evidence_graph import build_evidence_graph
from src.graph.graph_scorer import score_graph
from src.pipeline.orchestrator import score_from_rank
from src.reasoning.reason_generator import generate_reasoning
from src.utils.artifacts import atomic_write_text

HEADER = ["candidate_id", "rank", "score", "reasoning"]


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("raw_score", row.get("score", 0.0))),
            str(row["candidate_id"]),
        ),
    )


def _finalize_rows(
    rows: list[dict[str, Any]],
    *,
    profiles_by_id: dict | None = None,
    job_requirements: dict | None = None,
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for rank, row in enumerate(_sort_rows(rows), start=1):
        candidate_id = row["candidate_id"]
        score = score_from_rank(rank)
        reasoning = row.get("reasoning", "")

        if profiles_by_id is not None:
            profile = profiles_by_id[candidate_id]
            if profile.evidence_graph is None and job_requirements is not None:
                profile.evidence_graph = score_graph(
                    build_evidence_graph(profile, job_requirements)
                )
            reasoning = generate_reasoning(profile, rank=rank, score=score)

        finalized.append(
            {
                "candidate_id": candidate_id,
                "rank": rank,
                "score": score,
                "reasoning": reasoning,
            }
        )
    return finalized


def write_submission(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    profiles_by_id: dict | None = None,
    job_requirements: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finalized = _finalize_rows(
        rows,
        profiles_by_id=profiles_by_id,
        job_requirements=job_requirements,
    )

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEADER, lineterminator="\n")
    writer.writeheader()
    for row in finalized:
        writer.writerow(
            {
                "candidate_id": row["candidate_id"],
                "rank": row["rank"],
                "score": f"{float(row['score']):.4f}",
                "reasoning": row["reasoning"],
            }
        )
    atomic_write_text(path, buffer.getvalue())
