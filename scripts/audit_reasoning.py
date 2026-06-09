from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.reasoning.hallucination_guard import validate_reasoning
from src.understanding.parser import load_candidates


def audit_reasoning(submission_path: Path, candidates_path: Path) -> pd.DataFrame:
    submission = pd.read_csv(submission_path)
    if "reasoning" not in submission.columns:
        raise ValueError("Submission missing reasoning column")

    profiles = {profile.candidate_id: profile for profile in load_candidates(candidates_path)}
    rows: list[dict] = []

    for record in submission.to_dict(orient="records"):
        candidate_id = record["candidate_id"]
        reasoning = str(record.get("reasoning", ""))
        profile = profiles.get(candidate_id)
        if profile is None:
            rows.append(
                {
                    "rank": record.get("rank"),
                    "candidate_id": candidate_id,
                    "violations": "candidate not found",
                }
            )
            continue

        violations = validate_reasoning(reasoning, profile)
        if violations:
            rows.append(
                {
                    "rank": record.get("rank"),
                    "candidate_id": candidate_id,
                    "violations": "; ".join(violations),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit submission reasoning with hallucination guard.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()

    violations_df = audit_reasoning(args.submission, args.candidates)

    if violations_df.empty:
        print("No reasoning violations found.")
        raise SystemExit(0)

    print(f"Found {len(violations_df)} reasoning violation(s):")
    print(violations_df.to_string(index=False))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
