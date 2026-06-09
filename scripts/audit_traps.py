from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.understanding.parser import load_candidates

ONTOLOGY_PATH = ROOT / "src" / "ontology" / "title_classifier.json"
KEYWORD_STUFFER_THRESHOLD = 0.4
TOP_K = 10


def _load_non_technical_titles() -> list[str]:
    data = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    return [title.lower() for title in data.get("non_technical", [])]


def _is_non_technical_title(title: str, negative_titles: list[str]) -> bool:
    title_lower = title.lower().strip()
    return any(title_lower == entry or entry in title_lower for entry in negative_titles)


def _trap_flags(row: pd.Series, title: str, negative_titles: list[str]) -> list[str]:
    flags: list[str] = []
    if float(row.get("role_skill_mismatch_flag", 0.0)) >= 0.5:
        flags.append("role_skill_mismatch")
    if title and _is_non_technical_title(title, negative_titles):
        flags.append("non_technical_title")
    keyword_score = float(row.get("keyword_stuffing_score", row.get("keyword_stuffer_penalty", 0.0)))
    if keyword_score >= KEYWORD_STUFFER_THRESHOLD:
        flags.append("keyword_stuffer")
    return flags


def audit_traps(
    submission_path: Path,
    features_path: Path,
    *,
    candidates_path: Path | None = None,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    submission = pd.read_csv(submission_path).sort_values("rank").head(top_k)
    features = pd.read_parquet(features_path)

    titles: dict[str, str] = {}
    if candidates_path is not None:
        for profile in load_candidates(candidates_path):
            titles[profile.candidate_id] = profile.current_title

    negative_titles = _load_non_technical_titles()
    merged = submission.merge(features, on="candidate_id", how="left", suffixes=("", "_feat"))

    rows: list[dict] = []
    for record in merged.to_dict(orient="records"):
        candidate_id = record["candidate_id"]
        title = titles.get(candidate_id, "")
        flags = _trap_flags(pd.Series(record), title, negative_titles)
        if flags:
            rows.append(
                {
                    "rank": record["rank"],
                    "candidate_id": candidate_id,
                    "title": title,
                    "flags": ",".join(flags),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit top-10 submission for trap profiles.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=None)
    args = parser.parse_args()

    flagged = audit_traps(args.submission, args.features, candidates_path=args.candidates)

    if flagged.empty:
        print(f"No traps found in top {TOP_K}.")
        raise SystemExit(0)

    print(f"Flagged {len(flagged)} trap candidate(s) in top {TOP_K}:")
    print(flagged.to_string(index=False))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
