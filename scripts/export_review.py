from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.audit_traps import _load_non_technical_titles, _trap_flags
from src.understanding.parser import load_candidates

HONEYPOT_THRESHOLD = 0.6


def export_review(
    submission_path: Path,
    features_path: Path,
    labels_path: Path,
    out_path: Path,
    *,
    candidates_path: Path | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    submission = pd.read_csv(submission_path).sort_values("rank").head(top_n)
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    label_col = "tier" if "tier" in labels.columns else "label"
    labels = labels[["candidate_id", label_col]].rename(columns={label_col: "synthetic_tier"})

    titles: dict[str, str] = {}
    if candidates_path is not None:
        for profile in load_candidates(candidates_path):
            titles[profile.candidate_id] = profile.current_title

    negative_titles = _load_non_technical_titles()
    merged = submission.merge(features, on="candidate_id", how="left", suffixes=("", "_feat"))
    merged = merged.merge(labels, on="candidate_id", how="left")

    export_rows: list[dict] = []
    for record in merged.to_dict(orient="records"):
        candidate_id = record["candidate_id"]
        title = titles.get(candidate_id, "")
        flags: list[str] = []
        honeypot_prob = float(record.get("honeypot_probability", 0.0))
        if honeypot_prob >= HONEYPOT_THRESHOLD:
            flags.append("honeypot")
        flags.extend(_trap_flags(pd.Series(record), title, negative_titles))

        export_rows.append(
            {
                "rank": record["rank"],
                "candidate_id": candidate_id,
                "title": title,
                "synthetic_tier": record.get("synthetic_tier"),
                "honeypot_prob": round(honeypot_prob, 4),
                "score": record.get("score"),
                "reasoning": record.get("reasoning", ""),
                "flags": ",".join(sorted(set(flags))) if flags else "",
            }
        )

    frame = pd.DataFrame(export_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top-ranked candidates for manual review.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("eval/review_top20.csv"))
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    frame = export_review(
        args.submission,
        args.features,
        args.labels,
        args.out,
        candidates_path=args.candidates,
        top_n=args.top,
    )
    print(f"Wrote {len(frame)} rows to {args.out}")


if __name__ == "__main__":
    main()
