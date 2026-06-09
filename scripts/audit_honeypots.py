from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def audit_honeypots(
    submission_path: Path,
    features_path: Path,
    *,
    threshold: float = 0.6,
    top: int = 100,
) -> pd.DataFrame:
    submission = pd.read_csv(submission_path).sort_values("rank")
    features = pd.read_parquet(features_path)
    if "honeypot_probability" not in features.columns:
        raise ValueError("features parquet missing honeypot_probability column")

    top_rows = submission.head(top)
    merged = top_rows.merge(
        features[["candidate_id", "honeypot_probability"]],
        on="candidate_id",
        how="left",
    )
    flagged = merged[merged["honeypot_probability"].fillna(0.0) >= threshold].copy()
    return flagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit submission for honeypot candidates in top-N.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    flagged = audit_honeypots(
        args.submission,
        args.features,
        threshold=args.threshold,
        top=args.top,
    )

    if flagged.empty:
        print(f"No honeypots found in top {args.top} (threshold={args.threshold}).")
        raise SystemExit(0)

    print(f"Flagged {len(flagged)} honeypot candidate(s) in top {args.top}:")
    print(flagged.to_string(index=False))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
