from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.scoring.synthetic_labels import assign_synthetic_tier
from src.understanding.parser import load_candidates
from src.utils.artifacts import atomic_write_parquet


def build_synthetic_labels(
    features_path: Path,
    out_path: Path,
    candidates_path: Path | None = None,
) -> pd.DataFrame:
    features_df = pd.read_parquet(features_path)
    titles: dict[str, str] = {}

    if candidates_path is not None:
        for profile in load_candidates(candidates_path):
            titles[profile.candidate_id] = profile.current_title

    rows: list[dict] = []
    for record in features_df.to_dict(orient="records"):
        candidate_id = record.pop("candidate_id")
        title = titles.get(candidate_id, "")
        tier, label = assign_synthetic_tier(record, title=title)
        rows.append({"candidate_id": candidate_id, "tier": tier, "label": label})

    labels_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(out_path, labels_df)
    return labels_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign synthetic relevance tiers.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/synthetic_labels.parquet"),
    )
    parser.add_argument("--candidates", type=Path, default=None)
    args = parser.parse_args()
    labels_df = build_synthetic_labels(args.features, args.out, args.candidates)
    print(f"Wrote {len(labels_df)} labels to {args.out}")


if __name__ == "__main__":
    main()
