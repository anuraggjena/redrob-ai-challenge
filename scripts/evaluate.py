from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

DEFAULT_K = [10, 50]
HONEYPOT_THRESHOLD = 0.6
RELEVANCE_THRESHOLD = 3
COMPOSITE_WEIGHTS = {
    "ndcg@10": 0.50,
    "ndcg@50": 0.30,
    "map": 0.15,
    "p@10": 0.05,
}


def load_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "label" not in frame.columns and "tier" in frame.columns:
        frame = frame.rename(columns={"tier": "label"})
    return frame[["candidate_id", "label"]]


def load_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"candidate_id", "rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Submission missing columns: {sorted(missing)}")
    return frame.sort_values("rank").reset_index(drop=True)


def _prediction_scores(submission: pd.DataFrame) -> np.ndarray:
    ranks = submission["rank"].to_numpy(dtype=float)
    return -ranks


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(labels) == 0:
        return 0.0
    k = min(k, len(labels))
    return float(ndcg_score([labels], [scores], k=k))


def average_precision(labels: np.ndarray, scores: np.ndarray, *, threshold: int = RELEVANCE_THRESHOLD) -> float:
    if len(labels) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float))
    sorted_labels = np.asarray(labels, dtype=float)[order]
    relevant = sorted_labels >= threshold
    relevant_count = int(relevant.sum())
    if relevant_count == 0:
        return 0.0

    hits = 0
    precisions: list[float] = []
    for index, is_relevant in enumerate(relevant):
        if is_relevant:
            hits += 1
            precisions.append(hits / (index + 1))
    return float(sum(precisions) / relevant_count)


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int, *, threshold: int = RELEVANCE_THRESHOLD) -> float:
    if len(labels) == 0 or k <= 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float))
    sorted_labels = np.asarray(labels, dtype=float)[order]
    top_k = sorted_labels[: min(k, len(sorted_labels))]
    return float(np.mean(top_k >= threshold))


def honeypot_at_k(
    submission: pd.DataFrame,
    features: pd.DataFrame,
    k: int,
    *,
    threshold: float = HONEYPOT_THRESHOLD,
) -> int:
    if "honeypot_probability" not in features.columns:
        return 0
    top = submission.nsmallest(k, "rank")
    merged = top.merge(features[["candidate_id", "honeypot_probability"]], on="candidate_id", how="left")
    return int((merged["honeypot_probability"].fillna(0.0) >= threshold).sum())


def compute_composite(metrics: dict[str, float]) -> float:
    return sum(COMPOSITE_WEIGHTS[key] * metrics[key] for key in COMPOSITE_WEIGHTS)


def evaluate_ranking(
    labels_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    *,
    k_values: list[int] | None = None,
    features_df: pd.DataFrame | None = None,
) -> dict:
    k_values = k_values or DEFAULT_K
    merged = submission_df.merge(labels_df, on="candidate_id", how="inner")
    if merged.empty:
        raise ValueError("No overlapping candidate_id values between labels and submission")

    merged = merged.sort_values("rank").reset_index(drop=True)
    labels = merged["label"].to_numpy(dtype=float)
    scores = _prediction_scores(merged)

    report: dict[str, float | int | dict] = {}
    for k in k_values:
        report[f"ndcg@{k}"] = ndcg_at_k(labels, scores, k)

    report["map"] = average_precision(labels, scores)
    report["p@10"] = precision_at_k(labels, scores, 10)

    composite_inputs = {
        "ndcg@10": float(report.get("ndcg@10", 0.0)),
        "ndcg@50": float(report.get("ndcg@50", 0.0)),
        "map": float(report["map"]),
        "p@10": float(report["p@10"]),
    }
    report["composite"] = compute_composite(composite_inputs)
    report["num_candidates"] = int(len(merged))

    if features_df is not None and "honeypot_probability" in features_df.columns:
        report["honeypot@10"] = honeypot_at_k(submission_df, features_df, 10)
        report["honeypot@100"] = honeypot_at_k(submission_df, features_df, 100)

    return report


def parse_k_values(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranking submission against synthetic labels.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True, help="Submission CSV path")
    parser.add_argument("--features", type=Path, default=None, help="Features parquet for honeypot metrics")
    parser.add_argument("--k", type=str, default="10,50", help="Comma-separated K values for NDCG@K")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/reports/eval_report.json"),
    )
    args = parser.parse_args()

    labels_df = load_labels(args.labels)
    submission_df = load_submission(args.pred)
    features_df = pd.read_parquet(args.features) if args.features and args.features.exists() else None

    report = evaluate_ranking(
        labels_df,
        submission_df,
        k_values=parse_k_values(args.k),
        features_df=features_df,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
