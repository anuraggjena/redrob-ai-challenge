from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import ndcg_score

from src.scoring.component_scores import COMPONENT_KEYS
from src.scoring.ensemble_scorer import DEFAULT_WEIGHTS, score_candidate


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    if len(labels) == 0:
        return 0.0
    k = min(k, len(labels))
    return float(ndcg_score([labels], [scores], k=k))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: value / total for key, value in weights.items()}


def _perturbations(step: float = 0.05) -> list[float]:
    return [round(-step * 2, 3), -step, 0.0, step, round(step * 2, 3)]


def _score_ndcg(merged: pd.DataFrame, labels: np.ndarray, weights: dict[str, float]) -> float:
    scores = np.array(
        [
            score_candidate(
                {
                    key: float(value)
                    for key, value in row.items()
                    if key not in {"candidate_id", "tier", "label"}
                },
                weights,
            )
            for row in merged.to_dict(orient="records")
        ],
        dtype=float,
    )
    return ndcg_at_k(labels, scores, k=10)


def grid_search_weights(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    base_weights: dict[str, float] | None = None,
    step: float = 0.05,
) -> tuple[dict[str, float], float]:
    merged = features_df.merge(labels_df, on="candidate_id")
    labels = merged["label"].to_numpy(dtype=float)

    best_weights = _normalize_weights(base_weights or DEFAULT_WEIGHTS)
    best_ndcg = _score_ndcg(merged, labels, best_weights)

    deltas = _perturbations(step)
    improved = True
    while improved:
        improved = False
        for key in COMPONENT_KEYS:
            for delta in deltas:
                if delta == 0.0:
                    continue
                candidate = dict(best_weights)
                candidate[key] = max(0.01, candidate[key] + delta)
                candidate = _normalize_weights(candidate)
                metric = _score_ndcg(merged, labels, candidate)
                if metric > best_ndcg:
                    best_ndcg = metric
                    best_weights = candidate
                    improved = True

    return best_weights, best_ndcg


def tune_ensemble_weights(
    features_path: Path,
    labels_path: Path,
    weights_out: Path,
    step: float = 0.05,
) -> tuple[dict[str, float], float]:
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)
    best_weights, best_ndcg = grid_search_weights(features_df, labels_df, step=step)

    weights_out.parent.mkdir(parents=True, exist_ok=True)
    with weights_out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_weights, handle, sort_keys=False)

    return best_weights, best_ndcg


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune ensemble weights via grid search.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/features.parquet"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("artifacts/synthetic_labels.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("config/feature_weights.yaml"),
    )
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    best_weights, best_ndcg = tune_ensemble_weights(
        args.features,
        args.labels,
        args.out,
        step=args.step,
    )
    print(f"Best NDCG@10: {best_ndcg:.4f}")
    for key, value in best_weights.items():
        print(f"  {key}: {value:.4f}")
    print(f"Wrote weights to {args.out}")


if __name__ == "__main__":
    main()
