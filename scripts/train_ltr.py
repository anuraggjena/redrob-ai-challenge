from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import pandas as pd


def _lambdarank_params() -> dict:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "verbosity": -1,
        "force_col_wise": True,
    }


def _rank_xendcg_params() -> dict:
    return {
        "objective": "rank_xendcg",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 50],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "verbosity": -1,
        "force_col_wise": True,
    }


def train_ltr(
    features_path: Path,
    labels_path: Path,
    model_out: Path,
    num_boost_round: int = 200,
) -> lgb.Booster:
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)
    merged = features_df.merge(labels_df, on="candidate_id")

    feature_cols = [
        column
        for column in merged.columns
        if column not in {"candidate_id", "tier", "label"}
    ]
    x = merged[feature_cols]
    y = merged["label"].astype(float)
    group = [len(merged)]

    params = _lambdarank_params()
    try:
        train_data = lgb.Dataset(x, label=y, group=group)
        model = lgb.train(params, train_data, num_boost_round=num_boost_round)
    except (lgb.basic.LightGBMError, ValueError):
        params = _rank_xendcg_params()
        train_data = lgb.Dataset(x, label=y, group=group)
        model = lgb.train(params, train_data, num_boost_round=num_boost_round)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_out.with_suffix(model_out.suffix + ".tmp")
    model.save_model(str(tmp_path))
    tmp_path.replace(model_out)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM LTR model.")
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
        default=Path("artifacts/ltr_model.lgb"),
    )
    parser.add_argument("--rounds", type=int, default=200)
    args = parser.parse_args()
    model = train_ltr(args.features, args.labels, args.out, num_boost_round=args.rounds)
    print(f"Saved LTR model to {args.out} ({model.num_trees()} trees)")


if __name__ == "__main__":
    main()
