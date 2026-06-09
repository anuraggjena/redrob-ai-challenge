from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_step(command: list[str], *, cwd: Path) -> None:
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def _embeddings_exist(artifacts_dir: Path) -> bool:
    return (artifacts_dir / "embeddings.npy").exists() and (
        artifacts_dir / "candidate_ids.json"
    ).exists()


def precompute_all(
    candidates_path: Path,
    artifacts_dir: Path,
    config_dir: Path,
    *,
    skip_tune: bool = False,
    tune_step: float = 0.05,
) -> None:
    python = sys.executable
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    features_path = artifacts_dir / "features.parquet"
    labels_path = artifacts_dir / "synthetic_labels.parquet"
    model_path = artifacts_dir / "ltr_model.lgb"
    weights_path = config_dir / "feature_weights.yaml"

    _run_step(
        [
            python,
            "scripts/build_features.py",
            "--candidates",
            str(candidates_path),
            "--out",
            str(features_path),
        ],
        cwd=ROOT,
    )

    if not _embeddings_exist(artifacts_dir):
        _run_step(
            [
                python,
                "scripts/build_embeddings.py",
                "--candidates",
                str(candidates_path),
                "--out",
                str(artifacts_dir),
            ],
            cwd=ROOT,
        )
    else:
        print("\n>>> Skipping build_embeddings (artifacts already exist)")

    _run_step(
        [
            python,
            "scripts/build_indexes.py",
            "--candidates",
            str(candidates_path),
            "--artifacts",
            str(artifacts_dir),
            "--config",
            str(config_dir),
        ],
        cwd=ROOT,
    )

    _run_step(
        [
            python,
            "scripts/build_synthetic_labels.py",
            "--features",
            str(features_path),
            "--out",
            str(labels_path),
            "--candidates",
            str(candidates_path),
        ],
        cwd=ROOT,
    )

    _run_step(
        [
            python,
            "scripts/train_ltr.py",
            "--features",
            str(features_path),
            "--labels",
            str(labels_path),
            "--out",
            str(model_path),
        ],
        cwd=ROOT,
    )

    if skip_tune:
        print("\n>>> Skipping tune_ensemble_weights")
        return

    _run_step(
        [
            python,
            "scripts/tune_ensemble_weights.py",
            "--features",
            str(features_path),
            "--labels",
            str(labels_path),
            "--out",
            str(weights_path),
            "--step",
            str(tune_step),
        ],
        cwd=ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full offline precompute pipeline.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--skip-tune", action="store_true", help="Skip ensemble weight tuning")
    parser.add_argument("--tune-step", type=float, default=0.05)
    args = parser.parse_args()

    precompute_all(
        args.candidates,
        args.artifacts,
        args.config,
        skip_tune=args.skip_tune,
        tune_step=args.tune_step,
    )
    print("\nPrecompute pipeline complete.")


if __name__ == "__main__":
    main()
