from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent / "artifacts" / "ltr_model.lgb"
)


class LTRRanker:
    def __init__(self, model_path: Path | None = None):
        resolved = model_path or _DEFAULT_MODEL_PATH
        self.model_path = Path(resolved)
        self._model: lgb.Booster | None = None
        self._feature_names: list[str] | None = None

        if self.model_path.exists():
            self._model = lgb.Booster(model_file=str(self.model_path))
            self._feature_names = self._model.feature_name()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict_scores(self, feature_rows: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise FileNotFoundError(f"LTR model not found at {self.model_path}")

        frame = feature_rows.copy()
        if self._feature_names:
            for name in self._feature_names:
                if name not in frame.columns:
                    frame[name] = 0.0
            frame = frame[self._feature_names]

        return np.asarray(self._model.predict(frame), dtype=float)

    def rank(
        self,
        candidate_ids: list[str],
        features_by_id: dict[str, dict[str, float]],
        top_n: int = 100,
    ) -> list[tuple[str, float]]:
        if not candidate_ids:
            return []

        rows = [
            {"candidate_id": candidate_id, **features_by_id[candidate_id]}
            for candidate_id in candidate_ids
            if candidate_id in features_by_id
        ]
        if not rows:
            return []

        frame = pd.DataFrame(rows)
        ids = frame.pop("candidate_id").tolist()
        scores = self.predict_scores(frame)
        ranked = sorted(zip(ids, scores.tolist(), strict=True), key=lambda item: (-item[1], item[0]))
        return [(candidate_id, float(score)) for candidate_id, score in ranked[:top_n]]
