from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.scoring.component_scores import COMPONENT_KEYS, compute_components

DEFAULT_WEIGHTS: dict[str, float] = {
    "technical_fit": 0.30,
    "semantic_fit": 0.20,
    "experience_fit": 0.15,
    "behavioral_fit": 0.10,
    "recruiter_signal_fit": 0.10,
    "availability_fit": 0.10,
    "trustworthiness": 0.05,
}

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "feature_weights.yaml"


def load_weights(path: Path | None = None) -> dict[str, float]:
    weights_path = path or _WEIGHTS_PATH
    with weights_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return {key: float(loaded[key]) for key in COMPONENT_KEYS}


def apply_gates(features: dict[str, float]) -> float:
    multiplier = 1.0

    if features.get("consulting_only_flag", 0.0) > 0.5 and features.get(
        "product_company_ratio", 0.0
    ) < 0.1:
        multiplier = min(multiplier, 0.2)

    domain_penalty = max(
        features.get("cv_only_penalty", 0.0),
        features.get("speech_only_penalty", 0.0),
        features.get("robotics_only_penalty", 0.0),
    )
    nlp_ir_signal = max(
        features.get("nlp_skill_depth", 0.0),
        features.get("retrieval_experience_ratio", 0.0),
        features.get("ranking_experience_ratio", 0.0),
    )
    if domain_penalty > 0.3 and nlp_ir_signal < 0.2:
        multiplier = min(multiplier, 0.3)

    response_rate = features.get("responsiveness_score")
    if response_rate is None:
        response_rate = features.get("behavioral_recruiter_response_rate", 1.0)
    days_inactive = features.get("behavioral_days_since_active", 0.0)
    if float(response_rate) < 0.1 and float(days_inactive) > 180:
        multiplier = min(multiplier, 0.5)

    return multiplier


def _penalty(features: dict[str, float]) -> float:
    honeypot_penalty = features.get("honeypot_penalty", 0.0)
    keyword_stuffer_penalty = features.get(
        "keyword_stuffer_penalty", features.get("keyword_stuffing_score", 0.0)
    )
    return float(honeypot_penalty) + float(keyword_stuffer_penalty) * 0.3


def score_candidate(
    features: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    if features.get("honeypot_probability", 0.0) > 0.7:
        return 0.0

    active_weights = weights or DEFAULT_WEIGHTS
    components = compute_components(features)
    base = sum(active_weights[key] * components[key] for key in active_weights)
    gated = base * apply_gates(features)
    return max(0.0, gated - _penalty(features))


def _features_for_id(features_source: Any, candidate_id: str) -> dict[str, float]:
    if hasattr(features_source, "loc"):
        row = features_source.loc[features_source["candidate_id"] == candidate_id]
        if row.empty:
            raise KeyError(candidate_id)
        record = row.iloc[0].to_dict()
        record.pop("candidate_id", None)
        return {key: float(value) for key, value in record.items()}

    if candidate_id not in features_source:
        raise KeyError(candidate_id)
    return features_source[candidate_id]


def score_batch(
    candidate_ids: list[str],
    features_source: Any,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    return {
        candidate_id: score_candidate(_features_for_id(features_source, candidate_id), weights)
        for candidate_id in candidate_ids
    }
