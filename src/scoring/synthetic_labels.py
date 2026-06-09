from __future__ import annotations

import json
from pathlib import Path

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"

HIGH_HONEYPOT_THRESHOLD = 0.6
TIMELINE_HONEYPOT_THRESHOLD = 0.4
INACTIVE_DAYS_THRESHOLD = 180.0
LOW_RESPONSE_THRESHOLD = 0.1
RECRUITER_SAVE_THRESHOLD = 0.3


def _load_non_technical_titles() -> list[str]:
    data = json.loads((ONTOLOGY_DIR / "title_classifier.json").read_text(encoding="utf-8"))
    return [title.lower() for title in data.get("non_technical", [])]


def _is_non_technical_title(title: str) -> bool:
    title_lower = title.lower().strip()
    return any(title_lower == entry or entry in title_lower for entry in _load_non_technical_titles())


def _domain_only_penalty(features: dict[str, float]) -> float:
    return max(
        features.get("cv_only_penalty", 0.0),
        features.get("speech_only_penalty", 0.0),
        features.get("robotics_only_penalty", 0.0),
    )


def _has_nlp_ir_signal(features: dict[str, float]) -> bool:
    return max(
        features.get("nlp_skill_depth", 0.0),
        features.get("retrieval_experience_ratio", 0.0),
        features.get("ranking_experience_ratio", 0.0),
        features.get("retrieval_experience_score", 0.0),
        features.get("ranking_relevance_score", 0.0),
    ) >= 0.2


def _is_data_engineer_pattern(features: dict[str, float]) -> bool:
    relevant = features.get("relevant_experience_years", 0.0)
    ranking = features.get("ranking_experience_ratio", 0.0)
    retrieval = features.get("retrieval_experience_ratio", 0.0)
    production = features.get("production_ml_ratio", 0.0)
    return (
        3.0 <= relevant <= 5.0
        and production < 0.35
        and ranking < 0.25
        and retrieval < 0.25
        and features.get("technical_depth_score", 0.0) >= 0.2
    )


def _base_tier(features: dict[str, float], title: str) -> int:
    honeypot = float(features.get("honeypot_probability", 0.0))
    impossible = float(features.get("impossible_timeline_flag", 0.0)) >= 0.5
    inconsistent = float(features.get("timeline_inconsistent_flag", 0.0)) >= 0.5

    if honeypot >= HIGH_HONEYPOT_THRESHOLD or impossible:
        return 0
    if inconsistent and honeypot >= TIMELINE_HONEYPOT_THRESHOLD:
        return 0

    non_technical = _is_non_technical_title(title) or float(
        features.get("non_technical_title_flag", 0.0)
    ) >= 0.5
    consulting_only = float(features.get("consulting_only_flag", 0.0)) > 0.5
    no_product = features.get("product_company_ratio", 0.0) < 0.1
    langchain_penalty = features.get("langchain_skill_penalty", 0.0)
    low_production = features.get("production_ml_ratio", features.get("production_experience_score", 0.0)) < 0.2
    domain_only = _domain_only_penalty(features) > 0.3 and not _has_nlp_ir_signal(features)

    if (
        non_technical
        or (consulting_only and no_product)
        or (langchain_penalty > 0.5 and low_production)
        or domain_only
        or float(features.get("role_skill_mismatch_flag", 0.0)) >= 0.5
    ):
        return 1

    relevant_years = float(features.get("relevant_experience_years", 0.0))
    production_ml = float(features.get("production_ml_ratio", 0.0))
    partial_ml = 0.1 <= production_ml <= 0.4 or 1.0 <= features.get("ai_experience_years", 0.0) <= 3.0

    if (3.0 <= relevant_years <= 5.0 and partial_ml) or _is_data_engineer_pattern(features):
        return 2

    ranking_depth = max(
        features.get("ranking_relevance_score", 0.0),
        features.get("ranking_experience_ratio", 0.0),
    )
    retrieval_depth = max(
        features.get("retrieval_experience_score", 0.0),
        features.get("retrieval_experience_ratio", 0.0),
    )
    eval_signal = max(
        features.get("evaluation_maturity_score", 0.0),
        features.get("eval_skill_flag", 0.0),
    )
    product_ratio = float(features.get("product_company_ratio", 0.0))
    production_score = float(
        features.get("production_experience_score", features.get("production_ml_ratio", 0.0))
    )
    days_inactive = float(features.get("behavioral_days_since_active", 0.0))
    engagement = float(features.get("engagement_score", 0.5))
    location_fit = float(features.get("location_fit", 0.0))
    open_to_work = float(features.get("open_to_work_score", features.get("open_to_work", 0.0)))

    ideal_years = 6.0 <= relevant_years <= 8.0
    strong_evidence = (
        ranking_depth >= 0.35
        and retrieval_depth >= 0.35
        and production_score >= 0.4
        and eval_signal >= 0.4
        and product_ratio >= 0.3
        and location_fit >= 0.5
        and open_to_work >= 0.9
    )
    if ideal_years and strong_evidence:
        return 5

    active = days_inactive < 90.0 or engagement >= 0.5
    strong_tier = (
        ranking_depth >= 0.3
        and retrieval_depth >= 0.3
        and eval_signal >= 0.35
        and 5.0 <= relevant_years <= 9.0
        and active
    )
    if strong_tier:
        return 4

    good_fit = (
        relevant_years >= 5.0
        and production_score >= 0.25
        and product_ratio >= 0.25
        and ranking_depth < 0.5
    )
    if good_fit:
        return 3

    if relevant_years >= 5.0:
        return 3
    if relevant_years >= 3.0:
        return 2
    return 1


def _apply_behavioral_modifier(tier: int, features: dict[str, float]) -> int:
    adjusted = float(tier)
    days_inactive = float(features.get("behavioral_days_since_active", 0.0))
    response = float(
        features.get("responsiveness_score", features.get("behavioral_recruiter_response_rate", 1.0))
    )
    open_to_work = float(features.get("open_to_work_score", features.get("open_to_work", 0.0)))
    saved = float(features.get("saved_by_recruiters_norm", 0.0))

    if days_inactive > INACTIVE_DAYS_THRESHOLD and response < LOW_RESPONSE_THRESHOLD:
        adjusted -= 1.0
    if open_to_work >= 0.9 and saved >= RECRUITER_SAVE_THRESHOLD:
        adjusted += 0.5

    return int(max(0, min(5, round(adjusted))))


def assign_synthetic_tier(
    features: dict[str, float],
    *,
    title: str = "",
) -> tuple[int, int]:
    tier = _base_tier(features, title)
    tier = _apply_behavioral_modifier(tier, features)
    return tier, tier
