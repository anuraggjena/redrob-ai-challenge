from __future__ import annotations

from src.understanding.models import CandidateProfile


def _product(a: float, b: float) -> float:
    return a * b


def compute(profile: CandidateProfile) -> dict[str, float]:
    features = profile.features

    title_skill_coherence = (
        1.0 - features.get("role_skill_mismatch_flag", 0.0)
    ) * features.get("taxonomy_skill_count", 0.0) / max(features.get("skill_count", 1.0), 1.0)

    behavioral_x_technical = _product(
        features.get("engagement_score", 0.0),
        features.get("skill_evidence_retrieval_ranking", 0.0) / 10.0,
    )
    experience_x_production = _product(
        features.get("relevant_experience_ratio", 0.0),
        features.get("production_ml_ratio", 0.0),
    )
    ranking_x_retrieval = _product(
        features.get("ranking_experience_ratio", 0.0),
        features.get("retrieval_experience_ratio", 0.0),
    )
    engagement_x_availability = _product(
        features.get("engagement_score", 0.0),
        features.get("availability_score", 0.0),
    )
    trust_x_technical = _product(
        features.get("trustworthiness_score", 0.0),
        features.get("technical_depth_score", 0.0),
    )
    seniority_x_experience = _product(
        features.get("title_seniority_score", 0.0),
        features.get("total_experience_years", 0.0) / 10.0,
    )
    location_x_relocate = _product(
        features.get("location_fit", 0.0),
        features.get("willing_to_relocate_flag", 0.0) or 0.5,
    )
    open_to_work_x_response_rate = _product(
        features.get("open_to_work_score", 0.0),
        features.get("responsiveness_score", 0.0),
    )

    composite_hireability_index = (
        0.25 * features.get("hireability_score", 0.0)
        + 0.2 * trust_x_technical
        + 0.15 * ranking_x_retrieval
        + 0.15 * engagement_x_availability
        + 0.15 * experience_x_production
        + 0.1 * title_skill_coherence
    )

    return {
        "title_skill_coherence": title_skill_coherence,
        "behavioral_x_technical": behavioral_x_technical,
        "experience_x_production": experience_x_production,
        "ranking_x_retrieval": ranking_x_retrieval,
        "engagement_x_availability": engagement_x_availability,
        "trust_x_technical": trust_x_technical,
        "seniority_x_experience": seniority_x_experience,
        "location_x_relocate": location_x_relocate,
        "open_to_work_x_response_rate": open_to_work_x_response_rate,
        "composite_hireability_index": composite_hireability_index,
        "production_x_ranking_interaction": _product(
            features.get("production_experience_score", 0.0),
            features.get("ranking_relevance_score", 0.0),
        ),
    }
