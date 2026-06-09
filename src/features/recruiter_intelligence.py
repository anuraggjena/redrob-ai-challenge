from __future__ import annotations

from src.understanding.models import CandidateProfile


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _eval_maturity(profile: CandidateProfile, features: dict[str, float]) -> float:
    components = [
        float(profile.flags.get("eval_experience", False)),
        features.get("eval_skill_flag", 0.0),
        min(features.get("ml_skill_depth", 0.0), 1.0),
    ]
    return sum(components) / len(components)


def _ownership(profile: CandidateProfile) -> float:
    ownership_keywords = ("owned", "lead", "architect", "designed", "built")
    hits = 0
    for role in profile.career_roles:
        text = f"{role.get('title', '')} {role.get('description', '')}".lower()
        hits += sum(1 for keyword in ownership_keywords if keyword in text)
    return _clamp01(hits / 5.0)


def _startup_readiness(profile: CandidateProfile, features: dict[str, float]) -> float:
    return _clamp01(
        0.5 * features.get("startup_experience_ratio", 0.0)
        + 0.3 * features.get("product_company_ratio", 0.0)
        + 0.2 * (1.0 - features.get("consulting_years_ratio", 0.0))
    )


def compute(profile: CandidateProfile) -> dict[str, float]:
    features = profile.features
    experience = profile.experience
    total = max(experience.total_years, 0.1)

    technical_depth = (
        0.4 * features.get("ranking_experience_ratio", 0.0)
        + 0.3 * features.get("retrieval_experience_ratio", 0.0)
        + 0.3 * features.get("python_experience_ratio", 0.0)
    )

    production_experience = _clamp01(experience.ai_years / 4.0)
    ranking_relevance = _clamp01(experience.ranking_years / 3.0)
    retrieval_experience = _clamp01(experience.retrieval_years / 3.0)
    evaluation_maturity = _eval_maturity(profile, features)
    engineering_ownership = _ownership(profile)
    product_mindset = experience.product_company_years / total
    startup_readiness = _startup_readiness(profile, features)

    candidate_availability = features.get("availability_score", 0.0)
    candidate_reliability = (
        0.5 * features.get("trustworthiness_score", 0.5)
        + 0.3 * features.get("engagement_score", 0.0)
        + 0.2 * (1.0 - features.get("job_hopping_score", 0.0) / max(features.get("role_count", 1.0), 1.0))
    )
    career_stability = _clamp01(
        features.get("avg_tenure_months", 0.0) / 36.0 * (1.0 - features.get("job_hopping_score", 0.0) / 10.0)
    )

    hireability = _clamp01(
        0.2 * technical_depth
        + 0.15 * production_experience
        + 0.15 * ranking_relevance
        + 0.1 * retrieval_experience
        + 0.1 * candidate_availability
        + 0.15 * candidate_reliability
        + 0.15 * features.get("engagement_score", 0.0)
    )

    return {
        "technical_depth_score": technical_depth,
        "production_experience_score": production_experience,
        "ranking_relevance_score": ranking_relevance,
        "retrieval_experience_score": retrieval_experience,
        "evaluation_maturity_score": evaluation_maturity,
        "engineering_ownership_score": engineering_ownership,
        "product_mindset_score": product_mindset,
        "startup_readiness_score": startup_readiness,
        "candidate_availability_score": candidate_availability,
        "candidate_reliability_score": candidate_reliability,
        "career_stability_score": career_stability,
        "hireability_score": hireability,
    }
