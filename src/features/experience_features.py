from __future__ import annotations

from src.understanding.models import CandidateProfile


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 0.1)


def compute(profile: CandidateProfile) -> dict[str, float]:
    experience = profile.experience
    total = max(experience.total_years, 0.1)

    return {
        "total_experience_years": float(experience.total_years),
        "relevant_experience_years": float(experience.relevant_years),
        "relevant_experience_ratio": _safe_ratio(experience.relevant_years, total),
        "production_ml_ratio": _safe_ratio(experience.ai_years, total),
        "retrieval_experience_ratio": _safe_ratio(experience.retrieval_years, total),
        "ranking_experience_ratio": _safe_ratio(experience.ranking_years, total),
        "startup_experience_ratio": _safe_ratio(experience.startup_years, total),
        "python_experience_ratio": _safe_ratio(experience.python_years, total),
        "product_company_ratio": _safe_ratio(experience.product_company_years, total),
        "ai_experience_years": float(experience.ai_years),
        "ranking_experience_years": float(experience.ranking_years),
        "retrieval_experience_years": float(experience.retrieval_years),
        "python_experience_years": float(experience.python_years),
        "startup_experience_years": float(experience.startup_years),
        "product_company_years": float(experience.product_company_years),
        "experience_density_score": _safe_ratio(experience.relevant_years, total),
    }
