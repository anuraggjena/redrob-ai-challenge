from __future__ import annotations

from src.understanding.models import CandidateProfile

SEMANTIC_FEATURE_NAMES = (
    "semantic_bm25_score",
    "semantic_dense_score",
    "semantic_rrf_score",
    "semantic_skill_match",
    "semantic_responsibility_match",
    "semantic_career_match",
    "semantic_domain_match",
    "semantic_seniority_match",
    "semantic_intent_match",
    "semantic_production_match",
    "semantic_ranking_match",
    "semantic_retrieval_match",
    "semantic_behavioral_match",
    "semantic_availability_match",
    "semantic_fit_composite",
)


def compute(profile: CandidateProfile) -> dict[str, float]:
    del profile
    return {name: 0.0 for name in SEMANTIC_FEATURE_NAMES}
