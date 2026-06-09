from __future__ import annotations

COMPONENT_KEYS = (
    "technical_fit",
    "semantic_fit",
    "experience_fit",
    "behavioral_fit",
    "recruiter_signal_fit",
    "availability_fit",
    "trustworthiness",
)

SEMANTIC_MATCH_FEATURES = (
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
)

TECHNICAL_FEATURES = (
    "technical_depth_score",
    "ranking_relevance_score",
    "retrieval_experience_score",
    "production_experience_score",
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _technical_fit(features: dict[str, float]) -> float:
    values = [_clamp01(features.get(name, 0.0)) for name in TECHNICAL_FEATURES]
    return _mean(values)


def _semantic_fit(features: dict[str, float]) -> float:
    composite = features.get("semantic_fit_composite", 0.0)
    if composite > 0.0:
        return _clamp01(composite)

    match_values = [features.get(name, 0.0) for name in SEMANTIC_MATCH_FEATURES]
    if any(value > 0.0 for value in match_values):
        return _clamp01(_mean(match_values))

    retrieval_evidence = features.get("skill_evidence_retrieval_ranking", 0.0)
    return _clamp01(retrieval_evidence / 10.0)


def _experience_fit(features: dict[str, float]) -> float:
    years = features.get("relevant_experience_years", 0.0)
    years_component = min(1.0, years / 6.0) * 0.5
    growth_component = _clamp01(features.get("career_growth_score", 0.0)) * 0.3
    product_component = _clamp01(features.get("product_company_ratio", 0.0)) * 0.2
    return _clamp01(years_component + growth_component + product_component)


def _behavioral_fit(features: dict[str, float]) -> float:
    engagement = features.get("behavioral_engagement_score")
    if engagement is None or engagement == 0.0:
        engagement = features.get("engagement_score", 0.0)
    return _clamp01(float(engagement))


def _recruiter_signal_fit(features: dict[str, float]) -> float:
    hireability = features.get("hireability_score")
    if hireability is None or hireability == 0.0:
        hireability = features.get("composite_hireability_index", 0.0)
    return _clamp01(float(hireability))


def _availability_fit(features: dict[str, float]) -> float:
    return _clamp01(features.get("availability_score", 0.0))


def _trustworthiness_fit(features: dict[str, float]) -> float:
    trust = features.get("trustworthiness")
    if trust is None:
        trust = features.get("trustworthiness_score", 0.5)
    return _clamp01(float(trust))


def compute_components(features: dict[str, float]) -> dict[str, float]:
    return {
        "technical_fit": _technical_fit(features),
        "semantic_fit": _semantic_fit(features),
        "experience_fit": _experience_fit(features),
        "behavioral_fit": _behavioral_fit(features),
        "recruiter_signal_fit": _recruiter_signal_fit(features),
        "availability_fit": _availability_fit(features),
        "trustworthiness": _trustworthiness_fit(features),
    }
