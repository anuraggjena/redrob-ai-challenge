import pytest

from src.scoring.component_scores import COMPONENT_KEYS, compute_components
from src.scoring.ensemble_scorer import (
    DEFAULT_WEIGHTS,
    apply_gates,
    score_batch,
    score_candidate,
)


def _base_features(**overrides) -> dict[str, float]:
    features = {
        "technical_depth_score": 0.5,
        "ranking_relevance_score": 0.5,
        "retrieval_experience_score": 0.5,
        "production_experience_score": 0.5,
        "semantic_fit_composite": 0.5,
        "relevant_experience_years": 3.0,
        "career_growth_score": 0.5,
        "product_company_ratio": 0.5,
        "engagement_score": 0.5,
        "hireability_score": 0.5,
        "availability_score": 0.5,
        "trustworthiness": 0.5,
        "honeypot_probability": 0.0,
        "honeypot_penalty": 0.0,
        "keyword_stuffer_penalty": 0.0,
        "consulting_only_flag": 0.0,
        "cv_only_penalty": 0.0,
        "speech_only_penalty": 0.0,
        "robotics_only_penalty": 0.0,
        "nlp_skill_depth": 0.5,
        "retrieval_experience_ratio": 0.5,
        "ranking_experience_ratio": 0.5,
        "responsiveness_score": 0.5,
        "behavioral_days_since_active": 30.0,
    }
    features.update(overrides)
    return features


def _high_technical_features() -> dict[str, float]:
    return _base_features(
        technical_depth_score=0.92,
        ranking_relevance_score=0.88,
        retrieval_experience_score=0.85,
        production_experience_score=0.9,
        semantic_fit_composite=0.82,
        relevant_experience_years=6.0,
        career_growth_score=0.85,
        product_company_ratio=0.75,
        engagement_score=0.8,
        hireability_score=0.88,
        availability_score=0.9,
        trustworthiness=0.92,
        skill_evidence_retrieval_ranking=8.0,
    )


def _marketing_manager_features() -> dict[str, float]:
    return _base_features(
        technical_depth_score=0.08,
        ranking_relevance_score=0.05,
        retrieval_experience_score=0.0,
        production_experience_score=0.05,
        semantic_fit_composite=0.1,
        relevant_experience_years=1.0,
        career_growth_score=0.25,
        product_company_ratio=0.15,
        engagement_score=0.35,
        hireability_score=0.18,
        availability_score=0.45,
        trustworthiness=0.55,
        role_skill_mismatch_flag=1.0,
        skill_evidence_retrieval_ranking=0.2,
        cv_only_penalty=0.6,
        nlp_skill_depth=0.05,
        retrieval_experience_ratio=0.0,
        ranking_experience_ratio=0.0,
    )


def test_compute_components_returns_all_keys():
    components = compute_components(_base_features())
    assert set(components) == set(COMPONENT_KEYS)
    assert all(0.0 <= value <= 1.0 for value in components.values())


def test_high_technical_candidate_scores_higher_than_marketing_manager():
    high_score = score_candidate(_high_technical_features())
    marketing_score = score_candidate(_marketing_manager_features())
    assert high_score > marketing_score
    assert high_score >= 0.6
    assert marketing_score <= 0.45


def test_honeypot_probability_above_threshold_returns_zero():
    features = _base_features(honeypot_probability=0.75, honeypot_penalty=0.1)
    assert score_candidate(features) == 0.0


def test_weights_sum_produces_expected_range():
    perfect = {key: 1.0 for key in COMPONENT_KEYS}
    components = compute_components(_base_features())
    for key in COMPONENT_KEYS:
        components[key] = 1.0

    max_score = sum(DEFAULT_WEIGHTS[k] * components[k] for k in DEFAULT_WEIGHTS)
    assert max_score == pytest.approx(1.0)

    top = score_candidate(
        _base_features(
            technical_depth_score=1.0,
            ranking_relevance_score=1.0,
            retrieval_experience_score=1.0,
            production_experience_score=1.0,
            semantic_fit_composite=1.0,
            relevant_experience_years=12.0,
            career_growth_score=1.0,
            product_company_ratio=1.0,
            engagement_score=1.0,
            hireability_score=1.0,
            availability_score=1.0,
            trustworthiness=1.0,
        )
    )
    assert 0.95 <= top <= 1.0

    bottom = score_candidate(
        _base_features(
            technical_depth_score=0.0,
            ranking_relevance_score=0.0,
            retrieval_experience_score=0.0,
            production_experience_score=0.0,
            semantic_fit_composite=0.0,
            relevant_experience_years=0.0,
            career_growth_score=0.0,
            product_company_ratio=0.0,
            engagement_score=0.0,
            hireability_score=0.0,
            availability_score=0.0,
            trustworthiness=0.0,
            skill_evidence_retrieval_ranking=0.0,
        )
    )
    assert bottom == pytest.approx(0.0)


def test_apply_gates_consulting_only_without_product():
    features = _base_features(consulting_only_flag=1.0, product_company_ratio=0.0)
    assert apply_gates(features) == pytest.approx(0.2)


def test_apply_gates_domain_only_penalty():
    features = _base_features(
        cv_only_penalty=0.8,
        speech_only_penalty=0.0,
        robotics_only_penalty=0.0,
        nlp_skill_depth=0.0,
        retrieval_experience_ratio=0.0,
        ranking_experience_ratio=0.0,
    )
    assert apply_gates(features) == pytest.approx(0.3)


def test_apply_gates_availability_crush():
    features = _base_features(responsiveness_score=0.05, behavioral_days_since_active=200.0)
    assert apply_gates(features) == pytest.approx(0.5)


def test_score_batch_with_dict():
    features_by_id = {
        "c1": _high_technical_features(),
        "c2": _marketing_manager_features(),
    }
    scores = score_batch(["c1", "c2"], features_by_id)
    assert scores["c1"] > scores["c2"]


def test_score_batch_with_dataframe():
    pandas = pytest.importorskip("pandas")
    features_by_id = {
        "c1": _high_technical_features(),
        "c2": _marketing_manager_features(),
    }
    rows = [{"candidate_id": cid, **features} for cid, features in features_by_id.items()]
    frame = pandas.DataFrame(rows)
    scores = score_batch(["c1", "c2"], frame)
    assert set(scores) == {"c1", "c2"}
    assert scores["c1"] > scores["c2"]
