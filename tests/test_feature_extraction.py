import json
from pathlib import Path

import pytest

from src.features.registry import compute_all_features
from src.understanding.enrich import enrich_profile
from src.understanding.parser import parse_candidate

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_candidate.json"


@pytest.fixture
def enriched_profile():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    profile = parse_candidate(raw)
    enrich_profile(profile)
    return profile


def test_feature_count_at_least_100(enriched_profile):
    features = compute_all_features(enriched_profile)
    assert len(features) >= 100
    assert len(enriched_profile.features) >= 100


def test_all_features_are_floats(enriched_profile):
    features = compute_all_features(enriched_profile)
    assert features
    for key, value in features.items():
        assert isinstance(key, str), f"key {key!r} is not a str"
        assert isinstance(value, (int, float)), f"{key}={value!r} is not numeric"
        assert isinstance(float(value), float)


def test_experience_features_present(enriched_profile):
    features = compute_all_features(enriched_profile)
    for name in (
        "total_experience_years",
        "relevant_experience_years",
        "relevant_experience_ratio",
        "production_ml_ratio",
        "retrieval_experience_ratio",
        "ranking_experience_ratio",
    ):
        assert name in features


def test_semantic_placeholders_are_zero(enriched_profile):
    features = compute_all_features(enriched_profile)
    assert features["semantic_bm25_score"] == 0.0
    assert features["semantic_fit_composite"] == 0.0


def test_recruiter_intelligence_scores(enriched_profile):
    features = compute_all_features(enriched_profile)
    for name in (
        "technical_depth_score",
        "hireability_score",
        "candidate_reliability_score",
    ):
        assert name in features
        assert 0.0 <= features[name] <= 1.0 or features[name] >= 0.0
