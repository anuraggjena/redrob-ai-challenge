import json
from pathlib import Path

import pytest

from src.understanding.enrich import enrich_profile
from src.understanding.parser import parse_candidate

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_candidate.json"


@pytest.fixture
def minimal_candidate():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return parse_candidate(raw)


def test_enrich_populates_features(minimal_candidate):
    profile = enrich_profile(minimal_candidate)
    assert profile.features["skill_evidence_retrieval_ranking"] > 0
    assert "behavioral_engagement_score" in profile.features
    assert 0.0 <= profile.features["behavioral_engagement_score"] <= 1.0
    assert "availability_score" in profile.features
    assert profile.features["location_fit"] == pytest.approx(0.3)
    assert profile.features["title_progression_score"] >= 0.0


def test_role_skill_mismatch_detected():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["profile"]["current_title"] = "Marketing Manager"
    raw["skills"] = [
        {"name": "NLP", "proficiency": "expert", "endorsements": 50, "duration_months": 36},
        {"name": "Fine-tuning LLMs", "proficiency": "expert", "endorsements": 40, "duration_months": 30},
        {"name": "LoRA", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
        {"name": "Milvus", "proficiency": "expert", "endorsements": 45, "duration_months": 40},
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 30, "duration_months": 28},
        {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 25, "duration_months": 32},
        {"name": "RAG", "proficiency": "intermediate", "endorsements": 15, "duration_months": 18},
    ]
    profile = enrich_profile(parse_candidate(raw))
    assert profile.flags.get("role_skill_mismatch") is True


def test_github_sentinel_skipped():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["redrob_signals"]["github_activity_score"] = -1
    raw["redrob_signals"]["offer_acceptance_rate"] = -1
    profile = enrich_profile(parse_candidate(raw))
    assert "behavioral_github_activity_score" not in profile.features
    assert "behavioral_offer_acceptance_rate" not in profile.features
