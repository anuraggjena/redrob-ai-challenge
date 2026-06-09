import json
from pathlib import Path

import pytest

from src.features.registry import compute_all_features
from src.scoring.synthetic_labels import assign_synthetic_tier
from src.understanding.enrich import enrich_profile
from src.understanding.parser import parse_candidate

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_candidate.json"


def _tier(features: dict, title: str = "") -> int:
    tier, _label = assign_synthetic_tier(features, title=title)
    return tier


def test_honeypot_gets_tier_0():
    features = {
        "honeypot_probability": 0.75,
        "impossible_timeline_flag": 0.0,
        "timeline_inconsistent_flag": 0.0,
        "relevant_experience_years": 7.0,
        "ranking_relevance_score": 0.9,
    }
    tier, label = assign_synthetic_tier(features)
    assert tier == 0
    assert label == 0


def test_marketing_manager_low_tier():
    features = {
        "honeypot_probability": 0.1,
        "impossible_timeline_flag": 0.0,
        "timeline_inconsistent_flag": 0.0,
        "role_skill_mismatch_flag": 1.0,
        "relevant_experience_years": 2.0,
        "production_ml_ratio": 0.05,
        "product_company_ratio": 0.1,
        "ranking_relevance_score": 0.05,
        "retrieval_experience_score": 0.0,
        "langchain_skill_penalty": 0.2,
        "cv_only_penalty": 0.0,
        "speech_only_penalty": 0.0,
        "robotics_only_penalty": 0.0,
        "nlp_skill_depth": 0.05,
        "retrieval_experience_ratio": 0.0,
        "ranking_experience_ratio": 0.0,
        "consulting_only_flag": 0.0,
    }
    tier = _tier(features, title="Marketing Manager")
    assert tier <= 1


def test_strong_ai_engineer_high_tier():
    raw = {
        "candidate_id": "CAND_TEST_STRONG",
        "profile": {
            "headline": "Senior AI Engineer | Search & Ranking",
            "summary": (
                "7 years building production search, retrieval, and ranking systems "
                "with offline NDCG evaluation and hybrid retrieval."
            ),
            "location": "Pune",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior AI Engineer",
            "current_company": "Redrob Product Co",
            "current_company_size": "51-200",
            "current_industry": "HR Tech",
        },
        "career_history": [
            {
                "company": "Redrob Product Co",
                "title": "Senior AI Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 66,
                "is_current": True,
                "industry": "HR Tech",
                "company_size": "51-200",
                "description": (
                    "Owned learning-to-rank pipeline, hybrid BM25+dense retrieval, "
                    "and offline NDCG/MRR evaluation harness in production."
                ),
            },
            {
                "company": "Search Startup",
                "title": "ML Engineer",
                "start_date": "2018-06-01",
                "end_date": "2020-12-01",
                "duration_months": 30,
                "is_current": False,
                "industry": "Marketplace",
                "company_size": "11-50",
                "description": (
                    "Built recommendation and ranking systems with vector search "
                    "and production deployment."
                ),
            },
        ],
        "education": [],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 72},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
            {"name": "Ranking", "proficiency": "expert", "endorsements": 30, "duration_months": 48},
            {"name": "Retrieval", "proficiency": "expert", "endorsements": 28, "duration_months": 48},
            {"name": "NDCG", "proficiency": "advanced", "endorsements": 15, "duration_months": 24},
            {"name": "Milvus", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "last_active_date": "2026-05-28",
            "recruiter_response_rate": 0.6,
            "saved_by_recruiters_30d": 6,
            "notice_period_days": 30,
            "preferred_work_mode": "hybrid",
        },
    }
    profile = parse_candidate(raw)
    enrich_profile(profile)
    features = compute_all_features(profile)
    tier, label = assign_synthetic_tier(features, title=profile.current_title)
    assert tier >= 4
    assert label >= 4
