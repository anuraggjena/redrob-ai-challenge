import json
from pathlib import Path

from src.understanding.experience_extractor import enrich_experience
from src.understanding.parser import parse_candidate

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_candidate.json"


def test_ranking_years_from_description():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["career_history"][0]["description"] = (
        "Built learning-to-rank pipeline with NDCG evaluation at scale."
    )
    raw["career_history"][0]["duration_months"] = 24
    profile = parse_candidate(raw)
    enrich_experience(profile)
    assert profile.experience.ranking_years >= 2.0


def test_relevant_years_is_max_of_domains():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["skills"] = []
    raw["career_history"] = [
        {
            "company": "SearchCo",
            "title": "ML Engineer",
            "start_date": "2022-01-01",
            "end_date": "2024-01-01",
            "duration_months": 36,
            "is_current": False,
            "industry": "Technology",
            "company_size": "51-200",
            "description": "Owned learning-to-rank and NDCG benchmarks for search relevance.",
        },
        {
            "company": "VectorCo",
            "title": "Backend Engineer",
            "start_date": "2019-01-01",
            "end_date": "2022-01-01",
            "duration_months": 12,
            "is_current": False,
            "industry": "Technology",
            "company_size": "11-50",
            "description": "Built faiss vector search and hybrid search retrieval system.",
        },
    ]
    profile = parse_candidate(raw)
    enrich_experience(profile)
    assert profile.experience.ranking_years > profile.experience.retrieval_years
    assert profile.experience.relevant_years == profile.experience.ranking_years
