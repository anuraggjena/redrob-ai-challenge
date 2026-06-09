import json
from pathlib import Path

from src.honeypot.fusion import compute_honeypot_probability, compute_trustworthiness
from src.honeypot.keyword_stuffer_detector import score_keyword_stuffer
from src.honeypot.run import run_honeypot_detection
from src.honeypot.skill_inflation_detector import score_skill_inflation
from src.honeypot.timeline_detector import score_timeline
from src.understanding.enrich import enrich_profile
from src.understanding.parser import parse_candidate
from src.understanding.timeline_validator import validate_timeline

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_candidate.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_impossible_timeline_high_score():
    raw = {
        "candidate_id": "CAND_0009999",
        "profile": {
            "years_of_experience": 8,
            "current_title": "AI Engineer",
            "headline": "",
            "summary": "",
            "location": "",
            "country": "",
            "current_company": "",
            "current_company_size": "51-200",
            "current_industry": "",
        },
        "career_history": [
            {
                "company": "StartupX",
                "title": "ML Engineer",
                "start_date": "2010-01-01",
                "end_date": "2018-01-01",
                "duration_months": 96,
                "is_current": False,
                "industry": "",
                "company_size": "1-10",
                "description": "Founded company in 2016.",
            }
        ],
        "education": [],
        "skills": [],
        "redrob_signals": {},
    }
    profile = parse_candidate(raw)
    validate_timeline(profile)
    assert score_timeline(profile) >= 0.5


def test_marketing_manager_keyword_stuffer():
    raw = _load_fixture()
    raw["profile"]["current_title"] = "Marketing Manager"
    raw["skills"] = [
        {
            "name": f"skill_{i}",
            "proficiency": "expert",
            "endorsements": 50,
            "duration_months": 0,
        }
        for i in range(15)
    ]
    profile = parse_candidate(raw)
    assert score_keyword_stuffer(profile) >= 0.4


def test_expert_zero_duration_inflation():
    raw = _load_fixture()
    raw["skills"] = [
        {
            "name": f"skill_{i}",
            "proficiency": "expert",
            "endorsements": 10,
            "duration_months": 0,
        }
        for i in range(10)
    ]
    profile = parse_candidate(raw)
    assert score_skill_inflation(profile) >= 0.3


def test_honeypot_fusion_increases_with_multiple_detectors():
    single = compute_honeypot_probability({"timeline": 0.8})
    multiple = compute_honeypot_probability(
        {"timeline": 0.8, "skill_inflation": 0.7, "keyword_stuffer": 0.6}
    )
    assert multiple > single


def test_run_honeypot_detection_updates_features():
    raw = _load_fixture()
    profile = parse_candidate(raw)
    enrich_profile(profile)
    run_honeypot_detection(profile)
    assert "honeypot_probability" in profile.features
    assert "trustworthiness" in profile.features
    assert "profile_consistency_score" in profile.features
    assert "honeypot_penalty" in profile.features


def test_compute_trustworthiness_penalizes_honeypot_signals():
    scores = {
        "timeline": 0.8,
        "keyword_stuffer": 0.6,
    }
    honeypot_probability = compute_honeypot_probability(scores)
    trust = compute_trustworthiness(honeypot_probability, scores)
    assert trust < 0.5
