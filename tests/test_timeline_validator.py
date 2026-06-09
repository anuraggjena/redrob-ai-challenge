import json
from pathlib import Path

from src.understanding.parser import parse_candidate
from src.understanding.timeline_validator import validate_timeline

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_candidate.json"


def test_overlapping_roles_flag():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["career_history"] = [
        {
            "company": "Alpha",
            "title": "ML Engineer",
            "start_date": "2023-01-01",
            "end_date": None,
            "duration_months": 30,
            "is_current": True,
            "industry": "Technology",
            "company_size": "51-200",
            "description": "Ranking systems.",
        },
        {
            "company": "Beta",
            "title": "Data Scientist",
            "start_date": "2024-06-01",
            "end_date": None,
            "duration_months": 12,
            "is_current": True,
            "industry": "Technology",
            "company_size": "11-50",
            "description": "Retrieval pipelines.",
        },
    ]
    profile = parse_candidate(raw)
    validate_timeline(profile)
    assert profile.flags.get("timeline_inconsistent") is True


def test_impossible_founding_year():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["career_history"] = [
        {
            "company": "StartupX",
            "title": "ML Engineer",
            "start_date": "2010-01-01",
            "end_date": "2018-01-01",
            "duration_months": 96,
            "is_current": False,
            "industry": "Technology",
            "company_size": "1-10",
            "description": "Founded company in 2016.",
        }
    ]
    profile = parse_candidate(raw)
    validate_timeline(profile)
    assert profile.flags.get("impossible_timeline") is True
