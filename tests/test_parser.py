import json
from pathlib import Path
from src.understanding.parser import parse_candidate, load_candidates

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_candidate.json"


def test_parse_candidate_id():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    profile = parse_candidate(raw)
    assert profile.candidate_id == raw["candidate_id"]
    assert profile.experience.total_years == raw["profile"]["years_of_experience"]


def test_document_text_includes_headline():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    profile = parse_candidate(raw)
    assert profile.headline in profile.document_text


def test_load_candidates_json_array():
    sample = Path(__file__).parents[1] / "India_runs_data_and_ai_challenge" / "sample_candidates.json"
    profiles = load_candidates(sample)
    assert len(profiles) >= 1
    assert all(p.candidate_id.startswith("CAND_") for p in profiles)
