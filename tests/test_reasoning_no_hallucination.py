import json
import re
from pathlib import Path

from src.graph.evidence_graph import build_evidence_graph
from src.graph.graph_scorer import score_graph
from src.honeypot.run import run_honeypot_detection
from src.pipeline.config_loader import load_job_requirements
from src.reasoning.hallucination_guard import validate_reasoning
from src.reasoning.reason_generator import generate_reasoning
from src.understanding.enrich import enrich_profile
from src.understanding.parser import parse_candidate

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_candidate.json"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "job_requirements.yaml"


def _load_test_profile(**raw_overrides):
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw.update(raw_overrides)
    profile = enrich_profile(parse_candidate(raw))
    run_honeypot_detection(profile)
    job_requirements = load_job_requirements(CONFIG_PATH)
    profile.evidence_graph = score_graph(build_evidence_graph(profile, job_requirements))
    return profile


def test_reasoning_only_uses_profile_facts():
    profile = _load_test_profile()
    text = generate_reasoning(profile, rank=1, score=0.95)
    assert "LangChain" not in text or any(
        s.get("name") == "LangChain" for s in profile.skills
    )
    violations = validate_reasoning(text, profile)
    assert violations == []


def test_reasoning_includes_years_and_title():
    profile = _load_test_profile()
    years = round(
        profile.experience.relevant_years or profile.experience.total_years, 1
    )
    text = generate_reasoning(profile, rank=5, score=0.85)
    assert profile.current_title in text
    assert str(years) in text or f"{years:.1f}" in text


def test_hallucination_guard_catches_fake_skill():
    profile = _load_test_profile()
    violations = validate_reasoning('Strong on "LangChain" experience.', profile)
    assert violations
    assert any("LangChain" in violation for violation in violations)


def test_concern_for_long_notice_period():
    profile = _load_test_profile()
    profile.signals["notice_period_days"] = 120
    job_requirements = load_job_requirements(CONFIG_PATH)
    profile.evidence_graph = score_graph(build_evidence_graph(profile, job_requirements))
    text = generate_reasoning(profile, rank=15, score=0.7)
    assert re.search(r"notice", text, re.IGNORECASE)


def test_reasoning_is_one_or_two_sentences():
    profile = _load_test_profile()
    text = generate_reasoning(profile, rank=25, score=0.75)
    sentence_count = len(re.findall(r"[.!?]+", text.strip()))
    assert 1 <= sentence_count <= 2
