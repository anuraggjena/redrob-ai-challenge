import json
from pathlib import Path

import pytest

from src.features import graph_features
from src.graph.evidence_graph import build_evidence_graph
from src.graph.graph_scorer import score_graph
from src.honeypot.run import run_honeypot_detection
from src.pipeline.config_loader import load_job_requirements
from src.understanding.enrich import enrich_profile
from src.understanding.parser import parse_candidate

FIXTURE = Path(__file__).parent / "fixtures" / "minimal_candidate.json"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "job_requirements.yaml"


@pytest.fixture
def job_requirements():
    return load_job_requirements(CONFIG_PATH)


def test_graph_has_support_edges_for_matching_skills(job_requirements):
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["skills"] = [
        {
            "name": "Milvus",
            "proficiency": "advanced",
            "endorsements": 40,
            "duration_months": 35,
        }
    ]
    profile = enrich_profile(parse_candidate(raw))
    graph = score_graph(build_evidence_graph(profile, job_requirements))

    skill_nodes = [node for node in graph.nodes if node.type == "skill"]
    assert any(node.label == "Milvus" for node in skill_nodes)

    support_edges = [
        edge for edge in graph.edges if edge.edge_type == "supports_requirement"
    ]
    assert support_edges
    assert graph.support_count >= 1


def test_graph_has_risk_edge_for_long_notice_period(job_requirements):
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["redrob_signals"]["notice_period_days"] = 120
    profile = enrich_profile(parse_candidate(raw))
    graph = build_evidence_graph(profile, job_requirements)

    risk_nodes = [node for node in graph.nodes if node.type == "risk"]
    assert any("notice" in node.label.lower() for node in risk_nodes)

    contradict_edges = [
        edge for edge in graph.edges if edge.edge_type == "contradicts_requirement"
    ]
    assert contradict_edges


def test_graph_features_updated_after_build(job_requirements):
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["skills"] = [
        {
            "name": "Milvus",
            "proficiency": "advanced",
            "endorsements": 40,
            "duration_months": 35,
        },
        {
            "name": "Python",
            "proficiency": "advanced",
            "endorsements": 20,
            "duration_months": 48,
        },
    ]
    profile = enrich_profile(parse_candidate(raw))
    run_honeypot_detection(profile)
    graph = score_graph(build_evidence_graph(profile, job_requirements))
    profile.evidence_graph = graph

    features = graph_features.compute(profile)

    assert features["support_edge_count"] == float(graph.support_count)
    assert features["contradict_edge_count"] == float(graph.contradict_count)
    assert features["support_edge_count"] > 0
    assert 0.0 <= features["requirement_coverage_ratio"] <= 1.0
