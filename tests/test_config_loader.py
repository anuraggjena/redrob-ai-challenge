from pathlib import Path

from src.pipeline.config_loader import load_job_requirements, load_json_ontology

ROOT = Path(__file__).resolve().parents[1]


def test_load_job_requirements_has_required_skills():
    req = load_job_requirements(ROOT / "config" / "job_requirements.yaml")
    assert "embeddings" in req["required_skills"]
    assert "ranking" in req["required_skills"]
    assert len(req["negative_signals"]) > 0


def test_load_title_classifier():
    titles = load_json_ontology(ROOT / "src" / "ontology" / "title_classifier.json")
    assert "non_technical" in titles
    assert "Marketing Manager" in titles["non_technical"]
