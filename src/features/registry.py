from __future__ import annotations

from src.features import (
    availability_features,
    behavioral_features,
    career_features,
    experience_features,
    graph_features,
    interaction_features,
    recruiter_intelligence,
    semantic_features,
    technical_features,
    trust_features,
)
from pathlib import Path

from src.graph.evidence_graph import build_evidence_graph
from src.graph.graph_scorer import score_graph
from src.honeypot.run import run_honeypot_detection
from src.pipeline.config_loader import load_job_requirements
from src.understanding.models import CandidateProfile

JOB_REQUIREMENTS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "job_requirements.yaml"
)
_JOB_REQUIREMENTS = load_job_requirements(JOB_REQUIREMENTS_PATH)

COMPUTERS = [
    experience_features.compute,
    career_features.compute,
    technical_features.compute,
    behavioral_features.compute,
    availability_features.compute,
    trust_features.compute,
    semantic_features.compute,
    graph_features.compute,
    recruiter_intelligence.compute,
    interaction_features.compute,
]


def _attach_evidence_graph(profile: CandidateProfile) -> None:
    score_graph(build_evidence_graph(profile, _JOB_REQUIREMENTS))


def compute_all_features(profile: CandidateProfile) -> dict[str, float]:
    for compute_fn in COMPUTERS:
        if compute_fn is trust_features.compute:
            run_honeypot_detection(profile)
        if compute_fn is graph_features.compute:
            _attach_evidence_graph(profile)
        profile.features.update(compute_fn(profile))
    return dict(profile.features)
