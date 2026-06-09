from __future__ import annotations

from src.understanding.models import CandidateProfile


def _compute_from_graph(profile: CandidateProfile) -> dict[str, float]:
    graph = profile.evidence_graph
    assert graph is not None

    support_count = graph.support_count
    contradict_count = graph.contradict_count
    strong_count = sum(1 for edge in graph.edges if edge.edge_type == "strong_signal")
    weak_count = sum(1 for edge in graph.edges if edge.edge_type == "weak_signal")
    risk_count = sum(1 for edge in graph.edges if edge.edge_type == "risk_signal")

    requirement_ids = {node.id for node in graph.nodes if node.type == "requirement"}
    covered_requirements = {
        edge.target_id
        for edge in graph.edges
        if edge.edge_type == "supports_requirement" and edge.target_id in requirement_ids
    }

    total_signals = max(support_count + contradict_count + strong_count + weak_count, 1)
    positive_signals = support_count + strong_count
    negative_signals = contradict_count + weak_count + risk_count

    return {
        "support_edge_count": float(support_count),
        "contradict_edge_count": float(contradict_count),
        "strong_signal_ratio": positive_signals / total_signals,
        "weak_signal_ratio": negative_signals / total_signals,
        "evidence_confidence": positive_signals / max(len(requirement_ids), 1),
        "risk_signal_count": float(contradict_count + risk_count),
        "requirement_coverage_ratio": len(covered_requirements) / max(len(requirement_ids), 1),
        "graph_score": (positive_signals - negative_signals) / total_signals,
        "graph_connectivity_score": (support_count - contradict_count) / total_signals,
    }


def _compute_placeholder(profile: CandidateProfile) -> dict[str, float]:
    flags = profile.flags
    features = profile.features

    support_signals = [
        features.get("skill_evidence_retrieval_ranking", 0.0) > 0,
        features.get("ranking_experience_ratio", 0.0) > 0,
        features.get("retrieval_experience_ratio", 0.0) > 0,
        features.get("production_ml_ratio", 0.0) > 0,
        features.get("python_skill_flag", 0.0) > 0,
        features.get("vector_db_skill_flag", 0.0) > 0,
        not flags.get("role_skill_mismatch", False),
        not flags.get("timeline_inconsistent", False),
    ]
    contradict_signals = [
        flags.get("role_skill_mismatch", False),
        flags.get("timeline_inconsistent", False),
        flags.get("impossible_timeline", False),
        features.get("keyword_stuffing_score", 0.0) > 0.3,
        features.get("langchain_skill_penalty", 0.0) > 0.5,
    ]

    support_count = sum(1 for signal in support_signals if signal)
    contradict_count = sum(1 for signal in contradict_signals if signal)
    total_signals = max(support_count + contradict_count, 1)

    requirement_hits = sum(
        1
        for key in (
            "ranking_experience_ratio",
            "retrieval_experience_ratio",
            "production_ml_ratio",
            "python_experience_ratio",
            "eval_skill_flag",
        )
        if features.get(key, 0.0) > 0
    )

    return {
        "support_edge_count": float(support_count),
        "contradict_edge_count": float(contradict_count),
        "strong_signal_ratio": support_count / total_signals,
        "weak_signal_ratio": contradict_count / total_signals,
        "evidence_confidence": support_count / max(len(support_signals), 1),
        "risk_signal_count": float(contradict_count),
        "requirement_coverage_ratio": requirement_hits / 5.0,
        "graph_score": 0.5,
        "graph_connectivity_score": (support_count - contradict_count) / total_signals,
    }


def compute(profile: CandidateProfile) -> dict[str, float]:
    if profile.evidence_graph is not None:
        return _compute_from_graph(profile)
    return _compute_placeholder(profile)
