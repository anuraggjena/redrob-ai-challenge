from __future__ import annotations

from src.graph.evidence_graph import EvidenceGraph


def score_graph(graph: EvidenceGraph) -> EvidenceGraph:
    graph.support_count = sum(
        1 for edge in graph.edges if edge.edge_type == "supports_requirement"
    )
    graph.contradict_count = sum(
        1
        for edge in graph.edges
        if edge.edge_type in {"contradicts_requirement", "risk_signal"}
    )
    return graph
