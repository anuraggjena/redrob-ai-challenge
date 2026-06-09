from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.graph.node_types import EvidenceNode
from src.understanding.models import CandidateProfile

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "job_requirements.yaml"

STRONG_RESPONSE_RATE = 0.3
STRONG_COMPLETENESS = 0.8
HONEYPOT_RISK_THRESHOLD = 0.5
LONG_NOTICE_DAYS = 90


@dataclass
class EvidenceEdge:
    source_id: str
    target_id: str
    edge_type: str


@dataclass
class EvidenceGraph:
    nodes: list[EvidenceNode]
    edges: list[EvidenceEdge]
    support_count: int = 0
    contradict_count: int = 0


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((ONTOLOGY_DIR / name).read_text(encoding="utf-8"))


def _normalize_id(prefix: str, label: str) -> str:
    slug = label.lower().strip().replace(" ", "_").replace("/", "_")
    return f"{prefix}:{slug}"


def _skill_matches_term(skill_name: str, term: str) -> bool:
    skill_lower = skill_name.lower()
    term_lower = term.lower()
    return skill_lower == term_lower or term_lower in skill_lower or skill_lower in term_lower


def _matches_requirement(label: str, requirement: str) -> bool:
    return _skill_matches_term(label, requirement)


def _requirement_matches_skill(requirement: str, skill_name: str) -> bool:
    if _matches_requirement(skill_name, requirement):
        return True

    req_lower = requirement.lower()
    vector_db_terms = (
        "milvus",
        "faiss",
        "pinecone",
        "weaviate",
        "qdrant",
        "vector database",
    )
    if "vector" in req_lower and any(
        _skill_matches_term(skill_name, term) for term in vector_db_terms
    ):
        return True

    search_terms = ("elasticsearch", "opensearch", "bm25", "search")
    if "search" in req_lower and any(
        _skill_matches_term(skill_name, term) for term in search_terms
    ):
        return True

    taxonomy = _load_json("skills_taxonomy.json")
    for terms in taxonomy.values():
        skill_hits = [_skill_matches_term(skill_name, term) for term in terms]
        if not any(skill_hits):
            continue
        if any(_skill_matches_term(requirement, term) for term in terms):
            return True
        if any(term in req_lower or req_lower in term for term in terms if skill_hits):
            return True

    return False


def _skill_matches_any_requirement(skill_name: str, requirements: list[str]) -> bool:
    return any(_requirement_matches_skill(req, skill_name) for req in requirements)


def _taxonomy_terms() -> list[str]:
    taxonomy = _load_json("skills_taxonomy.json")
    terms: list[str] = []
    for key in ("retrieval_ranking", "embeddings_nlp", "production_ml"):
        terms.extend(taxonomy.get(key, []))
    return terms


def _skill_in_taxonomy(skill_name: str, terms: list[str]) -> bool:
    return any(_skill_matches_term(skill_name, term) for term in terms)


def _text_matches(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(pattern.lower() in text_lower for pattern in patterns)


def _role_hits(role: dict[str, Any]) -> list[str]:
    patterns = _load_json("responsibility_patterns.json")
    text = f"{role.get('title', '')} {role.get('description', '')}"
    hits: list[str] = []
    if _text_matches(text, patterns.get("ranking_evidence", [])):
        hits.append("ranking")
    if _text_matches(text, patterns.get("retrieval_evidence", [])):
        hits.append("retrieval")
    if _text_matches(text, patterns.get("production_evidence", [])):
        hits.append("production")
    return hits


def _requirement_nodes(job_requirements: dict[str, Any]) -> list[EvidenceNode]:
    required = list(job_requirements.get("required_skills", []))[:10]
    positive = list(job_requirements.get("positive_signals", []))
    seen: set[str] = set()
    nodes: list[EvidenceNode] = []

    for label in required + positive:
        key = label.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        nodes.append(
            EvidenceNode(
                id=_normalize_id("req", label),
                type="requirement",
                label=label,
                value=label,
                source_field="job_requirements",
            )
        )
    return nodes


def _add_edge(
    edges: list[EvidenceEdge],
    seen: set[tuple[str, str, str]],
    source_id: str,
    target_id: str,
    edge_type: str,
) -> None:
    key = (source_id, target_id, edge_type)
    if key in seen:
        return
    seen.add(key)
    edges.append(EvidenceEdge(source_id=source_id, target_id=target_id, edge_type=edge_type))


def build_evidence_graph(
    profile: CandidateProfile,
    job_requirements: dict[str, Any],
) -> EvidenceGraph:
    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    edge_seen: set[tuple[str, str, str]] = set()

    requirement_nodes = _requirement_nodes(job_requirements)
    nodes.extend(requirement_nodes)

    taxonomy_terms = _taxonomy_terms()
    all_required = list(job_requirements.get("required_skills", []))
    for skill in profile.skills:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        in_taxonomy = _skill_in_taxonomy(name, taxonomy_terms)
        in_requirements = _skill_matches_any_requirement(name, all_required)
        if not in_taxonomy and not in_requirements:
            continue

        skill_id = _normalize_id("skill", name)
        nodes.append(
            EvidenceNode(
                id=skill_id,
                type="skill",
                label=name,
                value=str(skill.get("proficiency", "")),
                source_field="skills",
            )
        )
        for req_node in requirement_nodes:
            if _requirement_matches_skill(req_node.label, name):
                _add_edge(
                    edges,
                    edge_seen,
                    skill_id,
                    req_node.id,
                    "supports_requirement",
                )

    for index, role in enumerate(profile.career_roles):
        hits = _role_hits(role)
        if not hits:
            continue

        title = str(role.get("title", "")).strip() or f"role_{index}"
        company = str(role.get("company", "")).strip()
        role_id = _normalize_id("role", f"{company}_{title}_{index}")
        nodes.append(
            EvidenceNode(
                id=role_id,
                type="role",
                label=title,
                value=",".join(hits),
                source_field="career_history",
            )
        )
        for req_node in requirement_nodes:
            req_lower = req_node.label.lower()
            matched = (
                ("ranking" in hits and "ranking" in req_lower)
                or ("retrieval" in hits and "retrieval" in req_lower)
                or ("production" in hits and "production" in req_lower)
                or ("search" in req_lower and ("retrieval" in hits or "ranking" in hits))
                or _requirement_matches_skill(req_node.label, title)
                or _requirement_matches_skill(req_node.label, str(role.get("description", "")))
            )
            if matched:
                _add_edge(
                    edges,
                    edge_seen,
                    role_id,
                    req_node.id,
                    "supports_requirement",
                )

    signals = profile.signals
    behavioral_specs = [
        (
            "recruiter_response_rate",
            float(signals.get("recruiter_response_rate") or 0.0),
            STRONG_RESPONSE_RATE,
            "redrob_signals.recruiter_response_rate",
        ),
        (
            "open_to_work",
            1.0 if signals.get("open_to_work_flag") else 0.0,
            0.5,
            "redrob_signals.open_to_work_flag",
        ),
        (
            "profile_completeness",
            float(signals.get("profile_completeness_score") or 0.0) / 100.0,
            STRONG_COMPLETENESS,
            "redrob_signals.profile_completeness_score",
        ),
    ]
    for name, value, threshold, source_field in behavioral_specs:
        if value < threshold:
            continue
        behavioral_id = _normalize_id("behavioral", name)
        nodes.append(
            EvidenceNode(
                id=behavioral_id,
                type="behavioral",
                label=name.replace("_", " "),
                value=str(value),
                source_field=source_field,
            )
        )
        for req_node in requirement_nodes:
            _add_edge(
                edges,
                edge_seen,
                behavioral_id,
                req_node.id,
                "strong_signal",
            )

    risk_specs: list[tuple[str, str, str, str]] = []
    notice_days = float(signals.get("notice_period_days") or 0.0)
    if notice_days > LONG_NOTICE_DAYS:
        risk_specs.append(
            (
                "long_notice_period",
                f"notice period {int(notice_days)} days",
                str(notice_days),
                "redrob_signals.notice_period_days",
            )
        )

    if profile.flags.get("role_skill_mismatch"):
        risk_specs.append(
            (
                "title_mismatch",
                "role skill mismatch",
                profile.current_title,
                "profile.current_title",
            )
        )

    honeypot_probability = profile.features.get("honeypot_probability", 0.0)
    if honeypot_probability >= HONEYPOT_RISK_THRESHOLD:
        risk_specs.append(
            (
                "honeypot_risk",
                "honeypot signals detected",
                str(honeypot_probability),
                "features.honeypot_probability",
            )
        )

    for risk_key, label, value, source_field in risk_specs:
        risk_id = _normalize_id("risk", risk_key)
        nodes.append(
            EvidenceNode(
                id=risk_id,
                type="risk",
                label=label,
                value=value,
                source_field=source_field,
            )
        )
        for req_node in requirement_nodes:
            _add_edge(
                edges,
                edge_seen,
                risk_id,
                req_node.id,
                "contradicts_requirement",
            )

    graph = EvidenceGraph(nodes=nodes, edges=edges)
    profile.evidence_graph = graph
    return graph
