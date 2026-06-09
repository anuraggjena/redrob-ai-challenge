from __future__ import annotations

from typing import Any

from src.graph.evidence_graph import HONEYPOT_RISK_THRESHOLD, LONG_NOTICE_DAYS, STRONG_RESPONSE_RATE
from src.reasoning import templates
from src.understanding.models import CandidateProfile

_ROLE_THEME_LABELS = {
    "ranking": "ranking systems",
    "retrieval": "retrieval/search",
    "production": "production ML",
}


def generate_reasoning(
    profile: CandidateProfile,
    rank: int,
    score: float = 0.0,
    components: dict[str, Any] | None = None,
) -> str:
    years = round(profile.experience.relevant_years or profile.experience.total_years, 1)
    title = profile.current_title or "Professional"
    strengths = _extract_strengths(profile)
    concern = _extract_concern(profile)

    concern_text = templates.format_concern(concern) if concern else None

    if rank <= 10:
        return templates.format_top_tier(title, years, strengths, concern_text)
    if rank <= 50:
        return templates.format_mid_tier(title, years, strengths, concern_text)
    return templates.format_lower_tier(title, years, strengths, concern_text)


def _profile_skill_names(profile: CandidateProfile) -> set[str]:
    return {str(skill.get("name", "")).strip() for skill in profile.skills if skill.get("name")}


def _extract_strengths(profile: CandidateProfile, max_count: int = 2) -> list[str]:
    skill_names = _profile_skill_names(profile)
    strengths: list[str] = []
    seen: set[str] = set()

    graph = profile.evidence_graph
    if graph is not None:
        node_by_id = {node.id: node for node in graph.nodes}
        edge_priority = ("strong_signal", "supports_requirement")
        for edge_type in edge_priority:
            for edge in graph.edges:
                if edge.edge_type != edge_type:
                    continue
                node = node_by_id.get(edge.source_id)
                if node is None:
                    continue
                label = _strength_label(node, skill_names)
                if not label or label in seen:
                    continue
                seen.add(label)
                strengths.append(label)
                if len(strengths) >= max_count:
                    return strengths

    if strengths:
        return strengths[:max_count]

    ranked_skills = sorted(
        profile.skills,
        key=lambda skill: (
            float(skill.get("endorsements") or 0),
            float(skill.get("duration_months") or 0),
        ),
        reverse=True,
    )
    for skill in ranked_skills:
        name = str(skill.get("name", "")).strip()
        if name and name not in seen:
            strengths.append(name)
            seen.add(name)
        if len(strengths) >= max_count:
            break

    return strengths[:max_count]


def _strength_label(node, skill_names: set[str]) -> str:
    if node.type == "skill" and node.label in skill_names:
        return node.label
    if node.type == "role":
        themes = [part.strip() for part in node.value.split(",") if part.strip()]
        for theme in themes:
            if theme in _ROLE_THEME_LABELS:
                return _ROLE_THEME_LABELS[theme]
        title = node.label.strip()
        return title if title else ""
    if node.type == "behavioral":
        return node.label.strip()
    return ""


def _extract_concern(profile: CandidateProfile) -> str:
    concerns: list[str] = []

    graph = profile.evidence_graph
    if graph is not None:
        for node in graph.nodes:
            if node.type != "risk":
                continue
            label = node.label.strip().lower()
            if "notice" in label:
                concerns.append(node.label)
            elif "honeypot" in label:
                concerns.append(node.label)
            elif "mismatch" in label:
                concerns.append(node.label)
            else:
                concerns.append(node.label)

    signals = profile.signals
    notice_days = float(signals.get("notice_period_days") or 0.0)
    if notice_days > LONG_NOTICE_DAYS and not any("notice" in c.lower() for c in concerns):
        concerns.append(f"notice period {int(notice_days)} days")

    response_rate = float(
        profile.features.get("behavioral_recruiter_response_rate")
        or signals.get("recruiter_response_rate")
        or 0.0
    )
    if response_rate < STRONG_RESPONSE_RATE and not any(
        "response" in c.lower() for c in concerns
    ):
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")

    honeypot_probability = float(profile.features.get("honeypot_probability", 0.0))
    honeypot_flags = (
        profile.flags.get("timeline_inconsistent")
        or profile.flags.get("impossible_timeline")
        or profile.flags.get("role_skill_mismatch")
    )
    if honeypot_probability < HONEYPOT_RISK_THRESHOLD and honeypot_flags:
        concerns.append("some profile consistency flags detected")

    if profile.flags.get("role_skill_mismatch") and not any(
        "mismatch" in c.lower() for c in concerns
    ):
        concerns.append("role skill mismatch")

    return concerns[0] if concerns else ""
