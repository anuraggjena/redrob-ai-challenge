from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.graph.evidence_graph import EvidenceGraph


@dataclass
class ExperienceStats:
    total_years: float = 0.0
    relevant_years: float = 0.0
    ai_years: float = 0.0
    ranking_years: float = 0.0
    retrieval_years: float = 0.0
    python_years: float = 0.0
    startup_years: float = 0.0
    product_company_years: float = 0.0


@dataclass
class CandidateProfile:
    candidate_id: str
    raw: dict[str, Any]
    headline: str = ""
    summary: str = ""
    current_title: str = ""
    current_company: str = ""
    location: str = ""
    country: str = ""
    experience: ExperienceStats = field(default_factory=ExperienceStats)
    career_roles: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    education: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    document_text: str = ""
    features: dict[str, float] = field(default_factory=dict)
    career_stats: dict[str, Any] = field(default_factory=dict)
    technical_stats: dict[str, float] = field(default_factory=dict)
    evidence_graph: EvidenceGraph | None = None
