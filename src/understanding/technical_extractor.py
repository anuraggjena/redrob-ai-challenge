from __future__ import annotations

import json
import math
from pathlib import Path

from src.understanding.models import CandidateProfile

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"

PROFICIENCY_WEIGHTS = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.75,
    "expert": 1.0,
}


def _load_json(name: str) -> dict:
    return json.loads((ONTOLOGY_DIR / name).read_text(encoding="utf-8"))


def _skill_matches_term(skill_name: str, term: str) -> bool:
    skill_lower = skill_name.lower()
    term_lower = term.lower()
    return skill_lower == term_lower or term_lower in skill_lower or skill_lower in term_lower


def _skill_in_taxonomy(skill_name: str, terms: list[str]) -> bool:
    return any(_skill_matches_term(skill_name, term) for term in terms)


def _skill_evidence(skill: dict) -> float:
    proficiency = str(skill.get("proficiency", "")).lower()
    weight = PROFICIENCY_WEIGHTS.get(proficiency, 0.5)
    duration_months = float(skill.get("duration_months") or 0)
    endorsements = float(skill.get("endorsements") or 0)
    return weight * min(duration_months / 12.0, 5.0) * math.log1p(endorsements)


def _is_non_technical_title(title: str, non_technical_titles: list[str]) -> bool:
    title_lower = title.lower().strip()
    return any(title_lower == entry.lower() for entry in non_technical_titles)


def enrich_technical(profile: CandidateProfile) -> None:
    taxonomy = _load_json("skills_taxonomy.json")
    title_classifier = _load_json("title_classifier.json")

    retrieval_ranking_terms = taxonomy.get("retrieval_ranking", [])
    embeddings_nlp_terms = taxonomy.get("embeddings_nlp", [])

    retrieval_ranking_evidence = 0.0
    taxonomy_skill_count = 0

    for skill in profile.skills:
        name = skill.get("name", "")
        if not name:
            continue

        evidence = _skill_evidence(skill)
        profile.technical_stats[name] = evidence

        in_retrieval_ranking = _skill_in_taxonomy(name, retrieval_ranking_terms)
        in_embeddings_nlp = _skill_in_taxonomy(name, embeddings_nlp_terms)

        if in_retrieval_ranking:
            retrieval_ranking_evidence += evidence
        if in_retrieval_ranking or in_embeddings_nlp:
            taxonomy_skill_count += 1

    profile.features["skill_evidence_retrieval_ranking"] = retrieval_ranking_evidence

    if _is_non_technical_title(
        profile.current_title, title_classifier.get("non_technical", [])
    ) and taxonomy_skill_count > 5:
        profile.flags["role_skill_mismatch"] = True
