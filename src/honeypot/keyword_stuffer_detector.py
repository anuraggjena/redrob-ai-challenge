from __future__ import annotations

from functools import lru_cache

from src.ontology.loader import load_ontology_json
from src.understanding.models import CandidateProfile


@lru_cache(maxsize=1)
def _load_title_classifier() -> dict:
    return load_ontology_json("title_classifier.json")


def _is_non_technical_title(title: str, non_technical_titles: list[str]) -> bool:
    title_lower = title.lower().strip()
    return any(title_lower == entry.lower() for entry in non_technical_titles)


def score_keyword_stuffer(profile: CandidateProfile) -> float:
    title_classifier = _load_title_classifier()
    non_technical = title_classifier.get("non_technical", [])
    skills = profile.skills

    if not _is_non_technical_title(profile.current_title, non_technical):
        return 0.0
    if len(skills) <= 10:
        return 0.0

    expert_count = sum(
        1 for skill in skills if str(skill.get("proficiency", "")).lower() == "expert"
    )
    expert_ratio = expert_count / len(skills)
    if expert_ratio <= 0.5:
        return 0.0

    ranking_years = profile.experience.ranking_years
    retrieval_years = profile.experience.retrieval_years
    if ranking_years >= 0.5 or retrieval_years >= 0.5:
        return 0.0

    return min(0.4 + expert_ratio * 0.4 + (len(skills) - 10) * 0.02, 1.0)
