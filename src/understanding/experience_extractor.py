from __future__ import annotations

import json
from pathlib import Path

from src.understanding.models import CandidateProfile

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"

PRODUCT_SIZES = frozenset({"1-10", "11-50", "51-200", "201-500"})
STARTUP_SIZES = frozenset({"1-10", "11-50"})

RANKING_TAXONOMY_TERMS = frozenset(
    {
        "ranking",
        "learning-to-rank",
        "ndcg",
        "mrr",
        "map",
        "reciprocal rank fusion",
    }
)
RETRIEVAL_TAXONOMY_TERMS = frozenset(
    {
        "retrieval",
        "faiss",
        "elasticsearch",
        "opensearch",
        "bm25",
        "vector database",
        "pinecone",
        "weaviate",
        "qdrant",
        "milvus",
        "hybrid search",
        "dense retrieval",
        "search",
    }
)


def _load_json(name: str) -> dict:
    return json.loads((ONTOLOGY_DIR / name).read_text(encoding="utf-8"))


def _role_years(role: dict) -> float:
    return float(role.get("duration_months") or 0) / 12.0


def _text_matches(text: str, patterns: list[str]) -> bool:
    return any(pattern.lower() in text for pattern in patterns)


def _is_consulting_company(company: str, consulting_list: list[str]) -> bool:
    company_lower = company.lower().strip()
    return any(
        company_lower == name.lower() or name.lower() in company_lower
        for name in consulting_list
    )


def _skill_matches_term(skill_name: str, term: str) -> bool:
    skill_lower = skill_name.lower()
    term_lower = term.lower()
    return skill_lower == term_lower or term_lower in skill_lower or skill_lower in term_lower


def _skill_in_terms(skill_name: str, terms: frozenset[str]) -> bool:
    return any(_skill_matches_term(skill_name, term) for term in terms)


def enrich_experience(profile: CandidateProfile) -> None:
    patterns = _load_json("responsibility_patterns.json")
    taxonomy = _load_json("skills_taxonomy.json")
    company_types = _load_json("company_types.json")
    consulting = company_types.get("consulting", [])

    ranking_patterns = patterns.get("ranking_evidence", [])
    retrieval_patterns = patterns.get("retrieval_evidence", [])
    production_patterns = patterns.get("production_evidence", [])
    eval_patterns = patterns.get("eval_evidence", [])

    for role in profile.career_roles:
        text = f"{role.get('title', '')} {role.get('description', '')}".lower()
        years = _role_years(role)

        if _text_matches(text, ranking_patterns):
            profile.experience.ranking_years += years
        if _text_matches(text, retrieval_patterns):
            profile.experience.retrieval_years += years
        if _text_matches(text, production_patterns):
            profile.experience.ai_years += years
        if _text_matches(text, eval_patterns):
            profile.flags["eval_experience"] = True
        if "python" in text:
            profile.experience.python_years += years

        company_size = role.get("company_size", "")
        industry = role.get("industry", "")
        company = role.get("company", "")

        is_consulting = _is_consulting_company(company, consulting)
        is_product = company_size in PRODUCT_SIZES or (
            industry != "IT Services" and not is_consulting
        )
        if is_product:
            profile.experience.product_company_years += years
        if company_size in STARTUP_SIZES:
            profile.experience.startup_years += years

    for skill in profile.skills:
        name = skill.get("name", "")
        skill_years = float(skill.get("duration_months") or 0) / 12.0
        if not name or skill_years <= 0:
            continue

        if "python" in name.lower():
            profile.experience.python_years += skill_years

        retrieval_ranking = taxonomy.get("retrieval_ranking", [])
        if any(_skill_matches_term(name, term) for term in retrieval_ranking):
            if _skill_in_terms(name, RANKING_TAXONOMY_TERMS) or _skill_matches_term(
                name, "ranking"
            ):
                profile.experience.ranking_years += skill_years
            if _skill_in_terms(name, RETRIEVAL_TAXONOMY_TERMS) or _skill_matches_term(
                name, "retrieval"
            ):
                profile.experience.retrieval_years += skill_years

        production_ml = taxonomy.get("production_ml", [])
        if any(_skill_matches_term(name, term) for term in production_ml):
            profile.experience.ai_years += skill_years

        embeddings_nlp = taxonomy.get("embeddings_nlp", [])
        if any(_skill_matches_term(name, term) for term in embeddings_nlp):
            profile.experience.ai_years += skill_years

    profile.experience.relevant_years = max(
        profile.experience.ai_years,
        profile.experience.ranking_years,
        profile.experience.retrieval_years,
    )
