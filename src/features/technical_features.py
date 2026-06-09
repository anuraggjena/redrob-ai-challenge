from __future__ import annotations

import math

from src.ontology.loader import load_ontology_json
from src.understanding.models import CandidateProfile

PROFICIENCY_WEIGHTS = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.75,
    "expert": 1.0,
}


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


def _depth_score(skills: list[dict], terms: list[str]) -> float:
    if not terms:
        return 0.0
    matched = [
        _skill_evidence(skill)
        for skill in skills
        if _skill_in_taxonomy(skill.get("name", ""), terms)
    ]
    return min(sum(matched) / 10.0, 1.0) if matched else 0.0


def compute(profile: CandidateProfile) -> dict[str, float]:
    taxonomy = load_ontology_json("skills_taxonomy.json")
    title_classifier = load_ontology_json("title_classifier.json")

    retrieval_ranking_terms = taxonomy.get("retrieval_ranking", [])
    embeddings_terms = taxonomy.get("embeddings_nlp", [])
    production_terms = taxonomy.get("production_ml", [])
    negative_terms = taxonomy.get("negative_only", [])

    retrieval_ranking_evidence = 0.0
    embeddings_evidence = 0.0
    production_evidence = 0.0
    expert_count = 0
    zero_duration_expert = 0
    endorsement_total = 0.0
    langchain_count = 0
    cv_only = 0
    speech_only = 0
    robotics_only = 0

    vector_db_flag = False
    python_flag = False
    eval_flag = False
    lora_flag = False

    for skill in profile.skills:
        name = skill.get("name", "")
        if not name:
            continue

        evidence = _skill_evidence(skill)
        endorsement_total += float(skill.get("endorsements") or 0)
        proficiency = str(skill.get("proficiency", "")).lower()

        if proficiency == "expert":
            expert_count += 1
            if float(skill.get("duration_months") or 0) == 0:
                zero_duration_expert += 1

        name_lower = name.lower()
        if "langchain" in name_lower:
            langchain_count += 1
        if any(token in name_lower for token in ("opencv", "image classification", "computer vision", "cv")):
            cv_only += 1
        if any(token in name_lower for token in ("speech", "tts", "asr", "voice")):
            speech_only += 1
        if any(token in name_lower for token in ("robot", "ros", "robotics")):
            robotics_only += 1
        if any(token in name_lower for token in ("faiss", "milvus", "pinecone", "weaviate", "qdrant", "vector")):
            vector_db_flag = True
        if "python" in name_lower:
            python_flag = True
        if any(token in name_lower for token in ("eval", "evaluation", "benchmark", "ndcg", "mrr")):
            eval_flag = True
        if "lora" in name_lower:
            lora_flag = True

        if _skill_in_taxonomy(name, retrieval_ranking_terms):
            retrieval_ranking_evidence += evidence
        if _skill_in_taxonomy(name, embeddings_terms):
            embeddings_evidence += evidence
        if _skill_in_taxonomy(name, production_terms):
            production_evidence += evidence

    skill_count = len(profile.skills)
    expert_ratio = expert_count / skill_count if skill_count else 0.0

    non_technical = title_classifier.get("non_technical", [])
    title_lower = profile.current_title.lower().strip()
    mismatch = any(title_lower == entry.lower() for entry in non_technical) and skill_count > 5

    taxonomy_skill_count = sum(
        1
        for skill in profile.skills
        if _skill_in_taxonomy(skill.get("name", ""), retrieval_ranking_terms)
        or _skill_in_taxonomy(skill.get("name", ""), embeddings_terms)
    )

    return {
        "skill_evidence_retrieval_ranking": retrieval_ranking_evidence,
        "skill_evidence_embeddings": embeddings_evidence,
        "skill_evidence_production": production_evidence,
        "skill_count": float(skill_count),
        "expert_skill_count": float(expert_count),
        "expert_skill_ratio": expert_ratio,
        "zero_duration_expert_count": float(zero_duration_expert),
        "skill_endorsement_total": endorsement_total,
        "vector_db_skill_flag": float(vector_db_flag),
        "python_skill_flag": float(python_flag),
        "eval_skill_flag": float(eval_flag),
        "lora_skill_flag": float(lora_flag),
        "langchain_skill_penalty": min(langchain_count / max(skill_count, 1), 1.0),
        "cv_only_penalty": min(cv_only / max(skill_count, 1), 1.0),
        "speech_only_penalty": min(speech_only / max(skill_count, 1), 1.0),
        "robotics_only_penalty": min(robotics_only / max(skill_count, 1), 1.0),
        "nlp_skill_depth": _depth_score(profile.skills, embeddings_terms),
        "ml_skill_depth": _depth_score(profile.skills, production_terms),
        "infra_skill_depth": _depth_score(
            profile.skills,
            [term for term in retrieval_ranking_terms if term in {"faiss", "elasticsearch", "milvus"}],
        ),
        "role_skill_mismatch_flag": float(mismatch or profile.flags.get("role_skill_mismatch", False)),
        "taxonomy_skill_count": float(taxonomy_skill_count),
        "negative_only_skill_penalty": min(
            sum(
                1
                for skill in profile.skills
                if _skill_in_taxonomy(skill.get("name", ""), negative_terms)
            )
            / max(skill_count, 1),
            1.0,
        ),
    }
