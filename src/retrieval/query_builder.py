from __future__ import annotations

from pathlib import Path

from src.pipeline.config_loader import load_job_requirements


def _tokenize_phrases(phrases: list[str]) -> list[str]:
    tokens: list[str] = []
    for phrase in phrases:
        tokens.extend(phrase.lower().split())
    return tokens


def build_query_from_jd(config_path: Path) -> tuple[str, list[str]]:
    """Build dense query text and BM25 token list from job requirements YAML."""
    req = load_job_requirements(config_path)

    dense_parts: list[str] = [
        req.get("role_title", ""),
        f"experience years {req.get('experience_range', [])}",
    ]
    for key in ("required_skills", "desired_skills", "positive_signals", "negative_signals"):
        dense_parts.extend(str(item) for item in req.get(key, []))

    dense_query_text = " ".join(part for part in dense_parts if part)

    bm25_tokens: list[str] = []
    for key in ("required_skills", "positive_signals", "negative_signals"):
        bm25_tokens.extend(_tokenize_phrases(req.get(key, [])))

    return dense_query_text, bm25_tokens
