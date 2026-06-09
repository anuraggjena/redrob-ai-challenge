from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import orjson

from src.understanding.models import CandidateProfile, ExperienceStats


def _build_document_text(raw: dict[str, Any]) -> str:
    profile = raw.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        f"{profile.get('current_title', '')} @ {profile.get('current_company', '')}",
    ]
    for role in raw.get("career_history", []):
        parts.append(role.get("description", ""))
        parts.append(role.get("title", ""))
    for skill in raw.get("skills", []):
        parts.append(f"{skill.get('name', '')} ({skill.get('proficiency', '')})")
    return " | ".join(x for x in parts if x)


def parse_candidate(raw: dict[str, Any]) -> CandidateProfile:
    profile = raw.get("profile", {})
    exp = ExperienceStats(total_years=float(profile.get("years_of_experience") or 0))
    return CandidateProfile(
        candidate_id=raw["candidate_id"],
        raw=raw,
        headline=profile.get("headline", ""),
        summary=profile.get("summary", ""),
        current_title=profile.get("current_title", ""),
        current_company=profile.get("current_company", ""),
        location=profile.get("location", ""),
        country=profile.get("country", ""),
        experience=exp,
        career_roles=list(raw.get("career_history", [])),
        skills=list(raw.get("skills", [])),
        education=list(raw.get("education", [])),
        signals=dict(raw.get("redrob_signals", {})),
        document_text=_build_document_text(raw),
    )


def parse_jsonl_line(line: str) -> CandidateProfile:
    return parse_candidate(orjson.loads(line))


def load_candidates(path: str | Path) -> list[CandidateProfile]:
    path = Path(path)
    name = path.name.lower()

    if name.endswith(".jsonl.gz"):
        with gzip.open(path, "rb") as handle:
            return [
                parse_jsonl_line(raw_line.decode("utf-8"))
                for raw_line in handle
                if raw_line.strip()
            ]

    if name.endswith(".jsonl"):
        with path.open("rb") as handle:
            return [
                parse_jsonl_line(raw_line.decode("utf-8"))
                for raw_line in handle
                if raw_line.strip()
            ]

    if name.endswith(".json"):
        data = orjson.loads(path.read_bytes())
        if isinstance(data, list):
            return [parse_candidate(item) for item in data]
        return [parse_candidate(data)]

    raise ValueError(f"Unsupported candidate file format: {path}")
