from __future__ import annotations

import json
from pathlib import Path

from src.understanding.models import CandidateProfile

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"

JUNIOR_KEYWORDS = ("junior", "associate", "intern", "entry", "graduate", "trainee")
SENIOR_KEYWORDS = ("senior", "staff", "principal", "lead", "head", "director", "vp", "chief")


def _load_json(name: str) -> dict:
    return json.loads((ONTOLOGY_DIR / name).read_text(encoding="utf-8"))


def _is_consulting_company(company: str, consulting_list: list[str]) -> bool:
    company_lower = company.lower().strip()
    return any(
        company_lower == name.lower() or name.lower() in company_lower
        for name in consulting_list
    )


def _title_level(title: str) -> float:
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in SENIOR_KEYWORDS):
        return 1.0
    if any(keyword in title_lower for keyword in JUNIOR_KEYWORDS):
        return 0.0
    return 0.5


def enrich_career(profile: CandidateProfile) -> None:
    company_types = _load_json("company_types.json")
    consulting = company_types.get("consulting", [])
    roles = profile.career_roles

    consulting_only = bool(roles) and all(
        _is_consulting_company(role.get("company", ""), consulting) for role in roles
    )
    profile.flags["consulting_only_flag"] = consulting_only

    sorted_roles = sorted(roles, key=lambda role: role.get("start_date", ""))
    levels = [_title_level(role.get("title", "")) for role in sorted_roles]
    if len(levels) <= 1:
        progression = levels[0] if levels else 0.0
    else:
        improvements = sum(
            1 for index in range(1, len(levels)) if levels[index] > levels[index - 1]
        )
        progression = improvements / (len(levels) - 1)

    job_hopping = sum(1 for role in roles if (role.get("duration_months") or 0) < 12)

    profile.career_stats = {
        "title_progression_score": progression,
        "job_hopping": job_hopping,
        "job_count": len(roles),
    }
    profile.features["title_progression_score"] = progression
    profile.features["job_hopping"] = float(job_hopping)
    profile.features["consulting_only_flag"] = float(consulting_only)
