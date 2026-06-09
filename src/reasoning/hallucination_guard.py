from __future__ import annotations

import re

from src.understanding.models import CandidateProfile

_YEARS_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:yrs?|years?)\b",
    re.IGNORECASE,
)
_QUOTED_SKILL_PATTERN = re.compile(r'"([^"]+)"')
_AT_EMPLOYER_PATTERN = re.compile(
    r"\b(?:at|with|from)\s+([A-Z][A-Za-z0-9&][A-Za-z0-9&\s.-]{1,40})\b"
)


def validate_reasoning(reasoning: str, profile: CandidateProfile) -> list[str]:
    violations: list[str] = []

    profile_years = profile.experience.relevant_years or profile.experience.total_years
    for match in _YEARS_PATTERN.finditer(reasoning):
        mentioned = float(match.group(1))
        if abs(mentioned - profile_years) > 1.0:
            violations.append(
                f"years mismatch: mentioned {mentioned}, profile has {profile_years:.1f}"
            )

    skill_names = {str(skill.get("name", "")).strip().lower() for skill in profile.skills}
    for match in _QUOTED_SKILL_PATTERN.finditer(reasoning):
        quoted = match.group(1).strip()
        if quoted.lower() not in skill_names:
            violations.append(f"unknown quoted skill: {quoted}")

    allowed_employers = {
        str(role.get("company", "")).strip().lower()
        for role in profile.career_roles
        if role.get("company")
    }
    if profile.current_company:
        allowed_employers.add(profile.current_company.strip().lower())

    for match in _AT_EMPLOYER_PATTERN.finditer(reasoning):
        employer = match.group(1).strip().rstrip(".,;")
        if employer.lower() not in allowed_employers:
            violations.append(f"unknown employer: {employer}")

    return violations
