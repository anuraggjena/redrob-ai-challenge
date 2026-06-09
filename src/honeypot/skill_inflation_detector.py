from __future__ import annotations

from src.understanding.models import CandidateProfile


def score_skill_inflation(profile: CandidateProfile) -> float:
    skills = profile.skills
    if not skills:
        return 0.0
    expert_zero = sum(
        1
        for skill in skills
        if str(skill.get("proficiency", "")).lower() == "expert"
        and float(skill.get("duration_months") or 0) == 0
    )
    ratio = expert_zero / len(skills)
    if ratio <= 0.3:
        return min(ratio / 0.3, 1.0) * 0.3
    return min(0.3 + (ratio - 0.3) / 0.7 * 0.7, 1.0)
