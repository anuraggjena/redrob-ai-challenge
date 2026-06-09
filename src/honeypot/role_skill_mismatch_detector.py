from __future__ import annotations

from src.understanding.models import CandidateProfile


def score_role_skill_mismatch(profile: CandidateProfile) -> float:
    if profile.flags.get("role_skill_mismatch", False):
        return 0.6
    return 0.0
