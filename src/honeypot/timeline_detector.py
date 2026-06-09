from __future__ import annotations

from src.understanding.models import CandidateProfile
from src.understanding.timeline_validator import validate_timeline


def score_timeline(profile: CandidateProfile) -> float:
    validate_timeline(profile)
    score = 0.0
    if profile.flags.get("impossible_timeline", False):
        score = max(score, 0.8)
    if profile.flags.get("timeline_inconsistent", False):
        score = max(score, 0.5)
    if profile.experience.total_years > 40:
        score = max(score, 0.9)
    return score
