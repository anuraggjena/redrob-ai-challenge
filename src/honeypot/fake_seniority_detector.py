from __future__ import annotations

from src.understanding.models import CandidateProfile

_SENIORITY_MARKERS = ("senior", "staff", "principal")


def score_fake_seniority(profile: CandidateProfile) -> float:
    title_lower = profile.current_title.lower()
    has_seniority_marker = any(marker in title_lower for marker in _SENIORITY_MARKERS)
    if has_seniority_marker and profile.experience.total_years < 3:
        return 0.8
    return 0.0
