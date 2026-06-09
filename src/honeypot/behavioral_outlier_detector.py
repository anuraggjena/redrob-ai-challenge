from __future__ import annotations

from src.understanding.models import CandidateProfile


def score_behavioral_outlier(profile: CandidateProfile) -> float:
    signals = profile.signals
    completeness = float(signals.get("profile_completeness_score") or 0)
    response_rate = float(signals.get("recruiter_response_rate") or 0)
    saved_30d = int(signals.get("saved_by_recruiters_30d") or 0)

    if completeness > 90 and response_rate < 0.05 and saved_30d == 0:
        return 0.7
    return 0.0
