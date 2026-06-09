from __future__ import annotations

from src.understanding.models import CandidateProfile


def score_education_anomaly(profile: CandidateProfile) -> float:
    for entry in profile.education:
        start_year = entry.get("start_year")
        end_year = entry.get("end_year")
        if end_year is None:
            continue
        end = int(end_year)
        if end > 2030:
            return 0.8
        if start_year is not None and end < int(start_year):
            return 0.8
    return 0.0
