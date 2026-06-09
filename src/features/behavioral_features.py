from __future__ import annotations

from datetime import datetime

from src.understanding.models import CandidateProfile

REFERENCE_DATE = datetime(2026, 6, 8)


def _normalize_count(value: float, divisor: float) -> float:
    return min(float(value) / divisor, 1.0)


def _activity_recency(days_since_active: float) -> float:
    return max(0.0, 1.0 - days_since_active / 180.0)


def compute(profile: CandidateProfile) -> dict[str, float]:
    signals = profile.signals
    features = profile.features

    response_rate = float(
        features.get("behavioral_recruiter_response_rate")
        or signals.get("recruiter_response_rate")
        or 0.0
    )
    completeness = float(
        features.get("behavioral_profile_completeness_score")
        or (float(signals.get("profile_completeness_score") or 0.0) / 100.0)
    )
    interview_rate = float(
        features.get("behavioral_interview_completion_rate")
        or signals.get("interview_completion_rate")
        or 0.0
    )

    github_norm = features.get("behavioral_github_activity_score")
    if github_norm is None:
        github_score = signals.get("github_activity_score")
        if github_score is not None and float(github_score) != -1:
            github_norm = float(github_score) / 100.0
        else:
            github_norm = 0.0
    else:
        github_norm = float(github_norm)

    saved_norm = float(
        features.get("behavioral_saved_by_recruiters_30d")
        or _normalize_count(float(signals.get("saved_by_recruiters_30d") or 0.0), 10.0)
    )
    search_norm = float(
        features.get("behavioral_search_appearance_30d")
        or _normalize_count(float(signals.get("search_appearance_30d") or 0.0), 20.0)
    )

    days_since_active = float(features.get("behavioral_days_since_active", 0.0))
    if not days_since_active and signals.get("last_active_date"):
        last_active_date = datetime.strptime(str(signals["last_active_date"])[:10], "%Y-%m-%d")
        days_since_active = float((REFERENCE_DATE - last_active_date).days)

    days_since_active_norm = _activity_recency(days_since_active)

    engagement = float(features.get("behavioral_engagement_score", 0.0))
    if not engagement:
        components = [
            value
            for value in (
                response_rate,
                completeness,
                interview_rate,
                github_norm if github_norm else None,
                saved_norm,
                search_norm,
                days_since_active_norm,
            )
            if value is not None
        ]
        engagement = sum(components) / len(components) if components else 0.0

    recruiter_interest = (saved_norm + search_norm) / 2.0

    return {
        "engagement_score": engagement,
        "responsiveness_score": response_rate,
        "recruiter_interest": recruiter_interest,
        "profile_completeness_norm": completeness,
        "interview_completion": interview_rate,
        "github_activity_norm": github_norm,
        "saved_by_recruiters_norm": saved_norm,
        "search_appearance_norm": search_norm,
        "verified_email_flag": float(bool(signals.get("verified_email"))),
        "verified_phone_flag": float(bool(signals.get("verified_phone"))),
        "linkedin_connected_flag": float(bool(signals.get("linkedin_connected"))),
        "days_since_active_norm": days_since_active_norm,
        "connection_count_norm": _normalize_count(float(signals.get("connection_count") or 0.0), 500.0),
        "applications_submitted_norm": _normalize_count(
            float(signals.get("applications_submitted_30d") or 0.0), 10.0
        ),
    }
