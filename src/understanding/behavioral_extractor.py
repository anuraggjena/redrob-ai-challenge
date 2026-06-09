from __future__ import annotations

from datetime import datetime

from src.understanding.models import CandidateProfile

REFERENCE_DATE = datetime(2026, 6, 8)


def _normalize_count(value: float, divisor: float) -> float:
    return min(float(value) / divisor, 1.0)


def _activity_recency(days_since_active: float) -> float:
    return max(0.0, 1.0 - days_since_active / 180.0)


def enrich_behavioral(profile: CandidateProfile) -> None:
    signals = profile.signals
    engagement_components: list[float] = []

    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    profile.features["behavioral_recruiter_response_rate"] = response_rate
    engagement_components.append(response_rate)

    completeness = float(signals.get("profile_completeness_score") or 0.0) / 100.0
    profile.features["behavioral_profile_completeness_score"] = completeness
    engagement_components.append(completeness)

    interview_rate = float(signals.get("interview_completion_rate") or 0.0)
    profile.features["behavioral_interview_completion_rate"] = interview_rate
    engagement_components.append(interview_rate)

    github_score = signals.get("github_activity_score")
    if github_score is not None and float(github_score) != -1:
        normalized_github = float(github_score) / 100.0
        profile.features["behavioral_github_activity_score"] = normalized_github
        engagement_components.append(normalized_github)

    saved_count = float(signals.get("saved_by_recruiters_30d") or 0.0)
    saved_normalized = _normalize_count(saved_count, 10.0)
    profile.features["behavioral_saved_by_recruiters_30d"] = saved_normalized
    engagement_components.append(saved_normalized)

    search_count = float(signals.get("search_appearance_30d") or 0.0)
    search_normalized = _normalize_count(search_count, 20.0)
    profile.features["behavioral_search_appearance_30d"] = search_normalized
    engagement_components.append(search_normalized)

    last_active = signals.get("last_active_date")
    if last_active:
        last_active_date = datetime.strptime(str(last_active)[:10], "%Y-%m-%d")
        days_since_active = (REFERENCE_DATE - last_active_date).days
        profile.features["behavioral_days_since_active"] = float(days_since_active)
        engagement_components.append(_activity_recency(days_since_active))

    offer_rate = signals.get("offer_acceptance_rate")
    if offer_rate is not None and float(offer_rate) != -1:
        profile.features["behavioral_offer_acceptance_rate"] = float(offer_rate)
        engagement_components.append(float(offer_rate))

    if engagement_components:
        profile.features["behavioral_engagement_score"] = sum(engagement_components) / len(
            engagement_components
        )
    else:
        profile.features["behavioral_engagement_score"] = 0.0
