from __future__ import annotations

from src.understanding.models import CandidateProfile


def _keyword_stuffing_score(profile: CandidateProfile) -> float:
    skills = profile.skills
    if not skills:
        return 0.0
    expert_zero = sum(
        1
        for skill in skills
        if str(skill.get("proficiency", "")).lower() == "expert"
        and float(skill.get("duration_months") or 0) == 0
    )
    return expert_zero / len(skills)


def _endorsement_anomaly(profile: CandidateProfile) -> float:
    if not profile.skills:
        return 0.0
    endorsements = [float(skill.get("endorsements") or 0) for skill in profile.skills]
    avg = sum(endorsements) / len(endorsements)
    if avg <= 0:
        return 0.0
    spikes = sum(1 for value in endorsements if value > avg * 3)
    return min(spikes / len(endorsements), 1.0)


def _skill_count_anomaly(profile: CandidateProfile) -> float:
    count = len(profile.skills)
    if count <= 20:
        return 0.0
    return min((count - 20) / 30.0, 1.0)


def compute(profile: CandidateProfile) -> dict[str, float]:
    timeline_inconsistent = float(profile.flags.get("timeline_inconsistent", False))
    impossible_timeline = float(profile.flags.get("impossible_timeline", False))
    keyword_stuffing = _keyword_stuffing_score(profile)

    inconsistency_signals = [
        timeline_inconsistent,
        impossible_timeline,
        keyword_stuffing,
        float(profile.flags.get("role_skill_mismatch", False)),
    ]
    inconsistency_score = sum(inconsistency_signals) / len(inconsistency_signals)
    profile_consistency_score = profile.features.get(
        "profile_consistency_score", 1.0 - inconsistency_score
    )
    honeypot_probability = profile.features.get("honeypot_probability", 0.0)
    honeypot_penalty = profile.features.get(
        "honeypot_penalty", honeypot_probability
    )
    trustworthiness_score = profile.features.get(
        "trustworthiness", profile_consistency_score
    )

    return {
        "timeline_inconsistent_flag": timeline_inconsistent,
        "impossible_timeline_flag": impossible_timeline,
        "keyword_stuffing_score": keyword_stuffing,
        "profile_consistency_score": profile_consistency_score,
        "trustworthiness_score": trustworthiness_score,
        "honeypot_probability": honeypot_probability,
        "honeypot_penalty": honeypot_penalty,
        "endorsement_anomaly": _endorsement_anomaly(profile),
        "skill_count_anomaly": _skill_count_anomaly(profile),
        "title_skill_mismatch_flag": float(profile.flags.get("role_skill_mismatch", False)),
        "expert_zero_duration_ratio": _keyword_stuffing_score(profile),
        "flag_risk_score": inconsistency_score,
        "eval_experience_flag": float(profile.flags.get("eval_experience", False)),
        "trust_signal_count": sum(1 for signal in inconsistency_signals if signal > 0),
        "profile_verification_score": (
            float(bool(profile.signals.get("verified_email")))
            + float(bool(profile.signals.get("verified_phone")))
        )
        / 2.0,
    }
