from __future__ import annotations

from src.honeypot.behavioral_outlier_detector import score_behavioral_outlier
from src.honeypot.education_anomaly_detector import score_education_anomaly
from src.honeypot.fake_seniority_detector import score_fake_seniority
from src.honeypot.fusion import (
    compute_honeypot_penalty,
    compute_honeypot_probability,
    compute_trustworthiness,
)
from src.honeypot.keyword_stuffer_detector import score_keyword_stuffer
from src.honeypot.role_skill_mismatch_detector import score_role_skill_mismatch
from src.honeypot.skill_inflation_detector import score_skill_inflation
from src.honeypot.timeline_detector import score_timeline
from src.understanding.models import CandidateProfile


def run_honeypot_detection(profile: CandidateProfile) -> dict[str, float]:
    scores = {
        "timeline": score_timeline(profile),
        "skill_inflation": score_skill_inflation(profile),
        "keyword_stuffer": score_keyword_stuffer(profile),
        "fake_seniority": score_fake_seniority(profile),
        "behavioral_outlier": score_behavioral_outlier(profile),
        "education_anomaly": score_education_anomaly(profile),
        "role_skill_mismatch": score_role_skill_mismatch(profile),
    }

    honeypot_probability = compute_honeypot_probability(scores)
    trustworthiness = compute_trustworthiness(honeypot_probability, scores)
    honeypot_penalty = compute_honeypot_penalty(honeypot_probability, scores)
    profile_consistency_score = trustworthiness

    result = {
        "honeypot_probability": honeypot_probability,
        "trustworthiness": trustworthiness,
        "profile_consistency_score": profile_consistency_score,
        "honeypot_penalty": honeypot_penalty,
    }
    profile.features.update(result)
    return result
