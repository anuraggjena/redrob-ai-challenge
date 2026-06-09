from __future__ import annotations

from src.understanding.models import CandidateProfile


def enrich_profile(profile: CandidateProfile) -> CandidateProfile:
    from src.understanding.availability_extractor import enrich_availability
    from src.understanding.behavioral_extractor import enrich_behavioral
    from src.understanding.career_extractor import enrich_career
    from src.understanding.experience_extractor import enrich_experience
    from src.understanding.technical_extractor import enrich_technical
    from src.understanding.timeline_validator import validate_timeline

    enrich_experience(profile)
    enrich_career(profile)
    validate_timeline(profile)
    enrich_technical(profile)
    enrich_behavioral(profile)
    enrich_availability(profile)
    return profile
