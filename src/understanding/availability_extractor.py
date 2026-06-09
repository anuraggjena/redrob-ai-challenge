from __future__ import annotations

from pathlib import Path

from src.pipeline.config_loader import load_job_requirements
from src.understanding.models import CandidateProfile

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "job_requirements.yaml"

FLEXIBLE_WORK_MODES = frozenset({"remote", "hybrid", "flexible"})


def _location_fit(location: str, country: str, preferred_locations: list[str]) -> float:
    country_lower = country.lower().strip()
    location_lower = location.lower().strip()

    if country_lower != "india":
        return 0.3

    preferred_lower = {entry.lower() for entry in preferred_locations}
    if location_lower in preferred_lower:
        return 1.0

    for city in preferred_locations:
        city_lower = city.lower()
        if city_lower in location_lower or location_lower in city_lower:
            return 1.0

    return 0.5


def _notice_penalty(notice_days: float) -> float:
    if notice_days <= 30:
        return 0.0
    if notice_days >= 120:
        return 1.0
    return (notice_days - 30.0) / 90.0


def _open_to_work_score(open_flag: bool) -> float:
    return 1.0 if open_flag else 0.3


def _work_mode_flex(preferred_work_mode: str) -> float:
    mode = preferred_work_mode.lower().strip()
    return 1.0 if mode in FLEXIBLE_WORK_MODES else 0.3


def enrich_availability(profile: CandidateProfile) -> None:
    requirements = load_job_requirements(CONFIG_PATH)
    preferred_locations = requirements.get("location_preferred", [])
    signals = profile.signals

    location_fit = _location_fit(profile.location, profile.country, preferred_locations)
    profile.features["location_fit"] = location_fit

    notice_days = float(signals.get("notice_period_days") or 0.0)
    notice_penalty = _notice_penalty(notice_days)
    profile.features["notice_penalty"] = notice_penalty

    open_to_work = _open_to_work_score(bool(signals.get("open_to_work_flag")))
    profile.features["open_to_work"] = open_to_work

    work_mode = str(signals.get("preferred_work_mode") or "")
    work_mode_flex = _work_mode_flex(work_mode)
    profile.features["work_mode_flex"] = work_mode_flex

    availability_components = [
        location_fit,
        1.0 - notice_penalty,
        open_to_work,
        work_mode_flex,
    ]
    profile.features["availability_score"] = sum(availability_components) / len(
        availability_components
    )
