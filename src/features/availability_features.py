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


def compute(profile: CandidateProfile) -> dict[str, float]:
    requirements = load_job_requirements(CONFIG_PATH)
    preferred_locations = requirements.get("location_preferred", [])
    signals = profile.signals
    features = profile.features

    location_fit = float(
        features.get("location_fit")
        or _location_fit(profile.location, profile.country, preferred_locations)
    )

    notice_days = float(signals.get("notice_period_days") or 0.0)
    notice_penalty = float(features.get("notice_penalty") or _notice_penalty(notice_days))
    notice_period_days_norm = min(notice_days / 120.0, 1.0)

    open_to_work = float(
        features.get("open_to_work")
        or _open_to_work_score(bool(signals.get("open_to_work_flag")))
    )

    work_mode_flex = float(
        features.get("work_mode_flex")
        or _work_mode_flex(str(signals.get("preferred_work_mode") or ""))
    )

    willing_to_relocate = float(bool(signals.get("willing_to_relocate")))

    salary_range = signals.get("expected_salary_range_inr_lpa") or {}
    salary_min = float(salary_range.get("min") or 0.0)
    salary_max = float(salary_range.get("max") or salary_min)
    salary_mid = (salary_min + salary_max) / 2.0
    salary_range_mid_norm = min(salary_mid / 50.0, 1.0)

    availability_components = [
        location_fit,
        1.0 - notice_penalty,
        open_to_work,
        work_mode_flex,
        1.0 if willing_to_relocate else 0.5,
    ]
    availability_score = sum(availability_components) / len(availability_components)

    return {
        "location_fit": location_fit,
        "notice_penalty": notice_penalty,
        "notice_period_days_norm": notice_period_days_norm,
        "open_to_work_score": open_to_work,
        "work_mode_flex": work_mode_flex,
        "availability_score": availability_score,
        "willing_to_relocate_flag": willing_to_relocate,
        "salary_range_mid_norm": salary_range_mid_norm,
        "relocation_availability_boost": 1.0 if willing_to_relocate else 0.0,
    }
