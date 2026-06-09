from __future__ import annotations

import json
from pathlib import Path

from src.understanding.models import CandidateProfile

ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontology"

JUNIOR_KEYWORDS = ("junior", "associate", "intern", "entry", "graduate", "trainee")
SENIOR_KEYWORDS = ("senior", "staff", "principal", "lead", "head", "director", "vp", "chief")

COMPANY_SIZE_SCORES = {
    "1-10": 0.9,
    "11-50": 0.85,
    "51-200": 0.75,
    "201-500": 0.65,
    "501-1000": 0.55,
    "1001-5000": 0.45,
    "5001-10000": 0.35,
    "10001+": 0.25,
}


def _load_consulting() -> list[str]:
    data = json.loads((ONTOLOGY_DIR / "company_types.json").read_text(encoding="utf-8"))
    return data.get("consulting", [])


def _is_consulting(company: str, consulting: list[str]) -> bool:
    company_lower = company.lower().strip()
    return any(
        company_lower == name.lower() or name.lower() in company_lower for name in consulting
    )


def _title_seniority(title: str) -> float:
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in SENIOR_KEYWORDS):
        return 1.0
    if any(keyword in title_lower for keyword in JUNIOR_KEYWORDS):
        return 0.0
    return 0.5


def compute(profile: CandidateProfile) -> dict[str, float]:
    roles = profile.career_roles
    consulting = _load_consulting()
    career_stats = profile.career_stats

    total_months = sum(float(role.get("duration_months") or 0) for role in roles)
    role_count = len(roles)
    avg_tenure = total_months / role_count if role_count else 0.0

    consulting_months = sum(
        float(role.get("duration_months") or 0)
        for role in roles
        if _is_consulting(role.get("company", ""), consulting)
    )
    product_months = sum(
        float(role.get("duration_months") or 0)
        for role in roles
        if role.get("company_size", "") in {"1-10", "11-50", "51-200", "201-500"}
        or (
            role.get("industry", "") != "IT Services"
            and not _is_consulting(role.get("company", ""), consulting)
        )
    )

    industries = {role.get("industry", "") for role in roles if role.get("industry")}
    industry_diversity = len(industries) / role_count if role_count else 0.0

    current_size = profile.raw.get("profile", {}).get("current_company_size", "")
    current_company_size_score = COMPANY_SIZE_SCORES.get(current_size, 0.5)

    title_seniority = _title_seniority(profile.current_title)
    senior_flag = title_seniority >= 1.0
    junior_flag = title_seniority <= 0.0

    return {
        "career_growth_score": float(career_stats.get("title_progression_score", 0.0)),
        "job_hopping_score": float(career_stats.get("job_hopping", 0.0)),
        "consulting_only_flag": float(profile.flags.get("consulting_only_flag", False)),
        "avg_tenure_months": avg_tenure,
        "role_count": float(role_count),
        "senior_title_flag": float(senior_flag),
        "junior_title_flag": float(junior_flag),
        "product_company_career_ratio": product_months / max(total_months, 1.0),
        "consulting_years_ratio": consulting_months / max(total_months, 1.0),
        "industry_diversity": industry_diversity,
        "current_company_size_score": current_company_size_score,
        "title_seniority_score": title_seniority,
        "longest_tenure_months": float(
            max((role.get("duration_months") or 0) for role in roles) if roles else 0
        ),
        "career_span_years": total_months / 12.0,
    }
