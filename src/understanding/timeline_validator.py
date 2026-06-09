from __future__ import annotations

import re
from datetime import datetime

from src.understanding.models import CandidateProfile

_FOUNDING_YEAR_RE = re.compile(
    r"(?:founded|founding|established)[\w\s]*?(\d{4})",
    re.IGNORECASE,
)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value[:10], "%Y-%m-%d")


def _overlap_months(
    start_a: datetime,
    end_a: datetime,
    start_b: datetime,
    end_b: datetime,
) -> int:
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    if overlap_start >= overlap_end:
        return 0
    return (overlap_end.year - overlap_start.year) * 12 + (
        overlap_end.month - overlap_start.month
    )


def validate_timeline(profile: CandidateProfile) -> None:
    roles = profile.career_roles
    inconsistent = False
    now = datetime.now()

    for role in roles:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        if start and end and end < start:
            inconsistent = True

    for index, role_a in enumerate(roles):
        start_a = _parse_date(role_a.get("start_date"))
        end_a = _parse_date(role_a.get("end_date")) or now
        if not start_a:
            continue
        for role_b in roles[index + 1 :]:
            start_b = _parse_date(role_b.get("start_date"))
            end_b = _parse_date(role_b.get("end_date")) or now
            if not start_b:
                continue
            if _overlap_months(start_a, end_a, start_b, end_b) > 3:
                inconsistent = True

    total_months = sum(role.get("duration_months") or 0 for role in roles)
    claimed_months = profile.experience.total_years * 12
    if abs(total_months - claimed_months) > 24:
        inconsistent = True

    profile.flags["timeline_inconsistent"] = inconsistent

    impossible = False
    for role in roles:
        description = role.get("description", "")
        start = _parse_date(role.get("start_date"))
        for match in _FOUNDING_YEAR_RE.finditer(description):
            founding_year = int(match.group(1))
            if start and founding_year > start.year:
                impossible = True

    profile.flags["impossible_timeline"] = impossible
