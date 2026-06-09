from __future__ import annotations


def format_top_tier(
    title: str,
    years: float,
    strengths: list[str],
    concern: str | None,
) -> str:
    strength_text = _format_strengths(strengths, positive=True)
    base = (
        f"Strong fit as {title} with {years} yrs experience; "
        f"excels in {strength_text}."
    )
    if concern:
        return f"{base} {concern}"
    return base


def format_mid_tier(
    title: str,
    years: float,
    strengths: list[str],
    concern: str | None,
) -> str:
    strength_text = _format_strengths(strengths, positive=True)
    base = (
        f"Solid {title} candidate ({years} yrs) with {strength_text} strengths."
    )
    if concern:
        return f"{base} {concern}"
    return base


def format_lower_tier(
    title: str,
    years: float,
    strengths: list[str],
    concern: str | None,
) -> str:
    if len(strengths) >= 2:
        strength_text = f"{strengths[0]} but gaps in {strengths[1]}"
    elif strengths:
        strength_text = strengths[0]
    else:
        strength_text = "limited role alignment"
    base = f"Partial fit for {title} ({years} yrs); has {strength_text}."
    if concern:
        return f"{base} {concern}"
    return base


def format_concern(text: str) -> str:
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return ""
    first = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
    if first.endswith("."):
        return first
    return f"{first}."


def _format_strengths(strengths: list[str], positive: bool = True) -> str:
    if not strengths:
        return "relevant experience" if positive else "skill coverage"
    if len(strengths) == 1:
        return strengths[0]
    return f"{strengths[0]} and {strengths[1]}"
