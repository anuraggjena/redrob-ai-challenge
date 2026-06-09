from __future__ import annotations


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_honeypot_probability(scores: dict[str, float]) -> float:
    prob = 1.0
    for score in scores.values():
        prob *= 1.0 - _clamp(score)
    return 1.0 - prob


def compute_trustworthiness(
    honeypot_probability: float,
    scores: dict[str, float],
) -> float:
    timeline = _clamp(scores.get("timeline", 0.0))
    keyword_stuffer = _clamp(scores.get("keyword_stuffer", 0.0))
    trust = (
        1.0
        - 0.5 * _clamp(honeypot_probability)
        - 0.3 * timeline
        - 0.2 * keyword_stuffer
    )
    return _clamp(trust)


def compute_honeypot_penalty(
    honeypot_probability: float,
    scores: dict[str, float],
) -> float:
    keyword_stuffer = _clamp(scores.get("keyword_stuffer", 0.0))
    return _clamp(0.5 * _clamp(honeypot_probability) + 0.3 * keyword_stuffer)
