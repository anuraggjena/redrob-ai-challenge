from __future__ import annotations

from typing import Protocol

from src.understanding.models import CandidateProfile


class HoneypotDetector(Protocol):
    name: str

    def score(self, profile: CandidateProfile) -> float:
        """Return honeypot suspicion score in [0, 1]."""
        ...
