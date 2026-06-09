from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvidenceNode:
    id: str
    type: str  # requirement, skill, role, behavioral, risk
    label: str
    value: str
    source_field: str
