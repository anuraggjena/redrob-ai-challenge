from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_job_requirements(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_ontology(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
