from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=None)
def load_job_requirements(path: Path) -> dict[str, Any]:
    with open(Path(path).resolve(), encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def load_json_ontology(path: Path) -> dict[str, Any]:
    with open(Path(path).resolve(), encoding="utf-8") as f:
        return json.load(f)
