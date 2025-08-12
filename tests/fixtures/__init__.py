"""Fixture utilities for tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

_FIXTURE_DIR = Path(__file__).resolve().parent


def load_fixture(name: str) -> Tuple[Dict, Dict]:
    """Load a single fixture by name.

    Parameters
    ----------
    name: str
        Name of the JSON file without extension.

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple of (inputs_dict, expected_dict).
    """
    path = _FIXTURE_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["inputs"], data["expected"]


def load_all() -> Iterable[Tuple[str, Tuple[Dict, Dict]]]:
    """Yield all case fixtures in alphabetical order."""
    for case_file in sorted(_FIXTURE_DIR.glob("case*.json")):
        name = case_file.stem
        yield name, load_fixture(name)
