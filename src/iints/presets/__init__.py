"""Built-in clinic-safe presets."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List


def load_presets() -> List[Dict[str, Any]]:
    if sys.version_info >= (3, 9):
        from importlib.resources import files
        content = files("iints.presets").joinpath("presets.json").read_text()
    else:
        from importlib import resources
        content = resources.read_text("iints.presets", "presets.json")
    return json.loads(content)


def get_preset(name: str) -> Dict[str, Any]:
    presets = load_presets()
    for preset in presets:
        if preset.get("name") == name:
            return preset
    raise KeyError(name)


__all__ = ["load_presets", "get_preset"]
