from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable, Dict, List

BiasHook = Callable[[Dict[str, Any]], Dict[str, Any]]


def discover_bias_hooks() -> Dict[str, BiasHook]:
    hooks: Dict[str, BiasHook] = {}
    for ep in entry_points(group="mdmp.bias_hooks"):
        loaded: Any | None = None
        try:
            loaded = ep.load()
        except Exception:
            loaded = None
        if callable(loaded):
            hooks[str(ep.name).strip()] = loaded
    return hooks


def run_bias_hooks(report_payload: Dict[str, Any]) -> Dict[str, Any]:
    hooks = discover_bias_hooks()
    results: List[Dict[str, Any]] = []
    for name, hook in hooks.items():
        try:
            output = hook(report_payload)
            if not isinstance(output, dict):
                output = {"status": "invalid_output", "detail": "hook did not return a mapping"}
            results.append({"hook": name, "ok": True, "result": output})
        except Exception as exc:
            results.append({"hook": name, "ok": False, "error": str(exc)})
    return {
        "hook_count": len(hooks),
        "results": results,
    }
