#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iints-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "iints-cache"))


def _current_surface() -> Dict[str, List[str]]:
    import iints

    symbols = sorted(set(getattr(iints, "__all__", [])))
    return {"module": "iints", "symbols": symbols}


def _load_baseline(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("API baseline must be a JSON object")
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise ValueError("API baseline is missing 'symbols' list")
    return {"module": str(payload.get("module", "iints")), "symbols": [str(item) for item in symbols]}


def _as_set(values: List[str]) -> Set[str]:
    return set(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check public API surface stability.")
    parser.add_argument("--baseline", type=Path, default=Path("tools/ci/api_surface_baseline.json"))
    parser.add_argument("--write-baseline", action="store_true", help="Write/update the baseline and exit.")
    parser.add_argument(
        "--fail-on-additions",
        action="store_true",
        help="Treat newly added public symbols as failure.",
    )
    args = parser.parse_args()

    current = _current_surface()
    if args.write_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"Wrote API baseline: {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"Missing API baseline: {args.baseline}", file=sys.stderr)
        print("Run with --write-baseline to create it.", file=sys.stderr)
        return 1

    baseline = _load_baseline(args.baseline)
    current_set = _as_set(current["symbols"])
    baseline_set = _as_set(baseline["symbols"])

    removed = sorted(baseline_set - current_set)
    added = sorted(current_set - baseline_set)

    if removed:
        print("API surface regression detected (removed symbols):", file=sys.stderr)
        for symbol in removed:
            print(f"- {symbol}", file=sys.stderr)
        return 1

    if added:
        print("API surface additions detected:")
        for symbol in added:
            print(f"- {symbol}")
        if args.fail_on_additions:
            return 1

    print(f"API surface check passed ({len(current_set)} symbols).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
