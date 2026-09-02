#!/usr/bin/env python3
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "iints"


@dataclass(frozen=True)
class ImportEdge:
    source: str
    imported: str


FORBIDDEN_PREFIXES: dict[str, tuple[str, ...]] = {
    "core": (
        "iints.analysis",
        "iints.research",
        "iints.cli",
        "iints.live_patient",
        "iints.jetson",
        "iints.visualization",
    ),
    "data": (
        "iints.analysis",
        "iints.research",
        "iints.ai",
        "iints.cli",
        "iints.live_patient",
        "iints.jetson",
        "iints.visualization",
    ),
    "validation": (
        "iints.cli",
        "iints.live_patient",
        "iints.jetson",
        "iints.visualization",
    ),
    "analysis": (
        "iints.cli",
        "iints.live_patient",
        "iints.jetson",
    ),
    "research": (
        "iints.cli",
        "iints.live_patient",
        "iints.jetson",
    ),
    "ai": (
        "iints.cli",
        "iints.live_patient",
        "iints.jetson",
    ),
}


# Keep this list short. Every exception must point to a tracked refactor issue.
ALLOWED_LEGACY_EDGES: set[ImportEdge] = set()


def _iter_python_files() -> Iterable[Path]:
    for path in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        # macOS AppleDouble sidecars ("._module.py") appear whenever the tree is
        # copied to or from a non-HFS filesystem. They are binary resource forks,
        # not source, and ast.parse used to raise UnicodeDecodeError on them --
        # which aborted the whole scan, so no boundary violation was ever
        # reported. Skip them instead of failing the check.
        if path.name.startswith("._"):
            continue
        yield path


def _imports_from(path: Path) -> set[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError) as exc:
        # Name the file: an anonymous decode error here previously looked like a
        # tooling bug rather than an unreadable source file.
        raise RuntimeError(f"cannot parse {path} for boundary analysis: {exc}") from exc
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("iints."):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("iints."):
                imports.add(node.module)

    return imports


def _owner_layer(relative_path: Path) -> str | None:
    if not relative_path.parts:
        return None
    first = relative_path.parts[0]
    return first if first in FORBIDDEN_PREFIXES else None


def _is_forbidden(layer: str, imported: str) -> bool:
    return any(
        imported == prefix or imported.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_PREFIXES[layer]
    )


def find_violations() -> list[ImportEdge]:
    violations: list[ImportEdge] = []

    for path in _iter_python_files():
        relative_path = path.relative_to(SRC_ROOT)
        layer = _owner_layer(relative_path)
        if layer is None:
            continue

        source = relative_path.as_posix()
        for imported in sorted(_imports_from(path)):
            edge = ImportEdge(source, imported)
            if edge in ALLOWED_LEGACY_EDGES:
                continue
            if _is_forbidden(layer, imported):
                violations.append(edge)

    return violations


def main() -> int:
    violations = find_violations()
    if not violations:
        print("Architecture boundary checks passed.")
        return 0

    print("Architecture boundary checks failed:")
    for edge in violations:
        print(f"- {edge.source} imports {edge.imported}")
    print("\nMove shared contracts down, or add a reviewed legacy exception.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
