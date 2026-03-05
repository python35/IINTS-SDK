#!/usr/bin/env python3
from __future__ import annotations

import re
import shlex
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import click
import typer

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iints-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "iints-cache"))

from iints.cli.cli import app


DOC_FILES = [
    Path("README.md"),
    Path("docs/COMPREHENSIVE_GUIDE.md"),
    Path("docs/TECHNICAL_README.md"),
    Path("research/README.md"),
]


def _iter_shell_commands(markdown: str) -> Iterable[str]:
    block_pattern = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)
    for match in block_pattern.finditer(markdown):
        lines = match.group(1).splitlines()
        merged: List[str] = []
        current = ""
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("\\"):
                current += line[:-1].rstrip() + " "
                continue
            current += line
            merged.append(current.strip())
            current = ""
        if current:
            merged.append(current.strip())
        for command in merged:
            yield command


def _extract_iints_help_target(command: str) -> Tuple[str, ...] | None:
    if command.startswith("$ "):
        command = command[2:].strip()
    if not command.startswith("iints "):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens or tokens[0] != "iints":
        return None
    path_tokens: List[str] = []
    for token in tokens[1:]:
        if token.startswith("-"):
            break
        path_tokens.append(token)
    return tuple(path_tokens)


def _command_exists(path_tokens: Tuple[str, ...]) -> Tuple[bool, str]:
    root = typer.main.get_command(app)
    command: click.Command = root
    for token in path_tokens:
        if not isinstance(command, click.Group):
            return False, f"'{token}' is not available under a leaf command"
        context = click.Context(command)
        next_command = command.get_command(context, token)
        if next_command is None:
            return False, f"subcommand '{token}' not found"
        command = next_command
    return True, ""


def main() -> int:
    targets: Set[Tuple[str, ...]] = set()
    for doc_path in DOC_FILES:
        if not doc_path.exists():
            continue
        content = doc_path.read_text(encoding="utf-8")
        for command in _iter_shell_commands(content):
            target = _extract_iints_help_target(command)
            if target is not None:
                targets.add(target)

    failures: List[str] = []
    for target in sorted(targets):
        ok, error = _command_exists(target)
        label = "iints" if not target else f"iints {' '.join(target)}"
        if not ok:
            failures.append(f"{label}: {error}")

    if failures:
        print("Documentation command checks failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print(f"Validated {len(targets)} documented CLI command paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
