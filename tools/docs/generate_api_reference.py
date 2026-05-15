#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT_DIR / "src" / "iints"
OUTPUT_PATH = ROOT_DIR / "docs" / "API_REFERENCE.md"


@dataclass(frozen=True)
class ClassInfo:
    name: str
    signature: str
    summary: str
    methods: tuple[str, ...]


@dataclass(frozen=True)
class ModuleInfo:
    module: str
    source_path: Path
    package: str
    summary: str
    classes: tuple[ClassInfo, ...]
    functions: tuple[str, ...]
    constants: tuple[str, ...]
    exports: tuple[str, ...]
    parse_error: str | None = None


def _module_name(path: Path) -> str:
    relative = path.relative_to(SOURCE_ROOT)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return "iints" if not parts else "iints." + ".".join(parts)


def _package_name(module: str) -> str:
    parts = module.split(".")
    return "root" if len(parts) == 1 else parts[1]


def _summary(text: str | None) -> str:
    if not text:
        return "No module docstring."
    paragraphs = [part.strip() for part in text.strip().split("\n\n") if part.strip()]
    return " ".join(paragraphs[0].split()) if paragraphs else "No module docstring."


def _render_annotation(annotation: ast.expr | None) -> str:
    return "" if annotation is None else f": {ast.unparse(annotation)}"


def _render_default(default: ast.expr | None) -> str:
    return "" if default is None else f" = {ast.unparse(default)}"


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    rendered: list[str] = []

    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        rendered.append(f"{argument.arg}{_render_annotation(argument.annotation)}{_render_default(default)}")
        if args.posonlyargs and index + 1 == len(args.posonlyargs):
            rendered.append("/")

    if args.vararg:
        rendered.append(f"*{args.vararg.arg}{_render_annotation(args.vararg.annotation)}")
    elif args.kwonlyargs:
        rendered.append("*")

    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        rendered.append(f"{argument.arg}{_render_annotation(argument.annotation)}{_render_default(default)}")

    if args.kwarg:
        rendered.append(f"**{args.kwarg.arg}{_render_annotation(args.kwarg.annotation)}")

    return_annotation = "" if node.returns is None else f" -> {ast.unparse(node.returns)}"
    return f"{node.name}({', '.join(rendered)}){return_annotation}"


def _class_signature(node: ast.ClassDef) -> str:
    bases = [ast.unparse(base) for base in node.bases]
    keywords = [f"{keyword.arg}={ast.unparse(keyword.value)}" for keyword in node.keywords if keyword.arg]
    inheritance = bases + keywords
    suffix = f"({', '.join(inheritance)})" if inheritance else ""
    return f"{node.name}{suffix}"


def _public_methods(node: ast.ClassDef) -> tuple[str, ...]:
    methods = [
        _function_signature(child)
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not child.name.startswith("_")
    ]
    return tuple(methods)


def _constant_names(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for node in tree.body:
        targets: Iterable[ast.expr]
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper() and not target.id.startswith("_"):
                names.append(target.id)
    return tuple(sorted(set(names)))


def _exports(tree: ast.Module) -> tuple[str, ...]:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple)):
            items = [
                item.value
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
            return tuple(items)
    return ()


def _module_info(path: Path) -> ModuleInfo:
    module = _module_name(path)
    relative_path = path.relative_to(ROOT_DIR)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return ModuleInfo(
            module=module,
            source_path=relative_path,
            package=_package_name(module),
            summary="Source could not be parsed as normal Python.",
            classes=(),
            functions=(),
            constants=(),
            exports=(),
            parse_error=f"{exc.msg} at line {exc.lineno}",
        )

    classes = tuple(
        ClassInfo(
            name=node.name,
            signature=_class_signature(node),
            summary=_summary(ast.get_docstring(node)),
            methods=_public_methods(node),
        )
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    )
    functions = tuple(
        _function_signature(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    )
    return ModuleInfo(
        module=module,
        source_path=relative_path,
        package=_package_name(module),
        summary=_summary(ast.get_docstring(tree)),
        classes=classes,
        functions=functions,
        constants=_constant_names(tree),
        exports=_exports(tree),
    )


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _render_module(module: ModuleInfo) -> list[str]:
    lines = [
        f"## `{module.module}`",
        "",
        f"- Source: `{module.source_path}`",
        f"- Summary: {_escape_cell(module.summary)}",
    ]
    if module.parse_error:
        lines.extend(
            [
                f"- Parse note: `{module.parse_error}`",
                "",
                "This file is documented as a source artifact rather than a normal importable Python module.",
                "",
            ]
        )
        return lines

    if module.exports:
        lines.append(f"- Explicit exports: `{', '.join(module.exports)}`")
    lines.append("")

    if module.classes:
        lines.extend(
            [
                "### Public Classes",
                "",
                "| Class | Signature | Summary |",
                "| --- | --- | --- |",
            ]
        )
        for item in module.classes:
            lines.append(
                f"| `{item.name}` | `{_escape_cell(item.signature)}` | {_escape_cell(item.summary)} |"
            )
        lines.append("")
        for item in module.classes:
            if item.methods:
                lines.extend(
                    [
                        f"#### `{item.name}` methods",
                        "",
                        *[f"- `{method}`" for method in item.methods],
                        "",
                    ]
                )

    if module.functions:
        lines.extend(
            [
                "### Public Functions",
                "",
                *[f"- `{function}`" for function in module.functions],
                "",
            ]
        )

    if module.constants:
        lines.extend(
            [
                "### Public Constants",
                "",
                *[f"- `{constant}`" for constant in module.constants],
                "",
            ]
        )

    if not module.classes and not module.functions and not module.constants:
        lines.extend(
            [
                "No public classes, functions, or all-caps constants are declared directly in this module.",
                "",
            ]
        )
    return lines


def build_reference() -> str:
    modules = tuple(_module_info(path) for path in sorted(SOURCE_ROOT.rglob("*.py")))
    by_package: dict[str, list[ModuleInfo]] = {}
    for module in modules:
        by_package.setdefault(module.package, []).append(module)

    lines = [
        "# API Reference",
        "",
        "This page is generated from the Python source tree by `tools/docs/generate_api_reference.py`.",
        "Do not edit it by hand; regenerate it after public module changes.",
        "",
        f"Documented modules: **{len(modules)}**",
        "",
        "## Package Index",
        "",
        "| Package | Modules |",
        "| --- | ---: |",
    ]
    for package, package_modules in sorted(by_package.items()):
        lines.append(f"| `{package}` | {len(package_modules)} |")

    lines.extend(["", "## Module Index", ""])
    for package, package_modules in sorted(by_package.items()):
        lines.append(f"### `{package}`")
        lines.append("")
        lines.extend(f"- [`{module.module}`](#{module.module.replace('.', '')})" for module in package_modules)
        lines.append("")

    for module in modules:
        lines.extend(_render_module(module))

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the IINTS-AF API reference markdown page.")
    parser.add_argument("--check", action="store_true", help="Fail if the generated page is not current.")
    args = parser.parse_args()

    rendered = build_reference()
    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if current != rendered:
            print("API reference is out of date. Run: python3 tools/docs/generate_api_reference.py")
            return 1
        print("API reference is up to date.")
        return 0

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
