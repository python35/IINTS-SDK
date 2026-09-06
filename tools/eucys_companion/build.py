"""Assemble the EUCYS companion: concatenate the parts, inline every figure."""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"
FIGS = HERE / "figures"
OUT = HERE / "iints-af-model-companion.html"

PARTS = [
    "00_head.html",
    "01_shell.html",
    "02_orientation.html",
    "03_physio_a.html",
    "04_physio_b.html",
    "05_physio_c.html",
    "06_physio_d.html",
    "07_compute.html",
    "08_evidence.html",
    "09_reference.html",
    "10_script.html",
]

# Matplotlib writes the resolved family name; swap in a stack that exists
# everywhere, since the SVG renders in an <img> with no access to page CSS.
FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def load_figure(name: str) -> str:
    path = FIGS / f"{name}.svg"
    if not path.exists():
        raise SystemExit(f"missing figure: {path}")
    svg = path.read_text(encoding="utf-8")

    # Strip the XML prolog / DOCTYPE so the data URI is a bare <svg> document.
    svg = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)

    svg = svg.replace("DejaVu Sans", FONT_STACK)

    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (
        f'<img alt="{name.replace("_", " ")}" loading="lazy" '
        f'src="data:image/svg+xml;base64,{encoded}">'
    )


def main() -> None:
    chunks = []
    for part in PARTS:
        path = SRC / part
        if not path.exists():
            raise SystemExit(f"missing source part: {path}")
        chunks.append(path.read_text(encoding="utf-8"))
    html = "\n".join(chunks)

    used: list[str] = []

    def sub(match: re.Match[str]) -> str:
        name = match.group(1)
        used.append(name)
        return load_figure(name)

    html = re.sub(r"\{\{FIG:([a-z0-9_]+)\}\}", sub, html)

    leftover = re.findall(r"\{\{[^}]+\}\}", html)
    if leftover:
        raise SystemExit(f"unresolved placeholders: {leftover}")

    # Escape every non-ASCII character as a numeric entity so the page renders
    # identically regardless of the charset the host declares. Verified safe:
    # entities are not decoded inside <script>/<style>, and both are pure ASCII.
    for tag in ("script", "style"):
        for block in re.finditer(rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.S):
            if any(ord(c) > 127 for c in block.group(1)):
                raise SystemExit(f"non-ASCII inside <{tag}>; entity escaping unsafe")

    escaped = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
    OUT.write_text(escaped, encoding="utf-8")
    html = escaped

    available = {p.stem for p in FIGS.glob("*.svg")}
    unused = sorted(available - set(used))
    size_mb = OUT.stat().st_size / 1_048_576

    print(f"wrote {OUT.name}  ({size_mb:.2f} MB)")
    print(f"figures inlined: {len(used)}")
    if unused:
        print(f"figures not used: {', '.join(unused)}")


if __name__ == "__main__":
    main()
