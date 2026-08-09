#!/usr/bin/env python3
"""Build the browsable and print-ready EUCYS dossier.

The maintained source pages live in ``docs/eucys`` so MkDocs can render native
Mermaid and MathJax. This builder also creates:

- a merged Markdown master dossier;
- static PNG fallbacks generated from the Mermaid sources;
- an offline HTML reader with the Mermaid source preserved;
- PDF editions through pandoc/tectonic, with an fpdf fallback;
- a ZIP bundle that can be handed to a reviewer.
"""

from __future__ import annotations

import html
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from collections import OrderedDict, defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs" / "eucys"
PACK_DIR = ROOT / "research" / "eucys_pack"
BROWSE_DIR = PACK_DIR / "browse"
PACK_ASSET_DIR = PACK_DIR / "assets"
DIAGRAM_ASSET_DIR = PACK_ASSET_DIR / "diagrams"
PDF_DIR = PACK_DIR / "pdf"
BUILD_DIR = ROOT / "tmp" / "pdfs" / "eucys_dossier"

PAGES = (
    "index.md",
    "01_PROJECT_AND_QUESTION.md",
    "02_SYSTEM_ARCHITECTURE.md",
    "03_SCIENTIFIC_MODEL.md",
    "04_AI_DATA_AND_EVIDENCE.md",
    "05_RESULTS_VALIDATION_AND_LIMITATIONS.md",
    "06_DEMO_RUNBOOK.md",
    "07_JURY_QUICK_REFERENCE.md",
    "08_EVIDENCE_MAP.md",
)

PAGE_LABELS = (
    "Orientation",
    "Project and research question",
    "System architecture",
    "Scientific model and formulas",
    "AI, data and evidence",
    "Results, validation and limitations",
    "Live demonstration runbook",
    "Jury quick reference",
    "Claim-to-evidence map",
)

COLORS = {
    "ink": "#183043",
    "muted": "#5E7180",
    "line": "#8195A3",
    "navy": "#1F4D66",
    "teal": "#207B78",
    "blue_fill": "#EAF2F6",
    "teal_fill": "#E8F4F2",
    "warm_fill": "#F6F1E8",
    "paper": "#FCFCFA",
    "white": "#FFFFFF",
}

NODE_PATTERN = re.compile(r"([A-Za-z0-9_]+)\s*\[\s*\"([^\"]+)\"\s*\]")
DIAGRAM_MARKER = re.compile(r"<!--\s*diagram:([a-z0-9-]+)\s*-->")


def _command_output(*args: str, fallback: str) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback
    return completed.stdout.strip() or fallback


def _project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    return match.group(1) if match else "unknown"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _clean_label(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    return html.unescape(value).replace("\\n", "\n")


def _extract_flow(code: str) -> tuple[str, OrderedDict[str, str], list[tuple[str, str]]]:
    first = next((line.strip() for line in code.splitlines() if line.strip()), "flowchart LR")
    direction = "TD" if first.endswith(("TD", "TB")) else "LR"
    nodes: OrderedDict[str, str] = OrderedDict()
    for node_id, label in NODE_PATTERN.findall(code):
        nodes.setdefault(node_id, _clean_label(label))

    edges: list[tuple[str, str]] = []
    known_ids = list(nodes)
    for raw in code.splitlines():
        if "-->" not in raw and ".->" not in raw:
            continue
        present = [
            node_id
            for node_id in known_ids
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(node_id)}(?![A-Za-z0-9_])", raw)
        ]
        if len(present) >= 2:
            edge = (present[0], present[-1])
            if edge[0] != edge[1] and edge not in edges:
                edges.append(edge)
    return direction, nodes, edges


def _wrap_label(text: str, width: int) -> str:
    parts: list[str] = []
    for paragraph in text.splitlines() or [text]:
        parts.extend(textwrap.wrap(paragraph, width=max(8, width)) or [""])
    return "\n".join(parts)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    width: int = 4,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 15
    spread = math.pi / 7
    left = (
        end[0] - length * math.cos(angle - spread),
        end[1] - length * math.sin(angle - spread),
    )
    right = (
        end[0] - length * math.cos(angle + spread),
        end[1] - length * math.sin(angle + spread),
    )
    draw.polygon((end, left, right), fill=color)


def _flow_levels(
    nodes: OrderedDict[str, str],
    edges: list[tuple[str, str]],
) -> dict[str, int]:
    order = {node_id: index for index, node_id in enumerate(nodes)}
    levels = {node_id: 0 for node_id in nodes}
    # Mermaid sources are written in reading order. Only forward edges determine
    # columns/rows; feedback edges remain arrows without creating layout cycles.
    for node_id in nodes:
        forward_sources = [
            source
            for source, target in edges
            if target == node_id and order.get(source, 0) < order.get(target, 0)
        ]
        if forward_sources:
            levels[node_id] = max(levels[source] + 1 for source in forward_sources)
    unique = sorted(set(levels.values()))
    remap = {value: index for index, value in enumerate(unique)}
    return {key: remap[value] for key, value in levels.items()}


def _render_flowchart(code: str, output: Path, title: str) -> None:
    direction, nodes, edges = _extract_flow(code)
    width, height = 1800, 1000
    margin_x, margin_y = 90, 150
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.text((margin_x, 45), title, font=_font(34, bold=True), fill=COLORS["ink"])
    draw.line((margin_x, 100, width - margin_x, 100), fill=COLORS["teal"], width=5)

    if not nodes:
        draw.text((margin_x, margin_y), "Mermaid diagram source is included in the dossier.", font=_font(24), fill=COLORS["muted"])
        image.save(output)
        return

    levels = _flow_levels(nodes, edges)
    grouped: defaultdict[int, list[str]] = defaultdict(list)
    for node_id in nodes:
        grouped[levels[node_id]].append(node_id)
    level_values = sorted(grouped)
    positions: dict[str, tuple[float, float]] = {}

    if direction == "LR":
        x_span = width - 2 * margin_x
        for level in level_values:
            x = margin_x + (x_span * level / max(1, len(level_values) - 1))
            group = grouped[level]
            y_span = height - margin_y - 90
            for index, node_id in enumerate(group):
                y = margin_y + y_span * (index + 1) / (len(group) + 1)
                positions[node_id] = (x, y)
    else:
        y_span = height - margin_y - 90
        for level in level_values:
            y = margin_y + (y_span * level / max(1, len(level_values) - 1))
            group = grouped[level]
            x_span = width - 2 * margin_x
            for index, node_id in enumerate(group):
                x = margin_x + x_span * (index + 1) / (len(group) + 1)
                positions[node_id] = (x, y)

    max_in_level = max(len(group) for group in grouped.values())
    if direction == "LR":
        box_w = max(150, min(280, int((width - 2 * margin_x) / max(2, len(level_values)) * 0.78)))
        box_h = max(86, min(150, int((height - margin_y - 90) / max(2, max_in_level + 1) * 0.72)))
    else:
        box_w = max(200, min(330, int((width - 2 * margin_x) / max(2, max_in_level + 1) * 0.72)))
        box_h = max(76, min(120, int((height - margin_y - 90) / max(2, len(level_values)) * 0.62)))

    positions = {
        node_id: (
            min(max(x, margin_x + box_w / 2), width - margin_x - box_w / 2),
            min(max(y, margin_y + box_h / 2), height - 70 - box_h / 2),
        )
        for node_id, (x, y) in positions.items()
    }

    for source, target in edges:
        if source not in positions or target not in positions:
            continue
        sx, sy = positions[source]
        tx, ty = positions[target]
        if direction == "LR":
            start = (sx + box_w / 2, sy)
            end = (tx - box_w / 2 - 7, ty)
        else:
            start = (sx, sy + box_h / 2)
            end = (tx, ty - box_h / 2 - 7)
        _arrow(draw, start, end)

    body_font = _font(max(17, min(24, int(box_w / 11))))
    for index, (node_id, label) in enumerate(nodes.items()):
        x, y = positions[node_id]
        bounds = (
            x - box_w / 2,
            y - box_h / 2,
            x + box_w / 2,
            y + box_h / 2,
        )
        fill = COLORS["teal_fill"] if index % 2 else COLORS["blue_fill"]
        draw.rounded_rectangle(bounds, radius=18, fill=fill, outline=COLORS["navy"], width=3)
        wrapped = _wrap_label(label, max(12, int(box_w / 12)))
        text_box = draw.multiline_textbbox((0, 0), wrapped, font=body_font, spacing=5, align="center")
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        draw.multiline_text(
            (x - text_w / 2, y - text_h / 2),
            wrapped,
            font=body_font,
            fill=COLORS["ink"],
            spacing=5,
            align="center",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)


def _render_sequence(code: str, output: Path, title: str) -> None:
    participants: OrderedDict[str, str] = OrderedDict()
    messages: list[tuple[str, str, str]] = []
    for raw in code.splitlines():
        line = raw.strip()
        participant = re.match(r"participant\s+(\w+)\s+as\s+(.+)", line)
        if participant:
            participants[participant.group(1)] = participant.group(2).strip()
            continue
        message = re.match(r"(\w+)-+>>(\w+):\s*(.+)", line)
        if message:
            messages.append((message.group(1), message.group(2), message.group(3).strip()))

    width = 1900
    height = max(1000, 280 + len(messages) * 90)
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.text((80, 45), title, font=_font(34, bold=True), fill=COLORS["ink"])
    draw.line((80, 100, width - 80, 100), fill=COLORS["teal"], width=5)

    ids = list(participants)
    x_positions = {
        participant_id: 120 + index * (width - 240) / max(1, len(ids) - 1)
        for index, participant_id in enumerate(ids)
    }
    box_w, box_h = 220, 72
    for index, (participant_id, label) in enumerate(participants.items()):
        x = x_positions[participant_id]
        fill = COLORS["teal_fill"] if index % 2 else COLORS["blue_fill"]
        draw.rounded_rectangle(
            (x - box_w / 2, 140, x + box_w / 2, 140 + box_h),
            radius=14,
            fill=fill,
            outline=COLORS["navy"],
            width=3,
        )
        wrapped = _wrap_label(label, 18)
        text_box = draw.multiline_textbbox((0, 0), wrapped, font=_font(20), spacing=4, align="center")
        draw.multiline_text(
            (x - (text_box[2] - text_box[0]) / 2, 176 - (text_box[3] - text_box[1]) / 2),
            wrapped,
            font=_font(20),
            fill=COLORS["ink"],
            spacing=4,
            align="center",
        )
        draw.line((x, 212, x, height - 80), fill="#B4C0C8", width=3)

    for row, (source, target, label) in enumerate(messages):
        if source not in x_positions or target not in x_positions:
            continue
        y = 270 + row * 86
        start = (x_positions[source], y)
        end = (x_positions[target], y)
        _arrow(draw, start, end)
        wrapped = _wrap_label(label, 38)
        text_box = draw.multiline_textbbox((0, 0), wrapped, font=_font(17), spacing=3, align="center")
        center_x = (start[0] + end[0]) / 2
        draw.rectangle(
            (
                center_x - (text_box[2] - text_box[0]) / 2 - 8,
                y - (text_box[3] - text_box[1]) - 12,
                center_x + (text_box[2] - text_box[0]) / 2 + 8,
                y - 3,
            ),
            fill=COLORS["paper"],
        )
        draw.multiline_text(
            (center_x - (text_box[2] - text_box[0]) / 2, y - (text_box[3] - text_box[1]) - 9),
            wrapped,
            font=_font(17),
            fill=COLORS["ink"],
            spacing=3,
            align="center",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)


def render_mermaid_fallback(code: str, output: Path, title: str) -> None:
    if code.lstrip().startswith("sequenceDiagram"):
        _render_sequence(code, output, title)
    else:
        _render_flowchart(code, output, title)


def _increment_headings(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            lines.append(line)
            continue
        if not in_fence:
            heading = re.match(r"^(#{1,5})\s+(.*)$", line)
            if heading:
                lines.append(f"{heading.group(1)}# {heading.group(2)}")
                continue
        lines.append(line)
    return "\n".join(lines)


def _rewrite_page_for_browse(text: str) -> str:
    return text.replace("../assets/eucys/", "assets/eucys/")


def _rewrite_page_for_master(text: str) -> str:
    text = text.replace("../assets/eucys/", "assets/")
    for page, label in zip(PAGES, PAGE_LABELS):
        text = re.sub(
            rf"\[([^\]]+)\]\({re.escape(page)}\)",
            lambda match: match.group(1),
            text,
        )
    return text


def _extract_diagrams(
    text: str,
    *,
    page_stem: str,
    title_by_name: dict[str, str],
) -> list[tuple[str, str]]:
    diagrams: list[tuple[str, str]] = []
    pending_name: str | None = None
    in_mermaid = False
    block: list[str] = []
    counter = 0
    for line in text.splitlines():
        marker = DIAGRAM_MARKER.search(line)
        if marker:
            pending_name = marker.group(1)
            continue
        if line.strip() == "```mermaid":
            in_mermaid = True
            block = []
            continue
        if in_mermaid and line.strip() == "```":
            counter += 1
            name = pending_name or f"{page_stem}-diagram-{counter}"
            code = "\n".join(block).strip() + "\n"
            diagrams.append((name, code))
            title_by_name.setdefault(name, name.replace("-", " ").title())
            pending_name = None
            in_mermaid = False
            block = []
            continue
        if in_mermaid:
            block.append(line)
    return diagrams


def _replace_mermaid(
    text: str,
    *,
    page_stem: str,
    mode: str,
    title_by_name: dict[str, str],
) -> str:
    lines = text.splitlines()
    output: list[str] = []
    pending_name: str | None = None
    counter = 0
    index = 0
    while index < len(lines):
        marker = DIAGRAM_MARKER.search(lines[index])
        if marker:
            pending_name = marker.group(1)
            index += 1
            continue
        if lines[index].strip() != "```mermaid":
            output.append(lines[index])
            index += 1
            continue
        counter += 1
        block: list[str] = []
        index += 1
        while index < len(lines) and lines[index].strip() != "```":
            block.append(lines[index])
            index += 1
        index += 1
        name = pending_name or f"{page_stem}-diagram-{counter}"
        pending_name = None
        title = title_by_name.get(name, name.replace("-", " ").title())
        image_ref = f"assets/diagrams/{name}.png"
        image_line = f"![{title}]({image_ref})"
        if mode == "pdf":
            image_line += "{ width=96% }"
        output.extend([image_line, ""])
        if mode == "html":
            escaped = html.escape("\n".join(block))
            output.extend(
                [
                    "<details class=\"mermaid-source\">",
                    "<summary>View Mermaid source</summary>",
                    "",
                    f"```mermaid\n{escaped}\n```",
                    "",
                    "</details>",
                    "",
                ]
            )
    return "\n".join(output)


def _copy_assets() -> None:
    BROWSE_DIR.mkdir(parents=True, exist_ok=True)
    (BROWSE_DIR / "assets" / "eucys").mkdir(parents=True, exist_ok=True)
    (BROWSE_DIR / "diagrams").mkdir(parents=True, exist_ok=True)
    PACK_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DIAGRAM_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    for filename in ("EUCYS_MAIN_FIGURE.png", "EUCYS_MAIN_FIGURE.csv", "EUCYS_RESULTS_TABLE.csv"):
        source = ROOT / "docs" / "assets" / "eucys" / filename
        shutil.copy2(source, BROWSE_DIR / "assets" / "eucys" / filename)
        shutil.copy2(source, PACK_ASSET_DIR / filename)


def _markdown_to_html(markdown_text: str, output: Path) -> None:
    try:
        import markdown
    except ImportError as exc:  # pragma: no cover - developer environment guard
        raise RuntimeError("The 'markdown' package is required to build offline HTML.") from exc

    body = markdown.markdown(
        markdown_text,
        extensions=("fenced_code", "tables", "toc", "admonition"),
        output_format="html5",
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IINTS-AF EUCYS Jury Dossier</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: {COLORS['ink']};
      --muted: {COLORS['muted']};
      --line: #D5DEE3;
      --navy: {COLORS['navy']};
      --teal: {COLORS['teal']};
      --paper: {COLORS['paper']};
      --panel: #FFFFFF;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #EEF2F3;
      font: 17px/1.62 Georgia, "Times New Roman", serif;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(220px, 290px) minmax(0, 980px);
      gap: 34px;
      max-width: 1360px;
      margin: 0 auto;
      padding: 30px;
    }}
    nav {{
      position: sticky;
      top: 24px;
      align-self: start;
      padding: 24px;
      color: #EFF7F8;
      background: var(--ink);
      border-radius: 4px;
      font: 14px/1.45 Arial, sans-serif;
    }}
    nav strong {{ display: block; margin-bottom: 14px; font-size: 16px; }}
    nav a {{ display: block; padding: 7px 0; color: #D9E9EC; text-decoration: none; }}
    nav a:hover {{ color: white; text-decoration: underline; }}
    main {{
      min-width: 0;
      padding: 52px 62px 80px;
      background: var(--panel);
      border-top: 8px solid var(--teal);
      box-shadow: 0 8px 28px rgba(24, 48, 67, .10);
    }}
    h1, h2, h3, h4 {{ color: var(--navy); font-family: Arial, sans-serif; line-height: 1.22; }}
    h1 {{ margin-top: 0; font-size: 2.4rem; }}
    h2 {{ margin-top: 2.6em; padding-bottom: .35em; border-bottom: 2px solid var(--line); }}
    h3 {{ margin-top: 2em; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1.2em 0 1.8em; font: 14px/1.4 Arial, sans-serif; }}
    th, td {{ padding: 10px 12px; border: 1px solid var(--line); vertical-align: top; text-align: left; }}
    th {{ background: #EAF2F6; }}
    tr:nth-child(even) td {{ background: #FAFBFB; }}
    code, pre {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    code {{ padding: .1em .3em; background: #F0F3F4; }}
    pre {{ overflow-x: auto; padding: 18px; background: #F2F5F5; border-left: 4px solid var(--teal); }}
    blockquote {{ margin: 1.5em 0; padding: 10px 24px; color: var(--navy); background: #EFF6F5; border-left: 5px solid var(--teal); }}
    img {{ display: block; max-width: 100%; height: auto; margin: 22px auto; }}
    details {{ margin: 12px 0 24px; padding: 12px 16px; border: 1px solid var(--line); background: #FBFCFC; }}
    summary {{ cursor: pointer; font: 600 14px Arial, sans-serif; color: var(--navy); }}
    .admonition {{ margin: 1.4em 0; padding: 14px 18px; background: #FFF9E9; border-left: 5px solid #B58A2D; }}
    a {{ color: #0D6471; }}
    @media (max-width: 900px) {{
      .layout {{ display: block; padding: 0; }}
      nav {{ position: static; border-radius: 0; }}
      main {{ padding: 32px 22px 60px; box-shadow: none; }}
    }}
    @media print {{
      body {{ background: white; }}
      .layout {{ display: block; max-width: none; padding: 0; }}
      nav {{ display: none; }}
      main {{ box-shadow: none; border: 0; padding: 0; }}
      h2 {{ break-before: page; }}
      table, img, blockquote {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <nav>
      <strong>IINTS-AF EUCYS Dossier</strong>
      <a href="#eucys-jury-dossier">Orientation</a>
      <a href="#project-and-research-question">Research question</a>
      <a href="#system-architecture">Architecture</a>
      <a href="#scientific-model-and-formula-registry">Scientific model</a>
      <a href="#ai-data-and-evidence">AI and data</a>
      <a href="#results-validation-and-limitations">Results</a>
      <a href="#eucys-live-demonstration-runbook">Demo runbook</a>
      <a href="#jury-quick-reference">Quick reference</a>
      <a href="#claim-to-evidence-map">Evidence map</a>
    </nav>
    <main>{body}</main>
  </div>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")


def _render_pdf(source: Path, output: Path, *, title: str) -> None:
    pandoc = shutil.which("pandoc")
    tectonic = shutil.which("tectonic")
    if pandoc and tectonic:
        command = [
            pandoc,
            str(source),
            "--from",
            "markdown+tex_math_dollars+tex_math_single_backslash+raw_tex",
            "--pdf-engine",
            tectonic,
            "--toc",
            "--number-sections",
            "--resource-path",
            str(PACK_DIR),
            "-V",
            "geometry:margin=18mm",
            "-V",
            "fontsize=10pt",
            "-V",
            "papersize=a4",
            "-V",
            "linestretch=1.05",
            "-V",
            "colorlinks=true",
            "-V",
            "linkcolor=blue",
            "-V",
            "urlcolor=blue",
            "--metadata",
            f"title={title}",
            "--metadata",
            "author=Runebob Baers",
            "-o",
            str(output),
        ]
        environment = os.environ.copy()
        environment["XDG_CACHE_HOME"] = str(BUILD_DIR / "cache")
        Path(environment["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(command, cwd=PACK_DIR, env=environment, check=True)
            return
        except subprocess.CalledProcessError:
            print("[IINTS] pandoc/tectonic failed; using the offline fallback renderer.", file=sys.stderr)

    fallback = ROOT / "tools" / "research" / "render_eucys_report_pdf.py"
    subprocess.run(
        [sys.executable, str(fallback), "--input", str(source), "--output", str(output)],
        cwd=ROOT,
        check=True,
    )


def _write_readme(version: str, _commit: str) -> None:
    readme = f"""# IINTS-AF EUCYS Jury Bundle

Build date: {date.today().isoformat()}

SDK version: `{version}`

## Start here

1. Open `BROWSE_ME.html` for the offline browsable dossier.
2. Open `pdf/IINTS_AF_EUCYS_MASTER_DOSSIER.pdf` for the print edition.
3. Keep `pdf/IINTS_AF_EUCYS_JURY_QUICK_REFERENCE.pdf` beside the live demo.
4. Use `pdf/IINTS_AF_EUCYS_DEMO_RUNBOOK.pdf` to rehearse.

## Included

- Nine linked dossier chapters.
- Mermaid source files for every architecture and workflow diagram.
- Static diagram fallbacks for offline reading and PDF.
- All 15 deterministic formulas from formula registry v5.
- Benchmark figure and CSV source tables.
- A demo script, failure plan and jury Q&A.
- A claim-to-code, claim-to-output and claim-to-literature evidence map.

## Boundary

IINTS-AF is research and educational software. It is not a medical device and
must not be used for treatment decisions or real medication delivery.

## Rebuild

```bash
tools/research/build_eucys_dossier.sh
```
"""
    (PACK_DIR / "README.md").write_text(readme, encoding="utf-8")


def _build_zip() -> None:
    target = PACK_DIR / "IINTS_AF_EUCYS_BROWSE_BUNDLE.zip"
    include_paths = [
        PACK_DIR / "README.md",
        PACK_DIR / "BROWSE_ME.html",
        PACK_DIR / "EUCYS_MASTER_DOSSIER.md",
        BROWSE_DIR,
        PACK_ASSET_DIR,
        PACK_DIR / "diagrams",
        PDF_DIR / "IINTS_AF_EUCYS_MASTER_DOSSIER.pdf",
        PDF_DIR / "IINTS_AF_EUCYS_DEMO_RUNBOOK.pdf",
        PDF_DIR / "IINTS_AF_EUCYS_JURY_QUICK_REFERENCE.pdf",
    ]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in include_paths:
            if not path.exists():
                continue
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        archive.write(child, child.relative_to(PACK_DIR))
            else:
                archive.write(path, path.relative_to(PACK_DIR))


def build() -> None:
    _copy_assets()
    version = _project_version()
    commit = _command_output("git", "rev-parse", "--short", "HEAD", fallback="unknown")
    title_by_name: dict[str, str] = {}
    page_texts: list[str] = []
    all_diagrams: OrderedDict[str, str] = OrderedDict()

    for page_name, page_label in zip(PAGES, PAGE_LABELS):
        source = DOCS_DIR / page_name
        text = source.read_text(encoding="utf-8")
        page_texts.append(text)
        title_by_name.update(
            {
                "01_project_and_question-diagram-1": "Three-part experimental structure",
                "system-architecture": "End-to-end IINTS-AF architecture",
                "simulation-step": "One simulation step",
                "numeric-authority": "Numeric authority boundary",
                "02_system_architecture-diagram-4": "Source-layer architecture",
                "evidence-lifecycle": "Evidence lifecycle",
                "desktop-bridge": "Desktop application boundary",
                "02_system_architecture-diagram-7": "Bench hardware boundary",
                "03_scientific_model-diagram-1": "State and observation separation",
                "ai-boundary": "AI authority boundary",
                "04_ai_data_and_evidence-diagram-2": "Glucose forecasting workflow",
                "data-lifecycle": "Research data lifecycle",
                "04_ai_data_and_evidence-diagram-4": "Cross-scale evidence boundary",
                "validation-ladder": "Validation ladder",
            }
        )
        for diagram_name, code in _extract_diagrams(
            text,
            page_stem=Path(page_name).stem.lower(),
            title_by_name=title_by_name,
        ):
            all_diagrams.setdefault(diagram_name, code)

        browse_text = _rewrite_page_for_browse(text)
        (BROWSE_DIR / page_name).write_text(browse_text, encoding="utf-8")

    diagram_source_dir = PACK_DIR / "diagrams"
    diagram_source_dir.mkdir(parents=True, exist_ok=True)
    for diagram_name, code in all_diagrams.items():
        (diagram_source_dir / f"{diagram_name}.mmd").write_text(code, encoding="utf-8")
        (BROWSE_DIR / "diagrams" / f"{diagram_name}.mmd").write_text(code, encoding="utf-8")
        render_mermaid_fallback(
            code,
            DIAGRAM_ASSET_DIR / f"{diagram_name}.png",
            title_by_name.get(diagram_name, diagram_name.replace("-", " ").title()),
        )
        shutil.copy2(
            DIAGRAM_ASSET_DIR / f"{diagram_name}.png",
            BROWSE_DIR / "assets" / "eucys" / f"{diagram_name}.png",
        )

    header = "\n".join(
        [
            "# IINTS-AF EUCYS Jury Dossier",
            "",
            f"**Author:** Runebob Baers",
            "",
            f"**Build date:** {date.today().isoformat()}",
            "",
            f"**SDK version:** `{version}`",
            "",
            f"**Repository commit:** `{commit}`",
            "",
            "> Research and educational software only. Not a medical device,",
            "> not treatment advice, and not for real medication delivery.",
            "",
            "---",
            "",
        ]
    )
    master_sections: list[str] = [header]
    pdf_sections: list[str] = [header]
    html_sections: list[str] = [header]
    for page_name, text in zip(PAGES, page_texts):
        transformed = _increment_headings(_rewrite_page_for_master(text))
        master_sections.extend([transformed, "\n\n---\n"])
        pdf_sections.extend(
            [
                _replace_mermaid(
                    transformed,
                    page_stem=Path(page_name).stem.lower(),
                    mode="pdf",
                    title_by_name=title_by_name,
                ),
                "\n\n---\n",
            ]
        )
        html_sections.extend(
            [
                _replace_mermaid(
                    transformed,
                    page_stem=Path(page_name).stem.lower(),
                    mode="html",
                    title_by_name=title_by_name,
                ),
                "\n\n---\n",
            ]
        )

    master = "\n".join(master_sections)
    pdf_master = "\n".join(pdf_sections)
    html_master = "\n".join(html_sections)
    master_path = PACK_DIR / "EUCYS_MASTER_DOSSIER.md"
    pdf_source = BUILD_DIR / "EUCYS_MASTER_DOSSIER_PDF.md"
    master_path.write_text(master, encoding="utf-8")
    pdf_source.write_text(pdf_master, encoding="utf-8")
    _markdown_to_html(html_master, PACK_DIR / "BROWSE_ME.html")
    _write_readme(version, commit)

    _render_pdf(
        pdf_source,
        PDF_DIR / "IINTS_AF_EUCYS_MASTER_DOSSIER.pdf",
        title="IINTS-AF EUCYS Jury Dossier",
    )
    _render_pdf(
        DOCS_DIR / "06_DEMO_RUNBOOK.md",
        PDF_DIR / "IINTS_AF_EUCYS_DEMO_RUNBOOK.pdf",
        title="IINTS-AF EUCYS Live Demonstration Runbook",
    )
    _render_pdf(
        DOCS_DIR / "07_JURY_QUICK_REFERENCE.md",
        PDF_DIR / "IINTS_AF_EUCYS_JURY_QUICK_REFERENCE.pdf",
        title="IINTS-AF EUCYS Jury Quick Reference",
    )
    _build_zip()

    print(f"[IINTS] EUCYS dossier ready: {PACK_DIR}")
    print(f"[IINTS] Browse: {PACK_DIR / 'BROWSE_ME.html'}")
    print(f"[IINTS] PDF: {PDF_DIR / 'IINTS_AF_EUCYS_MASTER_DOSSIER.pdf'}")
    print(f"[IINTS] ZIP: {PACK_DIR / 'IINTS_AF_EUCYS_BROWSE_BUNDLE.zip'}")


if __name__ == "__main__":
    build()
