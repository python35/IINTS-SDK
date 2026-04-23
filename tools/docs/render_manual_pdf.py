#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "--",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2192": "->",
    "\u00a0": " ",
}

HEADING_COLOR = (21, 78, 121)
MUTED_TEXT = (95, 95, 95)
BORDER_COLOR = (210, 216, 224)
HEADER_FILL = (232, 240, 247)
TITLE_FILL = (240, 246, 252)
QUOTE_FILL = (246, 249, 252)

INLINE_BOLD_LABEL = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
IMAGE_PATTERN = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")


def normalize_text(text: str) -> str:
    for source, target in REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", text)
    text = text.replace("**", "")
    text = text.replace("`", "")
    return text.encode("latin-1", "replace").decode("latin-1")


class ManualPdf(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(8)
        self.set_font("Helvetica", size=9)
        self.set_text_color(*MUTED_TEXT)
        self.cell(0, 6, "IINTS-AF SDK Manual", new_x="LMARGIN", new_y="NEXT", align="R")
        self.set_draw_color(*BORDER_COLOR)
        self.line(self.l_margin, self.y, self.w - self.r_margin, self.y)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_draw_color(*BORDER_COLOR)
        self.line(self.l_margin, self.y, self.w - self.r_margin, self.y)
        self.set_y(-10)
        self.set_font("Helvetica", size=9)
        self.set_text_color(*MUTED_TEXT)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def add_paragraph(
    pdf: ManualPdf,
    text: str,
    *,
    style: str = "",
    size: int = 11,
    color: tuple[int, int, int] = (0, 0, 0),
    indent: float = 0.0,
) -> None:
    pdf.set_text_color(*color)
    pdf.set_font("Helvetica", style=style, size=size)
    if indent:
        pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(pdf.epw - indent, 5.5 if size >= 11 else 4.8, normalize_text(text))
    pdf.ln(0.7)
    pdf.set_text_color(0, 0, 0)


def add_title_page(pdf: ManualPdf, title: str, metadata_lines: list[str]) -> None:
    pdf.add_page()
    pdf.set_fill_color(*TITLE_FILL)
    pdf.rect(pdf.l_margin, 22, pdf.epw, 54, style="F")
    pdf.set_xy(pdf.l_margin + 8, 30)
    pdf.set_text_color(*HEADING_COLOR)
    pdf.set_font("Helvetica", style="B", size=24)
    pdf.multi_cell(pdf.epw - 16, 10, normalize_text(title), align="L")
    pdf.ln(2)
    pdf.set_text_color(*MUTED_TEXT)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(
        pdf.epw - 16,
        6,
        "Research, simulation, benchmarking, and edge operations manual",
        align="L",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(84)
    for line in metadata_lines:
        if line.strip():
            add_paragraph(pdf, line, size=11)
    pdf.ln(6)
    add_paragraph(
        pdf,
        "This PDF was rendered from the markdown manual inside the IINTS-AF SDK workspace. It is research software documentation, not clinical guidance.",
        size=10,
        color=MUTED_TEXT,
    )


def add_heading(pdf: ManualPdf, level: int, text: str) -> None:
    size_map = {1: 20, 2: 15, 3: 13, 4: 12, 5: 11, 6: 11}
    line_height_map = {1: 9, 2: 7, 3: 6, 4: 6, 5: 5.5, 6: 5.5}
    pdf.ln(2)
    pdf.set_text_color(*HEADING_COLOR)
    pdf.set_font("Helvetica", style="B", size=size_map.get(level, 11))
    pdf.multi_cell(0, line_height_map.get(level, 6), normalize_text(text))
    pdf.set_draw_color(*BORDER_COLOR)
    if level <= 2:
        pdf.line(pdf.l_margin, pdf.y, pdf.w - pdf.r_margin, pdf.y)
    pdf.ln(1.2)
    pdf.set_text_color(0, 0, 0)


def parse_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) is not None for cell in cells)


def add_table(pdf: ManualPdf, lines: list[str]) -> None:
    rows = [parse_table_row(line) for line in lines]
    rows = [row for row in rows if row]
    if len(rows) < 2:
        for row in rows:
            add_paragraph(pdf, " | ".join(row), size=10)
        return
    if is_separator_row(rows[1]):
        header = rows[0]
        body = rows[2:]
    else:
        header = rows[0]
        body = rows[1:]

    col_count = max(len(header), *(len(row) for row in body))
    normalized_header = header + [""] * (col_count - len(header))
    normalized_body = [row + [""] * (col_count - len(row)) for row in body]
    col_width = pdf.epw / max(col_count, 1)

    pdf.ln(1)
    pdf.set_font("Helvetica", size=8)
    with pdf.table(
        width=pdf.epw,
        col_widths=[col_width] * col_count,
        text_align="LEFT",
        line_height=4.2,
        headings_style=FontFace(emphasis="B", fill_color=HEADER_FILL),
        borders_layout="HORIZONTAL_LINES",
        cell_fill_mode="ROWS",
        cell_fill_color=(250, 252, 255),
    ) as table:
        row = table.row()
        for cell in normalized_header:
            row.cell(normalize_text(cell))
        for body_row in normalized_body:
            row = table.row()
            for cell in body_row:
                row.cell(normalize_text(cell))
    pdf.ln(2)


def add_code_block(pdf: ManualPdf, lines: list[str]) -> None:
    pdf.set_fill_color(247, 247, 247)
    pdf.set_draw_color(*BORDER_COLOR)
    text = "\n".join(normalize_text(line) for line in lines) or " "
    x = pdf.l_margin
    y = pdf.y
    pdf.set_font("Courier", size=8)
    height = max(8, 4.2 * max(1, len(lines)) + 4)
    pdf.rect(x, y, pdf.epw, height, style="DF")
    pdf.set_xy(x + 2, y + 2)
    pdf.multi_cell(pdf.epw - 4, 4.2, text)
    pdf.ln(2)


def add_quote_block(pdf: ManualPdf, text: str) -> None:
    pdf.set_fill_color(*QUOTE_FILL)
    pdf.set_draw_color(*BORDER_COLOR)
    x = pdf.l_margin
    y = pdf.y
    pdf.rect(x, y, pdf.epw, 16, style="DF")
    pdf.set_xy(x + 4, y + 3)
    pdf.set_font("Helvetica", style="I", size=13)
    pdf.set_text_color(*HEADING_COLOR)
    pdf.multi_cell(pdf.epw - 8, 6, normalize_text(text))
    pdf.ln(2)
    pdf.set_text_color(0, 0, 0)


def add_rule(pdf: ManualPdf) -> None:
    pdf.ln(1)
    pdf.set_draw_color(*BORDER_COLOR)
    pdf.line(pdf.l_margin, pdf.y, pdf.w - pdf.r_margin, pdf.y)
    pdf.ln(2)


def render_markdown(input_path: Path, output_path: Path) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    pdf = ManualPdf(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(16, 16, 16)
    pdf.set_title("IINTS-AF SDK Manual")
    pdf.set_author("IINTS-AF SDK")

    title = "IINTS-AF SDK Manual"
    metadata_lines: list[str] = []
    start_index = 0
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("## "):
            start_index = idx
            break
        metadata_lines.append(stripped.lstrip("> ").strip())
    else:
        start_index = len(lines)

    add_title_page(pdf, title, [line for line in metadata_lines if line])

    i = start_index
    while i < len(lines):
        raw_line = lines[i]
        stripped = raw_line.strip()

        if not stripped:
            pdf.ln(2.5)
            i += 1
            continue

        if stripped == "---":
            add_rule(pdf)
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            add_code_block(pdf, code_lines)
            i += 1
            continue

        image_match = IMAGE_PATTERN.match(stripped)
        if image_match:
            alt_text, image_ref = image_match.groups()
            image_path = Path(image_ref)
            if not image_path.is_absolute():
                image_path = (input_path.parent / image_path).resolve()
            if image_path.exists():
                current_y = pdf.y
                max_h = pdf.h - pdf.b_margin - current_y - 18
                if max_h < 60:
                    pdf.add_page()
                    current_y = pdf.y
                    max_h = pdf.h - pdf.b_margin - current_y - 18
                pdf.image(str(image_path), x=pdf.l_margin, y=current_y, w=pdf.epw, h=max_h, keep_aspect_ratio=True)
                pdf.y = current_y + max_h + 2
                if alt_text:
                    add_paragraph(pdf, f"Figure: {alt_text}", size=9, color=MUTED_TEXT)
            else:
                add_paragraph(pdf, f"[Missing figure] {image_path}", size=10, color=(160, 40, 40))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            add_table(pdf, table_lines)
            continue

        if stripped.startswith(">"):
            add_quote_block(pdf, stripped.lstrip("> ").strip())
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            add_heading(pdf, level, text)
            i += 1
            continue

        label_match = INLINE_BOLD_LABEL.match(stripped)
        if label_match:
            label, rest = label_match.groups()
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.set_text_color(*MUTED_TEXT)
            pdf.write(5.5, normalize_text(f"{label}: "))
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(pdf.epw - pdf.get_x(), 5.5, normalize_text(rest))
            pdf.ln(0.7)
            i += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            bullet_text = re.sub(r"^[-*]\s+", "", stripped)
            add_paragraph(pdf, f"- {bullet_text}", indent=3)
            i += 1
            continue

        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            add_paragraph(pdf, f"{numbered.group(1)}. {numbered.group(2)}", indent=2)
            i += 1
            continue

        add_paragraph(pdf, stripped)
        i += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the SDK manual markdown to a styled offline PDF.")
    parser.add_argument("--input", default="docs/manuals/IINTS-AF_SDK_Manual.md", help="Input markdown manual path")
    parser.add_argument("--output", default="docs/manuals/IINTS-AF_SDK_Manual.pdf", help="Output PDF path")
    args = parser.parse_args()

    render_markdown(Path(args.input), Path(args.output))
    print(f"[IINTS] Offline manual PDF ready: {args.output}")


if __name__ == "__main__":
    main()
