#!/usr/bin/env python3
"""
Generate a professionally styled PDF manual from Markdown.

Example:
  python3 tools/generate_manual_from_markdown.py \
    --input docs/manuals/IINTS-AF_SDK_Manual_Improved.md \
    --output docs/manuals/IINTS-AF_SDK_Manual_Improved.pdf
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

from fpdf import FPDF
from markdown_it import MarkdownIt


UNICODE_FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
MONO_FONT = Path("/System/Library/Fonts/Supplemental/Andale Mono.ttf")


class StyledPDF(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.set_auto_page_break(auto=True, margin=18)
        self.set_left_margin(16)
        self.set_right_margin(16)
        self.set_top_margin(16)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font(self._body_font, size=9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, self.title, 0, 0, "L")
        self.ln(10)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(self._body_font, size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "C")
        self.set_text_color(0, 0, 0)


def _sanitize_codeblocks(text: str) -> str:
    pattern = re.compile(r"```(.*?)\n(.*?)```", re.S)

    def _repl(match: re.Match) -> str:
        lang = match.group(1).strip()
        code = match.group(2)
        code = code.encode("ascii", "ignore").decode("ascii")
        header = f"```{lang}\n" if lang else "```\n"
        return header + code + "```"

    return pattern.sub(_repl, text)


def _extract_title(md_text: str) -> str:
    for line in md_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "IINTS-AF Manual"


def build_pdf(input_path: Path, output_path: Path) -> None:
    md_text = input_path.read_text(encoding="utf-8")
    md_text = _sanitize_codeblocks(md_text)
    title = _extract_title(md_text)

    pdf = StyledPDF(title)
    if UNICODE_FONT.exists():
        pdf.add_font("ArialUnicode", "", str(UNICODE_FONT), uni=True)
        pdf.add_font("ArialUnicode", "B", str(UNICODE_FONT), uni=True)
        pdf.add_font("ArialUnicode", "I", str(UNICODE_FONT), uni=True)
        pdf.add_font("ArialUnicode", "BI", str(UNICODE_FONT), uni=True)
        pdf._body_font = "ArialUnicode"
    else:
        pdf._body_font = "Helvetica"

    if MONO_FONT.exists():
        pdf.add_font("AndaleMono", "", str(MONO_FONT), uni=True)
        pdf._mono_font = "AndaleMono"
    else:
        pdf._mono_font = "Courier"

    # Cover page
    pdf.add_page()
    pdf.set_font(pdf._body_font, "B", 20)
    pdf.cell(0, 12, title, ln=1)
    pdf.set_font(pdf._body_font, "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Research Use Only — Not for Clinical Care", ln=1)
    pdf.cell(0, 8, f"Generated: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ln=1)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(4)
    pdf.set_font(pdf._body_font, "", 11)

    md = MarkdownIt("commonmark")
    tokens = md.parse(md_text)
    list_stack: list[dict] = []

    def write_paragraph(text: str, indent: float = 0.0) -> None:
        if not text.strip():
            pdf.ln(4)
            return
        x = pdf.get_x()
        pdf.set_x(x + indent)
        pdf.multi_cell(0, 5.5, text.strip())
        pdf.set_x(x)
        pdf.ln(1)

    def write_codeblock(text: str) -> None:
        pdf.set_font(pdf._mono_font, "", 9)
        pdf.set_fill_color(245, 245, 245)
        pdf.multi_cell(0, 4.6, text.strip(), fill=True)
        pdf.set_font(pdf._body_font, "", 11)
        pdf.ln(1)

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1])
            inline = tokens[i + 1]
            text = inline.content.strip()
            size = 16 if level == 1 else 13 if level == 2 else 11
            pdf.set_font(pdf._body_font, "B", size)
            pdf.ln(2)
            pdf.multi_cell(0, 6, text)
            pdf.set_font(pdf._body_font, "", 11)
            pdf.ln(1)
            i += 3
            continue

        if tok.type == "paragraph_open":
            inline = tokens[i + 1]
            text = inline.content
            indent = 6 * len(list_stack) if list_stack else 0
            write_paragraph(text, indent=indent)
            i += 3
            continue

        if tok.type == "bullet_list_open":
            list_stack.append({"type": "bullet"})
            i += 1
            continue
        if tok.type == "ordered_list_open":
            list_stack.append({"type": "ordered", "idx": 1})
            i += 1
            continue
        if tok.type in {"bullet_list_close", "ordered_list_close"}:
            if list_stack:
                list_stack.pop()
            i += 1
            continue

        if tok.type == "list_item_open":
            i += 1
            continue
        if tok.type == "list_item_close":
            i += 1
            continue

        if tok.type in {"code_block", "fence"}:
            write_codeblock(tok.content)
            i += 1
            continue

        if tok.type == "inline":
            if list_stack:
                prefix = "- "
                if list_stack[-1]["type"] == "ordered":
                    prefix = f"{list_stack[-1]['idx']}. "
                    list_stack[-1]["idx"] += 1
                indent = 6 * (len(list_stack) - 1)
                write_paragraph(prefix + tok.content, indent=indent)
            i += 1
            continue

        i += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF from Markdown.")
    parser.add_argument("--input", type=Path, required=True, help="Input Markdown file")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF file")
    args = parser.parse_args()
    build_pdf(args.input, args.output)


if __name__ == "__main__":
    main()
