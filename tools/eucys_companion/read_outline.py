"""Map each chapter to its printed page, using the PDF outline Chrome built.

The outline nests h3 sections under their h2 chapter, so the chapter list is
exactly the depth-1 run: ["Contents", <chapter titles in document order>].
Matching positionally rather than by title text keeps this working when a
heading's wording differs from the short title used in the contents list.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pypdf import PdfReader

pdf_path, chapters_path, out_path = sys.argv[1:4]
reader = PdfReader(pdf_path)
chapters: dict[str, str] = json.loads(Path(chapters_path).read_text())


def flatten(items, depth=0, acc=None):
    acc = [] if acc is None else acc
    for item in items:
        if isinstance(item, list):
            flatten(item, depth + 1, acc)
        else:
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                page = None
            acc.append((depth, str(item.title).strip(), page))
    return acc


flat = flatten(reader.outline)
level1 = [(t, p) for d, t, p in flat if d == 1 and p is not None]

if level1 and level1[0][0].lower().startswith("contents"):
    level1 = level1[1:]

ids = list(chapters.keys())
pages: dict[str, int] = {}

if len(level1) == len(ids):
    for cid, (_title, page) in zip(ids, level1):
        pages[cid] = page
else:
    # Fall back to title matching if the structure is not what we expect.
    lookup = {t.lower(): p for t, p in level1}
    for cid, title in chapters.items():
        hit = lookup.get(title.lower())
        if hit:
            pages[cid] = hit

Path(out_path).write_text(json.dumps(pages, indent=1))
print(f"total pages: {len(reader.pages)}")
print(f"outline chapters: {len(level1)}  document chapters: {len(ids)}  mapped: {len(pages)}")
if len(pages) != len(ids):
    missing = [c for c in ids if c not in pages]
    print(f"UNMAPPED: {missing}")
