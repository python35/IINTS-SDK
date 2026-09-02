"""Parity between the desktop workbench's science.js and the Python SDK.

The Tauri app ports Clarke error-grid classification to JavaScript so charts
can be computed in the renderer. Two implementations of a published rule are
two chances to be wrong, so this test pins them to each other: if either side
is edited and the zone assignments diverge on any pair, this fails.

Skipped when node is unavailable (the port is still exercised by the app).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from iints.analysis.error_grid import clarke_zones

REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENCE_JS = REPO_ROOT / "apps" / "iints-tauri" / "frontend" / "science.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not SCIENCE_JS.exists(),
    reason="node runtime or desktop frontend not available",
)

# Boundary pairs chosen to land on every zone edge in Clarke et al. (1987):
# the 20% band, the 70 mg/dL hypo corner, the C upper/lower cuts at 110 mg/dL
# offset and (7/5)*ref - 182, the D cuts at 240 and 175/3, and the E corners.
EDGE_CASES = [
    (70, 70), (70, 56), (70, 84), (69.9, 69.9), (70.1, 84.2), (70, 180),
    (70, 179), (60, 200), (50, 250), (300, 60), (180, 70), (181, 69),
    (240, 180), (240, 181), (240, 69), (58.34, 70), (58.33, 70), (58, 84),
    (130, 0), (130, 1), (290, 400), (289, 399), (100, 210), (100, 209),
    (175 / 3, 180), (175 / 3, 70), (400, 400), (400, 320), (400, 319),
    (40, 40), (20, 20), (600, 600), (1, 1), (100, 100), (180, 70.1),
    (35, 70), (35, 181), (300, 300), (250, 50), (55, 190), (65, 84), (65, 78),
]


def _run_js(pairs: list[tuple[float, float]]) -> list[str]:
    script = textwrap.dedent(
        f"""
        import {{ clarkeZone }} from "file://{SCIENCE_JS}";
        const pairs = {json.dumps(pairs)};
        console.log(JSON.stringify(pairs.map(([r, p]) => clarkeZone(r, p))));
        """
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return json.loads(proc.stdout)


def test_edge_cases_match_python():
    js = _run_js([(float(r), float(p)) for r, p in EDGE_CASES])
    py = list(clarke_zones([r for r, _ in EDGE_CASES], [p for _, p in EDGE_CASES]))
    assert js == py


def test_edge_cases_reach_every_zone():
    """A parity test over cases that never reach C/D/E would prove little."""
    py = set(clarke_zones([r for r, _ in EDGE_CASES], [p for _, p in EDGE_CASES]))
    assert py == {"A", "B", "C", "D", "E"}


def test_random_sweep_matches_python():
    import random

    rng = random.Random(7)
    pairs = []
    for _ in range(2000):
        ref = 30 + rng.random() * 420
        pred = max(10.0, ref * (0.6 + rng.random() * 0.9) + (rng.random() - 0.5) * 60)
        pairs.append((ref, pred))

    js = _run_js(pairs)
    py = list(clarke_zones([r for r, _ in pairs], [p for _, p in pairs]))
    mismatches = [(p, a, b) for p, a, b in zip(pairs, js, py) if a != b]
    assert not mismatches, f"{len(mismatches)} divergent pairs, first: {mismatches[:3]}"
