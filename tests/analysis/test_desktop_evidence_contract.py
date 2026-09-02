"""Pin the contract between the evidence exporter and the desktop app.

The desktop Clarke chart reads a JSON file produced by
``research/export_desktop_evidence.py``. If a field is renamed on one side the
chart does not crash — it silently falls back to the synthetic demonstration,
which is exactly the failure mode this project is trying to get rid of. These
tests fail loudly instead.

The subject-disjointness test is the one that matters scientifically: an error
grid computed over subjects the model trained on, or over the subjects used for
early stopping, is not an out-of-sample result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "apps/iints-tauri/frontend/evidence/forecast_evidence.json"
FRONTEND_MAIN = REPO_ROOT / "apps/iints-tauri/frontend/main.js"
TAURI_CONF = REPO_ROOT / "apps/iints-tauri/src-tauri/tauri.conf.json"

# Every path the frontend dereferences on the evidence payload.
REQUIRED_PATHS = [
    ("provenance", "horizon_minutes"),
    ("provenance", "test_subjects"),
    ("provenance", "train_subjects"),
    ("provenance", "val_subjects"),
    ("provenance", "model_sha256"),
    ("provenance", "data_sha256"),
    ("pooled", "model", "n_pairs"),
    ("pooled", "model", "mae"),
    ("pooled", "model", "hazardous_pct"),
    ("pooled", "model", "zone_percentages", "A"),
    ("pooled", "model", "zone_percentages", "B"),
    ("pooled", "persistence", "mae"),
    ("pooled", "persistence", "zone_percentages", "A"),
    ("subject_level_zone_a", "n_subjects"),
    ("subject_level_zone_a", "min"),
    ("subject_level_zone_a", "max"),
    ("scatter", "reference"),
    ("scatter", "predicted"),
]


@pytest.fixture(scope="module")
def evidence() -> dict:
    if not EVIDENCE.exists():
        pytest.skip(
            "No forecast evidence exported; run research/export_desktop_evidence.py"
        )
    return json.loads(EVIDENCE.read_text())


def _dig(payload, path):
    node = payload
    for key in path:
        assert isinstance(node, dict), f"{'.'.join(path)}: {key} not reachable"
        assert key in node, f"missing evidence field: {'.'.join(path)}"
        node = node[key]
    return node


@pytest.mark.parametrize("path", REQUIRED_PATHS, ids=lambda p: ".".join(p))
def test_frontend_required_fields_present(evidence, path):
    assert _dig(evidence, path) is not None


def test_zone_percentages_are_a_partition(evidence):
    for arm in ("model", "persistence"):
        pct = evidence["pooled"][arm]["zone_percentages"]
        assert set(pct) == set("ABCDE")
        assert sum(pct.values()) == pytest.approx(100.0, abs=1e-6), arm


def test_hazardous_matches_zone_sum(evidence):
    for arm in ("model", "persistence"):
        block = evidence["pooled"][arm]
        pct = block["zone_percentages"]
        assert block["hazardous_pct"] == pytest.approx(
            pct["C"] + pct["D"] + pct["E"], abs=1e-6
        ), arm


def test_counts_agree_with_percentages(evidence):
    block = evidence["pooled"]["model"]
    total = sum(block["zone_counts"].values())
    assert total == block["n_pairs"]
    for zone, count in block["zone_counts"].items():
        assert block["zone_percentages"][zone] == pytest.approx(
            100.0 * count / total, abs=1e-6
        )


def test_held_out_subjects_are_disjoint_from_training(evidence):
    prov = evidence["provenance"]
    test = {str(s) for s in prov["test_subjects"]}
    train = {str(s) for s in prov["train_subjects"]}
    val = {str(s) for s in prov["val_subjects"]}
    assert test, "no held-out subjects recorded"
    assert not (test & train), f"test subjects seen in training: {sorted(test & train)}"
    assert not (test & val), (
        f"test subjects were used for model selection: {sorted(test & val)}"
    )


def test_pooled_pairs_exceed_displayed_scatter(evidence):
    """Percentages must come from all pairs, never from the thinned scatter."""
    n_shown = len(evidence["scatter"]["reference"])
    assert n_shown == len(evidence["scatter"]["predicted"])
    assert n_shown <= evidence["pooled"]["model"]["n_pairs"]


def test_per_subject_zone_a_range_matches_summary(evidence):
    zone_a = [
        block["model"]["zone_percentages"]["A"]
        for block in evidence["per_subject"].values()
    ]
    summary = evidence["subject_level_zone_a"]
    assert summary["n_subjects"] == len(zone_a)
    assert summary["min"] == pytest.approx(min(zone_a))
    assert summary["max"] == pytest.approx(max(zone_a))


def test_csp_permits_loading_the_evidence_file():
    """Without 'self' in connect-src the packaged app cannot fetch the JSON."""
    csp = json.loads(TAURI_CONF.read_text())["app"]["security"]["csp"]
    connect = next(
        (d for d in csp.split(";") if d.strip().startswith("connect-src")), ""
    )
    assert "'self'" in connect, f"connect-src must allow 'self', got: {connect.strip()}"


def test_frontend_falls_back_instead_of_inventing_numbers():
    source = FRONTEND_MAIN.read_text()
    assert "loadForecastEvidence" in source
    # The synthetic branch must still label itself.
    assert "drawSyntheticBanner" in source
    # No hand-written zone percentage may survive in the Clarke label.
    assert not re.search(r'Zone A:\s*\d+\.\d+%', source), (
        "a literal zone percentage is present in the frontend source"
    )
