"""The gates that make an arm comparison a comparison.

``export_arm_comparison.py`` exists to answer one question: did changing the
training loss change held-out behaviour? Every way that question can be
answered wrongly is a way two arms can differ other than in the loss — a
different held-out split, a different early-stopping set, different evaluation
windows, or a second config key that moved along with the loss. Each of those
raises here instead of producing a number.

The last two tests cover the opposite failure: not a confounded comparison, but
an honest one that is over-read. A two-subject split supports a direction, not
an interval, and a hypo gain obtained by predicting lower everywhere is a bias
rather than a forecast.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="export_arm_comparison.py loads checkpoints via _evidence_common, which needs torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

_spec = importlib.util.spec_from_file_location(
    "export_arm_comparison", ROOT / "research/export_arm_comparison.py"
)
eac = importlib.util.module_from_spec(_spec)
sys.modules["export_arm_comparison"] = eac
_spec.loader.exec_module(eac)

from _evidence_common import Fold  # noqa: E402

STEP = 5.0
FINGERPRINT = {
    "horizon_minutes": 120, "history_minutes": 120, "loss": "band_weighted",
    "hidden_size": 128, "learning_rate": 0.001, "batch_size": 256,
}


def _fold(name: str,
          *,
          loss: str = "band_weighted",
          test=("540", "575"),
          val=("570", "588"),
          best_epoch: int = 5,
          epoch_cap: int = 12,
          predicted: np.ndarray | None = None,
          reference: np.ndarray | None = None,
          subjects: np.ndarray | None = None,
          extra_fingerprint: dict | None = None) -> Fold:
    ref = reference if reference is not None else np.tile(np.linspace(120.0, 90.0, 25), (8, 1))
    subs = subjects if subjects is not None else np.array(["540"] * 4 + ["575"] * 4)
    pred = predicted if predicted is not None else ref - 5.0
    fingerprint = dict(FINGERPRINT, loss=loss, **(extra_fingerprint or {}))
    return Fold(
        name=name,
        reference=ref,
        predicted=pred,
        persistence=np.tile(ref[:, :1], (1, ref.shape[1])),
        subjects=subs,
        step_minutes=STEP,
        provenance={
            "test_subjects": list(test), "val_subjects": list(val),
            "best_epoch": best_epoch, "epoch_cap": epoch_cap,
            "config_fingerprint": fingerprint, "horizon_minutes": 120,
        },
    )


def test_identical_arms_are_accepted_and_report_what_varied():
    design = eac._check_arms_are_comparable({
        "a": _fold("a", loss="band_weighted"),
        "b": _fold("b", loss="safety_weighted"),
    })
    assert design["varied_fingerprint_keys"] == ["loss"]
    assert design["shared_test_subjects"] == ["540", "575"]
    assert "horizon_minutes" in design["held_constant"]


def test_different_held_out_subjects_are_refused():
    with pytest.raises(SystemExit, match="different subjects"):
        eac._check_arms_are_comparable({
            "a": _fold("a", test=("540", "575")),
            "b": _fold("b", test=("563", "570"), loss="safety_weighted"),
        })


def test_different_validation_subjects_are_refused():
    """Early stopping consumed them, so an easier val set is a hidden advantage."""
    with pytest.raises(SystemExit, match="validation subjects"):
        eac._check_arms_are_comparable({
            "a": _fold("a", val=("570", "588")),
            "b": _fold("b", val=("563", "591"), loss="safety_weighted"),
        })


def test_a_second_varying_config_key_is_refused_as_confounded():
    with pytest.raises(SystemExit, match="confound"):
        eac._check_arms_are_comparable({
            "a": _fold("a"),
            "b": _fold("b", loss="safety_weighted",
                       extra_fingerprint={"horizon_minutes": 60}),
        })


def test_arms_trained_on_different_data_are_refused():
    """The data path stays equal while the pack behind it is rebuilt."""
    a = _fold("a", loss="band_weighted")
    b = _fold("b", loss="safety_weighted")
    a.provenance["data_sha256_at_training"] = "a" * 64
    b.provenance["data_sha256_at_training"] = "b" * 64
    with pytest.raises(SystemExit, match="trained on different data"):
        eac._check_arms_are_comparable({"a": a, "b": b})


def test_arms_trained_on_the_same_data_are_accepted():
    a = _fold("a", loss="band_weighted")
    b = _fold("b", loss="safety_weighted")
    for f in (a, b):
        f.provenance["data_sha256_at_training"] = "c" * 64
    design = eac._check_arms_are_comparable({"a": a, "b": b})
    assert design["varied_fingerprint_keys"] == ["loss"]


def test_the_ingredient_under_test_can_be_the_target_parameterization():
    """A delta-vs-level contrast is a different experiment, not a wider loss set."""
    design = eac._check_arms_are_comparable(
        {
            "level": _fold("level", extra_fingerprint={"predict_delta": False}),
            "delta": _fold("delta", extra_fingerprint={"predict_delta": True}),
        },
        under_test="target_parameterization",
    )
    assert design["varied_fingerprint_keys"] == ["predict_delta"]
    assert design["under_test"] == "target_parameterization"
    assert "loss" in design["held_constant"]


def test_varying_the_parameterization_is_refused_when_the_loss_is_under_test():
    """The presets must not blur into each other in either direction."""
    with pytest.raises(SystemExit, match="confound"):
        eac._check_arms_are_comparable({
            "a": _fold("a", extra_fingerprint={"predict_delta": False}),
            "b": _fold("b", extra_fingerprint={"predict_delta": True}),
        })


def test_changing_both_loss_and_parameterization_is_refused():
    """Two ingredients at once answers neither question."""
    with pytest.raises(SystemExit, match="confound"):
        eac._check_arms_are_comparable(
            {
                "a": _fold("a", loss="band_weighted",
                           extra_fingerprint={"predict_delta": False}),
                "b": _fold("b", loss="safety_weighted",
                           extra_fingerprint={"predict_delta": True}),
            },
            under_test="target_parameterization",
        )


def test_arms_that_differ_in_nothing_are_refused():
    """Otherwise run-to-run noise is reported as a null result for the ingredient."""
    with pytest.raises(SystemExit, match="identical in every fingerprinted setting"):
        eac._check_arms_are_comparable({
            "a": _fold("a", loss="band_weighted"),
            "b": _fold("b", loss="band_weighted"),
        })


def test_an_unknown_ingredient_is_refused():
    with pytest.raises(SystemExit, match="unknown ingredient"):
        eac._check_arms_are_comparable(
            {"a": _fold("a"), "b": _fold("b", loss="safety_weighted")},
            under_test="hidden_size",
        )


def test_mismatched_evaluation_windows_are_refused():
    other = np.tile(np.linspace(150.0, 100.0, 25), (8, 1))
    with pytest.raises(SystemExit, match="reference trajectories differ"):
        eac._check_arms_are_comparable({
            "a": _fold("a"),
            "b": _fold("b", loss="safety_weighted", reference=other),
        })


def test_training_that_hit_the_epoch_cap_is_refused():
    with pytest.raises(SystemExit, match="epoch cap"):
        eac._check_arms_are_comparable({
            "a": _fold("a"),
            "b": _fold("b", loss="safety_weighted", best_epoch=12, epoch_cap=12),
        })


def test_no_interval_is_reported_below_the_subject_minimum():
    subs = np.array(["540"] * 4 + ["575"] * 4)
    block = eac._estimate(np.arange(8, dtype=float), subs)
    assert block["interval"] is None
    assert "below the minimum" in block["interval_omitted_because"]
    assert block["n_subjects"] == 2 and block["n_pairs"] == 8


def test_an_interval_appears_once_there_are_enough_subjects():
    subs = np.repeat(["540", "575", "591", "563"], 3)
    block = eac._estimate(np.arange(12, dtype=float), subs)
    assert block["interval"] is not None
    assert block["interval"]["n_subjects"] == 4


def test_offset_control_removes_a_pure_bias():
    """A control shifted by the fitted constant must equal a purely shifted arm."""
    control = np.tile(np.linspace(120.0, 90.0, 25), (8, 1))
    treatment = control - 17.0
    shifted, offset = eac._offset_control(control, treatment)
    assert offset == pytest.approx(-17.0)
    assert np.allclose(shifted, treatment), (
        "a treatment that is exactly the control minus a constant must be "
        "reproduced by the offset control, otherwise the control cannot detect "
        "a bias-only improvement"
    )


def test_offset_control_does_not_absorb_a_genuine_dynamics_change():
    control = np.tile(np.linspace(120.0, 120.0, 25), (8, 1))
    treatment = np.tile(np.linspace(120.0, 60.0, 25), (8, 1))
    shifted, _ = eac._offset_control(control, treatment)
    assert not np.allclose(shifted, treatment), (
        "an arm that forecasts a descent the control does not must survive the "
        "offset control, otherwise the control would explain away real signal"
    )
