"""Losses that carry a glucose threshold, tested against one silent failure mode.

``band_weighted``, ``safety_weighted`` and the physiology penalties all express
themselves in mg/dL: weight the sample more if the glucose is below 70, penalise
a rate above 3 mg/dL/min, and so on. Those statements are about a *level*.

When the predictor is trained on deltas (``predictor.predict_delta``) the target
is a change instead, and a level threshold applied to a change does not fail --
it degenerates. Almost every change is below 70 and none is above 180, so the
band weighting collapses to one constant weight for the whole dataset, which is
plain MSE with a rescaled learning rate. The run still prints
``Loss: band_weighted`` and the config still says the hypo band is weighted 2.5x.

These tests therefore do not check that the losses run. They check that the
degeneration is gone and cannot come back unnoticed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from iints.research.losses import (  # noqa: E402
    BandWeightedMSE,
    BandWeightedPINNLoss,
    PhysiologicalPINNLoss,
    SafetyWeightedMSE,
)

FEATURES = ["glucose_actual_mgdl", "insulin_units", "carb_grams"]
GLUCOSE_IDX = FEATURES.index("glucose_actual_mgdl")


def _window(anchors: list[float], history: int = 4) -> "torch.Tensor":
    """An unscaled input window whose last glucose reading is ``anchors[i]``."""
    x = torch.zeros(len(anchors), history, len(FEATURES))
    for i, a in enumerate(anchors):
        x[i, :, GLUCOSE_IDX] = float(a)
    return x


class TestBandWeightedAnchoring:
    # One sample sits in the hypo band after reconstruction, the other in the
    # hyper band. Both have a delta target of zero, so nothing distinguishes them
    # unless the loss looks at the anchor.
    ANCHORS = [60.0, 200.0]
    DELTA_TARGETS = torch.zeros(2, 1)
    DELTA_PREDS = torch.full((2, 1), 10.0)  # 10 mg/dL of error on both samples

    def test_without_anchor_delta_targets_collapse_to_a_constant_weight(self):
        """The bug, stated as an equality: unanchored, this is just scaled MSE."""
        loss = BandWeightedMSE()
        assert loss.anchor_index is None
        value = loss(self.DELTA_PREDS, self.DELTA_TARGETS)
        mse = ((self.DELTA_PREDS - self.DELTA_TARGETS) ** 2).mean()
        # Both deltas are "below 70", so both get 1 + low_weight and the band
        # structure is gone: 100 * 3.
        assert value == pytest.approx(float(mse) * 3.0)
        assert value == pytest.approx(300.0)

    def test_anchoring_restores_the_band_structure(self):
        loss = BandWeightedMSE(anchor_index=GLUCOSE_IDX)
        value = loss(self.DELTA_PREDS, self.DELTA_TARGETS, _window(self.ANCHORS))
        # Reconstructed levels are 60 (low band, weight 1 + 2.0) and 200 (high
        # band, weight 1 + 1.5): mean(100 * 3, 100 * 2.5).
        assert value == pytest.approx(275.0)

    def test_anchored_loss_is_not_proportional_to_mse(self):
        """The property that distinguishes a real weighting from a rescaling."""
        anchored = BandWeightedMSE(anchor_index=GLUCOSE_IDX)
        window = _window(self.ANCHORS)
        a = anchored(self.DELTA_PREDS, self.DELTA_TARGETS, window)
        # Same errors, same targets, only the glucose level differs -> a loss
        # that weights by level must move; a constant weight cannot.
        b = anchored(self.DELTA_PREDS, self.DELTA_TARGETS, _window([120.0, 120.0]))
        assert float(a) != pytest.approx(float(b))

    def test_hypo_sample_outweighs_euglycemic_sample(self):
        loss = BandWeightedMSE(anchor_index=GLUCOSE_IDX)
        hypo = loss(torch.full((1, 1), 10.0), torch.zeros(1, 1), _window([60.0]))
        eugly = loss(torch.full((1, 1), 10.0), torch.zeros(1, 1), _window([120.0]))
        assert float(hypo) > float(eugly)

    def test_refuses_to_guess_the_level_when_the_window_is_missing(self):
        loss = BandWeightedMSE(anchor_index=GLUCOSE_IDX)
        with pytest.raises(ValueError, match="received no input window"):
            loss(self.DELTA_PREDS, self.DELTA_TARGETS)

    def test_absolute_targets_are_unchanged(self):
        """Regression guard: the default path must behave exactly as before."""
        loss = BandWeightedMSE()
        targets = torch.tensor([[60.0], [200.0]])
        preds = targets + 10.0
        # weights 3.0 and 2.5 as before, computed straight from the targets.
        assert float(loss(preds, targets)) == pytest.approx(275.0)


class TestSafetyWeightedAnchoring:
    def test_without_anchor_a_zero_delta_looks_maximally_hypoglycemic(self):
        loss = SafetyWeightedMSE()
        value = loss(torch.full((2, 1), 10.0), torch.zeros(2, 1))
        # clamp(80 - 0) / 80 = 1 -> weight 1 + 2 * 1 = 3 for every sample.
        assert float(value) == pytest.approx(300.0)

    def test_anchoring_grades_the_weight_by_level(self):
        loss = SafetyWeightedMSE(anchor_index=GLUCOSE_IDX)
        value = loss(torch.full((2, 1), 10.0), torch.zeros(2, 1), _window([60.0, 200.0]))
        # Level 60: clamp(20)/80 = 0.25 -> weight 1.5. Level 200: weight 1.0.
        assert float(value) == pytest.approx(125.0)

    def test_refuses_to_guess_the_level_when_the_window_is_missing(self):
        loss = SafetyWeightedMSE(anchor_index=GLUCOSE_IDX)
        with pytest.raises(ValueError, match="received no input window"):
            loss(torch.full((2, 1), 10.0), torch.zeros(2, 1))


class TestPhysiologyPenaltyAnchoring:
    ANCHOR = 150.0

    def test_unanchored_penalty_fires_on_a_correct_flat_delta(self):
        """A delta of zero at 150 mg/dL is physiologically unremarkable."""
        loss = PhysiologicalPINNLoss(feature_columns=FEATURES)
        assert loss.predict_delta is False
        penalty = loss.physiology_penalty(torch.zeros(1, 3), _window([self.ANCHOR]))
        # Read as a level, a delta of 0 is 0 mg/dL of blood glucose: the lower
        # bound fires, and the first step reads as a 30 mg/dL/min crash.
        assert float(penalty) > 1000.0

    def test_anchoring_leaves_a_flat_delta_unpenalised(self):
        loss = PhysiologicalPINNLoss(feature_columns=FEATURES, predict_delta=True)
        penalty = loss.physiology_penalty(torch.zeros(1, 3), _window([self.ANCHOR]))
        assert float(penalty) == pytest.approx(0.0, abs=1e-6)

    def test_anchoring_still_penalises_a_genuinely_impossible_rate(self):
        """The constraint must stay a constraint, not be switched off."""
        loss = PhysiologicalPINNLoss(feature_columns=FEATURES, predict_delta=True)
        # +100 mg/dL in one 5-minute step is 20 mg/dL/min, well over the limit.
        penalty = loss.physiology_penalty(torch.full((1, 1), 100.0), _window([self.ANCHOR]))
        assert float(penalty) > 0.0

    def test_band_pinn_threads_the_anchor_to_both_terms(self):
        loss = BandWeightedPINNLoss(feature_columns=FEATURES, anchor_index=GLUCOSE_IDX)
        assert loss.predict_delta is True
        assert loss.band_loss.anchor_index == GLUCOSE_IDX
        # Runs end to end on delta targets without the bounds penalty firing.
        value = loss(torch.zeros(1, 3), torch.zeros(1, 3), _window([self.ANCHOR]))
        assert float(value) == pytest.approx(0.0, abs=1e-6)


class TestTrainerWiring:
    """The anchor must be decided by the trainer, not left to the config.

    A forgotten flag here is not a crash but a quietly different experiment, so
    the wiring is pinned rather than documented.
    """

    def _cfg(self, **kw):
        from iints.research.config import PredictorConfig

        defaults = dict(
            history_minutes=240,
            horizon_minutes=60,
            time_step_minutes=5,
            feature_columns=list(FEATURES),
            target_column="glucose_actual_mgdl",
        )
        defaults.update(kw)
        return PredictorConfig(**defaults)

    def test_absolute_targets_need_no_anchor(self):
        from train_predictor import _resolve_anchor_index

        assert _resolve_anchor_index(self._cfg()) is None

    def test_delta_targets_anchor_on_the_glucose_column(self):
        from train_predictor import _resolve_anchor_index

        assert _resolve_anchor_index(self._cfg(predict_delta=True)) == GLUCOSE_IDX

    def test_delta_targets_refuse_an_unobservable_anchor(self):
        from train_predictor import _resolve_anchor_index

        cfg = self._cfg(predict_delta=True, feature_columns=["insulin_units", "carb_grams"])
        with pytest.raises(ValueError, match="among the feature columns"):
            _resolve_anchor_index(cfg)

    def test_anchored_losses_are_handed_the_input_window(self):
        from train_predictor import _loss_needs_inputs

        assert _loss_needs_inputs(BandWeightedMSE(anchor_index=GLUCOSE_IDX)) is True
        assert _loss_needs_inputs(SafetyWeightedMSE(anchor_index=GLUCOSE_IDX)) is True
        assert _loss_needs_inputs(BandWeightedMSE()) is False
        assert _loss_needs_inputs(PhysiologicalPINNLoss(feature_columns=FEATURES)) is True
