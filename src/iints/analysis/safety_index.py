"""
IINTS-AF Safety Index
=====================
A single composite metric (0–100) that summarises the clinical safety of an
insulin dosing algorithm's output.  Higher is safer.

The index is designed to be:
* **Clinically meaningful** — components map directly to FDA SaMD guidance
  and ATTD/ADA consensus recommendations.
* **Configurable** — researchers can adjust component weights via CLI flags
  or direct API arguments to match the emphasis of their study protocol.
* **Transparent** — every component and its contribution is reported
  separately so the index can be audited and reproduced.

Grades
------
A  ≥ 90   Excellent — suitable for well-controlled clinical setting
B  ≥ 75   Good      — minor improvement recommended
C  ≥ 60   Acceptable — targeted optimisation required
D  ≥ 40   Poor      — significant safety concerns
F  < 40   Fail      — unsafe; do not proceed to clinical evaluation

References
----------
* Battelino et al. (2019) ATTD International Consensus on CGM Metrics.
* Kovatchev et al. (2006) Symmetrisation of the Blood Glucose Measurement Scale.
* FDA Guidance: Software as a Medical Device (SaMD).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default component weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_below54":     0.40,   # time-below-range critical (<54 mg/dL) — most dangerous
    "w_below70":     0.25,   # time-below-range mild (<70 mg/dL)
    "w_supervisor":  0.20,   # safety supervisor trigger rate (interventions/hr)
    "w_recovery":    0.10,   # mean hypoglycaemia episode duration
    "w_tail":        0.05,   # tail-risk binary (any glucose ever < 54 mg/dL)
}

# Normalisation scales — the raw value at which the component contributes
# its FULL 100-point penalty.  Values above the scale are clamped to 100.
NORM_SCALES: Dict[str, float] = {
    "w_below54":    5.0,    # 5 % TBR critical  →  full penalty
    "w_below70":   20.0,    # 20 % TBR low      →  full penalty
    "w_supervisor": 12.0,   # 12 supervisor triggers/hr → full penalty
    "w_recovery":    1.0,   # 1 hour mean episode duration → full penalty
    "w_tail":        1.0,   # binary 0 or 1
}


def _validate_weights(weights: Dict[str, float]) -> None:
    for key in DEFAULT_WEIGHTS:
        if key not in weights:
            raise ValueError(f"Missing weight key: {key!r}")
        if weights[key] < 0:
            raise ValueError(f"Weight {key!r} must be >= 0, got {weights[key]}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Sum of all weights must be > 0")


def _norm(value: float, scale: float) -> float:
    """Normalise raw value to [0, 100] given the full-penalty scale."""
    if scale <= 0:
        return 0.0
    return min(value / scale, 1.0) * 100.0


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _interpret(score: float, penalties: Dict[str, float], weights: Dict[str, float]) -> str:
    """Generate a one-sentence human-readable interpretation."""
    grade = _grade(score)
    grade_labels = {"A": "Excellent", "B": "Good", "C": "Acceptable", "D": "Poor", "F": "Fail"}
    label = grade_labels[grade]

    # Identify the top penalty contributor
    weighted = {k: weights[k] * _norm(v, NORM_SCALES[k]) for k, v in penalties.items()}
    top_key = max(weighted, key=weighted.get)
    key_labels = {
        "w_below54":    "time critically below 54 mg/dL",
        "w_below70":    "time below 70 mg/dL",
        "w_supervisor": "safety supervisor trigger rate",
        "w_recovery":   "hypoglycaemia episode duration",
        "w_tail":       "occurrence of critical hypoglycaemia",
    }

    if score >= 90:
        return f"{label}: Minimal safety concerns across all components."
    dominant = key_labels[top_key]
    return f"{label}: Primary concern is {dominant} (score={score:.1f}/100)."


# ---------------------------------------------------------------------------
# Main result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SafetyIndexResult:
    """Result of the IINTS Safety Index computation."""

    score: float
    """Composite safety score in [0, 100].  Higher is safer."""

    grade: str
    """Letter grade: A, B, C, D, or F."""

    components: Dict[str, float]
    """Normalised (0–100) per-component penalty before weighting."""

    penalties: Dict[str, float]
    """Raw penalty values (before normalisation and weighting)."""

    weights: Dict[str, float]
    """Component weights used in this computation."""

    interpretation: str
    """Human-readable one-sentence summary."""

    def to_dict(self) -> dict:
        return {
            "safety_index": self.score,
            "grade": self.grade,
            "interpretation": self.interpretation,
            "components": self.components,
            "penalties": self.penalties,
            "weights": self.weights,
        }

    def __str__(self) -> str:
        return (
            f"Safety Index: {self.score:.1f}/100  (Grade: {self.grade})\n"
            f"  {self.interpretation}\n"
            f"  Components: "
            + ", ".join(f"{k.replace('w_', '')}={v:.1f}" for k, v in self.components.items())
        )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_hypo_episode_duration(glucose: pd.Series, time_step_minutes: float) -> float:
    """
    Compute mean hypoglycaemia episode duration in minutes.

    An episode starts when glucose drops below 70 mg/dL and ends when it
    rises back above 70 mg/dL.  Returns 0.0 if no episodes.
    """
    below = (glucose < 70.0).values
    if not below.any():
        return 0.0

    episode_lengths: list[int] = []
    in_episode = False
    length = 0
    for b in below:
        if b:
            in_episode = True
            length += 1
        else:
            if in_episode:
                episode_lengths.append(length)
                in_episode = False
                length = 0
    if in_episode and length > 0:
        episode_lengths.append(length)

    if not episode_lengths:
        return 0.0
    return float(np.mean(episode_lengths)) * time_step_minutes


def compute_safety_index(
    results_df: pd.DataFrame,
    safety_report: Dict,
    duration_minutes: int,
    weights: Optional[Dict[str, float]] = None,
    time_step_minutes: float = 5.0,
) -> SafetyIndexResult:
    """
    Compute the IINTS Safety Index from a simulation run.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``Simulator.run_batch()``.  Must contain
        ``glucose_actual_mgdl`` and ``safety_triggered`` columns.
    safety_report : dict
        Safety report dict returned by ``Simulator.run_batch()``.
        Must contain ``bolus_interventions_count`` (int).
    duration_minutes : int
        Total simulation duration in minutes.
    weights : dict, optional
        Component weights.  If None, ``DEFAULT_WEIGHTS`` are used.
        Must contain keys: ``w_below54``, ``w_below70``, ``w_supervisor``,
        ``w_recovery``, ``w_tail``.
    time_step_minutes : float
        Simulator time step in minutes (default 5).

    Returns
    -------
    SafetyIndexResult
    """
    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)
    else:
        weights = dict(weights)  # defensive copy
    _validate_weights(weights)

    glucose = results_df["glucose_actual_mgdl"].astype(float)
    n = max(len(glucose), 1)

    # --- Component 1: TBR critical < 54 mg/dL (%) ---
    tbr_critical = float((glucose < 54.0).sum()) / n * 100.0

    # --- Component 2: TBR low < 70 mg/dL (%) ---
    tbr_low = float((glucose < 70.0).sum()) / n * 100.0

    # --- Component 3: Supervisor trigger rate (triggers / hour) ---
    duration_hours = max(duration_minutes / 60.0, 1e-9)
    interventions = int(safety_report.get("bolus_interventions_count", 0))
    supervisor_rate = interventions / duration_hours

    # --- Component 4: Mean hypo episode duration (hours) ---
    mean_episode_hr = _compute_hypo_episode_duration(glucose, time_step_minutes) / 60.0

    # --- Component 5: Tail-risk binary (1 if any glucose < 54, else 0) ---
    tail_risk = 1.0 if (glucose < 54.0).any() else 0.0

    penalties: Dict[str, float] = {
        "w_below54":    tbr_critical,
        "w_below70":    tbr_low,
        "w_supervisor": supervisor_rate,
        "w_recovery":   mean_episode_hr,
        "w_tail":       tail_risk,
    }

    components: Dict[str, float] = {
        k: _norm(penalties[k], NORM_SCALES[k]) for k in penalties
    }

    # --- Weighted sum of penalties ---
    total_weight = sum(weights.values())
    weighted_penalty = sum(
        weights[k] * components[k] for k in weights
    ) / total_weight

    score = float(np.clip(100.0 - weighted_penalty, 0.0, 100.0))
    grade = _grade(score)
    interpretation = _interpret(score, penalties, weights)

    return SafetyIndexResult(
        score=score,
        grade=grade,
        components=components,
        penalties=penalties,
        weights=dict(weights),
        interpretation=interpretation,
    )


def safety_weights_from_cli(
    w_below54: Optional[float] = None,
    w_below70: Optional[float] = None,
    w_supervisor: Optional[float] = None,
    w_recovery: Optional[float] = None,
    w_tail: Optional[float] = None,
) -> Dict[str, float]:
    """
    Build a weights dict from CLI arguments.  Any ``None`` values fall back
    to the corresponding default weight.

    Intended usage in CLI commands::

        weights = safety_weights_from_cli(
            w_below54=args.safety_w_below54,
            w_below70=args.safety_w_below70,
            ...
        )
        result = compute_safety_index(df, report, duration, weights=weights)
    """
    defaults = dict(DEFAULT_WEIGHTS)
    return {
        "w_below54":    w_below54    if w_below54    is not None else defaults["w_below54"],
        "w_below70":    w_below70    if w_below70    is not None else defaults["w_below70"],
        "w_supervisor": w_supervisor if w_supervisor is not None else defaults["w_supervisor"],
        "w_recovery":   w_recovery   if w_recovery   is not None else defaults["w_recovery"],
        "w_tail":       w_tail       if w_tail       is not None else defaults["w_tail"],
    }
