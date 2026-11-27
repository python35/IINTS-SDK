"""Where the research-layer constants come from, and which ones are guesses.

Several constants in the transplant and C-peptide models are plausible but not
derived from a cited measurement. They were documented in
``docs/DEFECT_REGISTER_NL.md`` as open points, which means a reader of the
documentation knows about them and a user of the code does not.

This module closes that gap. Every constant that is not traceable to a source
is registered here with its kind, and the code paths whose *absolute* output
depends on it call :func:`warn_unsourced` once per process. Relative
comparisons at a fixed parameter set are unaffected and never warn.

Kinds
-----
``literature``
    Traceable to a cited measurement. Listed for completeness; never warns.
``convention``
    A deliberate unit-defining choice. The number is arbitrary but the unit it
    creates is meaningful, and results scale with it.
``assumption``
    A number chosen to express a qualitative claim. Replacing it changes the
    conclusion, not only the scale.
``uncalibrated``
    A parameter whose value sets the size of an effect that has never been
    fitted to data.

Research and education only; not a medical device.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "UnsourcedConstantWarning",
    "ProvenanceEntry",
    "RESEARCH_PROVENANCE",
    "warn_unsourced",
    "reset_warning_state",
    "unsourced_names",
]


class UnsourcedConstantWarning(UserWarning):
    """An absolute result depends on a constant that has no cited source."""


@dataclass(frozen=True)
class ProvenanceEntry:
    """One constant, where it comes from, and what depends on it."""

    name: str
    module: str
    kind: str
    value: float
    note: str

    @property
    def sourced(self) -> bool:
        return self.kind == "literature"


RESEARCH_PROVENANCE: Tuple[ProvenanceEntry, ...] = (
    ProvenanceEntry(
        name="REFERENCE_BETA_MASS_IEQ_PER_KG",
        module="iints.research.stem_cell_transplant",
        kind="convention",
        value=14000.0,
        note=(
            "Derived from the order of magnitude of about one million islet "
            "equivalents in an adult pancreas. It defines what a graft mass of "
            "1.0 means, so every dose expressed in IEQ/kg scales with it. "
            "Replace with a derived value before interpreting an IEQ/kg dose "
            "quantitatively."
        ),
    ),
    ProvenanceEntry(
        name="DEFAULT_SHORT_FRACTION",
        module="iints.research.c_peptide",
        kind="assumption",
        value=0.76,
        note=(
            "The share of C-peptide clearance carried by the short half-life "
            "component. A common value, not yet checked against the original "
            "Van Cauter table. Absolute C-peptide concentrations depend on it; "
            "relative comparisons at a fixed parameter set do not."
        ),
    ),
    ProvenanceEntry(
        name="ISLET_FUNCTION_PRESETS['sc_islet_stage6'].first_phase_fraction",
        module="iints.research.stem_cell_transplant",
        kind="assumption",
        value=0.25,
        note=(
            "Expresses that stem-cell derived islets differ from primary "
            "islets by a first-phase deficit. The existence of the deficit is "
            "reported; this particular depth is not. It should come from "
            "published C-peptide curves."
        ),
    ),
    ProvenanceEntry(
        name="potentiation_gain_per_mgdl",
        module="iints.research.stem_cell_transplant",
        kind="uncalibrated",
        value=0.02,
        note=(
            "Together with first_phase_capacity_u_per_kg this sets the height "
            "of the early secretion peak. Plausible in order of magnitude but "
            "never fitted; an intravenous glucose tolerance test with a "
            "reported first-phase response is needed to pin it down."
        ),
    ),
    ProvenanceEntry(
        name="immunosuppression_effect",
        module="iints.research.stem_cell_transplant",
        kind="assumption",
        value=0.0,
        note=(
            "Immunosuppression lowers immune-mediated graft loss. Its known "
            "costs — beta-cell toxicity and impaired secretion — are available "
            "as separate parameters but default to zero, so a run that only "
            "sets immunosuppression_effect overstates the net benefit."
        ),
    ),
    ProvenanceEntry(
        name="CPEPTIDE_PMOL_PER_INSULIN_UNIT",
        module="iints.research.c_peptide",
        kind="literature",
        value=6000.0,
        note=(
            "Equimolar C-peptide and insulin secretion, with one unit of "
            "insulin taken as about 6 nmol. Standard conversion."
        ),
    ),
)

_PROVENANCE_BY_NAME: Dict[str, ProvenanceEntry] = {
    entry.name: entry for entry in RESEARCH_PROVENANCE
}
_ALREADY_WARNED: set = set()


def unsourced_names() -> Tuple[str, ...]:
    """Names of every registered constant that is not traceable to a source."""
    return tuple(entry.name for entry in RESEARCH_PROVENANCE if not entry.sourced)


def reset_warning_state() -> None:
    """Forget which constants have already warned. For tests."""
    _ALREADY_WARNED.clear()


def _article(kind: str) -> str:
    return "an" if kind[0] in "aeiou" else "a"


def warn_unsourced(name: str, *, stacklevel: int = 3) -> None:
    """Warn once per process that an absolute result rests on a guess.

    Raises :class:`KeyError` for an unregistered name, so a constant cannot be
    flagged without also being documented here.
    """
    entry = _PROVENANCE_BY_NAME[name]
    if entry.sourced or name in _ALREADY_WARNED:
        return
    _ALREADY_WARNED.add(name)
    warnings.warn(
        f"{entry.name} is {_article(entry.kind)} {entry.kind}, not a sourced "
        f"measurement "
        f"({entry.value:g}). {entry.note} See docs/DEFECT_REGISTER_NL.md.",
        UnsourcedConstantWarning,
        stacklevel=stacklevel,
    )
