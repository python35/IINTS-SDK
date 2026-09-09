---
title: 'IINTS-AF: an in-silico bench for testing diabetes control algorithms under degraded data and device conditions'
tags:
  - Python
  - diabetes technology
  - artificial pancreas
  - glucose simulation
  - reproducible research
authors:
  - name: Rune Bobbaers
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent researcher, Belgium
    index: 1
date: 9 September 2026
bibliography: paper.bib
---

# Summary

IINTS-AF is a Python simulation framework for testing insulin-dosing algorithms
against virtual patients. It integrates ordinary differential equation models of
glucose–insulin physiology — a Bergman-style minimal model [@bergman1979] and a
19-state Hovorka-style model [@hovorka2004], each extended with declared
research terms — and couples them to explicit models of the two devices that sit
between an algorithm and a patient: a continuous glucose monitor with
interstitial lag, autocorrelated noise and compression-low artefacts, and an
insulin pump with dose quantisation, delivery error and interruption.

Every physiological, sensor, safety and training-loss equation in the SDK is
registered in machine-readable form with its canonical expression, LaTeX, units,
implementing source paths, literature DOIs, and an `evidence_class` of
`canonical`, `adapted` or `heuristic`. Sixteen formulas are registered; nine are
marked heuristic, meaning the coefficients are explicit research assumptions
requiring calibration rather than identified values. A deterministic supervisor
sits between any controller and the pump, is configured from a single
fingerprinted safety contract, and can be ablated as an experimental arm.
Analysis utilities implement the standard consensus metrics together with the
Kovatchev symmetrised risk indices [@kovatchev1997; @kovatchev1998].

# Statement of need

In-silico testing is established practice in diabetes technology, and mature
simulators exist: the UVA/Padova simulator [@dallaman2014] is accepted by the
FDA as a substitute for animal trials, and open reimplementations such as
`simglucose` [@xie2018] make its meal model [@dallaman2007] widely available.
These tools answer the question *does this controller work when it is given the
information it expects*.

They are less suited to a different question that matters for deployed systems:
*what happens when the data and the devices are wrong*. Real closed-loop systems
run on uncertified CGM streams, sensors that read falsely low under compression,
pumps that under-deliver, and meals that are announced incorrectly or not at
all. Failures of this kind are a recognised source of harm, yet they are usually
outside the simulation and therefore outside the evidence.

IINTS-AF is built around that question. Data entering the SDK carries a
provenance and certification record; scenarios can inject sensor corruption,
device faults and carbohydrate-announcement error; and experiments are organised
into arms so that a clean, certified condition can be compared against a
deliberately corrupted one and against a supervisor-off ablation on the same
patients, scenarios, seeds and algorithms. A locked 3600-run benchmark ships
with the software as a worked example.

The intended users are researchers who want to characterise how an algorithm
degrades rather than how it performs at its best, and instructors teaching why
sensor lag and insulin-on-board make glucose control difficult. The framework is
deliberately auditable: the same expressions that produce a trajectory also emit
the per-step compartment and flux values, so a reported number can be traced
back to the equation that produced it.

The software is explicitly research and educational. It is not a medical device,
its parameterisation is not clinically validated, and it must not be used to
determine a real insulin dose. The registry, the evidence classes and the
reporting layer are designed to keep that boundary visible in the output rather
than only in the documentation.

# Acknowledgements

Development made use of the OhioT1DM dataset [@marling2020] for held-out
forecasting evaluation.

# References
