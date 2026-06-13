---
tags:
  - iints/sources
  - iints/writing
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# How To Use Sources In A Report

Use this page when writing EUCYS text, a README, a technical report, or a slide.

## Safe Claim Templates

| Claim type | Good wording | Avoid |
| --- | --- | --- |
| SDK purpose | "IINTS is a pre-clinical research SDK for simulation, validation, and reproducible review." | "IINTS controls diabetes treatment." |
| Metrics | "Reports use TIR/TBR/TAR terminology aligned with ADA/ATTD references." | "A good TIR proves the algorithm is safe for patients." |
| Data realism | "Synthetic traces are compared against empirical public-data envelopes." | "The simulator exactly represents every person with diabetes." |
| Pump lab | "The Pico flow packages bench-only firmware artifacts and dry-run upload metadata." | "The SDK can upload a safe insulin pump controller." |
| Local AI | "Local models can assist research workflows and summarize outputs." | "AI autonomously doses insulin." |

## Citation Workflow

1. Open [[Sources by SDK Feature]].
2. Pick the feature you are discussing.
3. Open the linked source notes.
4. Copy the citation from the note.
5. Add a sentence that states the limitation.

## One Paragraph You Can Reuse

IINTS is grounded in published diabetes metrics, public dataset provenance, simulator literature, and explicit hardware-safety boundaries. Its outputs are intended for research simulation, reproducible benchmarking, and education. They are not treatment recommendations and do not replace clinical validation or medical-device certification.
