---
tags:
  - iints/glossary
  - iints/physiology
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Physiological Parameter Cheat Sheet

> [!warning] Interpretation
> These terms help users understand simulation configs. They are not instructions for therapy.

| Parameter | Unit | What it changes in simulation | Watch out for |
| --- | --- | --- | --- |
| `initial_glucose_mgdl` | mg/dL | starting glucose | extreme starts can dominate short runs |
| `basal_rate_u_per_hour` | U/h | background insulin assumption | too high can force hypoglycemia |
| `isf_mgdl_per_unit` | mg/dL/U | glucose drop per insulin unit | lower ISF means stronger insulin effect |
| `icr_g_per_unit` | g/U | carbs covered by one unit | small ICR means larger meal boluses |
| `glucose_decay_rate` | per minute or model-specific | return/drift behavior | large values can crash glucose unrealistically |
| meal carbs | grams | meal disturbance | needs absorption timing context |
| exercise event | scenario event | can raise hypo risk | must be interpreted cautiously |

## Source Links

- [[Physiology and Safety Sources]]
- [[Sources by SDK Feature]]
- [[Realism Reference Envelopes]]
