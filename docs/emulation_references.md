# Commercial Algorithm Emulation References (v0.1.15)

This document summarizes the **best‑effort** references used for the emulator models in IINTS‑AF.
These emulators are **approximations**, intended for research comparisons — not clinical use.

---

## What’s Implemented

- Medtronic MiniMed 780G: `src/iints/emulation/medtronic_780g.py`
- Tandem t:slim X2 Control‑IQ: `src/iints/emulation/tandem_controliq.py`
- Omnipod 5: `src/iints/emulation/omnipod_5.py`

Each emulator exposes `emulate_decision(...)` and includes `get_sources()` citations.

---

## Methodology (High‑Level)

1. **Parameter extraction** from public clinical papers, FDA summaries, and user manuals.
2. **Behavioral modeling** of typical decision patterns (auto‑basal, correction bolus, suspend logic).
3. **Safety constraints** encoded via `SafetyLimits`.

**Limitations**
- Algorithms are proprietary; these are informed approximations.
- Parameter ranges may differ by region / firmware.
- Emulators represent a “typical” behavior, not individual personalization.

---

## References (Examples)

### Medtronic 780G
- Bergenstal et al. (2020), NEJM: DOI 10.1056/NEJMoa2003479
- FDA 510(k) K193510
- Medtronic 780G User Guide (public manual)

### Tandem Control‑IQ
- Brown et al. (2019), Diabetes Technology & Therapeutics: DOI 10.1089/dia.2019.0226
- FDA 510(k) K191289
- Control‑IQ User Guide

### Omnipod 5
- Omnipod 5 Pivotal / ASSERT trial publications
- FDA 510(k) K203467
- Omnipod 5 User Guide

---

## Example Usage

```python
from iints.emulation.medtronic_780g import Medtronic780GEmulator

emulator = Medtronic780GEmulator()

# Example inputs
decision = emulator.emulate_decision(
    glucose=180.0,
    velocity=1.2,           # mg/dL/min
    insulin_on_board=1.4,   # units
    carbs=0.0,
    current_time=0.0,
)

print(decision.insulin_delivered)
print(decision.action)
print(decision.reasoning)
```

---

## Next Improvements

- Verify parameters against updated manuals and peer‑reviewed sources.
- Add explicit citations into emulator output metadata.
- Add regression tests against published outcome ranges.
