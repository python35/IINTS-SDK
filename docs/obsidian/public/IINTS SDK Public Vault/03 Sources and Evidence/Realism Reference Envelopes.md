---
tags:
  - iints/sources
  - iints/realism
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Realism Reference Envelopes

These are empirical envelopes used by the SDK to decide whether simulated data looks plausible compared with local/public T1D data packs. They are not clinical targets; they are **sanity bands** for simulation quality.

| Reference ID | Label | Datasets | Source | Key medians |
| --- | --- | --- | --- | --- |
| free_living_t1d | Free-Living T1D Daily Envelope | azt1d, hupa_ucm | Derived from local AZT1D and HUPA-UCM public packs bundled with the workspace | mean_glucose_mgdl: median 141.306<br>cv_pct: median 28.083<br>tir_70_180_pct: median 78.819<br>tir_below_70_pct: median 0.694<br>median_peak_lag_minutes: median 82.5<br>median_meal_rise_mgdl: median 53.0 |
| azt1d_daily | AZT1D Daily Envelope | azt1d | Derived from the local AZT1D public pack in data_packs/public/azt1d | mean_glucose_mgdl: median 143.868<br>cv_pct: median 26.602<br>tir_70_180_pct: median 80.696<br>tir_below_70_pct: median 0.0<br>median_peak_lag_minutes: median 82.5<br>median_meal_rise_mgdl: median 53.0 |
| hupa_ucm_daily | HUPA-UCM Daily Envelope | hupa_ucm | Derived from the local HUPA-UCM public pack in data_packs/public/hupa_ucm | mean_glucose_mgdl: median 136.481<br>cv_pct: median 29.937<br>tir_70_180_pct: median 75.868<br>tir_below_70_pct: median 2.778 |

## How Users Should Read This

- A passing realism check means "the synthetic trace is not obviously weird against these reference envelopes".
- It does not mean the algorithm is clinically safe.
- For reports, pair this page with [[Dataset Source Library]] and [[Physiology and Safety Sources]].
