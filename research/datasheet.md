# Dataset Datasheet: AZT1D (Real-World T1D Dataset)

## Dataset Name
AZT1D: A Real-World Dataset for Type 1 Diabetes

## Source / Owner
Mendeley Data (Khamesian et al.)

## Description
Real-world dataset from 25 individuals using AID systems. We use the CGM Records subject CSVs for predictor training.

## Data Dictionary
- `timestamp` (EventDateTime)
- `glucose` (CGM, mg/dL)
- `basal` (Basal)
- `bolus` (TotalBolusInsulinDelivered)
- `correction` (CorrectionDelivered)
- `carb_grams` (CarbSize/FoodDelivered)
- `trend` (derived)
- `derived_iob_units`, `derived_cob_grams` (derived)
- `time_of_day_sin`, `time_of_day_cos` (derived)

## Preprocessing
- Parsed `EventDateTime` as timestamp.
- Converted glucose to mg/dL and removed non-positive values.
- Clipped extreme basal/bolus/carb values (to avoid outliers).
- Derived glucose trend (mg/dL per minute).
- Derived IOB/COB using exponential decay (DIA=240 min, carb absorption=120 min).
- Derived time-of-day sinusoidal features.

## Licensing & Access
- Access: manual download from Mendeley Data.
- License: CC BY 4.0.

## Known Limitations
- Derived IOB/COB are approximate (not manufacturer IOB).
- Device mode values are sparse and lightly encoded.

## Citation
Khamesian S, Arefeen A, Thompson BM, Grando A, Ghasemzadeh H. AZT1D: A Real-World Dataset for Type 1 Diabetes. Mendeley Data, v1, 2025. doi:10.17632/gk9m674wcx.1
