# Model Card: IINTS-Predictor (AZT1D Real-World)

## Model Details
- **Model name**: IINTS-Predictor
- **Version**: v2 (AZT1D Real-World)
- **Architecture**: LSTM (multi-step sequence-to-horizon)
- **Inputs**: CGM history, glucose trend, insulin/carbs, derived IOB/COB, time-of-day features
- **Outputs**: 60 minute glucose prediction (configurable horizon)

## Intended Use
- **Primary**: Safety Supervisor foresight (proactive hypo prevention).
- **Secondary**: Research benchmarking of control policies.
- **Not** intended for direct insulin dosing without supervision.

## Training Data
- **Dataset**: AZT1D: A Real-World Dataset for Type 1 Diabetes (Mendeley Data).
- **Subset**: CGM Records (Subject CSVs with insulin + carbs).
- **Records**: 288,498 rows across 24 subjects.
- **Time range**: 2023-12-08 to 2024-04-07.
- **Quality report**: `data_packs/public/azt1d/quality_report.json`.

## Evaluation
- Metrics: MSE (training), MAE/RMSE (evaluation).
- **Final val RMSE**: ~20.67 mg/dL (training split).
- **Evaluation subset (60k rows)**: MAE 13.07 mg/dL, RMSE 19.49 mg/dL.
- Compare against heuristic baseline predictor in future runs.

## Ethical Considerations
- Model does **not** replace clinical decision-making.
- All results must be validated against real-world datasets and safety constraints.

## Known Limitations
- IOB/COB are derived with a simplified exponential decay model.
- Predictor is a forecast signal, not a controller.
