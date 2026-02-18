# Model Card: IINTS-Predictor (Template)

## Model Details
- **Model name**: IINTS-Predictor
- **Version**: v1
- **Architecture**: LSTM (multi-step sequence-to-horizon)
- **Inputs**: CGM history, IOB, COB, dynamic ratios (ISF/ICR/Basal), trend
- **Outputs**: 30–120 minute glucose predictions (configurable horizon)

## Intended Use
- **Primary**: Safety Supervisor foresight (proactive hypo prevention).
- **Secondary**: Research benchmarking of control policies.
- **Not** intended for direct insulin dosing without supervision.

## Training Data
- **Phase 1**: Synthetic simulator data (IINTS-AF).
- **Phase 2**: Real-world T1D datasets (e.g., OhioT1DM, Jaeb).

## Evaluation
- Metrics: MAE, RMSE, TIR impact when integrated with Supervisor.
- Compare against heuristic baseline predictor.

## Ethical Considerations
- Model does **not** replace clinical decision-making.
- All results must be validated against real-world datasets and safety constraints.
