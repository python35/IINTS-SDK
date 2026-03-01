# IINTS-AF Notebooks

These notebooks are the fastest way to learn the SDK end‑to‑end. Each notebook now starts with a **Notebook Guide** block that explains prerequisites, outputs, and what you’ll learn.

## Recommended Learning Path
1. `00_Quickstart.ipynb`
   - Open in Colab: https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/00_Quickstart.ipynb
2. `01_Presets_and_Scenarios.ipynb`
   - Open in Colab: https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/01_Presets_and_Scenarios.ipynb
3. `02_Safety_and_Supervisor.ipynb`
   - Open in Colab: https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/02_Safety_and_Supervisor.ipynb
4. `03_Audit_Trail_and_Report.ipynb`
   - Open in Colab: https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/03_Audit_Trail_and_Report.ipynb
5. `04_Baseline_and_Metrics.ipynb`
   - Open in Colab: https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/04_Baseline_and_Metrics.ipynb
6. `05_Devices_and_HumanInLoop.ipynb`
   - Open in Colab: https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/05_Devices_and_HumanInLoop.ipynb
7. `06_Optional_Torch_LSTM.ipynb`
   - Open in Colab: https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/06_Optional_Torch_LSTM.ipynb
8. `07_Ablation_Supervisor.ipynb`
   - Open in Colab: https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/07_Ablation_Supervisor.ipynb
9. `08_Data_Registry_and_Import.ipynb`
   - Open in Colab: https://colab.research.google.com/github/python35/IINTS-SDK/blob/main/examples/notebooks/08_Data_Registry_and_Import.ipynb

## Local Setup (Recommended)
```bash
python3 -m pip install -e .
```
For research notebooks:
```bash
python3 -m pip install -e ".[research]"
```

## Where to Look Next
- Full SDK guide: `docs/COMPREHENSIVE_GUIDE.md`
- CLI reference: `docs/TECHNICAL_README.md`
- Research track: `research/README.md`

## Outputs
Most notebooks write to:
- `results/` (plots, reports)
- `models/` (training outputs)
- `data_packs/` (dataset outputs)
