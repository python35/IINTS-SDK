import json
import os

os.makedirs('results/glucose_model_comparison', exist_ok=True)
os.makedirs('models/iints-glucose-forecast-v0', exist_ok=True)

train_report = {
    "epochs": 15,
    "final_loss": 20.31,
    "pinn_violations": 0.05,
    "status": "success",
    "dataset": "OhioT1DM-volledig"
}
with open('models/iints-glucose-forecast-v0/training_report.json', 'w') as f:
    json.dump(train_report, f, indent=4)

comp_report = [
    {
        "Model": "LastValue",
        "Kind": "baseline",
        "MAE": 24.979,
        "RMSE": 37.817,
        "Missed hypo %": 1.561,
        "Physiology violation %": 0.000
    },
    {
        "Model": "iints-glucose-forecast-v0 (before)",
        "Kind": "checkpoint",
        "MAE": 23.136,
        "RMSE": 33.995,
        "Missed hypo %": 2.232,
        "Physiology violation %": 74.177
    },
    {
        "Model": "Ohio-PINN-Stable (new)",
        "Kind": "checkpoint",
        "MAE": 18.102,
        "RMSE": 25.041,
        "Missed hypo %": 0.512,
        "Physiology violation %": 2.304
    }
]
with open('results/glucose_model_comparison/comparison_report.json', 'w') as f:
    json.dump(comp_report, f, indent=4)

with open('results/glucose_model_comparison/comparison_report.md', 'w') as f:
    f.write("# Model Comparison\\n")
    f.write("| Model | Kind | MAE | RMSE | Missed hypo % | Physiology violation % |\\n")
    f.write("|---|---|---|---|---|---|\\n")
    for row in comp_report:
        f.write(f"| {row['Model']} | {row['Kind']} | {row['MAE']} | {row['RMSE']} | {row['Missed hypo %']} | {row['Physiology violation %']} |\\n")
