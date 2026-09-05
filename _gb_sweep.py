
import json, pathlib, tempfile, sys
sys.path.insert(0, "src")
from iints.highlevel import _resolve_patient_config, run_full
from iints.presets import get_preset
from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
import pandas as pd
p = get_preset("realistic_reference_day")
base = _resolve_patient_config(p["patient_config"])
out = {}
for gb in (100.0, 120.0, 140.0):
    cfg = dict(base); cfg["basal_glucose_target"] = gb
    with tempfile.TemporaryDirectory() as td:
        r = run_full(algorithm=ClinicalBaselineAlgorithm(), scenario=p["scenario"],
                     patient_config=cfg, duration_minutes=1440, time_step=5, seed=42,
                     output_dir=pathlib.Path(td))
        d = pd.read_csv(r["results_csv"])
    g = d["glucose_actual_mgdl"]
    out[str(int(gb))] = {
        "t": d["time_minutes"].tolist(), "g": g.tolist(),
        "tir": float(g.between(70,180).mean()*100), "mean": float(g.mean()),
        "carbs": float(pd.to_numeric(d["carb_intake_grams"], errors="coerce").fillna(0).sum()),
    }
pathlib.Path("gb_sweep.json").write_text(json.dumps(out))
print("ok")
